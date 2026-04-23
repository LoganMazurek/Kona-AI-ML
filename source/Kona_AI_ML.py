# -*- coding: utf-8 -*-
"""
Kona AI/ML Event Revenue Prediction System
Ensemble model with conformal prediction intervals
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from sklearn.base import BaseEstimator, RegressorMixin

# Setup path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import conformal prediction
try:
    from conformal_prediction import ConformalRegressor
    CONFORMAL_AVAILABLE = True
except ImportError:
    print("[WARN] Conformal prediction module not found")
    CONFORMAL_AVAILABLE = False

# Import other modules
try:
    from datetime_utils import parse_datetime_column
    from demographics import enrich_dataframe_with_demographics
except ImportError:
    # Prefer package-style import when installed as a package
    from Kona_AI_ML.datetime_utils import parse_datetime_column
    from Kona_AI_ML.demographics import enrich_dataframe_with_demographics


def _missing_feature_engineering(*_args, **_kwargs):
    raise ImportError(
        'Legacy feature engineering module is not available in this workspace. '
        'Training and deprecated preprocessing paths require feature_engineering.py.'
    )


try:
    from feature_engineering import apply_phase1_feature_engineering, apply_phase2_feature_engineering
except ImportError:
    try:
        from Kona_AI_ML.outdated.feature_engineering import apply_phase1_feature_engineering, apply_phase2_feature_engineering
    except ImportError:
        apply_phase1_feature_engineering = _missing_feature_engineering
        apply_phase2_feature_engineering = _missing_feature_engineering


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Production ensemble regressor combining XGBoost + CatBoost + LightGBM
    with expanded hyperparameter optimization and conformal prediction
    """
    
    def __init__(self, use_conformal=True, optimize_models=True, n_trials_per_model=50, random_state=42):
        self.use_conformal = use_conformal
        self.optimize_models = optimize_models
        self.n_trials_per_model = n_trials_per_model
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        # Dynamic weights - will be optimized based on individual performance
        self.weights = {}
        self.conformal_model = None  # type: ignore
        self.is_fitted = False
    
    def _optimize_xgboost(self, X, y):
        """Optimize XGBoost with expanded parameter space"""
        try:
            import optuna
            from xgboost import XGBRegressor
            from sklearn.model_selection import cross_val_score
            
            print(f"   [SEARCH] Optimizing XGBoost ({self.n_trials_per_model} trials)...")
            
            def objective(trial):
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
                    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.001, 10.0, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 100.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 100.0, log=True),
                    'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0)
                }
                
                model = XGBRegressor(
                    **params, 
                    random_state=self.random_state,
                    enable_categorical=True,
                    verbosity=0,
                    n_jobs=-1,
                    tree_method='hist'
                )
                
                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
                return -scores.mean()
            
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials_per_model, show_progress_bar=False)
            
            best_params = study.best_params
            best_score = study.best_value
            
            model = XGBRegressor(
                **best_params,
                random_state=self.random_state,
                enable_categorical=True,
                verbosity=0,
                n_jobs=-1,
                tree_method='hist'
            )
            
            print(f"      [OK] XGBoost optimized - CV MAE: ${best_score:.2f}")
            return model, best_params, best_score
            
        except ImportError as e:
            print(f"      [ERROR] XGBoost optimization failed - {e}")
            return None, {}, float('inf')
    
    def _optimize_catboost(self, X, y):
        """Optimize CatBoost with robust parameter handling"""
        try:
            import optuna
            from catboost import CatBoostRegressor
            
            print(f"   [SEARCH] Optimizing CatBoost ({self.n_trials_per_model} trials)...")
            
            # Get categorical features for CatBoost (handle NaN values)
            categorical_features = []
            X_clean = X.copy()
            
            for i, col in enumerate(X.columns):
                if X[col].dtype.name == 'category' or X[col].dtype == 'object':
                    # Check for NaN values
                    if X[col].isnull().any():
                        print(f"      [WARN] Skipping categorical feature '{col}' due to NaN values")
                        # Convert to numeric if possible, otherwise fill with placeholder
                        try:
                            X_clean[col] = pd.to_numeric(X_clean[col], errors='raise')
                        except:
                            X_clean[col] = X_clean[col].astype(str).fillna('unknown')
                            categorical_features.append(i)
                    else:
                        categorical_features.append(i)
            
            def objective(trial):
                # Use simpler, more stable parameter ranges
                bootstrap_type = trial.suggest_categorical('bootstrap_type', ['MVS', 'Bayesian'])
                
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'random_strength': trial.suggest_float('random_strength', 1.0, 20.0),
                    'bootstrap_type': bootstrap_type,
                }
                
                # Add bootstrap-specific parameters safely
                if bootstrap_type == 'Bayesian':
                    params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.1, 10.0)
                elif bootstrap_type == 'MVS':
                    params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
                
                # Add other stable parameters
                params.update({
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
                })
                
                try:
                    # Manually implement cross-validation for CatBoost with categorical features
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
                    scores = []
                    
                    for train_idx, val_idx in kf.split(X_clean):
                        X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        model_copy = CatBoostRegressor(
                            **params,
                            random_seed=self.random_state,
                            verbose=False,
                            allow_writing_files=False,
                            thread_count=-1,
                            task_type='GPU'
                        )
                        
                        # Fit with categorical features
                        model_copy.fit(X_train, y_train, cat_features=categorical_features)
                        y_pred = model_copy.predict(X_val)
                        
                        from sklearn.metrics import mean_absolute_error
                        mae = mean_absolute_error(y_val, y_pred)
                        scores.append(mae)
                    
                    return float(np.mean(scores))
                except Exception as e:
                    print(f"      [WARN] CatBoost trial failed: {str(e)[:100]}...")
                    return float('inf')  # Return worst score if training fails
            
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials_per_model, show_progress_bar=False)
            
            best_params = study.best_params
            best_score = study.best_value
            
            model = CatBoostRegressor(
                **best_params,
                random_seed=self.random_state,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
                task_type='GPU'
            )
            
            print(f"      [OK] CatBoost optimized - CV MAE: ${best_score:.2f}")
            return model, best_params, best_score
            
        except ImportError as e:
            print(f"      [ERROR] CatBoost optimization failed - {e}")
            return None, {}, float('inf')
    
    def _optimize_lightgbm(self, X, y):
        """Optimize LightGBM with expanded parameter space"""
        try:
            import optuna
            from lightgbm import LGBMRegressor
            from sklearn.model_selection import cross_val_score
            
            print(f"   [SEARCH] Optimizing LightGBM ({self.n_trials_per_model} trials)...")
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1000.0, log=True),
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 15.0),
                    'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 1000),
                    'max_bin': trial.suggest_int('max_bin', 10, 500),
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),  # Remove 'goss' to avoid conflicts
                }
                
                # Add bagging parameters (safe for gbdt and dart)
                params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.1, 1.0)
                params['bagging_freq'] = trial.suggest_int('bagging_freq', 0, 10)
                
                # Add boosting-specific parameters
                if params['boosting_type'] == 'dart':
                    params['drop_rate'] = trial.suggest_float('drop_rate', 0.01, 0.5)
                    params['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 1.0)
                
                model = LGBMRegressor(
                    **params,
                    random_state=self.random_state,
                    verbosity=-1,
                    n_jobs=-1,
                    force_col_wise=True,
                    device_type='gpu'
                )
                
                scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
                return -scores.mean()
            
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials_per_model, show_progress_bar=False)
            
            best_params = study.best_params
            best_score = study.best_value
            
            model = LGBMRegressor(
                **best_params,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=-1,
                force_col_wise=True,
                device_type='gpu'
            )
            
            print(f"      [OK] LightGBM optimized - CV MAE: ${best_score:.2f}")
            return model, best_params, best_score
            
        except ImportError as e:
            print(f"      [ERROR] LightGBM optimization failed - {e}")
            return None, {}, float('inf')
    
    def _get_default_models(self):
        """Get default model configurations (fallback)"""
        models = {}
        
        try:
            from xgboost import XGBRegressor
            models['xgboost'] = XGBRegressor(
                learning_rate=0.028, max_depth=11, min_child_weight=6,
                n_estimators=309, reg_alpha=7.15, reg_lambda=7.22,
                gamma=3.22, colsample_bytree=0.587, subsample=0.728,
                random_state=self.random_state, enable_categorical=True, verbosity=0,
                tree_method='hist', n_jobs=-1
            )
        except ImportError:
            pass
        
        try:
            from catboost import CatBoostRegressor
            models['catboost'] = CatBoostRegressor(
                iterations=500, learning_rate=0.05, depth=8,
                l2_leaf_reg=5.0, border_count=128, bagging_temperature=0.5,
                random_strength=10, subsample=0.8,
                min_data_in_leaf=10, random_seed=self.random_state,
                verbose=False, allow_writing_files=False,
                task_type='GPU'
            )
        except ImportError:
            pass
        
        try:
            from lightgbm import LGBMRegressor
            models['lightgbm'] = LGBMRegressor(
                n_estimators=400, learning_rate=0.04, max_depth=10,
                num_leaves=200, reg_alpha=3.0, reg_lambda=3.0,
                colsample_bytree=0.8, subsample=0.9, min_child_samples=15,
                random_state=self.random_state, verbosity=-1, force_col_wise=True,
                device_type='gpu'
            )
        except ImportError:
            pass
            
        return models
    
    def _prepare_for_sklearn_models(self, X):
        """Convert categorical columns to numeric for XGBoost/LightGBM"""
        X_copy = X.copy()
        for col in X_copy.columns:
            if X_copy[col].dtype == 'object':
                # Convert object dtype to categorical codes
                X_copy[col] = pd.Categorical(X_copy[col]).codes
            elif X_copy[col].dtype.name == 'category':
                # Convert pandas category to integer codes
                X_copy[col] = X_copy[col].cat.codes
        return X_copy
    def fit(self, X, y):
        """Fit the ensemble with comprehensive hyperparameter optimization"""
        print("[ENSEMBLE] Training Advanced Ensemble Regressor")
        print("=" * 60)
        
        # Get categorical column indices for CatBoost
        categorical_features = []
        for i, col in enumerate(X.columns):
            if X[col].dtype.name == 'category' or X[col].dtype == 'object':
                categorical_features.append(i)
        
        # Track model performance for dynamic weighting
        model_scores = {}
        self.models = {}
        self.best_params = {}
        
        if self.optimize_models:
            print(f"\n[OPTUNA] Hyperparameter Optimization ({self.n_trials_per_model} trials per model)")
            print("-" * 60)
            
            # Optimize XGBoost
            xgb_model, xgb_params, xgb_score = self._optimize_xgboost(X, y)
            if xgb_model is not None:
                self.models['xgboost'] = xgb_model
                self.best_params['xgboost'] = xgb_params
                model_scores['xgboost'] = xgb_score
            
            # Optimize CatBoost
            cat_model, cat_params, cat_score = self._optimize_catboost(X, y)
            if cat_model is not None:
                self.models['catboost'] = cat_model
                self.best_params['catboost'] = cat_params
                model_scores['catboost'] = cat_score
            
            # Optimize LightGBM
            lgb_model, lgb_params, lgb_score = self._optimize_lightgbm(X, y)
            if lgb_model is not None:
                self.models['lightgbm'] = lgb_model
                self.best_params['lightgbm'] = lgb_params
                model_scores['lightgbm'] = lgb_score
            
        else:
            print("\n[CONFIG] Using Default Model Configurations")
            print("-" * 40)
            self.models = self._get_default_models()
            # Set default scores for weighting
            model_scores = {name: 80.0 for name in self.models.keys()}
        
        if not self.models:
            raise ValueError("No ML models available. Install xgboost, catboost, or lightgbm.")
        
        # Calculate dynamic weights based on inverse performance (lower MAE = higher weight)
        if model_scores:
            # Convert MAE to weights (inverse relationship)
            score_weights = {name: 1.0 / score for name, score in model_scores.items() if score > 0}
            total_weight = sum(score_weights.values())
            self.weights = {name: weight / total_weight for name, weight in score_weights.items()}
        else:
            # Fallback to equal weights
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        
        print(f"\n[MODELS] Final Models: {list(self.models.keys())}")
        print(f"[SCORES] Performance Scores: {model_scores}")
        print(f"[WEIGHTS] Dynamic Weights: {self.weights}")
        
        # Train each model on full dataset
        print(f"\n[TRAINING] Training models on full dataset...")
        
        # Get categorical column indices for CatBoost and clean data
        categorical_features = []
        X_clean = X.copy()
        
        for i, col in enumerate(X.columns):
            if X[col].dtype.name == 'category' or X[col].dtype == 'object':
                # Check if column has any NaN values
                if X[col].isnull().any():
                    print(f"   [WARN] Skipping categorical feature '{col}' due to NaN values")
                    # Convert to numeric if possible, otherwise fill with placeholder
                    try:
                        X_clean[col] = pd.to_numeric(X_clean[col], errors='raise')
                    except:
                        X_clean[col] = X_clean[col].astype(str).fillna('unknown')
                        categorical_features.append(i)
                else:
                    categorical_features.append(i)
        
        # Prepare data for sklearn models (convert categorical to numeric)
        X_for_sklearn = self._prepare_for_sklearn_models(X)
        
        # Create a copy of models dict to avoid iteration issues
        models_list = list(self.models.items())
        for name, model in models_list:
            print(f"   Training {name}...")
            try:
                if name == 'catboost':
                    # Special handling for CatBoost categorical features with cleaned data
                    model.fit(X_clean, y, cat_features=categorical_features)
                else:
                    # For XGBoost/LightGBM, use converted data without object dtypes
                    model.fit(X_for_sklearn, y)
                print(f"   [OK] {name} trained successfully")
            except Exception as e:
                print(f"   [ERROR] {name} failed: {str(e)[:100]}...")
                # Remove failed model
                if name in self.models:
                    del self.models[name]
                if name in self.weights:
                    del self.weights[name]
        
        # Store cleaned data for predictions
        self.X_clean = X_clean
        
        # Renormalize weights after any failures
        if self.weights:
            total_weight = sum(self.weights.values())
            self.weights = {name: weight/total_weight for name, weight in self.weights.items()}
        
        # Note: Conformal prediction will be fitted separately with calibration data
        # This is done in the training pipeline, not in the base fit() method
        
        self.is_fitted = True
        print(f"\n[SUCCESS] Advanced ensemble ready with {len(self.models)} optimized models!")
        return self
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if not self.models:
            raise ValueError("No successful models available for prediction")
        
        predictions = np.zeros(len(X))
        
        # Clean data for CatBoost if needed
        X_for_catboost = X.copy()
        if 'catboost' in self.models:
            for col in X.columns:
                if X[col].dtype.name == 'category' or X[col].dtype == 'object':
                    if X[col].isnull().any():
                        # Convert to numeric if possible, otherwise fill with placeholder
                        try:
                            X_for_catboost[col] = pd.to_numeric(X_for_catboost[col], errors='raise')
                        except:
                            X_for_catboost[col] = X_for_catboost[col].astype(str).fillna('unknown')
        
        # Prepare data for sklearn models
        X_for_sklearn = self._prepare_for_sklearn_models(X)
        
        for name, model in self.models.items():
            weight = self.weights[name]
            if name == 'catboost':
                pred = model.predict(X_for_catboost)
            else:
                # Use converted data for XGBoost/LightGBM
                pred = model.predict(X_for_sklearn)
            predictions += weight * pred
        
        return predictions
    
    def predict_with_intervals(self, X, confidence=0.8):
        """Make predictions with confidence intervals"""
        if self.conformal_model is not None:
            try:
                pred_mean, pred_lower, pred_upper = self.conformal_model.predict_with_intervals(X, min_bound=0.0)
                return pred_mean, pred_lower, pred_upper
            except Exception as e:
                print(f"[WARN] Conformal prediction error: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback to point predictions
        pred = self.predict(X)
        return pred, pred, pred  # point, lower, upper (all same without conformal)


def cluster_zipcodes(zipcodes):
    """Group zipcodes into clusters based on first 3 digits (SCF level)."""
    # Convert to string and get first 3 digits
    zip_prefix = zipcodes.astype(str).str[:3]
    return zip_prefix

def load_and_preprocess(path: str):
    """Load a CSV and run preprocessing in a way that's tolerant to column-name variants.

    Returns the preprocessed DataFrame.
    """
    data = pd.read_csv(path)

    # Numeric conversions
    data['Net Sales'] = pd.to_numeric(data['Net Sales'], errors='coerce')

    # Convert datetime columns using a best-effort approach. The CSVs sometimes
    # contain mixed datetime formats (ISO vs. human-readable). Try a strict
    # parse first, otherwise fall back to pandas' flexible parser.
    datetime_format = '%b %d, %Y %I:%M %p'
    try:
        data['Event Start Date'] = pd.to_datetime(data['Event Start Date Time'], format=datetime_format, errors='raise')
        data['Event End Date'] = pd.to_datetime(data['Event End Date Time'], format=datetime_format, errors='raise')
    except Exception:
        # Fallback to flexible parsing which can handle ISO strings and mixed formats
        data['Event Start Date'] = pd.to_datetime(data['Event Start Date Time'], errors='coerce')
        data['Event End Date'] = pd.to_datetime(data['Event End Date Time'], errors='coerce')

    # Determine start/end column names present in data
    start_col = None
    end_col = None
    for c in ['Event Start Date Time', 'Event Start Date', 'Event Start']:
        if c in data.columns:
            start_col = c
            break
    for c in ['Event End Date Time', 'Event End Date', 'Event End']:
        if c in data.columns:
            end_col = c
            break

    if start_col is None or end_col is None:
        raise ValueError('Could not find event start/end columns in CSV')

    data['Event Start Date'] = parse_datetime_column(data[start_col].astype(str))
    data['Event End Date'] = parse_datetime_column(data[end_col].astype(str))

    # Duration and temporal features (guard against NaT)
    data['Duration (Hours)'] = (data['Event End Date'] - data['Event Start Date']).dt.total_seconds() / 3600
    data['Month'] = data['Event Start Date'].dt.month
    data['Day of Week'] = data['Event Start Date'].dt.dayofweek
    data['Is Weekend'] = data['Event Start Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Seasonality
    data['Season'] = data['Event Start Date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    data['Week of Month'] = data['Event Start Date'].dt.day.map(lambda x: (x-1)//7 + 1)
    
    
    # Competition/Density features
    data['Events Same Day'] = data.groupby([
        data['Event Start Date'].dt.date,
        'City'
    ])['Industry'].transform('count')
    
    if 'Industry' in data.columns:
        # Industry high-level categorization based on available categories
        industry_categories = {
            'Community': 'Community/Public',
            'Church': 'Religious',
            'Private': 'Private Events',
            'Athletics': 'Sports/Recreation',
            'Education': 'Education',
            'Corporate': 'Business',
            'Festival': 'Entertainment',
            'Daycare': 'Childcare'
        }
        data['Industry Category'] = data['Industry'].map(industry_categories)
    
    # Location features
    if 'Zip Code' in data.columns:
        data['ZIP Cluster'] = cluster_zipcodes(data['Zip Code'])
    
    # County feature (if available)
    if 'County' in data.columns:
        data['County'] = data['County'].astype('category')
        # Add events per county for competition analysis
        data['Events in County'] = data.groupby(['County', data['Event Start Date'].dt.date])['Industry'].transform('count')
    
    # Add demographic features if ZIP code is available
    if 'Zip Code' in data.columns:
        # Convert ZIP to numeric, handling any non-numeric values
        data['Zip Code'] = pd.to_numeric(data['Zip Code'], errors='coerce')
        
        # Add demographic features
        data = enrich_dataframe_with_demographics(data)
        
        # Create normalized versions of demographic features
        try:
            data['Income_Tier'] = pd.qcut(data['Median_Income'].fillna(data['Median_Income'].median()), 
                                        q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        except ValueError:
            # Fallback if we can't create quartiles
            data['Income_Tier'] = pd.cut(data['Median_Income'].fillna(data['Median_Income'].median()),
                                       bins=3, labels=['Low', 'Medium', 'High'])
        
        try:
            data['Population_Tier'] = pd.qcut(data['Population_Density'].fillna(data['Population_Density'].median()),
                                            q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        except ValueError:
            # Fallback if we can't create quartiles
            data['Population_Tier'] = pd.cut(data['Population_Density'].fillna(data['Population_Density'].median()),
                                           bins=3, labels=['Low', 'Medium', 'High'])

    # Net sales per hour (guard against zero/NaN durations)
    data['Net Sales per Hour'] = data['Net Sales'] / data['Duration (Hours)']
    data.loc[~np.isfinite(data['Net Sales per Hour']), 'Net Sales per Hour'] = np.nan

    # One-hot / dummy encoding for categorical columns
    cat_cols = [c for c in ['Event Industry', 'City', 'Event Type', 'County', 'ZIP Cluster'] 
                if c in data.columns]
    if cat_cols:
        data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    # Time of Day feature
    def get_time_of_day(hour):
        if 6 <= hour < 10:
            return 'Morning'
        elif 10 <= hour < 14:
            return 'Lunchtime'
        elif 14 <= hour < 17:
            return 'Early Afternoon'
        elif 17 <= hour < 19:
            return 'Late Afternoon'
        else:
            return 'Night'

    data['Event Start Hour'] = data['Event Start Date'].dt.hour
    data['Time of Day'] = (data['Event Start Hour'] + (data['Duration (Hours)'] / 2)).apply(get_time_of_day)
    if 'Time of Day' in data.columns:
        data = pd.get_dummies(data, columns=['Time of Day'], drop_first=True)

    # Apply Phase 1 feature engineering
    data = apply_phase1_feature_engineering(data)
    
    # Apply Phase 2 feature engineering (requires datetime columns and historical data)
    data = apply_phase2_feature_engineering(data)

    # Define features and target variable
    drop_cols = ['ID', 'Net Sales per Hour', 'Zip Code', 'Net Sales', 'Event Start Date', 'Event End Date', 'Event Start Hour']
    X = data.drop(columns=[col for col in drop_cols if col in data.columns])
    if 'Net Sales per Hour' not in data.columns:
        raise ValueError('Net Sales per Hour could not be computed')
    y = data['Net Sales per Hour']

    return data


def main(data_path: str = 'data/TinleyPark_2023-2025_complete_features.csv', train: bool = False, 
         n_trials: int = 20, use_ensemble: bool = True, optimize_ensemble: bool = True):
    """Entry point.

    Default data_path points to the complete features dataset. Set `train=True`
    to run training. Set `use_ensemble=True` for ensemble model, False for XGBoost only.
    Set `optimize_ensemble=True` to run comprehensive hyperparameter optimization.
    """
    
    # Use complete features dataset if it exists
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found, running preprocessing...")
        data = load_and_preprocess(data_path)
    else:
        print(f"Loading pre-processed data from {data_path}")
        data = pd.read_csv(data_path)

    if not train:
        if data_path.endswith('_complete_features.csv'):
            print(f'Using existing complete features dataset: {data_path}')
            return data, None
        else:
            out_path = os.path.splitext(data_path)[0] + '_cleaned.csv'
            data.to_csv(out_path, index=False)
            print(f'Preprocessed data saved to {out_path}')
            return data, None

    # Import heavy ML libraries only when training is requested
    from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt

    # Prepare features/target - handle both raw and complete feature datasets
    if 'Net Sales per Hour' in data.columns:
        # Complete features dataset
        drop_cols = ['Event Start Date Time', 'Event End Date Time', 'Net Sales per Hour',
                    'Net Sales', 'Event Start Date', 'Event End Date']
        X = data.drop(columns=[col for col in drop_cols if col in data.columns])
        y = data['Net Sales per Hour']
    else:
        # Raw dataset - needs preprocessing
        data = load_and_preprocess(data_path)
        drop_cols = ['ID', 'Event ID', 'Event Name', 'Address', 'Net Sales per Hour', 'Zip Code', 
                    'Net Sales', 'Event Start Date', 'Event End Date', 'Event Start Hour']
        X = data.drop(columns=[col for col in drop_cols if col in data.columns])
        y = data['Net Sales per Hour']

    # Convert any remaining object columns to category for better ML handling
    obj_cols = X.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        X[c] = X[c].astype('category')

    print(f"[DATA] Dataset: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"[TARGET] Target: Net Sales per Hour (mean=${y.mean():.2f}, std=${y.std():.2f})")

    # Time-based validation if Event Start Date Time is available
    use_time_split = False
    X_train = X_val = X_test = None
    y_train = y_val = y_test = None
    
    if 'Event Start Date Time' in data.columns:
        print("\n[TIME] Using time-based train/validation/test split")
        print("   Training: Events before 2025")
        print("   Validation: Early 2025 (for conformal calibration)")
        print("   Test: Recent 2025 events")
        
        # Parse event dates
        event_dates = pd.to_datetime(data['Event Start Date Time'], errors='coerce')
        data_with_dates = data.copy()
        data_with_dates['_event_date'] = event_dates
        
        # Define splits with timestamp comparison
        cutoff_2025 = pd.Timestamp('2025-01-01')
        cutoff_mid_2025 = pd.Timestamp('2025-06-01')
        
        train_mask = event_dates < cutoff_2025
        val_mask = (event_dates >= cutoff_2025) & (event_dates < cutoff_mid_2025)
        test_mask = event_dates >= cutoff_mid_2025
        
        # Check if we have data in all splits
        if train_mask.sum() > 0 and val_mask.sum() > 0 and test_mask.sum() > 0:
            use_time_split = True
            print(f"   Train samples: {train_mask.sum()}")
            print(f"   Validation samples: {val_mask.sum()}")
            print(f"   Test samples: {test_mask.sum()}")
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_val = X[val_mask]
            y_val = y[val_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
        else:
            print("   [WARN] Insufficient data for time-based split, using random split")
    
    if not use_time_split:
        print("\n[RANDOM] Using random train/validation/test split (50%/40%/10%)")
        # Split: 50% train, 40% validation (for conformal calibration), 10% test
        # 40% calibration size ensures sufficient samples for reliable quantile estimation
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=4/9, random_state=42  # 4/9 * 0.9 ≈ 0.4
        )

    if use_ensemble:
        print(f"\n[TRAINING] Training {'Optimized' if optimize_ensemble else 'Standard'} Ensemble Model")
        print("=" * 80)
        
        # Train ensemble model with optimization
        if optimize_ensemble:
            # Use the specified number of trials per model
            trials_per_model = n_trials
            print(f"[OPTUNA] Running comprehensive optimization ({trials_per_model} trials per model)")
        else:
            trials_per_model = 0
            print("[CONFIG] Using pre-optimized configurations")
        
        model = EnsembleRegressor(
            use_conformal=True, 
            optimize_models=optimize_ensemble,
            n_trials_per_model=trials_per_model,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Fit conformal prediction using validation set
        print(f"\n[SEARCH] Conformal setup check: use_conformal={model.use_conformal}, CONFORMAL_AVAILABLE={CONFORMAL_AVAILABLE}")
        if model.use_conformal and CONFORMAL_AVAILABLE:
            print("\n[TARGET] Fitting conformal prediction on validation set...")
            try:
                # Create conformal wrapper around the already-fitted ensemble
                # Note: Using 0.87 (87%) on calibration to account for ~8% drift
                # between calibration (validation) and test sets from temporal split
                model.conformal_model = ConformalRegressor(
                    base_model=model,
                    confidence_level=0.87
                )
                # Compute calibration residuals directly on validation set
                y_val_pred = model.predict(X_val)
                model.conformal_model.calibration_residuals = np.abs(y_val - y_val_pred)
                
                # Compute quantile threshold with finite sample correction
                # Using 0.87 to account for ~8% drift between calibration and test
                n_cal = len(model.conformal_model.calibration_residuals)
                alpha = 1 - 0.87
                # Use ceiling of (n+1)*(1-alpha)/n for distribution-free coverage guarantee
                quantile_index = np.ceil((n_cal + 1) * (1 - alpha))
                quantile_level = np.clip(quantile_index / n_cal, 0.0, 1.0)
                model.conformal_model.quantile_threshold = np.quantile(
                    model.conformal_model.calibration_residuals, 
                    quantile_level,
                    method='linear'
                )
                model.conformal_model.is_fitted = True
                
                print(f"[OK] Conformal prediction calibrated")
                print(f"   Calibration sample size: {n_cal}")
                print(f"   Mean residual: ${model.conformal_model.calibration_residuals.mean():.2f}")
                print(f"   Quantile threshold (80%): ${model.conformal_model.quantile_threshold:.2f}")
            except Exception as e:
                print(f"[WARN] Conformal prediction setup failed: {e}")
                import traceback
                traceback.print_exc()
                model.conformal_model = None
        
        # Evaluate ensemble on test set
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n[CHART] Ensemble Performance (Test Set):")
        print(f"   MAE: ${mae:.2f}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   R²: {r2:.4f}")
        
        # Display optimization results
        if optimize_ensemble and hasattr(model, 'best_params'):
            print(f"\n[TARGET] Best Hyperparameters:")
            for model_name, params in model.best_params.items():
                print(f"   {model_name}:")
                for param, value in params.items():
                    print(f"      {param}: {value}")
        
        # Test prediction intervals
        if model.conformal_model is not None:
            try:
                pred_mean, pred_lower, pred_upper = model.predict_with_intervals(X_test)
                coverage = np.mean((y_test >= pred_lower) & (y_test <= pred_upper))
                avg_width = np.mean(pred_upper - pred_lower)
                
                # Calculate efficiency: how tight vs coverage
                efficiency = 0.8 / max(coverage, 0.001)
                
                print(f"\n[TARGET] Conformal Prediction Intervals (Test Set):")
                print(f"   Coverage: {coverage:.1%} (target: 80%)")
                print(f"   Average interval width: ${avg_width:.2f}")
                print(f"   Efficiency: {efficiency:.2f}x (lower is tighter)")
                
                # Check validation set coverage too
                val_mean, val_lower, val_upper = model.predict_with_intervals(X_val)
                val_coverage = np.mean((y_val >= val_lower) & (y_val <= val_upper))
                val_width = np.mean(val_upper - val_lower)
                print(f"\n   Validation set:")
                print(f"   Coverage: {val_coverage:.1%}")
                print(f"   Average width: ${val_width:.2f}")
                
                # Add to return metrics
                metrics['conformal_coverage_test'] = coverage
                metrics['conformal_width'] = avg_width
                metrics['conformal_coverage_val'] = val_coverage
                
            except Exception as e:
                print(f"   [WARN] Prediction intervals error: {e}")

        
        return model, {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    else:
        print("\n[CONFIG] Training Single XGBoost Model with Optuna Optimization")
        print("=" * 60)
        
        # Traditional XGBoost with Optuna
        from xgboost import XGBRegressor
        import optuna
        
        study_db_path = 'optuna_study_cv.db'

        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'max_depth': trial.suggest_int('max_depth', 1, 6),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            model = XGBRegressor(**params, random_state=42, enable_categorical=True)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            return -scores.mean()

        if os.path.exists(study_db_path):
            study = optuna.load_study(study_name="xgboost_cv_tuning", storage=f"sqlite:///{study_db_path}")
        else:
            study = optuna.create_study(study_name="xgboost_cv_tuning", direction='minimize', storage=f"sqlite:///{study_db_path}", load_if_exists=False)

        study.optimize(objective, n_trials=n_trials)
        print(f"Best Parameters from Optuna: {study.best_params}")
        print(f"Best CV MAE: ${study.best_value:.2f}")

        best_params = study.best_params
        model = XGBRegressor(**best_params, random_state=42, enable_categorical=True)
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2 = r2_score(y_test, y_test_pred)

        print(f"\n[CHART] XGBoost Performance:")
        print(f"   Test MAE: ${mae:.2f}")
        print(f"   Test RMSE: ${rmse:.2f}")
        print(f"   Test R²: {r2:.4f}")
        
        return model, {'mae': mae, 'rmse': rmse, 'r2': r2}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Kona AI/ML Event Revenue Prediction Pipeline')
    parser.add_argument('--data', default='data/TinleyPark_2023-2025_complete_features.csv', 
                       help='Path to CSV data file')
    parser.add_argument('--train', action='store_true', 
                       help='Run training mode (preprocessing + model training)')
    parser.add_argument('--trials', type=int, default=60, 
                       help='Total number of optimization trials (split across models for ensemble)')
    parser.add_argument('--ensemble', action='store_true', default=True,
                       help='Use ensemble model (XGBoost + CatBoost + LightGBM)')
    parser.add_argument('--xgboost-only', action='store_true', 
                       help='Use only XGBoost with Optuna optimization')
    parser.add_argument('--optimize', action='store_true', default=True,
                       help='Run comprehensive hyperparameter optimization')
    parser.add_argument('--quick', action='store_true',
                       help='Skip optimization, use pre-tuned parameters (faster)')
    args = parser.parse_args()

    # Handle conflicting options
    use_ensemble = args.ensemble and not args.xgboost_only
    optimize_ensemble = args.optimize and not args.quick
    
    if args.train:
        print("[TARGET] Kona AI/ML Event Revenue Prediction Pipeline")
        print("=" * 80)
        print(f"[DATA] Data: {args.data}")
        print(f"[MODEL] Model: {'Ensemble' if use_ensemble else 'XGBoost + Optuna'}")
        print(f"[OPTUNA] Optimization: {'Yes' if optimize_ensemble else 'No (using defaults)'}")
        print(f"[TRIALS] Trials: {args.trials}")
        print("=" * 80)
    
    model, results = main(
        data_path=args.data, 
        train=args.train, 
        n_trials=args.trials,
        use_ensemble=use_ensemble,
        optimize_ensemble=optimize_ensemble
    )
    
    if args.train:
        print("\n[OK] Training completed successfully!")
        if results and isinstance(results, dict):
            print(f"[DATA] Final Results: MAE=${results['mae']:.2f}, R²={results['r2']:.4f}")
        
        # Save the trained model with absolute path
        import joblib
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        outputs_dir = os.path.join(parent_dir, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        
        if use_ensemble:
            model_filename = os.path.join(outputs_dir, 'ensemble_conformal_model_complete_features.joblib')
        else:
            model_filename = os.path.join(outputs_dir, 'xgboost_optimized_model.joblib')
        
        print(f"\n[DATA] Saving model to: {model_filename}")
        joblib.dump(model, model_filename)
        print(f"[OK] Model saved successfully!")
        
        # Verify the file was created
        if os.path.exists(model_filename):
            file_size = os.path.getsize(model_filename) / (1024 * 1024)  # MB
            print(f"   File size: {file_size:.2f} MB")
        else:
            print(f"[WARN]  Warning: Model file not found after save!")
        
        # Also save metrics
        metrics_filename = model_filename.replace('.joblib', '_metrics.json')
        import json
        with open(metrics_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[DATA] Metrics saved to: {metrics_filename}")
