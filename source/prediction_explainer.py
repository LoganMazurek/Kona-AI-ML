"""
Advanced prediction explanation system using SHAP and per-prediction feature contributions.
Provides substantial, interpretable explanations for individual predictions.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any
import joblib

warnings.filterwarnings('ignore')


class PredictionExplainer:
    """
    Advanced explainer for ensemble predictions using SHAP and feature contribution analysis.
    Provides per-prediction explanations rather than just global importance.
    """
    
    def __init__(self, model, X_background: Optional[pd.DataFrame] = None, use_shap: bool = True):
        """
        Initialize the explainer.
        
        Args:
            model: Trained ensemble model (EnsembleRegressor)
            X_background: Background data for SHAP (sample of training data for reference)
            use_shap: Whether to use SHAP explanations (requires shap package)
        """
        self.model = model
        self.use_shap = use_shap
        self.explainers = {}
        self.X_background = X_background
        self._shap_available = False
        
        # Try to initialize SHAP explainers if available
        if use_shap:
            try:
                import shap
                self._shap_available = True
                self._initialize_shap_explainers()
            except ImportError:
                print("[INFO] SHAP not installed. Install with: pip install shap")
                self._shap_available = False
    
    def _initialize_shap_explainers(self):
        """Initialize SHAP explainers for each model in the ensemble."""
        try:
            import shap
            
            # Check if model has individual models
            model_dict = None
            if hasattr(self.model, 'models'):
                model_dict = self.model.models
            elif hasattr(self.model, 'merged_models'):
                model_dict = self.model.merged_models

            if model_dict:
                for name, sub_model in model_dict.items():
                    try:
                        # Use TreeExplainer for tree-based models
                        if name.lower() in ['xgboost', 'catboost', 'lightgbm', 'xgb', 'cat', 'lgb']:
                            if self.X_background is not None:
                                # Use small sample for efficiency
                                X_sample = self.X_background.sample(
                                    n=min(100, len(self.X_background)),
                                    random_state=42
                                )
                                self.explainers[name] = shap.TreeExplainer(sub_model)
                            else:
                                self.explainers[name] = shap.TreeExplainer(sub_model)
                    except Exception as e:
                        print(f"[WARN] Could not initialize SHAP for {name}: {e}")
        except Exception as e:
            print(f"[WARN] SHAP initialization failed: {e}")
            self._shap_available = False
    
    def explain_prediction(self, X: pd.DataFrame, prediction: float, 
                          feature_names: Optional[List[str]] = None,
                          top_k: int = 10) -> Dict[str, Any]:
        """
        Explain a single prediction with multiple methods.
        
        Args:
            X: Feature vector as DataFrame (single row)
            prediction: The model's prediction
            feature_names: Optional list of feature names for better output
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation data including SHAP values, feature contributions,
            confidence metrics, and human-readable explanations.
        """
        explanation = {
            'prediction': float(prediction),
            'top_features': [],
            'feature_contributions': [],
            'model_agreements': {},
            'confidence_indicators': {},
            'explanations': []
        }
        
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame([X])
        
        if len(X) > 1:
            X = X.iloc[[0]]  # Take only first row
        
        # 1. Get SHAP values if available
        if self._shap_available:
            shap_explanation = self._get_shap_explanation(X, top_k)
            explanation['shap_values'] = shap_explanation
        
        # 2. Get per-model contributions
        model_contributions = self._get_model_contributions(X, top_k)
        explanation['model_agreements'] = model_contributions
        
        # 3. Calculate feature contributions (permutation-based)
        feature_contrib = self._get_feature_contributions(X, prediction, top_k)
        explanation['feature_contributions'] = feature_contrib['contributions']
        explanation['confidence_indicators'] = feature_contrib['confidence']
        
        # 4. Generate human-readable explanations
        explanations = self._generate_explanations(X, explanation)
        explanation['explanations'] = explanations
        
        return explanation

    def _normalize_prediction(self, prediction: Any) -> Optional[float]:
        """Normalize model predictions to a single float value."""
        if isinstance(prediction, tuple) and prediction:
            prediction = prediction[0]

        if isinstance(prediction, (list, tuple, np.ndarray, pd.Series)):
            if len(prediction) == 0:
                return None
            prediction = prediction[0]

        try:
            return float(prediction)
        except (TypeError, ValueError):
            return None
    
    def _get_shap_explanation(self, X: pd.DataFrame, top_k: int) -> Dict[str, Any]:
        """Get SHAP-based explanation."""
        try:
            import shap
            
            shap_data = {
                'available': False,
                'base_values': {},
                'shap_values': {},
                'top_features': []
            }
            
            all_shap_values = {}
            
            # Get SHAP values from each model
            for name, explainer in self.explainers.items():
                try:
                    sv = explainer.shap_values(X)
                    
                    # Handle multiple outputs (take first if list)
                    if isinstance(sv, list):
                        sv = sv[0]
                    
                    # Store SHAP values
                    if hasattr(explainer, 'expected_value'):
                        base_value = explainer.expected_value
                        if isinstance(base_value, (list, np.ndarray)):
                            base_value = base_value[0]
                    else:
                        base_value = 0
                    
                    shap_data['base_values'][name] = float(base_value)
                    
                    # Get absolute SHAP values for importance
                    abs_shap = np.abs(sv[0])
                    all_shap_values[name] = abs_shap
                    
                except Exception as e:
                    print(f"[WARN] SHAP values failed for {name}: {e}")
            
            if all_shap_values:
                # Combine across models (average absolute SHAP)
                shap_data['available'] = True
                
                combined_shap = np.zeros(len(X.columns))
                for sv_array in all_shap_values.values():
                    combined_shap += sv_array
                combined_shap /= len(all_shap_values)
                
                # Get top features
                top_indices = np.argsort(combined_shap)[-top_k:][::-1]
                
                for idx in top_indices:
                    feature_name = X.columns[idx] if idx < len(X.columns) else f"feature_{idx}"
                    shap_data['top_features'].append({
                        'feature': feature_name,
                        'importance': float(combined_shap[idx]),
                        'value': float(X.iloc[0, idx]) if isinstance(X.iloc[0, idx], (int, float, np.number)) else str(X.iloc[0, idx])
                    })
            
            return shap_data
            
        except Exception as e:
            print(f"[WARN] SHAP explanation failed: {e}")
            return {'available': False, 'error': str(e)}
    
    def _get_model_contributions(self, X: pd.DataFrame, top_k: int) -> Dict[str, Any]:
        """Get per-model contributions and check for agreement."""
        contributions = {}
        
        if not hasattr(self.model, 'models') and not hasattr(self.model, 'merged_models'):
            return contributions
        
        # Get predictions from each model
        predictions = {}
        try:
            model_dict = self.model.models if hasattr(self.model, 'models') else self.model.merged_models
            for name, sub_model in model_dict.items():
                try:
                    pred = sub_model.predict(X)
                    pred_value = self._normalize_prediction(pred)
                    if pred_value is not None:
                        predictions[name] = float(pred_value)
                except Exception as e:
                    print(f"[WARN] Prediction from {name} failed: {e}")
        except Exception as e:
            print(f"[WARN] Model contribution extraction failed: {e}")
        
        if predictions:
            # Calculate agreement metrics
            pred_values = list(predictions.values())
            contributions['model_predictions'] = predictions
            contributions['mean_prediction'] = float(np.mean(pred_values))
            contributions['std_prediction'] = float(np.std(pred_values))
            contributions['coefficient_of_variation'] = float(
                np.std(pred_values) / (np.mean(pred_values) + 1e-6)
            )
            contributions['num_models'] = len(predictions)
            
            # Check for high agreement (all models within 10% of mean)
            mean_pred = contributions['mean_prediction']
            within_10pct = sum(1 for p in pred_values if abs(p - mean_pred) / (abs(mean_pred) + 1) < 0.10)
            contributions['agreement_score'] = float(within_10pct) / len(predictions)
        
        return contributions
    
    def _get_feature_contributions(self, X: pd.DataFrame, baseline_pred: float, 
                                 top_k: int) -> Dict[str, Any]:
        """
        Calculate feature contributions using permutation-based importance.
        Shows how much each feature impacts the prediction.
        """
        result = {'contributions': [], 'confidence': {}}
        
        try:
            # For each feature, permute and see how prediction changes
            original_pred = baseline_pred
            feature_impacts = []
            
            for col_idx, col_name in enumerate(X.columns):
                try:
                    # Create permuted version
                    X_permuted = X.copy()
                    
                    # Permute by shuffling the value slightly or using median
                    if X[col_name].dtype in [np.float64, np.float32, int, np.int64, np.int32, np.int16]:
                        # Use a small perturbation so single-row inputs still change
                        current_val = X[col_name].iloc[0]
                        if current_val == 0:
                            X_permuted[col_name] = 1.0
                        else:
                            X_permuted[col_name] = current_val * 0.9
                    else:
                        # For categorical, flip to a different placeholder value
                        current_val = X[col_name].iloc[0]
                        if isinstance(current_val, (int, np.integer)):
                            X_permuted[col_name] = 0 if current_val != 0 else 1
                        else:
                            X_permuted[col_name] = 'other'
                    
                    # Get prediction with permuted feature
                    try:
                        permuted_pred = self.model.predict(X_permuted)
                    except:
                        permuted_pred = self.model.predict(X_permuted)

                    permuted_value = self._normalize_prediction(permuted_pred)
                    if permuted_value is None:
                        continue
                    
                    # Calculate impact (how much prediction changed)
                    impact = abs(float(original_pred - permuted_value))
                    
                    # Calculate relative impact
                    relative_impact = impact / (abs(baseline_pred) + 1e-6)
                    
                    feature_impacts.append({
                        'feature': col_name,
                        'impact': impact,
                        'relative_impact': relative_impact,
                        'value': float(X[col_name].iloc[0]) if isinstance(X[col_name].iloc[0], (int, float, np.number)) else str(X[col_name].iloc[0]),
                        'direction': 'increases' if float(original_pred - permuted_value) > 0 else 'decreases'
                    })
                
                except Exception as e:
                    print(f"[WARN] Feature impact calculation failed for {col_name}: {e}")
            
            # Sort by impact and return top_k
            feature_impacts.sort(key=lambda x: x['impact'], reverse=True)
            
            total_impact = sum(f['impact'] for f in feature_impacts) + 1e-6
            
            for contrib in feature_impacts[:top_k]:
                contrib['importance_pct'] = (contrib['impact'] / total_impact) * 100
                result['contributions'].append(contrib)
            
            # Confidence indicators
            if feature_impacts:
                top_5_impact = sum(f['impact'] for f in feature_impacts[:5])
                result['confidence']['top_5_coverage'] = float(top_5_impact / total_impact)
                result['confidence']['has_clear_driver'] = result['confidence']['top_5_coverage'] > 0.7
                result['confidence']['total_features_analyzed'] = len(feature_impacts)
        
        except Exception as e:
            print(f"[WARN] Feature contribution calculation failed: {e}")
        
        return result
    
    def _generate_explanations(self, X: pd.DataFrame, explanation: Dict) -> List[str]:
        """Generate human-readable explanations."""
        explanations = []
        
        try:
            # Top contributors
            if explanation['feature_contributions']:
                top_feature = explanation['feature_contributions'][0]
                explanations.append(
                    f"🎯 Top driver: {top_feature['feature']} ({top_feature['importance_pct']:.1f}% of variation)"
                )
                
                # List top 3
                top_3_text = ", ".join([
                    f"{f['feature']}" for f in explanation['feature_contributions'][:3]
                ])
                explanations.append(f"📊 Key factors: {top_3_text}")
            
            # Model agreement
            if 'model_agreements' in explanation and 'agreement_score' in explanation['model_agreements']:
                agreement = explanation['model_agreements']['agreement_score']
                if agreement > 0.75:
                    explanations.append("✅ Strong model consensus on this prediction")
                elif agreement > 0.5:
                    explanations.append("🟡 Moderate agreement between models")
                else:
                    explanations.append("⚠️ Models show some disagreement on this prediction")
            
            # Confidence based on feature clarity
            if 'confidence' in explanation and explanation['confidence'].get('has_clear_driver'):
                explanations.append("📈 Clear dominant factors drive this prediction")
            elif explanation['feature_contributions']:
                explanations.append("💡 Multiple factors contribute to this prediction")
            
            # SHAP explanation if available
            if explanation.get('shap_values', {}).get('available'):
                shap_features = explanation['shap_values'].get('top_features', [])
                if shap_features:
                    shap_text = ", ".join([f['feature'] for f in shap_features[:3]])
                    explanations.append(f"🔍 SHAP analysis highlights: {shap_text}")
        
        except Exception as e:
            print(f"[WARN] Explanation generation failed: {e}")
        
        return explanations
    
    def explain_batch(self, X: pd.DataFrame, predictions: np.ndarray, 
                     batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions efficiently.
        
        Args:
            X: Feature matrix (multiple rows)
            predictions: Model predictions for each row
            batch_size: Process predictions in batches for efficiency
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i in range(0, len(X), batch_size):
            batch_end = min(i + batch_size, len(X))
            
            for j in range(i, batch_end):
                try:
                    exp = self.explain_prediction(
                        X.iloc[[j]], 
                        predictions[j] if isinstance(predictions, np.ndarray) else predictions.iloc[j]
                    )
                    explanations.append(exp)
                except Exception as e:
                    print(f"[WARN] Explanation failed for sample {j}: {e}")
                    explanations.append({'error': str(e)})
        
        return explanations


def create_explainer_from_saved_model(model_path: str, 
                                      background_data_path: Optional[str] = None) -> PredictionExplainer:
    """
    Create a PredictionExplainer from a saved model file.
    
    Args:
        model_path: Path to saved model (joblib format)
        background_data_path: Path to background data CSV for SHAP initialization
        
    Returns:
        Initialized PredictionExplainer instance
    """
    # Load model
    model = joblib.load(model_path)
    print(f"[INFO] Model loaded from {model_path}")
    
    # Load background data if provided
    background_data = None
    if background_data_path:
        background_data = pd.read_csv(background_data_path)
        print(f"[INFO] Background data loaded: {len(background_data)} samples, {len(background_data.columns)} features")
    
    # Create explainer
    explainer = PredictionExplainer(
        model, 
        X_background=background_data,
        use_shap=True
    )
    
    return explainer
