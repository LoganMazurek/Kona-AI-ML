"""
Production model deployment strategy.

APPROACH: Staged Model Rollout
1. Phase 1 (Current): Use merged ensemble model across all franchises
2. Phase 2 (When franchise reaches threshold): Use franchise-specific model

This balances:
- Early predictions with global model (good generalization)
- Personalized predictions once enough franchise data accumulates
"""

import os
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class ProductionModelManager:
    """Manages production model selection and prediction."""
    
    # Thresholds for model graduation
    FRANCHISE_DATA_THRESHOLD = 100  # Events needed to train franchise model
    FRANCHISE_MODEL_CONFIDENCE_R2 = 0.25  # Minimum R2 for franchise model acceptance
    
    def __init__(self, models_dir='output_models', data_dir='Kona_AI_ML/data'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        
        # Load merged ensemble (production standard)
        self.merged_models = self._load_merged_models()
        self.merged_weights = self._load_merged_weights()
        self.feature_names = self._load_feature_names()
        
        # Franchise-specific models (optional, loaded on demand)
        self.franchise_models = {}
        
    def _load_merged_models(self):
        """Load the global merged ensemble models."""
        models = {}
        for model_name in ['xgboost', 'catboost', 'lightgbm']:
            path = os.path.join(self.models_dir, f'clean_layers_{model_name}_model.joblib')
            if os.path.exists(path):
                models[model_name] = joblib.load(path)
        return models
    
    def _load_merged_weights(self):
        """Load ensemble weights."""
        path = os.path.join(self.models_dir, 'clean_layers_ensemble_weights.json')
        with open(path) as f:
            return json.load(f)
    
    def _load_feature_names(self):
        """Load feature names."""
        path = os.path.join(self.models_dir, 'clean_layers_feature_names.json')
        with open(path) as f:
            return json.load(f)
    
    def load_training_metrics(self):
        """Load training metrics from the metrics JSON file."""
        path = os.path.join(self.models_dir, 'clean_layers_training_metrics.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None
    
    def load_franchise_model(self, franchise_id):
        """Load franchise-specific model if available."""
        if franchise_id in self.franchise_models:
            return self.franchise_models[franchise_id]
        
        path = os.path.join(self.models_dir, f'franchise_{franchise_id}_model.joblib')
        if os.path.exists(path):
            self.franchise_models[franchise_id] = joblib.load(path)
            return self.franchise_models[franchise_id]
        
        return None
    
    def should_use_franchise_model(self, franchise_id, master_data):
        """Check if franchise has enough data to use franchise-specific model."""
        franchise_events = master_data[master_data['franchise_id'] == franchise_id]
        return len(franchise_events) >= self.FRANCHISE_DATA_THRESHOLD
    
    def predict(self, X, franchise_id=None, master_data=None):
        """
        Make prediction using appropriate model.
        
        Args:
            X: Feature matrix (DataFrame with properly typed categorical and numeric columns)
            franchise_id: (Optional) Franchise ID for potential model selection
            master_data: (Optional) Master data to check franchise event count
            
        Returns:
            tuple: (ensemble prediction, model type)
        """
        # Safety: Ensure no NaN values and proper dtypes
        if isinstance(X, pd.DataFrame):
            # Fill any remaining NaN with 0
            X = X.fillna(0)
            
            # Ensure categorical columns are int64 (not float)
            # CatBoost expects integer categorical features
            categorical_cols = [
                'Event_Industry', 'equipment_type', 'Duration_Bracket', 
                'Hour_Bucket', 'Season', 'School_Season',
                'Equipment_Industry', 'Equipment_Season'  # Interaction features
            ]
            for col in categorical_cols:
                if col in X.columns and X[col].dtype == 'float64':
                    X[col] = X[col].astype('int64')
        
        # Try franchise model first if available
        if franchise_id and master_data is not None:
            if self.should_use_franchise_model(franchise_id, master_data):
                franchise_model = self.load_franchise_model(franchise_id)
                if franchise_model is not None:
                    return franchise_model.predict(X), 'franchise_specific'
        
        # Fall back to merged ensemble
        preds = {}
        for model_name, model in self.merged_models.items():
            preds[model_name] = model.predict(X)
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(preds['xgboost'])
        for model_name, weight in self.merged_weights.items():
            if model_name in preds:
                ensemble_pred += weight * preds[model_name]
        
        return ensemble_pred, 'merged_ensemble'
    
    def predict_batch(self, df, franchise_col='franchise_id', master_data=None):
        """
        Predict for batch of events, using appropriate model per franchise.
        
        Args:
            df: DataFrame with features and franchise_col
            franchise_col: Column name for franchise ID
            master_data: Master data for franchise event count checking
            
        Returns:
            DataFrame with predictions and model used
        """
        results = []
        
        for franchise_id in df[franchise_col].unique():
            mask = df[franchise_col] == franchise_id
            X_franchise = df.loc[mask].drop(columns=[franchise_col])
            
            pred, model_used = self.predict(
                X_franchise,
                franchise_id=franchise_id,
                master_data=master_data
            )
            
            for idx, (_, row) in enumerate(df[mask].iterrows()):
                results.append({
                    'franchise_id': franchise_id,
                    'prediction': pred[idx] if isinstance(pred, np.ndarray) else pred,
                    'model_used': model_used
                })
        
        return pd.DataFrame(results)


class FranchiseDataTracker:
    """Track franchise data accumulation for model graduation decisions."""
    
    def __init__(self, tracking_file='franchise_model_status.json'):
        self.tracking_file = tracking_file
        self.status = self._load_status()
    
    def _load_status(self):
        """Load franchise model status."""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file) as f:
                return json.load(f)
        return {}
    
    def update_from_master(self, master_data):
        """Update franchise data counts from master data."""
        status = {}
        
        for franchise_id in master_data['franchise_id'].unique():
            events_count = len(master_data[master_data['franchise_id'] == franchise_id])
            status[str(franchise_id)] = {
                'event_count': int(events_count),
                'last_updated': datetime.now().isoformat(),
                'ready_for_franchise_model': events_count >= 100,
                'model_exists': os.path.exists(f'output_models/franchise_{franchise_id}_model.joblib')
            }
        
        self.status = status
        self.save()
        return status
    
    def save(self):
        """Save status to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def get_franchises_ready_for_model(self):
        """Get franchises with enough data for franchise-specific models."""
        return [
            fid for fid, status in self.status.items()
            if status.get('ready_for_franchise_model', False) and not status.get('model_exists', False)
        ]
    
    def summary(self):
        """Get summary of franchise data status."""
        summary = {
            'total_franchises': len(self.status),
            'franchises_with_model': sum(1 for s in self.status.values() if s.get('model_exists')),
            'franchises_ready_for_model': sum(1 for s in self.status.values() if s.get('ready_for_franchise_model')),
            'franchises_needing_more_data': sum(1 for s in self.status.values() if not s.get('ready_for_franchise_model'))
        }
        return summary


if __name__ == '__main__':
    print("\n" + "="*80)
    print("PRODUCTION MODEL DEPLOYMENT STRATEGY")
    print("="*80)
    
    # Load actual metrics
    manager = ProductionModelManager()
    metrics = manager.load_training_metrics()
    
    if metrics and 'model_scores' in metrics:
        scores = metrics['model_scores']
        weights = metrics.get('ensemble_weights', {})
        
        # Calculate ensemble MAE (weighted average)
        ensemble_mae = sum(
            scores[model]['MAE'] * weights.get(model, 1/3)
            for model in scores.keys()
        )
        
        # Calculate ensemble R² (weighted average)
        ensemble_r2 = sum(
            scores[model]['R2'] * weights.get(model, 1/3)
            for model in scores.keys()
        )
        
        print("\nPHASE 1 (CURRENT): Merged Ensemble Model")
        print(f"  - All franchises use global merged model")
        print(f"  - Trained on 1438 events across all franchises")
        print(f"  - Performance: MAE ~${ensemble_mae:.0f}/hour, R² ~{ensemble_r2:.2f} (weighted ensemble)")
        
        for model_name, model_scores in scores.items():
            weight = weights.get(model_name, 0) * 100
            print(f"    • {model_name.capitalize()}: MAE ${model_scores['MAE']:.0f}, R² {model_scores['R2']:.2f} (weight: {weight:.1f}%)")
    else:
        print("\nPHASE 1 (CURRENT): Merged Ensemble Model")
        print("  - All franchises use global merged model")
        print("  - Trained on 1438 events across all franchises")
        print("  - (Metrics file not found)")
    
    print("\nPHASE 2 (TRIGGERED): Franchise-Specific Models")
    print("  - When franchise reaches 100 events → train franchise-specific model")
    print("  - If franchise model R² >= 0.25 → use franchise model")
    print("  - Otherwise → continue using merged model")
    
    print("\nHOW TO USE:")
    print("  manager = ProductionModelManager()")
    print("  pred, model_type = manager.predict(X, franchise_id=123, master_data=df)")
    print("  # model_type will be 'merged_ensemble' or 'franchise_specific'")
    
    print("\nTRACKING FRANCHISE READINESS:")
    print("  tracker = FranchiseDataTracker()")
    print("  tracker.update_from_master(master_data)")
    print("  ready = tracker.get_franchises_ready_for_model()")
    
    print("\n" + "="*80 + "\n")
