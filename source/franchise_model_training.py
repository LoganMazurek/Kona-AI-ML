import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor


class FranchiseModelTrainer:
    """Train and publish franchise-specific models from stored feature snapshots."""

    def __init__(self, franchise_db, models_dir: str, threshold: int = 100, min_r2: float = 0.25):
        self.franchise_db = franchise_db
        self.models_dir = models_dir
        self.threshold = threshold
        self.min_r2 = min_r2
        os.makedirs(self.models_dir, exist_ok=True)

    def ensure_model(self, franchise_id: str, force: bool = False) -> Dict[str, Any]:
        progress = self.franchise_db.get_franchise_model_progress(franchise_id, threshold=self.threshold)
        latest_model = progress.get('latest_model')
        had_existing_model = latest_model is not None
        if latest_model and not force:
            return {
                'status': 'already_exists',
                'message': 'Franchise-specific model already available.',
                'model': latest_model,
                'progress': progress,
            }

        if not progress['ready_for_training']:
            return {
                'status': 'not_ready',
                'message': f"{progress['remaining_events']} more completed events needed before training starts.",
                'progress': progress,
            }

        if not progress['ready_with_features']:
            missing = max(0, self.threshold - progress['trainable_event_count'])
            expected_ready = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d')
            self.franchise_db.update_franchise_model_status(
                franchise_id,
                last_training_attempt_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                last_training_status='awaiting_feature_snapshots',
                last_training_message=(
                    f'Waiting for {missing} more completed events captured under the new training pipeline.'
                ),
                expected_ready_date=expected_ready,
            )
            return {
                'status': 'awaiting_feature_snapshots',
                'message': f'Waiting for {missing} more feature-complete events before the dedicated model can train.',
                'expected_ready_date': expected_ready,
                'progress': progress,
            }

        self.franchise_db.update_franchise_model_status(
            franchise_id,
            last_training_attempt_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            last_training_status='training',
            last_training_message=(
                'Refreshing franchise-specific model with newly completed events.'
                if had_existing_model else
                'Training franchise-specific model.'
            ),
            expected_ready_date=(datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
        )

        examples = self.franchise_db.get_franchise_training_examples(franchise_id)
        train_df = self._build_training_frame(examples)
        if len(train_df) < self.threshold:
            return {
                'status': 'insufficient_examples',
                'message': 'Not enough feature-complete examples to train the franchise-specific model yet.',
                'progress': self.franchise_db.get_franchise_model_progress(franchise_id, threshold=self.threshold),
            }

        X = train_df.drop(columns=['target_total_net_sales', 'duration_hours'])
        self._assert_numeric_features(X)
        X = X.fillna(0.0)
        y = train_df['target_total_net_sales'].astype(float)
        durations = train_df['duration_hours'].astype(float).replace(0, np.nan)

        fold_count = min(5, max(3, len(train_df) // 20))
        cv = KFold(n_splits=fold_count, shuffle=True, random_state=42)
        base_model = self._build_model()
        cv_predictions = cross_val_predict(base_model, X, y, cv=cv)

        mae_total = float(mean_absolute_error(y, cv_predictions))
        rmse_total = float(mean_squared_error(y, cv_predictions) ** 0.5)
        r2_total = float(r2_score(y, cv_predictions))
        per_hour_actual = (y / durations).replace([np.inf, -np.inf], np.nan)
        per_hour_pred = (pd.Series(cv_predictions, index=train_df.index) / durations).replace([np.inf, -np.inf], np.nan)
        valid_per_hour = per_hour_actual.notna() & per_hour_pred.notna()
        if valid_per_hour.any():
            per_hour_errors = (per_hour_actual[valid_per_hour] - per_hour_pred[valid_per_hour]).abs()
            mae_per_hour = float(per_hour_errors.mean())
            std_per_hour = float(per_hour_errors.std(ddof=0))
        else:
            mae_per_hour = mae_total
            std_per_hour = 0.0

        if r2_total < self.min_r2:
            message = f'Candidate franchise model scored R2={r2_total:.3f}; threshold is {self.min_r2:.2f}. Continuing with merged model.'
            self.franchise_db.update_franchise_model_status(
                franchise_id,
                last_training_attempt_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                last_training_status='rejected_low_r2',
                last_training_message=message,
                expected_ready_date=(datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
            )
            return {
                'status': 'rejected_low_r2',
                'message': message,
                'metrics': {
                    'cv_mae': mae_per_hour,
                    'cv_std': std_per_hour,
                    'r2': r2_total,
                    'total_mae': mae_total,
                    'rmse': rmse_total,
                },
                'progress': self.franchise_db.get_franchise_model_progress(franchise_id, threshold=self.threshold),
            }

        model = self._build_model()
        model.fit(X, y)

        model_path = os.path.join(self.models_dir, f'franchise_{franchise_id}_model.joblib')
        metrics_path = os.path.join(self.models_dir, f'franchise_{franchise_id}_metrics.json')
        training_history = self._build_training_history(latest_model, {
            'trained_at': datetime.now().isoformat(),
            'r2': r2_total,
            'cv_mae_per_hour': mae_per_hour,
            'trainable_event_count': int(len(train_df)),
        })
        metadata = {
            'franchise_id': franchise_id,
            'trained_at': datetime.now().isoformat(),
            'threshold_event_count': self.threshold,
            'trainable_event_count': int(len(train_df)),
            'feature_count': int(X.shape[1]),
            'cross_validation_folds': int(fold_count),
            'metrics': {
                'cv_mae_per_hour': mae_per_hour,
                'cv_std_per_hour': std_per_hour,
                'cv_mae_total': mae_total,
                'rmse_total': rmse_total,
                'r2': r2_total,
            },
            'feature_names': list(X.columns),
            'training_history': training_history,
        }

        joblib.dump(model, model_path)
        with open(metrics_path, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)

        model_id = f'franchise_{franchise_id}'
        self.franchise_db.upsert_model(
            model_id=model_id,
            franchise_id=franchise_id,
            model_name='Your Franchise Model',
            model_path=model_path,
            model_type='franchise_specific',
            training_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            cv_mae=mae_per_hour,
            cv_std=std_per_hour,
            feature_count=int(X.shape[1]),
            data_records_count=int(len(train_df)),
            is_default=False,
            training_metadata_json=json.dumps(metadata),
        )
        next_retrain_target = self._next_retrain_target(int(len(train_df)), self.threshold)
        self.franchise_db.update_franchise_model_status(
            franchise_id,
            last_training_attempt_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            last_training_status='trained',
            last_training_message=(
                f'Franchise-specific model refreshed at {len(train_df)} completed events. '
                f'Next auto-refresh at {next_retrain_target} events.'
                if had_existing_model else
                f'Franchise-specific model trained successfully at {len(train_df)} completed events. '
                f'Next auto-refresh at {next_retrain_target} events.'
            ),
            expected_ready_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            last_trained_event_count=int(len(train_df)),
            next_retrain_event_count=next_retrain_target,
            retrain_popup_shown_at=None,
        )

        return {
            'status': 'retrained' if had_existing_model else 'trained',
            'message': (
                'Franchise-specific model refreshed with new events.'
                if had_existing_model else
                'Franchise-specific model trained successfully.'
            ),
            'model_path': model_path,
            'metrics_path': metrics_path,
            'metrics': metadata['metrics'],
            'progress': self.franchise_db.get_franchise_model_progress(franchise_id, threshold=self.threshold),
        }

    def _build_training_frame(self, examples: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for example in examples:
            raw_snapshot = example.get('event_features_json')
            if not raw_snapshot:
                continue
            try:
                feature_snapshot = json.loads(raw_snapshot)
            except json.JSONDecodeError:
                continue
            if not isinstance(feature_snapshot, dict):
                continue
            row = dict(feature_snapshot)
            row['target_total_net_sales'] = float(example['actual_total_net_sales'])
            row['duration_hours'] = float(example['duration_hours'] or 0.0)
            rows.append(row)
        return pd.DataFrame(rows)

    def _assert_numeric_features(self, X: pd.DataFrame) -> None:
        bad_columns = [
            column for column in X.columns
            if not pd.api.types.is_numeric_dtype(X[column])
        ]
        if bad_columns:
            raise ValueError(f'Non-numeric feature snapshots found: {bad_columns}')

    def _build_model(self) -> XGBRegressor:
        return XGBRegressor(
            objective='reg:squarederror',
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
        )

    @staticmethod
    def _next_retrain_target(last_trained_event_count: int, threshold: int) -> int:
        milestones = [threshold, 175, 250, 350, 500]
        trained = max(0, int(last_trained_event_count or 0))
        for milestone in milestones:
            if milestone > trained:
                return milestone
        return 500 + (((trained - 500) // 250) + 1) * 250

    @staticmethod
    def _build_training_history(latest_model: Optional[Dict[str, Any]], new_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        if latest_model and latest_model.get('training_metadata_json'):
            try:
                previous_metadata = json.loads(latest_model.get('training_metadata_json') or '{}')
            except (TypeError, json.JSONDecodeError):
                previous_metadata = {}
            previous_history = previous_metadata.get('training_history')
            if isinstance(previous_history, list):
                for row in previous_history:
                    if isinstance(row, dict):
                        history.append({
                            'trained_at': row.get('trained_at'),
                            'r2': row.get('r2'),
                            'cv_mae_per_hour': row.get('cv_mae_per_hour'),
                            'trainable_event_count': row.get('trainable_event_count'),
                        })
            elif previous_metadata.get('metrics'):
                metrics = previous_metadata.get('metrics', {})
                history.append({
                    'trained_at': previous_metadata.get('trained_at'),
                    'r2': metrics.get('r2'),
                    'cv_mae_per_hour': metrics.get('cv_mae_per_hour'),
                    'trainable_event_count': previous_metadata.get('trainable_event_count'),
                })

        history.append(new_entry)
        return history[-12:]