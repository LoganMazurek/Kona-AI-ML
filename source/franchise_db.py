"""
Franchise database management module.
Handles SQLite database initialization, schema creation, and CRUD operations
for franchises, models, sessions, and prediction tracking.
"""

import sqlite3
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


class FranchiseDatabase:
    """Manage franchise database operations."""
    
    def __init__(self, db_path: str):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_schema()
    
    def get_connection(self):
        """Get database connection with row factory enabled."""
        # timeout lets a connection wait (rather than immediately raising
        # "database is locked") when the background franchise-model trainer and
        # a web request touch the DB at the same time.
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def init_schema(self):
        """Create database schema if not exists."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Franchises table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS franchises (
                franchise_id TEXT PRIMARY KEY,
                franchise_name TEXT NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active BOOLEAN DEFAULT 1,
                default_model_id TEXT,
                target_net_sales_per_hour REAL,
                FOREIGN KEY(default_model_id) REFERENCES models(model_id)
            )
        ''')
        
        # Models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                franchise_id TEXT NOT NULL,
                model_name TEXT,
                model_path TEXT NOT NULL,
                model_type TEXT,
                training_date TIMESTAMP,
                cv_mae REAL,
                cv_std REAL,
                feature_count INTEGER,
                data_records_count INTEGER,
                is_default BOOLEAN DEFAULT 0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(franchise_id) REFERENCES franchises(franchise_id)
            )
        ''')
        
        # Login sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_token TEXT PRIMARY KEY,
                franchise_id TEXT NOT NULL,
                login_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(franchise_id) REFERENCES franchises(franchise_id)
            )
        ''')
        
        # Prediction history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                franchise_id TEXT NOT NULL,
                model_id TEXT,
                event_name TEXT,
                duration_hours REAL,
                predicted_revenue_per_hour REAL,
                predicted_total_revenue REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                actual_total_net_sales REAL,
                actual_revenue_per_hour REAL,
                actual_updated_timestamp TIMESTAMP,
                is_test BOOLEAN DEFAULT 0,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(franchise_id) REFERENCES franchises(franchise_id),
                FOREIGN KEY(model_id) REFERENCES models(model_id)
            )
        ''')
        
        # Batch uploads table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_uploads (
                batch_id TEXT PRIMARY KEY,
                franchise_id TEXT NOT NULL,
                model_id TEXT,
                original_filename TEXT,
                upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_count INTEGER,
                result_csv_path TEXT,
                FOREIGN KEY(franchise_id) REFERENCES franchises(franchise_id),
                FOREIGN KEY(model_id) REFERENCES models(model_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS franchise_model_status (
                franchise_id TEXT PRIMARY KEY,
                threshold_event_count INTEGER DEFAULT 100,
                last_trained_event_count INTEGER,
                next_retrain_event_count INTEGER,
                threshold_reached_at TIMESTAMP,
                threshold_popup_shown_at TIMESTAMP,
                model_ready_popup_shown_at TIMESTAMP,
                retrain_popup_shown_at TIMESTAMP,
                last_training_attempt_at TIMESTAMP,
                last_training_status TEXT,
                last_training_message TEXT,
                expected_ready_date TEXT,
                FOREIGN KEY(franchise_id) REFERENCES franchises(franchise_id)
            )
        ''')

        # Ensure new columns exist for older databases
        self._ensure_column_exists(cursor, 'franchises', 'target_net_sales_per_hour', 'REAL')
        self._ensure_column_exists(cursor, 'predictions', 'duration_hours', 'REAL')
        self._ensure_column_exists(cursor, 'predictions', 'actual_total_net_sales', 'REAL')
        self._ensure_column_exists(cursor, 'predictions', 'actual_revenue_per_hour', 'REAL')
        self._ensure_column_exists(cursor, 'predictions', 'actual_updated_timestamp', 'TIMESTAMP')
        self._ensure_column_exists(cursor, 'predictions', 'is_test', 'BOOLEAN DEFAULT 0')
        self._ensure_column_exists(cursor, 'predictions', 'event_status', "TEXT DEFAULT 'predicted_only'")
        self._ensure_column_exists(cursor, 'predictions', 'include_in_training', 'BOOLEAN DEFAULT 0')
        self._ensure_column_exists(cursor, 'predictions', 'scheduled_event_date', 'TEXT')
        self._ensure_column_exists(cursor, 'predictions', 'event_features_json', 'TEXT')
        self._ensure_column_exists(cursor, 'models', 'training_metadata_json', 'TEXT')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'threshold_event_count', 'INTEGER DEFAULT 100')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'last_trained_event_count', 'INTEGER')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'next_retrain_event_count', 'INTEGER')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'threshold_reached_at', 'TIMESTAMP')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'threshold_popup_shown_at', 'TIMESTAMP')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'model_ready_popup_shown_at', 'TIMESTAMP')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'retrain_popup_shown_at', 'TIMESTAMP')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'last_training_attempt_at', 'TIMESTAMP')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'last_training_status', 'TEXT')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'last_training_message', 'TEXT')
        self._ensure_column_exists(cursor, 'franchise_model_status', 'expected_ready_date', 'TEXT')

        # Equipment mappings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS equipment_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                franchise_id TEXT NOT NULL,
                equipment_name TEXT NOT NULL,
                equipment_type TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(franchise_id, equipment_name),
                FOREIGN KEY(franchise_id) REFERENCES franchises(franchise_id)
            )
        ''')

        # Password reset tokens table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token TEXT PRIMARY KEY,
                franchise_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY(franchise_id) REFERENCES franchises(franchise_id)
            )
        ''')

        # Login events table -- append-only usage history.
        #
        # `sessions` cannot answer "how often does each franchise log in": rows are
        # deleted on logout and again when cleanup_expired_sessions() runs, so it
        # only ever shows who is currently signed in. This table keeps one durable
        # row per successful login instead.
        #
        # Deliberately holds no IP address or user agent -- franchise_id and a
        # timestamp answer the usage questions without storing anything that
        # identifies an individual person.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                franchise_id TEXT NOT NULL,
                login_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(franchise_id) REFERENCES franchises(franchise_id)
            )
        ''')
        # Reporting scans by time window, then groups by franchise.
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_login_events_timestamp
            ON login_events(login_timestamp)
        ''')

        conn.commit()
        conn.close()
    
    # ===== FRANCHISE OPERATIONS =====
    
    def create_franchise(self, franchise_id: str, franchise_name: str, email: str, 
                        password_hash: str, target_net_sales_per_hour: Optional[float] = None) -> bool:
        """
        Create a new franchise record.
        
        Args:
            franchise_id: Unique identifier for franchise
            franchise_name: Display name
            email: Contact email
            password_hash: Hashed password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO franchises 
                (franchise_id, franchise_name, email, password_hash, active, target_net_sales_per_hour)
                VALUES (?, ?, ?, ?, 1, ?)
            ''', (franchise_id, franchise_name, email, password_hash, target_net_sales_per_hour))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def get_franchise(self, franchise_id: str) -> Optional[Dict]:
        """Get franchise by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM franchises WHERE franchise_id = ?', (franchise_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_franchise_by_email(self, email: str) -> Optional[Dict]:
        """Get franchise by email."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM franchises WHERE email = ?', (email,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def list_franchises(self) -> List[Dict]:
        """List all active franchises."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM franchises WHERE active = 1')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def update_franchise_default_model(self, franchise_id: str, model_id: str) -> bool:
        """Update default model for franchise."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE franchises 
                SET default_model_id = ? 
                WHERE franchise_id = ?
            ''', (model_id, franchise_id))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def update_franchise_target(self, franchise_id: str, target_net_sales_per_hour: float) -> bool:
        """Update target net sales per hour for a franchise."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE franchises
                SET target_net_sales_per_hour = ?
                WHERE franchise_id = ?
            ''', (target_net_sales_per_hour, franchise_id))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def _ensure_column_exists(self, cursor, table_name: str, column_name: str, column_type: str) -> None:
        """Add a column if it does not exist (SQLite migration helper)."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing = {row[1] for row in cursor.fetchall()}
        if column_name not in existing:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
    
    # ===== MODEL OPERATIONS =====
    
    def create_model(self, model_id: str, franchise_id: str, model_name: str,
                    model_path: str, model_type: Optional[str] = None, 
                    training_date: Optional[datetime] = None, cv_mae: Optional[float] = None, 
                    cv_std: Optional[float] = None, feature_count: Optional[int] = None,
                    data_records_count: Optional[int] = None, is_default: bool = False,
                    training_metadata_json: Optional[str] = None) -> bool:
        """
        Create a new model record.
        
        Args:
            model_id: Unique identifier for model
            franchise_id: Associated franchise
            model_name: Display name
            model_path: Path to model file (.joblib)
            model_type: Type of model (ensemble, xgboost, catboost, lightgbm)
            training_date: When model was trained
            cv_mae: Mean Absolute Error from cross-validation
            cv_std: Standard deviation of error
            feature_count: Number of features used
            data_records_count: Number of training records
            is_default: Whether this is the default model for franchise
            
        Returns:
            True if successful
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO models 
                (model_id, franchise_id, model_name, model_path, model_type,
                 training_date, cv_mae, cv_std, feature_count, data_records_count, is_default,
                 training_metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (model_id, franchise_id, model_name, model_path, model_type,
                  training_date, cv_mae, cv_std, feature_count, data_records_count, is_default,
                  training_metadata_json))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False

    def upsert_model(self, model_id: str, franchise_id: str, model_name: str,
                    model_path: str, model_type: Optional[str] = None,
                    training_date: Optional[datetime] = None, cv_mae: Optional[float] = None,
                    cv_std: Optional[float] = None, feature_count: Optional[int] = None,
                    data_records_count: Optional[int] = None, is_default: bool = False,
                    training_metadata_json: Optional[str] = None) -> bool:
        """Create or update a model record."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO models
                (model_id, franchise_id, model_name, model_path, model_type,
                 training_date, cv_mae, cv_std, feature_count, data_records_count, is_default,
                 training_metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    franchise_id = excluded.franchise_id,
                    model_name = excluded.model_name,
                    model_path = excluded.model_path,
                    model_type = excluded.model_type,
                    training_date = excluded.training_date,
                    cv_mae = excluded.cv_mae,
                    cv_std = excluded.cv_std,
                    feature_count = excluded.feature_count,
                    data_records_count = excluded.data_records_count,
                    is_default = excluded.is_default,
                    training_metadata_json = excluded.training_metadata_json
            ''', (model_id, franchise_id, model_name, model_path, model_type,
                  training_date, cv_mae, cv_std, feature_count, data_records_count, is_default,
                  training_metadata_json))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """Get model by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM models WHERE model_id = ?', (model_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def list_franchise_models(self, franchise_id: str) -> List[Dict]:
        """List all models for a franchise."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM models 
            WHERE franchise_id = ? 
            ORDER BY created_date DESC
        ''', (franchise_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_default_franchise_model(self, franchise_id: str) -> Optional[Dict]:
        """Get the default model for a franchise."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # First try to get explicitly set default
        cursor.execute('''
            SELECT m.* FROM models m
            JOIN franchises f ON m.model_id = f.default_model_id
            WHERE f.franchise_id = ?
        ''', (franchise_id,))
        row = cursor.fetchone()
        
        # If no explicit default, get latest model for franchise
        if not row:
            cursor.execute('''
                SELECT * FROM models 
                WHERE franchise_id = ?
                ORDER BY created_date DESC
                LIMIT 1
            ''', (franchise_id,))
            row = cursor.fetchone()
        
        conn.close()
        return dict(row) if row else None

    def get_latest_model_by_type(self, franchise_id: str, model_type: str) -> Optional[Dict]:
        """Get the latest model for a franchise by type."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM models
            WHERE franchise_id = ? AND model_type = ?
            ORDER BY COALESCE(training_date, created_date) DESC, created_date DESC
            LIMIT 1
        ''', (franchise_id, model_type))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_franchise_model_status(self, franchise_id: str, **fields: Any) -> bool:
        """Upsert franchise model status fields for a franchise."""
        allowed_fields = {
            'threshold_event_count',
            'last_trained_event_count',
            'next_retrain_event_count',
            'threshold_reached_at',
            'threshold_popup_shown_at',
            'model_ready_popup_shown_at',
            'retrain_popup_shown_at',
            'last_training_attempt_at',
            'last_training_status',
            'last_training_message',
            'expected_ready_date',
        }
        updates = {key: value for key, value in fields.items() if key in allowed_fields}
        if not updates:
            return True

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR IGNORE INTO franchise_model_status (franchise_id) VALUES (?)',
                (franchise_id,)
            )
            assignments = ', '.join(f'{column} = ?' for column in updates.keys())
            params = list(updates.values()) + [franchise_id]
            cursor.execute(
                f'''UPDATE franchise_model_status
                    SET {assignments}
                    WHERE franchise_id = ?''',
                params
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_franchise_model_progress(self, franchise_id: str, threshold: int = 100) -> Dict[str, Any]:
        """Return dashboard-ready franchise model progress and latest status."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT
                COUNT(CASE
                    WHEN is_test = 0
                     AND include_in_training = 1
                     AND actual_total_net_sales IS NOT NULL
                    THEN 1 END) AS eligible_event_count,
                COUNT(CASE
                    WHEN is_test = 0
                     AND include_in_training = 1
                     AND actual_total_net_sales IS NOT NULL
                     AND event_features_json IS NOT NULL
                     AND TRIM(event_features_json) != ''
                    THEN 1 END) AS trainable_event_count
            FROM predictions
            WHERE franchise_id = ?
            ''',
            (franchise_id,)
        )
        counts = cursor.fetchone()

        cursor.execute(
            '''SELECT * FROM franchise_model_status WHERE franchise_id = ?''',
            (franchise_id,)
        )
        status_row = cursor.fetchone()
        conn.close()

        eligible_event_count = int((counts['eligible_event_count'] if counts else 0) or 0)
        trainable_event_count = int((counts['trainable_event_count'] if counts else 0) or 0)
        effective_threshold = max(1, int((dict(status_row)['threshold_event_count'] if status_row and status_row['threshold_event_count'] else threshold)))
        progress_pct = min(100.0, round((eligible_event_count / effective_threshold) * 100, 1))
        remaining_events = max(0, effective_threshold - eligible_event_count)
        ready_for_training = eligible_event_count >= effective_threshold
        ready_with_features = trainable_event_count >= effective_threshold

        if ready_for_training and status_row is None:
            self.update_franchise_model_status(
                franchise_id,
                threshold_event_count=effective_threshold,
                threshold_reached_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            status_dict = self.get_franchise_model_progress(franchise_id, threshold)
            return status_dict

        if ready_for_training and status_row and not status_row['threshold_reached_at']:
            self.update_franchise_model_status(
                franchise_id,
                threshold_reached_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            status_row = None

        latest_model = self.get_latest_model_by_type(franchise_id, 'franchise_specific')
        status = dict(status_row) if status_row else {}

        if status.get('last_trained_event_count') is not None:
            last_trained_event_count = int(status.get('last_trained_event_count') or 0)
        elif latest_model and latest_model.get('data_records_count') is not None:
            last_trained_event_count = int(latest_model.get('data_records_count') or 0)
        else:
            last_trained_event_count = 0

        next_retrain_event_count = int(status.get('next_retrain_event_count') or 0)
        if ready_for_training:
            if not next_retrain_event_count or next_retrain_event_count <= last_trained_event_count:
                next_retrain_event_count = self._next_retrain_target(last_trained_event_count, effective_threshold)
        else:
            next_retrain_event_count = effective_threshold

        if status.get('next_retrain_event_count') != next_retrain_event_count:
            self.update_franchise_model_status(
                franchise_id,
                threshold_event_count=effective_threshold,
                next_retrain_event_count=next_retrain_event_count,
            )

        remaining_to_next_retrain = max(0, next_retrain_event_count - eligible_event_count)
        should_retrain_now = bool(
            latest_model
            and eligible_event_count >= next_retrain_event_count
            and trainable_event_count >= next_retrain_event_count
        )

        if should_retrain_now and status.get('retrain_popup_shown_at'):
            # Clear popup marker when a new retrain cycle is due.
            self.update_franchise_model_status(franchise_id, retrain_popup_shown_at=None)

        return {
            'threshold_event_count': effective_threshold,
            'eligible_event_count': eligible_event_count,
            'trainable_event_count': trainable_event_count,
            'progress_pct': progress_pct,
            'remaining_events': remaining_events,
            'ready_for_training': ready_for_training,
            'ready_with_features': ready_with_features,
            'threshold_reached_at': status.get('threshold_reached_at'),
            'threshold_popup_shown_at': status.get('threshold_popup_shown_at'),
            'model_ready_popup_shown_at': status.get('model_ready_popup_shown_at'),
            'retrain_popup_shown_at': status.get('retrain_popup_shown_at'),
            'last_training_attempt_at': status.get('last_training_attempt_at'),
            'last_training_status': status.get('last_training_status'),
            'last_training_message': status.get('last_training_message'),
            'expected_ready_date': status.get('expected_ready_date'),
            'last_trained_event_count': last_trained_event_count,
            'next_retrain_event_count': next_retrain_event_count,
            'remaining_to_next_retrain': remaining_to_next_retrain,
            'should_retrain_now': should_retrain_now,
            'model_exists': latest_model is not None,
            'latest_model': latest_model,
        }

    @staticmethod
    def _next_retrain_target(last_trained_event_count: int, threshold: int) -> int:
        """Return the next event-count target for automatic retraining."""
        milestones = [threshold, 175, 250, 350, 500]
        trained = max(0, int(last_trained_event_count or 0))
        for milestone in milestones:
            if milestone > trained:
                return milestone

        # After 500 events, retrain every +250 events.
        return 500 + (((trained - 500) // 250) + 1) * 250
    
    # ===== SESSION OPERATIONS =====
    
    def create_session(self, session_token: str, franchise_id: str, 
                      expires_in_hours: int = 24) -> bool:
        """
        Create a new login session.
        
        Args:
            session_token: Unique session token
            franchise_id: Associated franchise
            expires_in_hours: Session expiration time (default 24 hours)
            
        Returns:
            True if successful
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
            expires_at_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO sessions 
                (session_token, franchise_id, expires_at)
                VALUES (?, ?, ?)
            ''', (session_token, franchise_id, expires_at_str))
            conn.commit()
            conn.close()
            self.record_login_event(franchise_id)
            return True
        except sqlite3.IntegrityError:
            return False

    def record_login_event(self, franchise_id: str) -> bool:
        """
        Append a login to the durable usage history.

        Called from create_session so every successful login is captured
        regardless of which entry point produced it.

        Analytics must never cost a franchise their login, so a failure here is
        swallowed rather than raised -- the caller has already committed the
        session by this point.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO login_events (franchise_id) VALUES (?)',
                (franchise_id,)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[ANALYTICS] Failed to record login event for {franchise_id}: {e}")
            return False
    
    def get_session(self, session_token: str) -> Optional[Dict]:
        """Get session by token if still valid."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM sessions 
            WHERE session_token = ? AND expires_at > datetime('now')
        ''', (session_token,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def update_session_activity(self, session_token: str) -> bool:
        """Update last activity timestamp for session."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions 
                SET last_activity = datetime('now')
                WHERE session_token = ?
            ''', (session_token,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def delete_session(self, session_token: str) -> bool:
        """Delete session (logout)."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE session_token = ?', (session_token,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def delete_expired_sessions(self) -> int:
        """Delete expired sessions and return count deleted."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM sessions 
                WHERE expires_at < datetime('now')
            ''')
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            return deleted
        except Exception:
            return 0
    
    # ===== PREDICTION HISTORY =====
    
    def record_prediction(self, prediction_id: str, franchise_id: str, model_id: str,
                         event_name: str, duration_hours: float,
                         predicted_revenue_per_hour: float,
                         predicted_total_revenue: float, confidence_lower: float,
                         confidence_upper: float, is_test: bool = False,
                         actual_total_net_sales: Optional[float] = None,
                         event_status: str = 'predicted_only',
                         include_in_training: bool = False,
                         scheduled_event_date: Optional[str] = None,
                         event_features_json: Optional[str] = None) -> bool:
        """Record a prediction for history tracking."""
        try:
            actual_per_hour = None
            if actual_total_net_sales is not None and duration_hours:
                actual_per_hour = actual_total_net_sales / duration_hours
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (prediction_id, franchise_id, model_id, event_name, duration_hours,
                 predicted_revenue_per_hour, predicted_total_revenue,
                 confidence_lower, confidence_upper, actual_total_net_sales,
                     actual_revenue_per_hour, actual_updated_timestamp, is_test,
                     event_status, include_in_training, scheduled_event_date,
                     event_features_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        CASE WHEN ? IS NULL THEN NULL ELSE datetime('now') END,
                        ?, ?, ?, ?, ?)
            ''', (prediction_id, franchise_id, model_id, event_name, duration_hours,
                  predicted_revenue_per_hour, predicted_total_revenue,
                  confidence_lower, confidence_upper, actual_total_net_sales,
                    actual_per_hour, actual_total_net_sales, int(is_test),
                    event_status, int(include_in_training), scheduled_event_date,
                    event_features_json))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_franchise_training_examples(self, franchise_id: str) -> List[Dict[str, Any]]:
        """Return completed franchise events with stored feature snapshots."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT prediction_id, duration_hours, actual_total_net_sales, event_features_json,
                   scheduled_event_date, actual_updated_timestamp, created_timestamp
            FROM predictions
            WHERE franchise_id = ?
              AND is_test = 0
              AND include_in_training = 1
              AND actual_total_net_sales IS NOT NULL
              AND event_features_json IS NOT NULL
              AND TRIM(event_features_json) != ''
            ORDER BY COALESCE(scheduled_event_date, actual_updated_timestamp, created_timestamp) ASC
            ''',
            (franchise_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_recent_predictions(self, franchise_id: str, limit: int = 10) -> List[Dict]:
        """Get recent predictions for a franchise."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE franchise_id = ?
            ORDER BY created_timestamp DESC
            LIMIT ?
        ''', (franchise_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_recent_predictions_page(
        self,
        franchise_id: str,
        page: int = 1,
        page_size: int = 20,
        status_filter: str = 'all',
        sort_by: str = 'created_timestamp',
        sort_dir: str = 'desc',
        month_start=None,
        month_end_exclusive=None,
    ) -> Tuple[List[Dict], int]:
        """Get paginated recent predictions for a franchise with filtering and sorting."""
        safe_page = max(1, int(page or 1))
        safe_page_size = max(1, min(100, int(page_size or 20)))
        offset = (safe_page - 1) * safe_page_size

        allowed_statuses = {
            'predicted_only',
            'booked_confirmed',
            'needs_outcome',
            'completed',
            'cancelled',
            'rescheduled',
        }
        normalized_status_filter = str(status_filter or 'all').strip().lower()
        if normalized_status_filter not in allowed_statuses:
            normalized_status_filter = 'all'

        sort_column_map = {
            'created_timestamp': 'created_timestamp',
            'event_name': 'event_name',
            'event_status': 'event_status',
            'event_date': 'scheduled_event_date',
            'predicted_revenue_per_hour': 'predicted_revenue_per_hour',
            'predicted_total_revenue': 'predicted_total_revenue',
            'actual_total_net_sales': 'actual_total_net_sales',
            'predicted_vs_actual_diff': '(predicted_total_revenue - actual_total_net_sales)',
            'actual_revenue_per_hour': 'actual_revenue_per_hour',
            'confidence_lower': 'confidence_lower',
            'confidence_upper': 'confidence_upper',
            'duration_hours': 'duration_hours',
        }
        numeric_sort_fields = {
            'predicted_revenue_per_hour',
            'predicted_total_revenue',
            'actual_total_net_sales',
            'predicted_vs_actual_diff',
            'actual_revenue_per_hour',
            'confidence_lower',
            'confidence_upper',
            'duration_hours',
        }

        safe_sort_by = str(sort_by or 'created_timestamp').strip().lower()
        sort_column = sort_column_map.get(safe_sort_by, 'created_timestamp')
        safe_sort_dir = 'asc' if str(sort_dir or '').strip().lower() == 'asc' else 'desc'

        event_date_expr = "COALESCE(date(NULLIF(scheduled_event_date, '')), date(actual_updated_timestamp), date(created_timestamp))"

        where_clause = 'WHERE franchise_id = ?'
        where_params: List = [franchise_id]

        if month_start and month_end_exclusive:
            where_clause += f' AND {event_date_expr} >= ? AND {event_date_expr} < ?'
            where_params.append(str(month_start))
            where_params.append(str(month_end_exclusive))

        if normalized_status_filter != 'all':
            where_clause += ' AND event_status = ?'
            where_params.append(normalized_status_filter)

        if safe_sort_by in numeric_sort_fields:
            order_by_clause = f"{sort_column} IS NULL ASC, {sort_column} {safe_sort_dir.upper()}, created_timestamp DESC"
        else:
            order_by_clause = f"{sort_column} {safe_sort_dir.upper()}, created_timestamp DESC"

        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            f'''
            SELECT COUNT(*) AS total_count
            FROM predictions
            {where_clause}
            ''',
            tuple(where_params),
        )
        total_count_row = cursor.fetchone()
        total_count = int(total_count_row['total_count'] or 0) if total_count_row else 0

        cursor.execute(
            f'''
            SELECT *
            FROM predictions
            {where_clause}
            ORDER BY {order_by_clause}
            LIMIT ? OFFSET ?
            ''',
            tuple(where_params + [safe_page_size, offset]),
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows], total_count

    def get_prediction_dashboard_stats(self, franchise_id: str, month_start=None,
                                       month_end_exclusive=None) -> Dict[str, object]:
        """Return month-scoped lifecycle and completed-event realized vs predicted stats."""
        conn = self.get_connection()
        cursor = conn.cursor()

        event_date_expr = "COALESCE(date(NULLIF(scheduled_event_date, '')), date(actual_updated_timestamp), date(created_timestamp))"
        where_clause = 'franchise_id = ?'
        params: List = [franchise_id]
        if month_start and month_end_exclusive:
            where_clause += f' AND {event_date_expr} >= ? AND {event_date_expr} < ?'
            params.append(str(month_start))
            params.append(str(month_end_exclusive))

        cursor.execute(
            f'''
            SELECT
                COUNT(CASE
                    WHEN is_test = 0
                     AND event_status = 'completed'
                     AND actual_total_net_sales IS NOT NULL
                    THEN 1 END) AS realized_count,
                COALESCE(SUM(CASE
                    WHEN is_test = 0
                     AND event_status = 'completed'
                     AND actual_total_net_sales IS NOT NULL
                    THEN actual_total_net_sales ELSE 0 END), 0) AS realized_total,
                COALESCE(SUM(CASE
                    WHEN is_test = 0
                     AND event_status = 'completed'
                     AND actual_total_net_sales IS NOT NULL
                    THEN predicted_total_revenue ELSE 0 END), 0) AS completed_predicted_total,
                COALESCE(SUM(CASE
                    WHEN is_test = 0
                     AND event_status = 'completed'
                     AND actual_total_net_sales IS NOT NULL
                     AND duration_hours IS NOT NULL
                     AND duration_hours > 0
                    THEN duration_hours ELSE 0 END), 0) AS completed_duration_hours,
                COUNT(CASE
                    WHEN is_test = 0
                     AND event_status = 'predicted_only'
                    THEN 1 END) AS forecast_count,
                COUNT(CASE
                    WHEN is_test = 0
                     AND event_status = 'booked_confirmed'
                    THEN 1 END) AS booked_count,
                COUNT(CASE
                    WHEN is_test = 0
                     AND event_status = 'needs_outcome'
                    THEN 1 END) AS needs_outcome_count
            FROM predictions
            WHERE {where_clause}
            ''',
            tuple(params)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {
                'realized_count': 0,
                'realized_total': 0.0,
                'completed_predicted_total': 0.0,
                'realized_vs_predicted_delta': 0.0,
                'realized_vs_predicted_pct': None,
                'monthly_actual_net_sales_per_hour': None,
                'forecast_count': 0,
                'booked_count': 0,
                'needs_outcome_count': 0,
            }

        realized_total = float(row['realized_total'] or 0.0)
        completed_predicted_total = float(row['completed_predicted_total'] or 0.0)
        completed_duration_hours = float(row['completed_duration_hours'] or 0.0)
        realized_vs_predicted_pct = None
        if completed_predicted_total != 0:
            realized_vs_predicted_pct = ((realized_total - completed_predicted_total) / completed_predicted_total) * 100

        monthly_actual_net_sales_per_hour = None
        if completed_duration_hours > 0:
            monthly_actual_net_sales_per_hour = realized_total / completed_duration_hours

        return {
            'realized_count': int(row['realized_count'] or 0),
            'realized_total': realized_total,
            'completed_predicted_total': completed_predicted_total,
            'realized_vs_predicted_delta': realized_total - completed_predicted_total,
            'realized_vs_predicted_pct': realized_vs_predicted_pct,
            'monthly_actual_net_sales_per_hour': monthly_actual_net_sales_per_hour,
            'forecast_count': int(row['forecast_count'] or 0),
            'booked_count': int(row['booked_count'] or 0),
            'needs_outcome_count': int(row['needs_outcome_count'] or 0),
        }

    def update_actual_net_sales(self, franchise_id: str, prediction_id: str,
                                actual_total_net_sales: float) -> bool:
        """Update actual net sales for a prediction and compute per-hour values."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT duration_hours FROM predictions
                   WHERE franchise_id = ? AND prediction_id = ?''',
                (franchise_id, prediction_id)
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False

            duration_hours = row['duration_hours']
            actual_per_hour = None
            if duration_hours and duration_hours > 0:
                actual_per_hour = actual_total_net_sales / duration_hours

            cursor.execute(
                '''UPDATE predictions
                   SET actual_total_net_sales = ?,
                       actual_revenue_per_hour = ?,
                       actual_updated_timestamp = datetime('now'),
                       event_status = 'completed',
                       include_in_training = 1
                   WHERE franchise_id = ? AND prediction_id = ?''',
                (actual_total_net_sales, actual_per_hour, franchise_id, prediction_id)
            )
            conn.commit()
            updated = cursor.rowcount > 0
            conn.close()
            return updated
        except Exception:
            return False

    def update_prediction_status(self, franchise_id: str, prediction_id: str,
                                 event_status: str,
                                 include_in_training: Optional[bool] = None) -> bool:
        """Update lifecycle status for a prediction."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if include_in_training is None:
                cursor.execute(
                    '''UPDATE predictions
                       SET event_status = ?
                       WHERE franchise_id = ? AND prediction_id = ?''',
                    (event_status, franchise_id, prediction_id)
                )
            else:
                cursor.execute(
                    '''UPDATE predictions
                       SET event_status = ?, include_in_training = ?
                       WHERE franchise_id = ? AND prediction_id = ?''',
                    (event_status, int(include_in_training), franchise_id, prediction_id)
                )

            conn.commit()
            updated = cursor.rowcount > 0
            conn.close()
            return updated
        except Exception:
            return False

    def delete_predictions(self, franchise_id: str, prediction_ids: List[str],
                           test_only: bool = True) -> int:
        """Delete predictions for a franchise and return count deleted."""
        if not prediction_ids:
            return 0

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(prediction_ids))
            params = [franchise_id] + prediction_ids
            test_filter = ' AND is_test = 1' if test_only else ''
            cursor.execute(
                f'''DELETE FROM predictions
                    WHERE franchise_id = ?
                    AND prediction_id IN ({placeholders}){test_filter}
                ''',
                params
            )
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            return deleted
        except Exception:
            return 0
    
    # ===== BATCH UPLOAD OPERATIONS =====
    
    def record_batch_upload(self, batch_id: str, franchise_id: str, model_id: str,
                           original_filename: str, event_count: int,
                           result_csv_path: str) -> bool:
        """Record a batch upload."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO batch_uploads 
                (batch_id, franchise_id, model_id, original_filename, event_count, result_csv_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (batch_id, franchise_id, model_id, original_filename, event_count, result_csv_path))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
    def get_recent_batch_uploads(self, franchise_id: str, limit: int = 10) -> List[Dict]:
        """Get recent batch uploads for a franchise."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM batch_uploads 
            WHERE franchise_id = ?
            ORDER BY upload_timestamp DESC
            LIMIT ?
        ''', (franchise_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # ===== EQUIPMENT MAPPING OPERATIONS =====

    def get_equipment_mappings(self, franchise_id: str) -> List[Dict]:
        """Return all equipment mappings for a franchise."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM equipment_mappings WHERE franchise_id = ? ORDER BY equipment_name',
            (franchise_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def upsert_equipment_mapping(self, franchise_id: str, equipment_name: str,
                                  equipment_type: str, notes: str = '') -> bool:
        """Insert or replace an equipment mapping for a franchise."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO equipment_mappings (franchise_id, equipment_name, equipment_type, notes)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(franchise_id, equipment_name)
                   DO UPDATE SET equipment_type = excluded.equipment_type,
                                 notes = excluded.notes''',
                (franchise_id, equipment_name, equipment_type, notes)
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def delete_equipment_mapping(self, franchise_id: str, equipment_name: str) -> bool:
        """Delete an equipment mapping for a franchise."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM equipment_mappings WHERE franchise_id = ? AND equipment_name = ?',
                (franchise_id, equipment_name)
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def lookup_equipment_type(self, franchise_id: str, equipment_name: str) -> Optional[str]:
        """Return the equipment_type for a given equipment name, or None if not found.

        Matching is case-insensitive and ignores leading/trailing whitespace.
        """
        if not equipment_name or not equipment_name.strip():
            return None
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT equipment_type FROM equipment_mappings
               WHERE franchise_id = ?
                 AND LOWER(TRIM(equipment_name)) = LOWER(TRIM(?))
               LIMIT 1''',
            (franchise_id, equipment_name)
        )
        row = cursor.fetchone()
        conn.close()
        return row['equipment_type'] if row else None

    # ===== PASSWORD RESET OPERATIONS =====

    def create_password_reset_token(self, token: str, franchise_id: str, expires_at: datetime) -> bool:
        """Store a password reset token."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            # Remove any existing tokens for this franchise first
            cursor.execute('DELETE FROM password_reset_tokens WHERE franchise_id = ?', (franchise_id,))
            cursor.execute(
                'INSERT INTO password_reset_tokens (token, franchise_id, expires_at) VALUES (?, ?, ?)',
                (token, franchise_id, expires_at.isoformat())
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def get_password_reset_token(self, token: str) -> Optional[Dict]:
        """Retrieve a reset token record if it exists and has not expired."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM password_reset_tokens WHERE token = ? AND expires_at > ?',
            (token, datetime.now(timezone.utc).isoformat())
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def delete_password_reset_token(self, token: str) -> bool:
        """Delete a password reset token (after use or expiry)."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM password_reset_tokens WHERE token = ?', (token,))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    def update_franchise_password(self, franchise_id: str, new_password_hash: str) -> bool:
        """Update password hash for a franchise."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE franchises SET password_hash = ? WHERE franchise_id = ?',
                (new_password_hash, franchise_id)
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False

    # ===== UTILITY OPERATIONS =====
    
    def clear_all_data(self):
        """DANGEROUS: Clear all database data. Use for testing only."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM batch_uploads')
        cursor.execute('DELETE FROM predictions')
        cursor.execute('DELETE FROM sessions')
        cursor.execute('DELETE FROM models')
        cursor.execute('DELETE FROM franchises')
        conn.commit()
        conn.close()
