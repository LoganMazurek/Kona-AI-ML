"""
Franchise database management module.
Handles SQLite database initialization, schema creation, and CRUD operations
for franchises, models, sessions, and prediction tracking.
"""

import sqlite3
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple


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
        conn = sqlite3.connect(self.db_path)
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
                predicted_revenue_per_hour REAL,
                predicted_total_revenue REAL,
                confidence_lower REAL,
                confidence_upper REAL,
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

        # Ensure new columns exist for older databases
        self._ensure_column_exists(cursor, 'franchises', 'target_net_sales_per_hour', 'REAL')
        
        conn.commit()
        conn.close()
    
    # ===== FRANCHISE OPERATIONS =====
    
    def create_franchise(self, franchise_id: str, franchise_name: str, email: str, 
                        password_hash: str, target_net_sales_per_hour: float = None) -> bool:
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
                    model_path: str, model_type: str = None, 
                    training_date: datetime = None, cv_mae: float = None, 
                    cv_std: float = None, feature_count: int = None,
                    data_records_count: int = None, is_default: bool = False) -> bool:
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
                 training_date, cv_mae, cv_std, feature_count, data_records_count, is_default)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (model_id, franchise_id, model_name, model_path, model_type,
                  training_date, cv_mae, cv_std, feature_count, data_records_count, is_default))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
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
            return True
        except sqlite3.IntegrityError:
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
                         event_name: str, predicted_revenue_per_hour: float,
                         predicted_total_revenue: float, confidence_lower: float,
                         confidence_upper: float) -> bool:
        """Record a prediction for history tracking."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (prediction_id, franchise_id, model_id, event_name,
                 predicted_revenue_per_hour, predicted_total_revenue,
                 confidence_lower, confidence_upper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (prediction_id, franchise_id, model_id, event_name,
                  predicted_revenue_per_hour, predicted_total_revenue,
                  confidence_lower, confidence_upper))
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
    
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
