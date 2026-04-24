"""
Authentication and session management module.
Handles password hashing, session token generation/validation, and login logic.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple
from werkzeug.security import generate_password_hash, check_password_hash

try:
    from Kona_AI_ML.franchise_db import FranchiseDatabase
except ImportError:
    from .franchise_db import FranchiseDatabase


class AuthManager:
    """Manage authentication and session operations."""
    
    def __init__(self, db: FranchiseDatabase):
        """
        Initialize auth manager with database connection.
        
        Args:
            db: FranchiseDatabase instance
        """
        self.db = db
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using werkzeug security.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        return generate_password_hash(password, method='pbkdf2:sha256')
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            password_hash: Hashed password to compare against
            
        Returns:
            True if password matches, False otherwise
        """
        return check_password_hash(password_hash, password)
    
    @staticmethod
    def generate_session_token() -> str:
        """
        Generate a secure random session token.
        
        Returns:
            Hex-encoded 32-byte random token
        """
        return secrets.token_hex(32)
    
    def login(self, franchise_id: str, password: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Authenticate a franchise and create a session.
        
        Args:
            franchise_id: Franchise ID or email address
            password: Plain text password
            
        Returns:
            Tuple of (session_token, error_message)
            On success: (session_token, None)
            On failure: (None, error_message)
        """
        # Find franchise by ID or email
        franchise = self.db.get_franchise(franchise_id)
        if not franchise:
            franchise = self.db.get_franchise_by_email(franchise_id)
        
        if not franchise:
            return None, "Franchise not found"
        
        if not franchise.get('active'):
            return None, "Franchise account is inactive"
        
        # Verify password
        if not self.verify_password(password, franchise['password_hash']):
            return None, "Invalid password"
        
        # Create session
        session_token = self.generate_session_token()
        if not self.db.create_session(session_token, franchise['franchise_id']):
            return None, "Failed to create session"
        
        return session_token, None
    
    def validate_session(self, session_token: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Validate a session token and return franchise ID.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            Tuple of (franchise_id, error_message)
            On success: (franchise_id, None)
            On failure: (None, error_message)
        """
        if not session_token:
            return None, "No session token provided"
        
        session = self.db.get_session(session_token)
        if not session:
            return None, "Session invalid or expired"
        
        # Update activity timestamp
        self.db.update_session_activity(session_token)
        
        return session['franchise_id'], None
    
    def logout(self, session_token: str) -> bool:
        """
        Logout a session by deleting it.
        
        Args:
            session_token: Session token to delete
            
        Returns:
            True if successful
        """
        return self.db.delete_session(session_token)
    
    def clean_expired_sessions(self) -> int:
        """
        Delete all expired sessions.
        
        Returns:
            Number of sessions deleted
        """
        return self.db.delete_expired_sessions()


class SessionManager:
    """Helper for managing session cookies in Flask."""
    
    SESSION_COOKIE_NAME = 'franchise_session'
    SESSION_COOKIE_DURATION = 24 * 60 * 60  # 24 hours in seconds
    
    @staticmethod
    def get_session_cookie_name() -> str:
        """Get the name of the session cookie."""
        return SessionManager.SESSION_COOKIE_NAME
    
    @staticmethod
    def get_session_duration() -> int:
        """Get session duration in seconds."""
        return SessionManager.SESSION_COOKIE_DURATION
