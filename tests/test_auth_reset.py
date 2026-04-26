"""
Unit tests for AuthManager password-reset methods.
"""
import pytest
from datetime import datetime, timedelta, timezone
from werkzeug.security import generate_password_hash

from source.auth import AuthManager
from source.franchise_db import FranchiseDatabase


# ---------------------------------------------------------------------------
# Minimal in-memory mock of FranchiseDatabase
# ---------------------------------------------------------------------------

class MockDB(FranchiseDatabase):
    """Lightweight stand-in that keeps data in plain dicts."""

    def __init__(self):
        self._franchises = {
            'acme': {
                'franchise_id': 'acme',
                'franchise_name': 'Acme Kona',
                'email': 'owner@acme.com',
                'password_hash': generate_password_hash('hunter2', method='pbkdf2:sha256'),
                'active': True,
            }
        }
        self._sessions: dict = {}
        self._reset_tokens: dict = {}

    # --- franchise ---
    def get_franchise(self, franchise_id):
        return self._franchises.get(franchise_id)

    def get_franchise_by_email(self, email):
        return next((f for f in self._franchises.values() if f['email'] == email), None)

    def update_franchise_password(self, franchise_id, new_hash):
        if franchise_id not in self._franchises:
            return False
        self._franchises[franchise_id]['password_hash'] = new_hash
        return True

    # --- sessions ---
    def create_session(self, session_token, franchise_id):
        self._sessions[session_token] = {'franchise_id': franchise_id}
        return True

    def get_session(self, session_token):
        return self._sessions.get(session_token)

    def update_session_activity(self, session_token):
        pass

    def delete_session(self, session_token):
        return self._sessions.pop(session_token, None) is not None

    def delete_expired_sessions(self):
        return 0

    # --- reset tokens ---
    def create_password_reset_token(self, token, franchise_id, expires_at):
        # Remove any existing token for this franchise (mirrors production logic)
        self._reset_tokens = {
            t: r for t, r in self._reset_tokens.items()
            if r['franchise_id'] != franchise_id
        }
        self._reset_tokens[token] = {'franchise_id': franchise_id, 'expires_at': expires_at}
        return True

    def get_password_reset_token(self, token):
        record = self._reset_tokens.get(token)
        if record is None:
            return None
        if record['expires_at'] < datetime.now(timezone.utc):
            return None
        return record

    def delete_password_reset_token(self, token):
        return self._reset_tokens.pop(token, None) is not None


@pytest.fixture
def db():
    return MockDB()


@pytest.fixture
def auth(db):
    return AuthManager(db)


# ---------------------------------------------------------------------------
# generate_password_reset_token
# ---------------------------------------------------------------------------

class TestGeneratePasswordResetToken:
    def test_returns_token_for_existing_franchise(self, auth):
        token = auth.generate_password_reset_token('acme')
        assert token is not None
        assert len(token) > 20

    def test_returns_none_for_unknown_franchise(self, auth):
        token = auth.generate_password_reset_token('ghost')
        assert token is None

    def test_token_is_stored_in_db(self, auth, db):
        token = auth.generate_password_reset_token('acme')
        record = db.get_password_reset_token(token)
        assert record is not None
        assert record['franchise_id'] == 'acme'

    def test_new_token_replaces_old_one(self, auth, db):
        token1 = auth.generate_password_reset_token('acme')
        token2 = auth.generate_password_reset_token('acme')
        assert token1 != token2
        # Old token should be gone
        assert db.get_password_reset_token(token1) is None
        # New token should exist
        assert db.get_password_reset_token(token2) is not None

    def test_tokens_are_unique(self, auth, db):
        """Generating tokens for different franchises doesn't stomp each other."""
        db._franchises['rival'] = {
            'franchise_id': 'rival', 'franchise_name': 'Rival', 'email': 'r@rival.com',
            'password_hash': generate_password_hash('abc', method='pbkdf2:sha256'),
            'active': True,
        }
        t1 = auth.generate_password_reset_token('acme')
        t2 = auth.generate_password_reset_token('rival')
        assert t1 != t2
        assert db.get_password_reset_token(t1) is not None
        assert db.get_password_reset_token(t2) is not None


# ---------------------------------------------------------------------------
# validate_and_consume_reset_token
# ---------------------------------------------------------------------------

class TestValidateAndConsumeResetToken:
    def test_valid_token_returns_franchise_id(self, auth):
        token = auth.generate_password_reset_token('acme')
        franchise_id, err = auth.validate_and_consume_reset_token(token)
        assert franchise_id == 'acme'
        assert err is None

    def test_token_is_single_use(self, auth, db):
        token = auth.generate_password_reset_token('acme')
        auth.validate_and_consume_reset_token(token)
        franchise_id, err = auth.validate_and_consume_reset_token(token)
        assert franchise_id is None
        assert err is not None

    def test_unknown_token_returns_error(self, auth):
        franchise_id, err = auth.validate_and_consume_reset_token('bogus-token')
        assert franchise_id is None
        assert 'invalid' in err.lower() or 'expired' in err.lower()

    def test_expired_token_returns_error(self, auth, db):
        token = auth.generate_password_reset_token('acme')
        # Back-date the expiry
        db._reset_tokens[token]['expires_at'] = datetime.now(timezone.utc) - timedelta(seconds=1)
        franchise_id, err = auth.validate_and_consume_reset_token(token)
        assert franchise_id is None
        assert err is not None


# ---------------------------------------------------------------------------
# update_password
# ---------------------------------------------------------------------------

class TestUpdatePassword:
    def test_updates_password_hash(self, auth, db):
        assert auth.update_password('acme', 'newpassword99')
        # New password should verify
        assert AuthManager.verify_password('newpassword99', db._franchises['acme']['password_hash'])

    def test_old_password_no_longer_works(self, auth, db):
        auth.update_password('acme', 'newpassword99')
        assert not AuthManager.verify_password('hunter2', db._franchises['acme']['password_hash'])

    def test_returns_false_for_unknown_franchise(self, auth):
        assert not auth.update_password('ghost', 'whatever')


# ---------------------------------------------------------------------------
# Full end-to-end reset flow
# ---------------------------------------------------------------------------

class TestFullResetFlow:
    def test_request_reset_then_set_new_password_then_login(self, auth, db):
        # 1. User requests a reset
        token = auth.generate_password_reset_token('acme')
        assert token is not None

        # 2. User follows the link — token is validated & consumed
        franchise_id, err = auth.validate_and_consume_reset_token(token)
        assert err is None
        assert franchise_id == 'acme'

        # 3. New password is saved
        assert auth.update_password(franchise_id, 'supersecret!')

        # 4. Old password no longer works
        session, error = auth.login('acme', 'hunter2')
        assert session is None
        assert error is not None

        # 5. New password works
        session, error = auth.login('acme', 'supersecret!')
        assert session is not None
        assert error is None

    def test_login_by_email_still_works_after_reset(self, auth, db):
        token = auth.generate_password_reset_token('acme')
        franchise_id, _ = auth.validate_and_consume_reset_token(token)
        auth.update_password(franchise_id, 'newpass42')
        session, error = auth.login('owner@acme.com', 'newpass42')
        assert session is not None
        assert error is None
