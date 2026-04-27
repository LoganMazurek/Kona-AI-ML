"""
Integration tests for login, logout, and password-reset web routes.
Uses Flask's test client with patched module-level globals so no real
database or SMTP connection is required.
"""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from werkzeug.security import generate_password_hash

from source.auth import AuthManager, SessionManager
from source.franchise_db import FranchiseDatabase


# ---------------------------------------------------------------------------
# In-memory mock DB (same helper pattern as test_auth_reset.py)
# ---------------------------------------------------------------------------

class MockDB(FranchiseDatabase):
    def __init__(self):
        self._franchises = {
            'acme': {
                'franchise_id': 'acme',
                'franchise_name': 'Acme Kona',
                'email': 'owner@acme.com',
                'password_hash': generate_password_hash('hunter2', method='pbkdf2:sha256'),
                'active': True,
                'default_model_id': None,
                'target_net_sales_per_hour': None,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    return MockDB()


@pytest.fixture
def flask_client(mock_db):
    """
    Return a Flask test client with module-level globals replaced by our
    in-memory mock so no real DB or SMTP calls happen.
    """
    import source.web_app as web_app_module

    real_auth = AuthManager(mock_db)

    with patch.object(web_app_module, 'franchise_db', mock_db), \
         patch.object(web_app_module, 'auth_manager', real_auth), \
         patch('source.web_app.send_password_reset_email') as mock_reset_email, \
         patch('source.web_app.send_username_reminder_email') as mock_remind_email:

        web_app_module.app.config['TESTING'] = True
        web_app_module.app.config['WTF_CSRF_ENABLED'] = False
        client = web_app_module.app.test_client()
        # Expose helpers so tests can inspect sent emails
        client._mock_db = mock_db
        client._real_auth = real_auth
        client._mock_reset_email = mock_reset_email
        client._mock_remind_email = mock_remind_email
        yield client


def _set_session_cookie(client, token):
    """Attach a session cookie to the test client."""
    client.set_cookie(SessionManager.get_session_cookie_name(), token)


# ---------------------------------------------------------------------------
# Login route
# ---------------------------------------------------------------------------

class TestLoginRoute:
    def test_get_returns_login_page(self, flask_client):
        res = flask_client.get('/login')
        assert res.status_code == 200
        assert b'Login' in res.data

    def test_valid_credentials_redirect_to_dashboard(self, flask_client):
        res = flask_client.post(
            '/login',
            data={'franchise_id': 'acme', 'password': 'hunter2'},
            follow_redirects=False,
        )
        assert res.status_code == 302
        assert '/dashboard' in res.headers['Location']

    def test_valid_credentials_set_session_cookie(self, flask_client):
        res = flask_client.post(
            '/login',
            data={'franchise_id': 'acme', 'password': 'hunter2'},
        )
        # Flask test client stores cookies in its internal jar accessible via get_cookie
        cookie = flask_client.get_cookie(SessionManager.get_session_cookie_name())
        assert cookie is not None and cookie.value

    def test_login_by_email(self, flask_client):
        res = flask_client.post(
            '/login',
            data={'franchise_id': 'owner@acme.com', 'password': 'hunter2'},
            follow_redirects=False,
        )
        assert res.status_code == 302
        assert '/dashboard' in res.headers['Location']

    def test_wrong_password_shows_error(self, flask_client):
        res = flask_client.post(
            '/login',
            data={'franchise_id': 'acme', 'password': 'wrongpass'},
        )
        assert res.status_code == 200
        assert b'Invalid password' in res.data

    def test_unknown_franchise_shows_error(self, flask_client):
        res = flask_client.post(
            '/login',
            data={'franchise_id': 'nobody', 'password': 'whatever'},
        )
        assert res.status_code == 200
        assert b'not found' in res.data.lower()

    def test_missing_fields_shows_error(self, flask_client):
        res = flask_client.post('/login', data={'franchise_id': '', 'password': ''})
        assert res.status_code == 200
        assert b'Please provide' in res.data

    def test_inactive_account_is_rejected(self, flask_client, mock_db):
        mock_db._franchises['acme']['active'] = False
        res = flask_client.post(
            '/login',
            data={'franchise_id': 'acme', 'password': 'hunter2'},
        )
        assert b'inactive' in res.data.lower()
        mock_db._franchises['acme']['active'] = True


# ---------------------------------------------------------------------------
# Logout route
# ---------------------------------------------------------------------------

class TestLogoutRoute:
    def test_logout_clears_cookie_and_redirects(self, flask_client):
        # Log in first
        flask_client.post('/login', data={'franchise_id': 'acme', 'password': 'hunter2'})
        res = flask_client.get('/logout', follow_redirects=False)
        assert res.status_code == 302
        assert '/login' in res.headers['Location']

    def test_logout_invalidates_session_in_db(self, flask_client, mock_db):
        flask_client.post('/login', data={'franchise_id': 'acme', 'password': 'hunter2'})
        session_token = next(iter(mock_db._sessions), None)
        flask_client.get('/logout')
        assert session_token not in mock_db._sessions


# ---------------------------------------------------------------------------
# Forgot-password route
# ---------------------------------------------------------------------------

class TestForgotPasswordRoute:
    def test_get_returns_page(self, flask_client):
        res = flask_client.get('/forgot-password')
        assert res.status_code == 200
        assert b'Forgot' in res.data

    def test_valid_email_shows_success_and_sends_email(self, flask_client):
        res = flask_client.post(
            '/forgot-password',
            data={'email': 'owner@acme.com'},
        )
        assert res.status_code == 200
        assert b'reset link has been sent' in res.data.lower()
        flask_client._mock_reset_email.assert_called_once()

    def test_unknown_email_still_shows_success_no_email_sent(self, flask_client):
        """Don't leak whether an account exists."""
        res = flask_client.post(
            '/forgot-password',
            data={'email': 'ghost@nowhere.com'},
        )
        assert res.status_code == 200
        assert b'reset link has been sent' in res.data.lower()
        flask_client._mock_reset_email.assert_not_called()

    def test_stores_reset_token_in_db(self, flask_client, mock_db):
        flask_client.post('/forgot-password', data={'email': 'owner@acme.com'})
        assert len(mock_db._reset_tokens) == 1

    def test_empty_email_shows_error(self, flask_client):
        res = flask_client.post('/forgot-password', data={'email': ''})
        assert res.status_code == 200
        assert b'Please enter' in res.data

    def test_smtp_failure_is_swallowed(self, flask_client):
        """An SMTP error must not surface a 500 to the user."""
        flask_client._mock_reset_email.side_effect = Exception('SMTP down')
        res = flask_client.post('/forgot-password', data={'email': 'owner@acme.com'})
        assert res.status_code == 200
        assert b'reset link has been sent' in res.data.lower()


# ---------------------------------------------------------------------------
# Reset-password route
# ---------------------------------------------------------------------------

class TestResetPasswordRoute:
    def _get_fresh_token(self, flask_client, mock_db):
        """Helper: request a reset and return the stored token."""
        flask_client.post('/forgot-password', data={'email': 'owner@acme.com'})
        return next(iter(mock_db._reset_tokens))

    def test_get_with_valid_token_shows_form(self, flask_client, mock_db):
        token = self._get_fresh_token(flask_client, mock_db)
        res = flask_client.get(f'/reset-password/{token}')
        assert res.status_code == 200
        assert b'Reset Password' in res.data
        # Error div must not be rendered (class appears in CSS but not in an active element)
        assert b'class="alert alert-error"' not in res.data

    def test_get_with_invalid_token_shows_error(self, flask_client):
        res = flask_client.get('/reset-password/bogus-token')
        assert res.status_code == 200
        assert b'invalid' in res.data.lower() or b'expired' in res.data.lower()

    def test_post_mismatched_passwords_shows_error(self, flask_client, mock_db):
        token = self._get_fresh_token(flask_client, mock_db)
        res = flask_client.post(
            f'/reset-password/{token}',
            data={'password': 'newpass1!', 'confirm_password': 'newpass2!'},
        )
        assert b'do not match' in res.data.lower()

    def test_post_too_short_password_shows_error(self, flask_client, mock_db):
        token = self._get_fresh_token(flask_client, mock_db)
        res = flask_client.post(
            f'/reset-password/{token}',
            data={'password': 'short', 'confirm_password': 'short'},
        )
        assert b'8 characters' in res.data

    def test_post_valid_data_updates_password(self, flask_client, mock_db):
        token = self._get_fresh_token(flask_client, mock_db)
        res = flask_client.post(
            f'/reset-password/{token}',
            data={'password': 'newpassword1', 'confirm_password': 'newpassword1'},
        )
        assert res.status_code == 200
        assert b'updated' in res.data.lower()
        # Verify actual hash changed
        assert AuthManager.verify_password(
            'newpassword1',
            mock_db._franchises['acme']['password_hash'],
        )

    def test_post_valid_data_token_is_consumed(self, flask_client, mock_db):
        token = self._get_fresh_token(flask_client, mock_db)
        flask_client.post(
            f'/reset-password/{token}',
            data={'password': 'newpassword1', 'confirm_password': 'newpassword1'},
        )
        assert mock_db.get_password_reset_token(token) is None

    def test_post_expired_token_shows_error(self, flask_client, mock_db):
        token = self._get_fresh_token(flask_client, mock_db)
        mock_db._reset_tokens[token]['expires_at'] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        res = flask_client.post(
            f'/reset-password/{token}',
            data={'password': 'newpassword1', 'confirm_password': 'newpassword1'},
        )
        assert b'invalid' in res.data.lower() or b'expired' in res.data.lower()

    def test_full_reset_then_login_succeeds(self, flask_client, mock_db):
        """End-to-end: request reset → set new password → log in with it."""
        # 1. Request reset
        flask_client.post('/forgot-password', data={'email': 'owner@acme.com'})
        token = next(iter(mock_db._reset_tokens))

        # 2. Submit new password via reset link
        flask_client.post(
            f'/reset-password/{token}',
            data={'password': 'brand_new_pw', 'confirm_password': 'brand_new_pw'},
        )

        # 3. Old password must fail
        res = flask_client.post('/login', data={'franchise_id': 'acme', 'password': 'hunter2'})
        assert b'Invalid password' in res.data

        # 4. New password must succeed
        res = flask_client.post(
            '/login',
            data={'franchise_id': 'acme', 'password': 'brand_new_pw'},
            follow_redirects=False,
        )
        assert res.status_code == 302
        assert '/dashboard' in res.headers['Location']


# ---------------------------------------------------------------------------
# Forgot-username route
# ---------------------------------------------------------------------------

class TestForgotUsernameRoute:
    def test_get_returns_page(self, flask_client):
        res = flask_client.get('/forgot-username')
        assert res.status_code == 200
        assert b'Login ID' in res.data or b'Forgot' in res.data

    def test_known_email_shows_success_and_sends_email(self, flask_client):
        res = flask_client.post('/forgot-username', data={'email': 'owner@acme.com'})
        assert res.status_code == 200
        assert b'login id has been sent' in res.data.lower()
        flask_client._mock_remind_email.assert_called_once_with(
            'owner@acme.com', 'acme', 'Acme Kona'
        )

    def test_unknown_email_shows_same_success_no_email(self, flask_client):
        res = flask_client.post('/forgot-username', data={'email': 'nobody@nowhere.com'})
        assert res.status_code == 200
        assert b'login id has been sent' in res.data.lower()
        flask_client._mock_remind_email.assert_not_called()

    def test_empty_email_shows_error(self, flask_client):
        res = flask_client.post('/forgot-username', data={'email': ''})
        assert res.status_code == 200
        assert b'Please enter' in res.data

    def test_smtp_failure_is_swallowed(self, flask_client):
        flask_client._mock_remind_email.side_effect = Exception('SMTP down')
        res = flask_client.post('/forgot-username', data={'email': 'owner@acme.com'})
        assert res.status_code == 200
        assert b'login id has been sent' in res.data.lower()
