import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from werkzeug.security import generate_password_hash
from source.auth import AuthManager, FranchiseDatabase


class MockFranchiseDatabase(FranchiseDatabase):
    def __init__(self):
        self.sessions = {}
        self.franchises = {
            'test_id': {
                'franchise_id': 'test_id',
                'email': 'test@example.com',
                'password_hash': generate_password_hash('password', method='pbkdf2:sha256'),
                'active': True
            }
        }

    def get_franchise(self, franchise_id):
        return self.franchises.get(franchise_id)

    def get_franchise_by_email(self, email):
        for franchise in self.franchises.values():
            if franchise['email'] == email:
                return franchise
        return None

    def create_session(self, session_token, franchise_id):
        self.sessions[session_token] = {
            'franchise_id': franchise_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        return True

    def get_session(self, session_token):
        return self.sessions.get(session_token)

    def update_session_activity(self, session_token):
        if session_token in self.sessions:
            self.sessions[session_token]['last_activity'] = datetime.now()

    def delete_session(self, session_token):
        if session_token in self.sessions:
            del self.sessions[session_token]
            return True
        return False

    def delete_expired_sessions(self):
        now = datetime.now()
        expired_sessions = [token for token, session in self.sessions.items() if (now - session['last_activity']).total_seconds() > 86400]
        for token in expired_sessions:
            del self.sessions[token]
        return len(expired_sessions)


@pytest.fixture
@patch('source.auth.FranchiseDatabase', new=MockFranchiseDatabase)
def auth_manager_fixture():
    return AuthManager(MockFranchiseDatabase())


@patch('source.auth.FranchiseDatabase', new=MockFranchiseDatabase)
def test_login_success(auth_manager_fixture):
    auth_manager = auth_manager_fixture
    session_token, error = auth_manager.login('test_id', 'password')
    assert session_token is not None
    assert error is None


@patch('source.auth.FranchiseDatabase', new=MockFranchiseDatabase)
def test_login_failure(auth_manager_fixture):
    auth_manager = auth_manager_fixture
    session_token, error = auth_manager.login('test_id', 'wrong_password')
    assert session_token is None
    assert error == 'Invalid password'


@patch('source.auth.FranchiseDatabase', new=MockFranchiseDatabase)
def test_validate_session_success(auth_manager_fixture):
    auth_manager = auth_manager_fixture
    session_token = auth_manager.login('test_id', 'password')[0]
    franchise_id, error = auth_manager.validate_session(session_token)
    assert franchise_id == 'test_id'
    assert error is None


@patch('source.auth.FranchiseDatabase', new=MockFranchiseDatabase)
def test_validate_session_failure(auth_manager_fixture):
    auth_manager = auth_manager_fixture
    franchise_id, error = auth_manager.validate_session('invalid_token')
    assert franchise_id is None
    assert error == 'Session invalid or expired'
