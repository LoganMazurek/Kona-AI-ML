"""
Tests for the /health monitoring endpoint.

The endpoint must stay reachable without authentication and must signal
dependency failures through the HTTP status code, so a monitor can alert on
the status alone without parsing the body.
"""
import sqlite3
from unittest.mock import patch

import pytest


@pytest.fixture
def flask_client(tmp_path):
    """
    Flask test client backed by a real (empty) SQLite file, so the endpoint's
    connectivity check exercises the same code path it does in production.
    """
    import source.web_app as web_app_module
    from source.franchise_db import FranchiseDatabase

    db_path = tmp_path / 'health_test.db'
    db = FranchiseDatabase(str(db_path))

    with patch.object(web_app_module, 'franchise_db', db):
        web_app_module.app.config['TESTING'] = True
        yield web_app_module.app.test_client()


class TestHealthEndpoint:
    def test_healthy_when_models_and_db_are_up(self, flask_client):
        import source.web_app as web_app_module

        with patch.object(web_app_module, 'PROD_MANAGER', object()):
            res = flask_client.get('/health')

        assert res.status_code == 200
        body = res.get_json()
        assert body['status'] == 'healthy'
        assert body['models_loaded'] is True
        assert body['database_ok'] is True
        assert body['flask_running'] is True

    def test_requires_no_authentication(self, flask_client):
        """No session cookie is set, so a redirect to /login would be a regression."""
        import source.web_app as web_app_module

        with patch.object(web_app_module, 'PROD_MANAGER', object()):
            res = flask_client.get('/health')

        assert res.status_code == 200

    def test_degraded_503_when_models_failed_to_load(self, flask_client):
        import source.web_app as web_app_module

        with patch.object(web_app_module, 'PROD_MANAGER', None):
            res = flask_client.get('/health')

        assert res.status_code == 503
        body = res.get_json()
        assert body['status'] == 'degraded'
        assert body['models_loaded'] is False

    def test_degraded_503_when_database_unreachable(self, flask_client):
        import source.web_app as web_app_module

        with patch.object(web_app_module, 'PROD_MANAGER', object()), \
             patch.object(
                 web_app_module.franchise_db,
                 'get_connection',
                 side_effect=sqlite3.OperationalError('unable to open database file'),
             ):
            res = flask_client.get('/health')

        assert res.status_code == 503
        body = res.get_json()
        assert body['status'] == 'degraded'
        assert body['database_ok'] is False

    def test_response_leaks_no_internal_details(self, flask_client):
        """The endpoint is public, so it must not expose paths or row counts."""
        import source.web_app as web_app_module

        with patch.object(web_app_module, 'PROD_MANAGER', object()):
            body = flask_client.get('/health').get_json()

        assert set(body) == {
            'status',
            'python_version',
            'flask_running',
            'models_loaded',
            'database_ok',
        }
