"""
Tests for session-key resolution.

The session cookie is signed with this key, so a predictable value lets anyone
forge a session for any franchise. These tests pin the two behaviours that
matter: an explicit key is always honoured, and production refuses to fall back
to anything guessable.
"""
import pytest

from source.web_app import resolve_secret_key


class TestResolveSecretKey:
    def test_uses_explicit_key_when_set(self):
        assert resolve_secret_key({'SECRET_KEY': 'a-real-key'}) == 'a-real-key'

    def test_explicit_key_wins_in_production(self):
        env = {'SECRET_KEY': 'a-real-key', 'FLASK_ENV': 'production'}
        assert resolve_secret_key(env) == 'a-real-key'

    def test_raises_in_production_when_missing(self):
        with pytest.raises(RuntimeError, match='SECRET_KEY'):
            resolve_secret_key({'FLASK_ENV': 'production'})

    def test_raises_in_production_when_empty(self):
        """An empty value in .env must not read as 'set'."""
        with pytest.raises(RuntimeError, match='SECRET_KEY'):
            resolve_secret_key({'SECRET_KEY': '', 'FLASK_ENV': 'production'})

    def test_production_check_is_case_insensitive(self):
        with pytest.raises(RuntimeError, match='SECRET_KEY'):
            resolve_secret_key({'FLASK_ENV': 'Production'})

    def test_generates_random_key_outside_production(self):
        key = resolve_secret_key({})
        assert key
        assert len(key) >= 32

    def test_generated_keys_are_not_reused(self):
        """The old failure mode was a shared constant -- guard against its return."""
        assert resolve_secret_key({}) != resolve_secret_key({})

    def test_no_hardcoded_fallback_remains(self):
        """The previously committed default must not come back."""
        assert resolve_secret_key({}) != 'dev-key-change-in-production'
