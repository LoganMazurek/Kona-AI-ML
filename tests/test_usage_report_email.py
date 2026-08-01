"""
Tests for emailing the usage report: SMTP config resolution, the .env loader
that makes cron work, and the --email code path in the CLI.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

from source import email_utils  # noqa: E402
import usage_report  # noqa: E402


SMTP_VARS = ('SMTP_HOST', 'SMTP_PORT', 'SMTP_USER', 'SMTP_PASSWORD',
             'SMTP_FROM', 'SMTP_USE_TLS')


@pytest.fixture(autouse=True)
def clean_smtp_env(monkeypatch):
    for var in SMTP_VARS:
        monkeypatch.delenv(var, raising=False)


class TestSmtpConfigDefaults:
    def test_uses_defaults_when_unset(self):
        cfg = email_utils._get_smtp_config()
        assert cfg['host'] == 'localhost'
        assert cfg['port'] == 587
        assert cfg['from_addr'] == 'noreply@kona-ml.app'
        assert cfg['use_tls'] is True

    def test_empty_values_fall_back_to_defaults(self, monkeypatch):
        """docker compose passes "${SMTP_HOST:-}", which sets an empty string."""
        for var in SMTP_VARS:
            monkeypatch.setenv(var, '')
        cfg = email_utils._get_smtp_config()
        assert cfg['host'] == 'localhost'
        assert cfg['port'] == 587          # int('') would raise without the guard
        assert cfg['from_addr'] == 'noreply@kona-ml.app'

    def test_real_values_are_used(self, monkeypatch):
        monkeypatch.setenv('SMTP_HOST', 'smtp.example.com')
        monkeypatch.setenv('SMTP_PORT', '2525')
        monkeypatch.setenv('SMTP_FROM', 'bot@example.com')
        monkeypatch.setenv('SMTP_USE_TLS', 'false')
        cfg = email_utils._get_smtp_config()
        assert cfg['host'] == 'smtp.example.com'
        assert cfg['port'] == 2525
        assert cfg['from_addr'] == 'bot@example.com'
        assert cfg['use_tls'] is False


class TestSendUsageReportEmail:
    def _sent(self, mock_smtp):
        return mock_smtp.return_value.__enter__.return_value

    @patch('source.email_utils.smtplib.SMTP')
    def test_sends_to_single_recipient(self, mock_smtp):
        email_utils.send_usage_report_email('a@example.com', 'Subj', 'the report')
        server = self._sent(mock_smtp)
        args = server.sendmail.call_args[0]
        assert args[1] == ['a@example.com']
        assert 'the report' in args[2]

    @patch('source.email_utils.smtplib.SMTP')
    def test_sends_to_multiple_recipients(self, mock_smtp):
        email_utils.send_usage_report_email(
            ['a@example.com', 'b@example.com'], 'Subj', 'body')
        assert self._sent(mock_smtp).sendmail.call_args[0][1] == [
            'a@example.com', 'b@example.com']

    @patch('source.email_utils.smtplib.SMTP')
    def test_report_text_is_html_escaped(self, mock_smtp):
        """Report content is not HTML; angle brackets must not break the markup."""
        email_utils.send_usage_report_email('a@example.com', 'S', 'a <b> & c')
        body = self._sent(mock_smtp).sendmail.call_args[0][2]
        assert '&lt;b&gt;' in body
        assert '&amp;' in body

    @patch('source.email_utils.smtplib.SMTP')
    def test_login_skipped_without_credentials(self, mock_smtp):
        email_utils.send_usage_report_email('a@example.com', 'S', 'body')
        self._sent(mock_smtp).login.assert_not_called()

    @patch('source.email_utils.smtplib.SMTP')
    def test_login_used_when_credentials_present(self, mock_smtp, monkeypatch):
        monkeypatch.setenv('SMTP_USER', 'u')
        monkeypatch.setenv('SMTP_PASSWORD', 'p')
        email_utils.send_usage_report_email('a@example.com', 'S', 'body')
        self._sent(mock_smtp).login.assert_called_once_with('u', 'p')

    def test_rejects_empty_recipient_list(self):
        with pytest.raises(ValueError):
            email_utils.send_usage_report_email([], 'S', 'body')

    def test_rejects_blank_recipient(self):
        with pytest.raises(ValueError):
            email_utils.send_usage_report_email(['  '], 'S', 'body')


class TestEnvFileLoader:
    def test_loads_keys(self, tmp_path, monkeypatch):
        env = tmp_path / '.env'
        env.write_text('SMTP_HOST=smtp.example.com\nSMTP_PORT=2525\n')
        monkeypatch.delenv('SMTP_HOST', raising=False)
        assert usage_report.load_env_file(env) == 2
        assert usage_report.load_env_file.__module__  # sanity
        import os
        assert os.environ['SMTP_HOST'] == 'smtp.example.com'

    def test_ignores_comments_and_blanks(self, tmp_path):
        env = tmp_path / '.env'
        env.write_text('# a comment\n\nSMTP_FROM=x@y.z\nnot-a-pair\n')
        assert usage_report.load_env_file(env) == 1

    def test_strips_surrounding_quotes(self, tmp_path, monkeypatch):
        env = tmp_path / '.env'
        env.write_text('SMTP_PASSWORD="s3cret"\n')
        monkeypatch.delenv('SMTP_PASSWORD', raising=False)
        usage_report.load_env_file(env)
        import os
        assert os.environ['SMTP_PASSWORD'] == 's3cret'

    def test_existing_environment_wins(self, tmp_path, monkeypatch):
        """An explicit export must override the file, not the other way round."""
        monkeypatch.setenv('SMTP_HOST', 'already-set')
        env = tmp_path / '.env'
        env.write_text('SMTP_HOST=from-file\n')
        usage_report.load_env_file(env)
        import os
        assert os.environ['SMTP_HOST'] == 'already-set'

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert usage_report.load_env_file(tmp_path / 'nope.env') == 0


class TestCliEmailPath:
    @pytest.fixture
    def db_path(self, tmp_path):
        from source.franchise_db import FranchiseDatabase
        p = tmp_path / 'cli.db'
        d = FranchiseDatabase(str(p))
        d.create_franchise('acme', 'Acme Kona', 'a@example.com', 'hash')
        return str(p)

    def test_prints_report_without_email_flag(self, db_path, capsys):
        assert usage_report.main(['--db', db_path, '--days', '7']) == 0
        assert 'Kona AI/ML usage report' in capsys.readouterr().out

    def test_email_flag_sends_and_prints_nothing(self, db_path, capsys, tmp_path):
        sender = MagicMock()
        with patch.dict(sys.modules, {'email_utils': MagicMock(
                send_usage_report_email=sender)}):
            rc = usage_report.main([
                '--db', db_path, '--days', '7',
                '--email', 'me@example.com',
                '--env-file', str(tmp_path / 'absent.env'),
            ])

        assert rc == 0
        # cron mails any output, so a successful run must stay silent.
        assert capsys.readouterr().out == ''
        recipients, subject, body = sender.call_args[0]
        assert recipients == ['me@example.com']
        assert 'trailing 7 days' in subject
        assert 'Kona AI/ML usage report' in body

    def test_email_flag_accepts_multiple_addresses(self, db_path, tmp_path):
        sender = MagicMock()
        with patch.dict(sys.modules, {'email_utils': MagicMock(
                send_usage_report_email=sender)}):
            usage_report.main([
                '--db', db_path, '--days', '7',
                '--email', 'a@example.com, b@example.com',
                '--env-file', str(tmp_path / 'absent.env'),
            ])
        assert sender.call_args[0][0] == ['a@example.com', 'b@example.com']

    def test_custom_subject_is_used(self, db_path, tmp_path):
        sender = MagicMock()
        with patch.dict(sys.modules, {'email_utils': MagicMock(
                send_usage_report_email=sender)}):
            usage_report.main([
                '--db', db_path, '--days', '7',
                '--email', 'me@example.com', '--subject', 'Weekly numbers',
                '--env-file', str(tmp_path / 'absent.env'),
            ])
        assert sender.call_args[0][1] == 'Weekly numbers'

    def test_send_failure_exits_nonzero_and_preserves_report(self, db_path, capsys, tmp_path):
        """A failed send must not silently lose the numbers."""
        sender = MagicMock(side_effect=OSError('connection refused'))
        with patch.dict(sys.modules, {'email_utils': MagicMock(
                send_usage_report_email=sender)}):
            rc = usage_report.main([
                '--db', db_path, '--days', '7',
                '--email', 'me@example.com',
                '--env-file', str(tmp_path / 'absent.env'),
            ])

        assert rc == 1
        err = capsys.readouterr().err
        assert 'connection refused' in err
        assert 'Kona AI/ML usage report' in err

    def test_json_can_be_emailed(self, db_path, tmp_path):
        sender = MagicMock()
        with patch.dict(sys.modules, {'email_utils': MagicMock(
                send_usage_report_email=sender)}):
            usage_report.main([
                '--db', db_path, '--days', '7', '--json',
                '--email', 'me@example.com',
                '--env-file', str(tmp_path / 'absent.env'),
            ])
        assert sender.call_args[0][2].lstrip().startswith('{')
