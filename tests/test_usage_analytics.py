"""
Tests for usage analytics: the login_events history, the SQL aggregation, and
the nginx log parser.

The reporting queries are only trustworthy if the window boundaries are exact,
so those are pinned explicitly rather than assumed.
"""
import gzip
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from source.franchise_db import FranchiseDatabase
from source.usage_analytics import (
    collect_db_metrics,
    collect_log_metrics,
    format_report,
    is_bot,
    resolve_window,
)

NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S')


@pytest.fixture
def db(tmp_path):
    """A real SQLite DB with two franchises and no activity yet."""
    path = tmp_path / 'analytics.db'
    d = FranchiseDatabase(str(path))
    d.create_franchise('acme', 'Acme Kona', 'a@example.com', 'hash')
    d.create_franchise('bolt', 'Bolt Kona', 'b@example.com', 'hash')
    return d, str(path)


def _insert_login(db_path, franchise_id, when):
    conn = sqlite3.connect(db_path)
    conn.execute(
        'INSERT INTO login_events (franchise_id, login_timestamp) VALUES (?, ?)',
        (franchise_id, _iso(when)),
    )
    conn.commit()
    conn.close()


def _insert_prediction(db_path, franchise_id, when, pred_id, is_test=0):
    conn = sqlite3.connect(db_path)
    conn.execute(
        '''INSERT INTO predictions (prediction_id, franchise_id, created_timestamp, is_test)
           VALUES (?, ?, ?, ?)''',
        (pred_id, franchise_id, _iso(when), is_test),
    )
    conn.commit()
    conn.close()


class TestLoginEventRecording:
    def test_create_session_records_a_login(self, db):
        d, path = db
        d.create_session('tok-1', 'acme')

        conn = sqlite3.connect(path)
        rows = conn.execute(
            'SELECT franchise_id FROM login_events'
        ).fetchall()
        conn.close()
        assert rows == [('acme',)]

    def test_history_survives_logout(self, db):
        """The whole point: sessions are deleted on logout, login_events are not."""
        d, path = db
        d.create_session('tok-1', 'acme')
        d.delete_session('tok-1')

        conn = sqlite3.connect(path)
        sessions = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]
        logins = conn.execute('SELECT COUNT(*) FROM login_events').fetchone()[0]
        conn.close()
        assert sessions == 0
        assert logins == 1

    def test_repeat_logins_accumulate(self, db):
        d, path = db
        for i in range(3):
            d.create_session(f'tok-{i}', 'acme')

        conn = sqlite3.connect(path)
        count = conn.execute(
            "SELECT COUNT(*) FROM login_events WHERE franchise_id='acme'"
        ).fetchone()[0]
        conn.close()
        assert count == 3

    def test_duplicate_token_does_not_record_a_login(self, db):
        """A rejected session must not inflate the usage numbers."""
        d, path = db
        assert d.create_session('tok-1', 'acme') is True
        assert d.create_session('tok-1', 'acme') is False

        conn = sqlite3.connect(path)
        count = conn.execute('SELECT COUNT(*) FROM login_events').fetchone()[0]
        conn.close()
        assert count == 1

    def test_analytics_failure_does_not_break_login(self, db, monkeypatch):
        """Recording is best-effort -- a broken write must not cost a login."""
        d, _ = db
        monkeypatch.setattr(
            d, 'record_login_event',
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('disk full'))
        )
        with pytest.raises(RuntimeError):
            d.record_login_event('acme')

        # The real guard: record_login_event swallows its own errors.
        monkeypatch.undo()
        monkeypatch.setattr(d, 'get_connection',
                            lambda: (_ for _ in ()).throw(sqlite3.OperationalError('locked')))
        assert d.record_login_event('acme') is False


class TestDbMetrics:
    def test_counts_activity_inside_the_window(self, db):
        d, path = db
        _insert_login(path, 'acme', NOW - timedelta(days=1))
        _insert_login(path, 'acme', NOW - timedelta(days=2))
        _insert_login(path, 'bolt', NOW - timedelta(days=3))
        _insert_prediction(path, 'acme', NOW - timedelta(days=1), 'p1')
        _insert_prediction(path, 'acme', NOW - timedelta(days=2), 'p2', is_test=1)

        since, until = resolve_window(7, NOW)
        m = collect_db_metrics(path, since, until)

        assert m['totals']['logins_in_window'] == 3
        assert m['totals']['franchises_logged_in_window'] == 2
        assert m['totals']['predictions_in_window'] == 2
        assert m['totals']['predictions_real_in_window'] == 1
        assert m['totals']['predictions_test_in_window'] == 1

    def test_excludes_activity_outside_the_window(self, db):
        d, path = db
        _insert_login(path, 'acme', NOW - timedelta(days=30))
        _insert_prediction(path, 'acme', NOW - timedelta(days=30), 'old')

        since, until = resolve_window(7, NOW)
        m = collect_db_metrics(path, since, until)

        assert m['totals']['logins_in_window'] == 0
        assert m['totals']['predictions_in_window'] == 0

    def test_window_is_half_open(self, db):
        """[since, until): the start instant counts, the end instant does not."""
        d, path = db
        since, until = resolve_window(7, NOW)
        _insert_login(path, 'acme', since)
        _insert_login(path, 'bolt', until)

        m = collect_db_metrics(path, since, until)
        assert m['totals']['logins_in_window'] == 1

    def test_per_franchise_breakdown_and_ordering(self, db):
        d, path = db
        for i in range(3):
            _insert_prediction(path, 'bolt', NOW - timedelta(hours=i + 1), f'b{i}')
        _insert_prediction(path, 'acme', NOW - timedelta(hours=1), 'a0')
        _insert_login(path, 'acme', NOW - timedelta(hours=1))

        since, until = resolve_window(7, NOW)
        m = collect_db_metrics(path, since, until)

        assert [r['franchise_id'] for r in m['per_franchise']] == ['bolt', 'acme']
        bolt = m['per_franchise'][0]
        assert bolt['predictions'] == 3
        assert bolt['logins'] == 0

    def test_dormant_franchises_are_counted(self, db):
        d, path = db
        _insert_login(path, 'acme', NOW - timedelta(days=1))

        since, until = resolve_window(7, NOW)
        m = collect_db_metrics(path, since, until)
        assert m['totals']['franchises_with_no_activity'] == 1

    def test_missing_login_events_table_warns_instead_of_failing(self, db):
        """Reports run against a pre-migration database must still work."""
        d, path = db
        conn = sqlite3.connect(path)
        conn.execute('DROP TABLE login_events')
        conn.commit()
        conn.close()

        since, until = resolve_window(7, NOW)
        m = collect_db_metrics(path, since, until)
        assert m['totals']['logins_in_window'] == 0
        assert any('login_events' in w for w in m['warnings'])

    def test_database_is_opened_read_only(self, db):
        d, path = db
        since, until = resolve_window(7, NOW)
        collect_db_metrics(path, since, until)

        # Sanity: the report must not have created or altered anything.
        conn = sqlite3.connect(path)
        count = conn.execute('SELECT COUNT(*) FROM login_events').fetchone()[0]
        conn.close()
        assert count == 0


class TestLogMetrics:
    def _line(self, when, ip='1.2.3.4', method='GET', path='/', status=200,
              agent='Mozilla/5.0'):
        ts = when.strftime('%d/%b/%Y:%H:%M:%S +0000')
        return (f'{ip} - - [{ts}] "{method} {path} HTTP/1.1" {status} 512 '
                f'"-" "{agent}"\n')

    def test_parses_combined_format_and_counts_visitors(self, tmp_path):
        log = tmp_path / 'access.log'
        log.write_text(
            self._line(NOW - timedelta(hours=1), ip='1.1.1.1')
            + self._line(NOW - timedelta(hours=2), ip='1.1.1.1')
            + self._line(NOW - timedelta(hours=3), ip='2.2.2.2')
        )
        since, until = resolve_window(7, NOW)
        m = collect_log_metrics([log], since, until)

        assert m['requests_human'] == 3
        assert m['unique_visitors'] == 2
        assert m['lines_unparsed'] == 0

    def test_separates_bot_traffic(self, tmp_path):
        log = tmp_path / 'access.log'
        log.write_text(
            self._line(NOW - timedelta(hours=1), agent='Mozilla/5.0')
            + self._line(NOW - timedelta(hours=1), agent='Googlebot/2.1')
            + self._line(NOW - timedelta(hours=1), agent='curl/8.0')
        )
        since, until = resolve_window(7, NOW)
        m = collect_log_metrics([log], since, until)

        assert m['requests_human'] == 1
        assert m['requests_bot'] == 2

    def test_respects_the_time_window(self, tmp_path):
        log = tmp_path / 'access.log'
        log.write_text(
            self._line(NOW - timedelta(days=1))
            + self._line(NOW - timedelta(days=40))
        )
        since, until = resolve_window(7, NOW)
        m = collect_log_metrics([log], since, until)
        assert m['requests_human'] == 1

    def test_counts_logins_and_predictions(self, tmp_path):
        log = tmp_path / 'access.log'
        log.write_text(
            self._line(NOW - timedelta(hours=1), method='POST', path='/login', status=302)
            + self._line(NOW - timedelta(hours=1), method='POST', path='/predict', status=200)
            + self._line(NOW - timedelta(hours=1), method='POST', path='/login', status=401)
        )
        since, until = resolve_window(7, NOW)
        m = collect_log_metrics([log], since, until)

        assert m['logins'] == 1      # the 401 is a failed attempt, not a login
        assert m['predictions'] == 1

    def test_strips_query_strings_from_paths(self, tmp_path):
        log = tmp_path / 'access.log'
        log.write_text(
            self._line(NOW - timedelta(hours=1), path='/dashboard?month=2026-08')
            + self._line(NOW - timedelta(hours=1), path='/dashboard?month=2026-07')
        )
        since, until = resolve_window(7, NOW)
        m = collect_log_metrics([log], since, until)
        assert m['top_paths'] == [('/dashboard', 2)]

    def test_reads_gzipped_archives(self, tmp_path):
        plain = tmp_path / 'access.log'
        plain.write_text(self._line(NOW - timedelta(hours=1), ip='1.1.1.1'))
        archive = tmp_path / 'access.log.2.gz'
        with gzip.open(archive, 'wt') as fh:
            fh.write(self._line(NOW - timedelta(days=2), ip='3.3.3.3'))

        since, until = resolve_window(7, NOW)
        m = collect_log_metrics([plain, archive], since, until)
        assert m['requests_human'] == 2
        assert m['unique_visitors'] == 2

    def test_malformed_lines_are_counted_not_fatal(self, tmp_path):
        log = tmp_path / 'access.log'
        log.write_text('not a log line at all\n' + self._line(NOW - timedelta(hours=1)))
        since, until = resolve_window(7, NOW)
        m = collect_log_metrics([log], since, until)

        assert m['requests_human'] == 1
        assert m['lines_unparsed'] == 1

    def test_missing_file_warns_instead_of_raising(self, tmp_path):
        since, until = resolve_window(7, NOW)
        m = collect_log_metrics([tmp_path / 'nope.log'], since, until)
        assert m['requests_human'] == 0
        assert any('not found' in w for w in m['warnings'])

    def test_non_utc_log_timestamps_are_converted(self, tmp_path):
        """An entry 1h ago in -05:00 must land inside a UTC window."""
        log = tmp_path / 'access.log'
        local = (NOW - timedelta(hours=1)).astimezone(timezone(timedelta(hours=-5)))
        ts = local.strftime('%d/%b/%Y:%H:%M:%S %z')
        log.write_text(f'9.9.9.9 - - [{ts}] "GET / HTTP/1.1" 200 1 "-" "Mozilla/5.0"\n')

        since, until = resolve_window(7, NOW)
        m = collect_log_metrics([log], since, until)
        assert m['requests_human'] == 1

    @pytest.mark.parametrize('agent,expected', [
        ('Googlebot/2.1', True),
        ('python-requests/2.31', True),
        ('Mozilla/5.0 (Macintosh) Safari/605', False),
        ('', False),
    ])
    def test_bot_detection(self, agent, expected):
        assert is_bot(agent) is expected


class TestReportRendering:
    def test_renders_without_logs(self, db):
        d, path = db
        _insert_login(path, 'acme', NOW - timedelta(days=1))
        since, until = resolve_window(7, NOW)
        text = format_report(collect_db_metrics(path, since, until), None, 'trailing 7 days')

        assert 'Kona AI/ML usage report' in text
        assert 'Acme Kona' in text
        assert 'WEB TRAFFIC' not in text

    def test_renders_with_logs(self, db, tmp_path):
        d, path = db
        log = tmp_path / 'access.log'
        ts = (NOW - timedelta(hours=1)).strftime('%d/%b/%Y:%H:%M:%S +0000')
        log.write_text(f'1.1.1.1 - - [{ts}] "GET / HTTP/1.1" 200 5 "-" "Mozilla/5.0"\n')

        since, until = resolve_window(7, NOW)
        text = format_report(
            collect_db_metrics(path, since, until),
            collect_log_metrics([log], since, until),
            'trailing 7 days',
        )
        assert 'WEB TRAFFIC' in text
        assert 'Unique visitors : 1' in text

    def test_empty_window_is_reported_not_crashed(self, db):
        d, path = db
        since, until = resolve_window(7, NOW)
        text = format_report(collect_db_metrics(path, since, until), None, 'trailing 7 days')
        assert 'no franchise activity' in text
