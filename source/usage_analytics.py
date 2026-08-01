"""
Usage analytics for the Kona prediction app.

Two independent sources, because neither is sufficient alone:

- The SQLite DB answers "what did franchises actually do" (predictions, bulk
  uploads, logins, signups) but only sees authenticated, successful actions.
- The nginx access log answers "who reached the app at all" -- unauthenticated
  traffic, error rates, crawler noise -- but knows nothing about franchises.

Both are read-only. Nothing here writes to the database.

Timestamps: every DB column read here is written by SQLite's CURRENT_TIMESTAMP
or datetime('now'), both of which store UTC as 'YYYY-MM-DD HH:MM:SS'. Windows
are therefore compared as UTC strings, and nginx timestamps are converted to
UTC before comparison so the two sources line up.
"""

import gzip
import re
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SQL_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Combined log format, which is what the kona-ml nginx site emits by default:
#   1.2.3.4 - - [01/Aug/2026:21:02:22 +0000] "GET /predict HTTP/1.1" 200 512 "-" "curl/8.0"
NGINX_LINE_RE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<path>\S+)[^"]*" '
    r'(?P<status>\d{3}) (?P<bytes>\S+)'
    r'(?: "(?P<referer>[^"]*)" "(?P<agent>[^"]*)")?'
)
NGINX_TIME_FORMAT = "%d/%b/%Y:%H:%M:%S %z"

# Substrings that identify non-human traffic. Matched case-insensitively against
# the user agent so crawler hits do not inflate the usage numbers.
BOT_MARKERS = (
    'bot', 'crawler', 'spider', 'slurp', 'bingpreview', 'headlesschrome',
    'python-requests', 'curl/', 'wget', 'go-http-client', 'scrapy',
    'facebookexternalhit', 'uptime', 'monitor', 'pingdom',
)


def _fmt(dt: datetime) -> str:
    """Render a datetime as the UTC string form SQLite stores."""
    return dt.astimezone(timezone.utc).strftime(SQL_TIME_FORMAT)


def resolve_window(days: int, until: Optional[datetime] = None) -> tuple:
    """Return the (since, until) UTC datetimes for a trailing window of N days."""
    end = (until or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return end - timedelta(days=days), end


# ---------------------------------------------------------------------------
# Database metrics
# ---------------------------------------------------------------------------

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def collect_db_metrics(db_path: str, since: datetime, until: datetime) -> Dict:
    """
    Aggregate franchise activity from the SQLite database for a time window.

    Opens the DB read-only so a report can never mutate production state, and
    so it is safe to run against the live file while the app is serving.
    """
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    lo, hi = _fmt(since), _fmt(until)

    try:
        metrics: Dict = {
            'window_start': lo,
            'window_end': hi,
            'totals': {},
            'per_franchise': [],
            'warnings': [],
        }

        metrics['totals']['franchises_total'] = conn.execute(
            'SELECT COUNT(*) FROM franchises'
        ).fetchone()[0]
        metrics['totals']['franchises_active_flag'] = conn.execute(
            'SELECT COUNT(*) FROM franchises WHERE active = 1'
        ).fetchone()[0]
        metrics['totals']['signups_in_window'] = conn.execute(
            'SELECT COUNT(*) FROM franchises WHERE created_date >= ? AND created_date < ?',
            (lo, hi),
        ).fetchone()[0]

        # Predictions. is_test rows are user-generated throwaways, so they are
        # counted separately rather than folded into real usage.
        row = conn.execute(
            '''SELECT
                 COUNT(*) AS total,
                 SUM(CASE WHEN is_test = 1 THEN 1 ELSE 0 END) AS tests,
                 COUNT(DISTINCT franchise_id) AS franchises
               FROM predictions
               WHERE created_timestamp >= ? AND created_timestamp < ?''',
            (lo, hi),
        ).fetchone()
        metrics['totals']['predictions_in_window'] = row['total'] or 0
        metrics['totals']['predictions_test_in_window'] = row['tests'] or 0
        metrics['totals']['predictions_real_in_window'] = (row['total'] or 0) - (row['tests'] or 0)
        metrics['totals']['franchises_predicting_in_window'] = row['franchises'] or 0

        # Outcome tracking: how many predictions got a real result recorded.
        metrics['totals']['actuals_recorded_in_window'] = conn.execute(
            '''SELECT COUNT(*) FROM predictions
               WHERE actual_updated_timestamp >= ? AND actual_updated_timestamp < ?''',
            (lo, hi),
        ).fetchone()[0]

        row = conn.execute(
            '''SELECT COUNT(*) AS uploads, COALESCE(SUM(event_count), 0) AS events
               FROM batch_uploads
               WHERE upload_timestamp >= ? AND upload_timestamp < ?''',
            (lo, hi),
        ).fetchone()
        metrics['totals']['bulk_uploads_in_window'] = row['uploads'] or 0
        metrics['totals']['bulk_upload_events_in_window'] = row['events'] or 0

        # Logins come from the append-only login_events table. Databases that
        # predate it report zero rather than failing the whole report.
        if _table_exists(conn, 'login_events'):
            row = conn.execute(
                '''SELECT COUNT(*) AS logins, COUNT(DISTINCT franchise_id) AS franchises
                   FROM login_events
                   WHERE login_timestamp >= ? AND login_timestamp < ?''',
                (lo, hi),
            ).fetchone()
            metrics['totals']['logins_in_window'] = row['logins'] or 0
            metrics['totals']['franchises_logged_in_window'] = row['franchises'] or 0
            login_counts = {
                r['franchise_id']: r['logins']
                for r in conn.execute(
                    '''SELECT franchise_id, COUNT(*) AS logins FROM login_events
                       WHERE login_timestamp >= ? AND login_timestamp < ?
                       GROUP BY franchise_id''',
                    (lo, hi),
                )
            }
            last_login = {
                r['franchise_id']: r['last_login']
                for r in conn.execute(
                    'SELECT franchise_id, MAX(login_timestamp) AS last_login '
                    'FROM login_events GROUP BY franchise_id'
                )
            }
        else:
            metrics['totals']['logins_in_window'] = 0
            metrics['totals']['franchises_logged_in_window'] = 0
            login_counts, last_login = {}, {}
            metrics['warnings'].append(
                'login_events table not present -- login history starts once the app '
                'with this migration has been deployed.'
            )

        # Per-franchise breakdown, so "how much is it being used by others"
        # can be answered per account rather than only in aggregate.
        pred_counts = {
            r['franchise_id']: r
            for r in conn.execute(
                '''SELECT franchise_id,
                          COUNT(*) AS predictions,
                          SUM(CASE WHEN is_test = 1 THEN 1 ELSE 0 END) AS tests
                   FROM predictions
                   WHERE created_timestamp >= ? AND created_timestamp < ?
                   GROUP BY franchise_id''',
                (lo, hi),
            )
        }
        upload_counts = {
            r['franchise_id']: r['uploads']
            for r in conn.execute(
                '''SELECT franchise_id, COUNT(*) AS uploads FROM batch_uploads
                   WHERE upload_timestamp >= ? AND upload_timestamp < ?
                   GROUP BY franchise_id''',
                (lo, hi),
            )
        }

        for f in conn.execute(
            'SELECT franchise_id, franchise_name, created_date, active FROM franchises'
        ):
            fid = f['franchise_id']
            pred = pred_counts.get(fid)
            total_preds = pred['predictions'] if pred else 0
            test_preds = (pred['tests'] or 0) if pred else 0
            metrics['per_franchise'].append({
                'franchise_id': fid,
                'franchise_name': f['franchise_name'],
                'active': bool(f['active']),
                'created_date': f['created_date'],
                'logins': login_counts.get(fid, 0),
                'last_login': last_login.get(fid),
                'predictions': total_preds,
                'predictions_real': total_preds - test_preds,
                'bulk_uploads': upload_counts.get(fid, 0),
            })

        # Busiest first; ties broken by name so output is stable between runs.
        metrics['per_franchise'].sort(
            key=lambda r: (-r['predictions'], -r['logins'], r['franchise_id'])
        )
        metrics['totals']['franchises_with_no_activity'] = sum(
            1 for r in metrics['per_franchise']
            if not r['logins'] and not r['predictions'] and not r['bulk_uploads']
        )
        return metrics
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# nginx access log metrics
# ---------------------------------------------------------------------------

def _open_log(path: Path):
    """Open a log file, transparently handling logrotate's .gz archives."""
    if path.suffix == '.gz':
        return gzip.open(path, 'rt', errors='replace')
    return open(path, 'r', errors='replace')


def iter_log_lines(paths: Iterable[Path]) -> Iterable[str]:
    """Yield lines from each readable log file, skipping ones we cannot open."""
    for path in paths:
        try:
            with _open_log(path) as fh:
                for line in fh:
                    yield line
        except OSError:
            # A rotated file can vanish mid-run, and the report should not die
            # because one archive is unreadable.
            continue


def is_bot(agent: str) -> bool:
    lowered = (agent or '').lower()
    return any(marker in lowered for marker in BOT_MARKERS)


def collect_log_metrics(log_paths: List[Path], since: datetime, until: datetime) -> Dict:
    """
    Summarise nginx access-log traffic within a window.

    Reports human and bot traffic separately -- crawler hits otherwise dominate
    the request count on a small public site and make usage look healthier than
    it is.
    """
    metrics: Dict = {
        'files_read': [str(p) for p in log_paths if p.exists()],
        'lines_total': 0,
        'lines_unparsed': 0,
        'lines_in_window': 0,
        'requests_human': 0,
        'requests_bot': 0,
        'unique_visitors': 0,
        'status_classes': {},
        'top_paths': [],
        'logins': 0,
        'predictions': 0,
        'warnings': [],
    }

    missing = [str(p) for p in log_paths if not p.exists()]
    if missing:
        metrics['warnings'].append(f"log file(s) not found: {', '.join(missing)}")

    visitors = set()
    status_classes: Counter = Counter()
    paths_seen: Counter = Counter()

    for line in iter_log_lines(log_paths):
        metrics['lines_total'] += 1
        m = NGINX_LINE_RE.match(line)
        if not m:
            metrics['lines_unparsed'] += 1
            continue
        try:
            ts = datetime.strptime(m.group('ts'), NGINX_TIME_FORMAT).astimezone(timezone.utc)
        except ValueError:
            metrics['lines_unparsed'] += 1
            continue
        if not (since <= ts < until):
            continue

        metrics['lines_in_window'] += 1
        agent = m.group('agent') or ''
        if is_bot(agent):
            metrics['requests_bot'] += 1
            continue

        metrics['requests_human'] += 1
        # IPs are counted, never retained or reported individually.
        visitors.add(m.group('ip'))
        status = m.group('status')
        status_classes[f"{status[0]}xx"] += 1

        path = m.group('path').split('?', 1)[0]
        paths_seen[path] += 1
        status_int = int(status)
        if m.group('method') == 'POST' and path == '/login' and status_int < 400:
            metrics['logins'] += 1
        if m.group('method') == 'POST' and path == '/predict' and status_int < 400:
            metrics['predictions'] += 1

    metrics['unique_visitors'] = len(visitors)
    metrics['status_classes'] = dict(sorted(status_classes.items()))
    metrics['top_paths'] = paths_seen.most_common(12)

    if metrics['lines_total'] and metrics['lines_unparsed'] == metrics['lines_total']:
        metrics['warnings'].append(
            'no log lines could be parsed -- the site may not be using the default '
            'combined log format.'
        )
    return metrics


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _bar(value: int, peak: int, width: int = 24) -> str:
    if peak <= 0:
        return ''
    return '#' * max(1, round(value / peak * width)) if value else ''


def format_report(db: Dict, logs: Optional[Dict], label: str, top_n: int = 15) -> str:
    """Render the collected metrics as a plain-text report."""
    t = db['totals']
    out: List[str] = []
    out.append(f"Kona AI/ML usage report -- {label}")
    out.append(f"Window (UTC): {db['window_start']}  ->  {db['window_end']}")
    out.append('=' * 72)
    out.append('')
    out.append('ACCOUNTS')
    out.append(f"  Franchises registered (all time) : {t['franchises_total']}")
    out.append(f"  Marked active                    : {t['franchises_active_flag']}")
    out.append(f"  New signups this window          : {t['signups_in_window']}")
    out.append(f"  Logged in this window            : {t['franchises_logged_in_window']}")
    out.append(f"  No activity this window          : {t['franchises_with_no_activity']}")
    out.append('')
    out.append('ACTIVITY')
    out.append(f"  Logins                           : {t['logins_in_window']}")
    out.append(f"  Predictions (real)               : {t['predictions_real_in_window']}")
    out.append(f"  Predictions (test rows)          : {t['predictions_test_in_window']}")
    out.append(f"  Franchises making predictions    : {t['franchises_predicting_in_window']}")
    out.append(f"  Bulk uploads                     : {t['bulk_uploads_in_window']} "
               f"({t['bulk_upload_events_in_window']} events)")
    out.append(f"  Actuals recorded                 : {t['actuals_recorded_in_window']}")
    out.append('')

    rows = [r for r in db['per_franchise'] if r['logins'] or r['predictions'] or r['bulk_uploads']]
    out.append(f'PER-FRANCHISE (active this window: {len(rows)})')
    if rows:
        peak = max(r['predictions'] for r in rows) or 0
        out.append(f"  {'franchise':<24} {'logins':>7} {'preds':>7} {'bulk':>5}  last login")
        for r in rows[:top_n]:
            name = (r['franchise_name'] or r['franchise_id'])[:24]
            last = (r['last_login'] or '-')[:16]
            out.append(f"  {name:<24} {r['logins']:>7} {r['predictions']:>7} "
                       f"{r['bulk_uploads']:>5}  {last}  {_bar(r['predictions'], peak)}")
        if len(rows) > top_n:
            out.append(f"  ... and {len(rows) - top_n} more")
    else:
        out.append('  (no franchise activity in this window)')
    out.append('')

    idle = [r for r in db['per_franchise']
            if not (r['logins'] or r['predictions'] or r['bulk_uploads'])]
    if idle:
        out.append(f'DORMANT ({len(idle)} with no activity this window)')
        for r in idle[:top_n]:
            name = (r['franchise_name'] or r['franchise_id'])[:24]
            out.append(f"  {name:<24} last login: {r['last_login'] or 'never'}")
        if len(idle) > top_n:
            out.append(f"  ... and {len(idle) - top_n} more")
        out.append('')

    if logs is not None:
        out.append('WEB TRAFFIC (nginx)')
        if logs['files_read']:
            out.append(f"  Files read      : {', '.join(logs['files_read'])}")
        out.append(f"  Requests (human): {logs['requests_human']}")
        out.append(f"  Requests (bot)  : {logs['requests_bot']}")
        out.append(f"  Unique visitors : {logs['unique_visitors']}")
        if logs['status_classes']:
            summary = '  '.join(f"{k}={v}" for k, v in logs['status_classes'].items())
            out.append(f"  Status          : {summary}")
        out.append(f"  POST /login ok  : {logs['logins']}")
        out.append(f"  POST /predict ok: {logs['predictions']}")
        if logs['top_paths']:
            out.append('  Top paths:')
            for path, count in logs['top_paths']:
                out.append(f"    {count:>7}  {path[:60]}")
        if logs['lines_unparsed']:
            out.append(f"  (unparsed lines: {logs['lines_unparsed']} of {logs['lines_total']})")
        for w in logs['warnings']:
            out.append(f"  ! {w}")
        out.append('')

    for w in db['warnings']:
        out.append(f"! {w}")

    return '\n'.join(out).rstrip() + '\n'
