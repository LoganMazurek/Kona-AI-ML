#!/usr/bin/env python3
"""
Generate a usage report for the Kona prediction app.

Read-only: it opens the database in SQLite read-only mode and only reads the
nginx logs, so it is safe to run against production while the app is serving.

Examples:
    # Trailing week, database only
    python scripts/usage_report.py --days 7

    # Trailing month, including nginx traffic (needs read access to the log)
    python scripts/usage_report.py --days 30 --logs /var/log/nginx/access.log

    # Machine-readable, for archiving a series of reports
    python scripts/usage_report.py --days 7 --json > reports/week.json

Cron example (Mondays 08:00, emailed via the local MTA):
    0 8 * * 1 cd /root/Kona-AI-ML && python3 scripts/usage_report.py --days 7 \
        --logs '/var/log/nginx/access.log*' | mail -s 'Kona weekly usage' you@example.com
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source'))

from usage_analytics import (  # noqa: E402
    collect_db_metrics,
    collect_log_metrics,
    format_report,
    resolve_window,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / 'source' / 'franchise_data.db'
DEFAULT_ENV = REPO_ROOT / '.env'


def load_env_file(path: Path) -> int:
    """
    Load KEY=VALUE pairs from a .env file into os.environ.

    cron runs with a nearly empty environment, so without this the SMTP settings
    the app gets from docker compose would be invisible to a scheduled report.
    Existing environment variables win, so an explicit export can still override
    the file.

    Returns the number of variables set.
    """
    if not path.exists():
        return 0
    count = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Usage report for the Kona prediction app.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--days', type=int, default=7,
                   help='Length of the trailing window in days (default: 7).')
    p.add_argument('--until', default=None,
                   help='End of the window as UTC YYYY-MM-DD (default: now).')
    p.add_argument('--db', default=str(DEFAULT_DB),
                   help=f'Path to the SQLite database (default: {DEFAULT_DB}).')
    p.add_argument('--logs', default=None,
                   help='nginx access log path or glob (e.g. "/var/log/nginx/access.log*"). '
                        'Omit to report on the database only.')
    p.add_argument('--top', type=int, default=15,
                   help='How many franchises to list per section (default: 15).')
    p.add_argument('--json', action='store_true',
                   help='Emit JSON instead of the text report.')
    p.add_argument('--email', default=None,
                   help='Email the report to this address (comma-separated for several) '
                        'using the SMTP_* settings, relaying through the same server the '
                        'app uses for password resets. Prints nothing on success, so a '
                        'cron run stays quiet unless something fails.')
    p.add_argument('--subject', default=None,
                   help='Subject line for --email (default: "Kona usage report -- <window>").')
    p.add_argument('--env-file', default=str(DEFAULT_ENV),
                   help=f'Read SMTP settings from this .env file (default: {DEFAULT_ENV}). '
                        'Needed under cron, which starts with an empty environment.')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not Path(args.db).exists():
        print(f"error: database not found: {args.db}", file=sys.stderr)
        return 1
    if args.days < 1:
        print('error: --days must be at least 1', file=sys.stderr)
        return 1

    until = None
    if args.until:
        try:
            until = datetime.strptime(args.until, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"error: --until must be YYYY-MM-DD, got {args.until!r}", file=sys.stderr)
            return 1

    since, end = resolve_window(args.days, until)

    db_metrics = collect_db_metrics(args.db, since, end)

    log_metrics = None
    if args.logs:
        # Accept a glob so rotated archives (access.log.1, access.log.2.gz) can be
        # included -- a monthly report needs more than the current file holds.
        paths = [Path(p) for p in sorted(glob.glob(args.logs))] or [Path(args.logs)]
        log_metrics = collect_log_metrics(paths, since, end)

    label = f'trailing {args.days} day{"s" if args.days != 1 else ""}'

    if args.json:
        payload = json.dumps({'database': db_metrics, 'logs': log_metrics},
                             indent=2, default=str)
    else:
        payload = format_report(db_metrics, log_metrics, label, top_n=args.top)

    if not args.email:
        print(payload, end='' if payload.endswith('\n') else '\n')
        return 0

    # SMTP settings live in the .env file that docker compose reads; cron has no
    # environment of its own, so load them before importing the sender.
    load_env_file(Path(args.env_file))
    from email_utils import send_usage_report_email  # noqa: E402

    recipients = [a.strip() for a in args.email.split(',') if a.strip()]
    if not recipients:
        print(f"error: --email given no valid address: {args.email!r}", file=sys.stderr)
        return 1

    subject = args.subject or f'Kona usage report -- {label}'
    try:
        send_usage_report_email(recipients, subject, payload)
    except Exception as e:
        # Fail loudly and print the report, so a scheduled run that cannot send
        # still surfaces the numbers via cron's own mail rather than losing them.
        print(f"error: failed to send report to {', '.join(recipients)}: {e}", file=sys.stderr)
        print(payload, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
