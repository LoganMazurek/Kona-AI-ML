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

DEFAULT_DB = Path(__file__).resolve().parent.parent / 'source' / 'franchise_data.db'


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

    if args.json:
        print(json.dumps({'database': db_metrics, 'logs': log_metrics}, indent=2, default=str))
    else:
        label = f'trailing {args.days} day{"s" if args.days != 1 else ""}'
        print(format_report(db_metrics, log_metrics, label, top_n=args.top), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
