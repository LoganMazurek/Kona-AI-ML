"""Safely clear one franchise's data from a Kona ML SQLite database.

This script is intended for testing workflows where you want to reset a single
franchise (for example, c_naper) without affecting other franchises.

By default, this runs in dry-run mode and only reports what would be deleted.
Use --apply to execute deletes.

Examples
--------
Dry run (default):
    python scripts/reset_franchise_data.py --franchise-id c_naper

Apply deletion with backup:
    python scripts/reset_franchise_data.py --franchise-id c_naper --apply

Full account reset (also removes franchise/models):
    python scripts/reset_franchise_data.py --franchise-id c_naper --mode full --apply
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "source" / "franchise_data.db"

EVENT_RESET_TABLES = [
    "predictions",
    "batch_uploads",
    "sessions",
    "equipment_mappings",
]

FULL_RESET_TABLES = [
    "predictions",
    "batch_uploads",
    "sessions",
    "equipment_mappings",
    "models",
    "franchises",
]


def table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    )
    return cursor.fetchone() is not None


def count_rows(cursor: sqlite3.Cursor, table_name: str, franchise_id: str) -> int:
    cursor.execute(
        f"SELECT COUNT(*) FROM {table_name} WHERE franchise_id = ?",
        (franchise_id,),
    )
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def make_backup(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_name(f"{db_path.stem}.backup_{stamp}{db_path.suffix}")
    shutil.copy2(db_path, backup)
    return backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear one franchise's rows from Kona ML database tables.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB),
        help="Path to SQLite DB. Defaults to source/franchise_data.db",
    )
    parser.add_argument(
        "--franchise-id",
        required=True,
        help="Franchise ID to clear (example: c_naper)",
    )
    parser.add_argument(
        "--mode",
        choices=["events", "full"],
        default="events",
        help="events: clear event/upload/session/mapping data; full: also remove model/franchise rows",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletions. If omitted, runs in dry-run mode.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation (only when --apply is provided).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path)

    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        return 1

    tables: List[str] = FULL_RESET_TABLES if args.mode == "full" else EVENT_RESET_TABLES

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        existing_tables = [table for table in tables if table_exists(cursor, table)]
        missing_tables = [table for table in tables if table not in existing_tables]

        if not existing_tables:
            print("No target tables exist in this DB; nothing to do.")
            return 0

        counts: Dict[str, int] = {
            table: count_rows(cursor, table, args.franchise_id) for table in existing_tables
        }

        print(f"Database: {db_path}")
        print(f"Franchise: {args.franchise_id}")
        print(f"Mode: {args.mode}")
        print(f"Action: {'APPLY' if args.apply else 'DRY-RUN'}")
        print("\nRow counts by table:")
        for table in existing_tables:
            print(f"  - {table}: {counts[table]}")

        if missing_tables:
            print("\nSkipped missing tables:")
            for table in missing_tables:
                print(f"  - {table}")

        total_rows = sum(counts.values())
        if total_rows == 0:
            print("\nNothing to delete for this franchise.")
            return 0

        if not args.apply:
            print("\nDry-run complete. Re-run with --apply to execute deletes.")
            return 0

        if not args.no_backup:
            backup_path = make_backup(db_path)
            print(f"\nBackup created: {backup_path}")

        conn.execute("BEGIN")
        try:
            deleted_counts: Dict[str, int] = {}
            for table in existing_tables:
                cursor.execute(
                    f"DELETE FROM {table} WHERE franchise_id = ?",
                    (args.franchise_id,),
                )
                deleted_counts[table] = int(cursor.rowcount or 0)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    print("\nDeleted rows:")
    for table in existing_tables:
        print(f"  - {table}: {deleted_counts.get(table, 0)}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
