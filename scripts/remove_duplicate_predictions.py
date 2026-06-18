#!/usr/bin/env python
"""
Remove duplicate prediction rows for a franchise.
Identifies complete duplicates (all columns identical) and removes extras,
keeping one of each.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

def get_db_path():
    """Get the database path."""
    repo_root = Path(__file__).parent.parent
    return str(repo_root / "source" / "franchise_data.db")

def find_duplicates(franchise_id: str, db_path: str):
    """Find all duplicate prediction rows for a franchise."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all predictions for the franchise
    cursor.execute(
        'SELECT * FROM predictions WHERE franchise_id = ? ORDER BY created_timestamp',
        (franchise_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    # Convert to dicts and find duplicates
    predictions = [dict(row) for row in rows]
    seen = {}
    duplicates = []

    for pred in predictions:
        # Create a hashable key from all columns except prediction_id
        key_data = {k: v for k, v in pred.items() if k != 'prediction_id'}
        key = str(sorted(key_data.items()))

        if key in seen:
            duplicates.append({
                'keep_id': seen[key]['prediction_id'],
                'remove_id': pred['prediction_id'],
                'keep_timestamp': seen[key]['created_timestamp'],
                'remove_timestamp': pred['created_timestamp'],
                'event_name': pred.get('event_name', 'N/A'),
            })
        else:
            seen[key] = pred

    return duplicates

def remove_duplicates(franchise_id: str, db_path: str, duplicates: list, dry_run: bool = False):
    """Remove duplicate predictions."""
    if not duplicates:
        print("No duplicates found.")
        return 0

    print(f"\nFound {len(duplicates)} duplicate rows:")
    print("-" * 100)
    for i, dup in enumerate(duplicates, 1):
        print(f"{i}. Event: {dup['event_name']}")
        print(f"   KEEPING:   {dup['keep_id']} (created {dup['keep_timestamp']})")
        print(f"   REMOVING:  {dup['remove_id']} (created {dup['remove_timestamp']})")
    print("-" * 100)

    if dry_run:
        print(f"\n[DRY RUN] Would delete {len(duplicates)} rows")
        return 0

    # Ask for confirmation
    response = input(f"\nDelete {len(duplicates)} duplicate rows? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Cancelled.")
        return 0

    # Create backup first
    import shutil
    backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    print(f"✓ Created backup: {backup_path}")

    # Delete duplicates
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    deleted_ids = [dup['remove_id'] for dup in duplicates]
    placeholders = ','.join(['?'] * len(deleted_ids))

    cursor.execute(
        f'DELETE FROM predictions WHERE prediction_id IN ({placeholders})',
        deleted_ids
    )

    conn.commit()
    deleted_count = cursor.rowcount
    conn.close()

    print(f"✓ Deleted {deleted_count} duplicate rows")
    return deleted_count

def main():
    if len(sys.argv) < 2:
        franchise_id = 'c_naper'
        print(f"Usage: python remove_duplicate_predictions.py [franchise_id] [--dry-run]")
        print(f"Using default franchise_id: {franchise_id}")
    else:
        franchise_id = sys.argv[1]

    dry_run = '--dry-run' in sys.argv

    db_path = get_db_path()

    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    print(f"Checking for duplicates in franchise '{franchise_id}'...")
    duplicates = find_duplicates(franchise_id, db_path)

    if duplicates:
        remove_duplicates(franchise_id, db_path, duplicates, dry_run=dry_run)
    else:
        print("No duplicates found.")

if __name__ == '__main__':
    main()
