#!/usr/bin/env bash
set -euo pipefail

# Restores DB from a snapshot and creates a pre-rollback safety snapshot first.
# Usage:
#   ./deploy/db_rollback.sh <db_path> <snapshot_path> [--yes]
# Example:
#   ./deploy/db_rollback.sh source/franchise_data.db backups/franchise_data_20260423_120000.db --yes

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <db_path> <snapshot_path> [--yes]" >&2
  exit 1
fi

DB_PATH="$1"
SNAPSHOT_PATH="$2"
AUTO_YES="${3:-}"

if [[ ! -f "$SNAPSHOT_PATH" ]]; then
  echo "ERROR: Snapshot not found: $SNAPSHOT_PATH" >&2
  exit 1
fi

BACKUP_DIR="$(dirname "$SNAPSHOT_PATH")"
mkdir -p "$BACKUP_DIR"

if [[ -f "$DB_PATH" ]]; then
  TS_UTC="$(date -u +%Y%m%d_%H%M%S)"
  DB_BASENAME="$(basename "$DB_PATH")"
  DB_NAME="${DB_BASENAME%.*}"
  DB_EXT="${DB_BASENAME##*.}"
  if [[ "$DB_EXT" == "$DB_BASENAME" ]]; then
    DB_EXT="db"
  fi

  PRE_ROLLBACK_PATH="$BACKUP_DIR/${DB_NAME}_pre_rollback_${TS_UTC}.${DB_EXT}"
  cp -a "$DB_PATH" "$PRE_ROLLBACK_PATH"
  sha256sum "$PRE_ROLLBACK_PATH" > "$PRE_ROLLBACK_PATH.sha256"
  echo "Pre-rollback snapshot created: $PRE_ROLLBACK_PATH"
fi

if [[ "$AUTO_YES" != "--yes" ]]; then
  echo
  read -r -p "Proceed restoring $DB_PATH from $SNAPSHOT_PATH? [y/N] " reply
  if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "Rollback canceled."
    exit 0
  fi
fi

cp -a "$SNAPSHOT_PATH" "$DB_PATH"
sha256sum "$DB_PATH" > "$DB_PATH.sha256"

echo "Rollback complete. Active DB restored from snapshot: $SNAPSHOT_PATH"
echo "Active DB checksum saved to: $DB_PATH.sha256"
