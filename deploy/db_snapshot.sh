#!/usr/bin/env bash
set -euo pipefail

# Creates a timestamped DB snapshot and prints an exact rollback command.
# Usage:
#   ./deploy/db_snapshot.sh [db_path] [backup_dir]
# Example:
#   ./deploy/db_snapshot.sh source/franchise_data.db backups

DB_PATH="${1:-source/franchise_data.db}"
BACKUP_DIR="${2:-backups}"

if [[ ! -f "$DB_PATH" ]]; then
  echo "ERROR: Database file not found: $DB_PATH" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"

DB_BASENAME="$(basename "$DB_PATH")"
DB_NAME="${DB_BASENAME%.*}"
DB_EXT="${DB_BASENAME##*.}"
if [[ "$DB_EXT" == "$DB_BASENAME" ]]; then
  DB_EXT="db"
fi

TS_UTC="$(date -u +%Y%m%d_%H%M%S)"
SNAPSHOT_PATH="$BACKUP_DIR/${DB_NAME}_${TS_UTC}.${DB_EXT}"
SHA_PATH="$SNAPSHOT_PATH.sha256"

cp -a "$DB_PATH" "$SNAPSHOT_PATH"
sha256sum "$SNAPSHOT_PATH" > "$SHA_PATH"

echo "Snapshot created: $SNAPSHOT_PATH"
echo "Checksum saved:   $SHA_PATH"
echo

echo "Rollback command:"
echo "./deploy/db_rollback.sh \"$DB_PATH\" \"$SNAPSHOT_PATH\""
