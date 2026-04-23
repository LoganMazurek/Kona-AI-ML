# DB Backup Snapshot and Rollback

This project stores application data in `source/franchise_data.db`.
Use the scripts below before deployment and for emergency rollback.

## 1) Create a snapshot before deploy

From repo root:

```bash
./deploy/db_snapshot.sh
```

Optional explicit paths:

```bash
./deploy/db_snapshot.sh source/franchise_data.db backups
```

What it does:
- Creates a UTC timestamped snapshot under `backups/`
- Writes a SHA256 checksum beside the snapshot
- Prints the exact rollback command to reuse later

## 2) Roll back to a snapshot

From repo root:

```bash
./deploy/db_rollback.sh source/franchise_data.db backups/<snapshot-file>.db
```

Non-interactive mode:

```bash
./deploy/db_rollback.sh source/franchise_data.db backups/<snapshot-file>.db --yes
```

What it does:
- Creates a pre-rollback safety snapshot first
- Restores the requested snapshot over active DB
- Writes a SHA256 checksum for the active DB

## 3) Typical production flow

1. Create snapshot: `./deploy/db_snapshot.sh`
2. Deploy application changes
3. Run smoke checks
4. If needed, execute printed rollback command
