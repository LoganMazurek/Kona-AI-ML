# Kona-AI-ML
Machine learning model training and deployment with a companion web app for interfacing with the various models. Make better decisions about which events to take based on historical, weather, and demographic data.

## DB Safety Commands

Before a production deploy, create a DB snapshot:

```bash
./deploy/db_snapshot.sh
```

If rollback is needed, restore from a snapshot:

```bash
./deploy/db_rollback.sh source/franchise_data.db backups/<snapshot-file>.db --yes
```

See full runbook: `deploy/DB_BACKUP_ROLLBACK.md`

## Deploy

On the droplet, from the repo directory:

```bash
./deploy/db_snapshot.sh
git pull
docker compose down && docker compose up -d --build
```

> Note: uses `docker compose` (plugin, not the legacy `docker-compose` standalone command).
