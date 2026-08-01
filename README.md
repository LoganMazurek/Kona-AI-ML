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

## Configuration

`SECRET_KEY` is read from a `.env` file next to `docker-compose.yml` (gitignored;
copy `.env.example` to start). It is **required** in production — `docker compose up`
aborts without it, and the app refuses to start when `FLASK_ENV=production`.

Session cookies are signed with this key, so anyone who knows it can forge a
session for any franchise. Treat it like a password: never commit it, and never
paste it into a shell that logs history.

### Rotating SECRET_KEY

Rotating **logs out every active user** — signed-in franchises are bounced to
`/login` on their next request. No stored data is affected. Rotate whenever the
key may have been exposed:

```bash
# On the droplet, from the repo directory:
./deploy/db_snapshot.sh

# Generate a new key and write it to .env (note the leading space, which keeps
# the key out of shell history on bash/zsh with HISTCONTROL=ignorespace):
 python3 -c "import secrets; print(f'SECRET_KEY={secrets.token_urlsafe(48)}')" > .env
chmod 600 .env

docker compose down && docker compose up -d --build
curl -s https://kona-ml.loganmazurek.com/health    # expect {"status":"healthy",...}
```

If `.env` already holds other settings, edit the `SECRET_KEY` line in place
rather than overwriting the file.
