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

## Usage reporting

`scripts/usage_report.py` summarises how much the app is actually being used —
per franchise, over a trailing window. It is read-only (the database is opened in
SQLite read-only mode), so it is safe to run against production while the app is
serving.

```bash
python scripts/usage_report.py --days 7                            # database only
python scripts/usage_report.py --days 30 \
    --logs '/var/log/nginx/access.log*'                            # + web traffic
python scripts/usage_report.py --days 7 --json > reports/week.json # machine-readable
```

It reads two independent sources, because neither is sufficient alone:

- **The database** — signups, logins, predictions (real vs. test), bulk uploads,
  and recorded actuals, broken down per franchise, plus which accounts have gone
  dormant. This only sees authenticated, successful actions.
- **The nginx access log** (`--logs`, optional) — total traffic, unique visitors,
  status-code mix, and top paths, with crawler hits reported separately so they
  don't inflate the numbers. Pass a glob to include rotated archives; `.gz` files
  are read directly.

Login history comes from the `login_events` table, which records one row per
successful login. The `sessions` table cannot answer this — rows there are
deleted on logout and again when expired sessions are cleaned up, so it only ever
shows who is signed in right now. **`login_events` starts collecting from the
deploy that introduced it**; earlier logins are not recoverable.

### Emailing the report

`--email` sends the report over the same authenticated SMTP relay the app uses
for password resets, rather than through a local MTA. Mail sent directly from the
droplet is routinely spam-filtered or rejected by large providers, because the
droplet IP has no matching SPF/DKIM/rDNS — so this is the reliable path, and it
means no `postfix`/`mailutils` install at all.

It requires the `SMTP_*` settings in `.env` (see `.env.example`). The script reads
that file itself, because cron starts with an empty environment; point it
elsewhere with `--env-file`.

```bash
python scripts/usage_report.py --days 7 --email you@example.com
python scripts/usage_report.py --days 7 --email 'a@example.com,b@example.com'
python scripts/usage_report.py --days 7 --email you@example.com --subject 'Weekly numbers'
```

On success it prints nothing, so a cron run stays quiet. If the send fails it
exits non-zero and writes the report to stderr, so the numbers reach you through
cron's own mail rather than being lost.

Weekly cron (run as root — the nginx logs are root-only):

```cron
0 8 * * 1 cd /root/Kona-AI-ML && /usr/bin/python3 scripts/usage_report.py --days 7 --logs '/var/log/nginx/access.log*' --email you@example.com
```

Use absolute paths (cron's `PATH` is minimal), keep the log glob quoted (unquoted,
the shell expands it and only the first file is passed), and escape any literal
`%` as `\%` — in crontab `%` means newline and truncates the command.

The report script imports only the standard library, so it needs no virtualenv and
does not run inside the container.

Note that nginx's default logrotate keeps ~14 days, so a `--days 30` report will
silently under-count web traffic unless retention is extended. The database
figures are unaffected.

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
