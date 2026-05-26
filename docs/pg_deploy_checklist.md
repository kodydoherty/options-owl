# PostgreSQL Deploy Checklist

## Pre-deploy (local)

- [x] `asyncpg>=0.29` in pyproject.toml
- [x] `options_owl/db/postgres.py` — shared PG module (trades, trade_events, agent_state, ml_signals)
- [x] `options_owl/sourcing/db.py` — sourcing PG module (scoring_audit, signals, cooldowns)
- [x] `options_owl/collectors/signal_consumer.py` — sourcing→entry bridge
- [x] Paper trader dual-write (fire-and-forget)
- [x] Discord collector: PG pool init + signal consumer task
- [x] docker-compose.yml: postgres service + all bots depend_on + env vars
- [x] All 40 PG tests passing
- [x] `ENABLE_POSTGRES=true` + `DATABASE_URL` in all bot environments

## Deploy steps

1. **Set POSTGRES_PASSWORD on droplet .env:**
   ```bash
   ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145
   echo 'POSTGRES_PASSWORD=<strong_password_here>' >> /root/options-owl/.env
   ```

2. **Create journal/postgres-data dir on droplet:**
   ```bash
   mkdir -p /root/options-owl/journal/postgres-data
   ```

3. **Deploy via rebuild.sh:**
   ```bash
   ./scripts/rebuild.sh
   ```

4. **Verify postgres is running:**
   ```bash
   ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145 \
     "cd /root/options-owl && docker compose ps postgres"
   ```

5. **Verify schema was created:**
   ```bash
   ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145 \
     "cd /root/options-owl && docker compose exec postgres psql -U owl options_owl -c '\dt'"
   ```

6. **Verify bots are writing:**
   ```bash
   ssh -i ~/.ssh/id_ed25519_do root@129.212.138.145 \
     "cd /root/options-owl && docker compose exec postgres psql -U owl options_owl -c 'SELECT agent_id, portfolio_size, last_heartbeat FROM agent_state'"
   ```

## Rollback

PG is dual-write only (Phase 1). SQLite remains primary. To disable:
- Set `ENABLE_POSTGRES=false` in each bot's environment
- Rebuild: `./scripts/rebuild.sh`
- PG container can be stopped independently: `docker compose stop postgres`

## Phase 2 (future)

- Switch signal reads to PG (replace SQLite queries)
- Cross-agent dashboard queries via PG
- Trade reconciliation between SQLite and PG
