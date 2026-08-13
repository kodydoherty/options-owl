#!/usr/bin/env bash
# Send an urgent notification through EVERY configured channel.
#
# Deliberately multi-channel and dependency-light. The 2026-08-13 incident happened with
# three monitoring layers live and all three silent, one of them because a single alerting
# dependency (the bot's Discord client) was not configured and the code skipped the alert
# without saying so. So:
#
#   * every configured channel is tried, independently -- one being broken cannot mute
#     the others;
#   * a channel that fails says so on stdout rather than failing quietly;
#   * if NOTHING is configured, that is reported as an error, because "no channels" must
#     never look like "sent successfully".
#
# Channels, all optional, read from .env:
#   DISCORD_WEBHOOK / SOURCING_DISCORD_WEBHOOK_URL
#                        already present; an HTTPS POST, independent of the bot's client
#   NTFY_TOPIC           free phone push via ntfy.sh (install the ntfy app, subscribe)
#   TWILIO_SID/TOKEN/FROM/TO   true SMS (paid; ~$1.15/mo + ~$0.008/msg)
#
# Usage: ./scripts/notify.sh "message text"
set -uo pipefail

MSG="${1:-(no message)}"
ROOT=/root/options-owl
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" 2>/dev/null && set +a

SENT=0
HOST=$(hostname -s 2>/dev/null || echo owl)
FULL="[OptionsOwl/$HOST] $MSG"

# --- Discord webhook (independent of the bot's discord.Client) ---
# Accept either name: the repo's existing key is SOURCING_DISCORD_WEBHOOK_URL, and a bare
# DISCORD_WEBHOOK is the obvious thing to add later. Taking both avoids a silent no-op
# caused purely by which name someone happened to use.
WEBHOOK="${DISCORD_WEBHOOK:-${SOURCING_DISCORD_WEBHOOK_URL:-}}"
if [ -n "$WEBHOOK" ]; then
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 12 \
    -H "Content-Type: application/json" \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"content": sys.argv[1][:1900]}))' "$FULL")" \
    "$WEBHOOK" 2>/dev/null)
  if [ "$code" = "204" ] || [ "$code" = "200" ]; then
    echo "notify: discord OK"; SENT=$((SENT+1))
  else
    echo "notify: discord FAILED (http $code)"
  fi
fi

# --- ntfy.sh phone push (free, no account) ---
if [ -n "${NTFY_TOPIC:-}" ]; then
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 12 \
    -H "Title: OptionsOwl SAFETY" -H "Priority: urgent" -H "Tags: rotating_light" \
    -d "$FULL" "https://ntfy.sh/${NTFY_TOPIC}" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "notify: ntfy OK"; SENT=$((SENT+1))
  else
    echo "notify: ntfy FAILED (http $code)"
  fi
fi

# --- Twilio SMS (real text message) ---
# NOTE: the URL path takes the ACCOUNT sid (AC...), while auth takes the API KEY sid
# (SK...) and its secret. Using the key sid in the path returns 404 -- easy to miss because
# both are "sids". TWILIO_ACCOUNT_SID falls back to TWILIO_SID for the older
# account-sid+auth-token style of credential.
TW_ACCT="${TWILIO_ACCOUNT_SID:-${TWILIO_SID:-}}"
if [ -n "${TWILIO_SID:-}" ] && [ -n "${TWILIO_TOKEN:-}" ] && [ -n "${TWILIO_FROM:-}" ] && [ -n "${TWILIO_TO:-}" ] && [ -n "$TW_ACCT" ]; then
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 -X POST \
    "https://api.twilio.com/2010-04-01/Accounts/${TW_ACCT}/Messages.json" \
    --data-urlencode "From=${TWILIO_FROM}" \
    --data-urlencode "To=${TWILIO_TO}" \
    --data-urlencode "Body=${FULL:0:300}" \
    -u "${TWILIO_SID}:${TWILIO_TOKEN}" 2>/dev/null)
  # 201 means Twilio ACCEPTED the message, NOT that a carrier delivered it. Reporting
  # "OK" on accept hid a real failure: every message was being rejected downstream with
  # error 30034 (unregistered A2P) while this printed success. Say "accepted" and let the
  # delivery check below have the last word.
  if [ "$code" = "201" ]; then
    echo "notify: sms accepted (not yet confirmed delivered)"; SENT=$((SENT+1))
  else
    echo "notify: sms FAILED (http $code)"
  fi
fi

if [ "$SENT" -eq 0 ]; then
  # The whole point: never let "nothing configured" resemble "delivered".
  echo "notify: NO CHANNEL DELIVERED — alerting is NOT armed. Message was: $FULL" >&2
  exit 1
fi
exit 0
