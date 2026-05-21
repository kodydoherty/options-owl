"""Find missing harvester data for signals, checking nearby expiry dates."""
import sqlite3
from datetime import datetime, timedelta

TRADES_DB = "journal/owlet-kody/raw_messages.db"
HARVESTER_DB = "journal/owlet-harvester/options_data.db"

sconn = sqlite3.connect(TRADES_DB)
sconn.row_factory = sqlite3.Row
signals = sconn.execute("""
    SELECT id, ticker, direction, strike, expiry, score, created_at, atm_premium
    FROM trade_signals ORDER BY id
""").fetchall()
sconn.close()

hconn = sqlite3.connect(HARVESTER_DB)

missing_contracts = []
found_alt = 0
found_exact = 0

for s in signals:
    ticker = s["ticker"]
    strike = float(s["strike"])
    expiry = s["expiry"]
    created = s["created_at"]
    direction = s["direction"] or "call"

    if expiry and "0DTE" in expiry.upper():
        dt = datetime.fromisoformat(created)
        base = (dt - timedelta(hours=4)).date()
    elif expiry and len(expiry) == 10:
        base = datetime.strptime(expiry, "%Y-%m-%d").date()
    else:
        continue

    opt_char = "C" if "call" in direction.lower() or "bull" in direction.lower() else "P"
    strike_int = int(strike * 1000)

    # Try signal date + 0..4 business days
    found_this = False
    for delta in range(0, 5):
        try_date = base + timedelta(days=delta)
        if try_date.weekday() >= 5:
            continue
        exp_str = try_date.strftime("%y%m%d")
        ct = "O:{t}{d}{o}{s:08d}".format(t=ticker, d=exp_str, o=opt_char, s=strike_int)
        count = hconn.execute(
            "SELECT COUNT(*) FROM harvest_snapshots WHERE contract_ticker = ?", (ct,)
        ).fetchone()[0]
        if count > 0:
            if delta == 0:
                found_exact += 1
            else:
                real_exp = try_date.strftime("%Y-%m-%d")
                print(
                    "  FOUND: #{} {} ${} {} sig={} -> exp={} ({} snaps)".format(
                        s["id"], ticker, strike, opt_char, base, real_exp, count
                    )
                )
                found_alt += 1
            found_this = True
            break

    if not found_this:
        missing_contracts.append(
            (s["id"], ticker, strike, opt_char, str(base))
        )

hconn.close()

print()
print("Found exact match: {}".format(found_exact))
print("Found alt expiry:  {}".format(found_alt))
print("Truly missing:     {}".format(len(missing_contracts)))
print()
for sid, tk, st, o, d in missing_contracts:
    print("  MISSING: #{} {} ${} {} date={}".format(sid, tk, st, o, d))
