"""Show pipeline rejections and blocked trades for today."""
import sqlite3

conn = sqlite3.connect("journal/raw_messages.db")
rows = conn.execute(
    "SELECT event_type, ticker, detail, created_at FROM trade_events "
    "WHERE date(created_at) >= '2026-05-11' "
    "AND (event_type LIKE '%rejected%' OR event_type LIKE '%pipeline%' "
    "OR event_type LIKE '%blocked%' OR event_type LIKE '%momentum%') "
    "ORDER BY id"
).fetchall()

if not rows:
    print("No rejections found today")
else:
    for r in rows:
        det = r[2] or ""
        if len(det) > 300:
            det = det[:300]
        print("{} {:25s} {:6s} {}".format(
            r[3][:19], r[0], r[1] or "", det))

print()
print("--- All event types today ---")
types = conn.execute(
    "SELECT event_type, COUNT(*) FROM trade_events "
    "WHERE date(created_at) >= '2026-05-11' "
    "GROUP BY event_type ORDER BY COUNT(*) DESC"
).fetchall()
for t in types:
    print("  {:30s} {:3d}".format(t[0], t[1]))

conn.close()
