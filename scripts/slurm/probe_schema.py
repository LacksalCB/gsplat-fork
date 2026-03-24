import sqlite3, sys
from pathlib import Path

# Pass sqlite path as argument
path = Path(sys.argv[1])
con = sqlite3.connect(path)

print("=== TABLES ===")
for row in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
    print(" ", row[0])

print("\n=== CUPTI_ACTIVITY_KIND_MEMSET columns ===")
try:
    cur = con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMSET)")
    for row in cur.fetchall():
        print(" ", row)
except Exception as e:
    print("  ERROR:", e)

print("\n=== CUPTI_ACTIVITY_KIND_MEMSET sample row ===")
try:
    cur = con.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET LIMIT 1")
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    print("  cols:", cols)
    print("  vals:", row)
except Exception as e:
    print("  ERROR:", e)

print("\n=== CUPTI_ACTIVITY_KIND_MEMCPY columns ===")
try:
    cur = con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")
    for row in cur.fetchall():
        print(" ", row)
except Exception as e:
    print("  ERROR:", e)

print("\n=== CUPTI_ACTIVITY_KIND_MEMCPY sample row ===")
try:
    cur = con.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY LIMIT 1")
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    print("  cols:", cols)
    print("  vals:", row)
except Exception as e:
    print("  ERROR:", e)

con.close()