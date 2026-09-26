"""
Test batched querying with SQLite to ensure maximum throughput for 1.7M S1 entities.
"""

import os
import time
import sqlite3

db_path = "test_batch.db"
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("PRAGMA synchronous = OFF;")
cur.execute("PRAGMA journal_mode = OFF;")
cur.execute("PRAGMA cache_size = 100000;")
cur.execute("CREATE TABLE blocking (bkey TEXT, eid TEXT);")
cur.execute("CREATE TABLE records (eid TEXT PRIMARY KEY, name TEXT, addr TEXT, country TEXT);")

# Insert 100k dummy rows
b_rows = [(f"k_{i % 5000}", f"S2-{i}") for i in range(100000)]
r_rows = [(f"S2-{i}", f"Company {i}", f"Address {i}", "US") for i in range(100000)]

t0 = time.time()
cur.executemany("INSERT INTO blocking VALUES (?, ?);", b_rows)
cur.executemany("INSERT INTO records VALUES (?, ?, ?, ?);", r_rows)
cur.execute("CREATE INDEX idx_bkey ON blocking(bkey);")
conn.commit()
print(f"Setup 100k rows in {time.time() - t0:.2f}s")

# Test batch query for 100 S1 entities, each having 5 keys
s1_keys_map = {f"S1-{i}": [f"k_{(i * 7 + j) % 5000}" for j in range(5)] for i in range(500)}

t_query = time.time()
all_keys = list({k for keys in s1_keys_map.values() for k in keys})

# Query in chunks of 500 keys
placeholders = ",".join(["?"] * len(all_keys))
cur.execute(f"SELECT bkey, eid FROM blocking WHERE bkey IN ({placeholders})", all_keys)
matches = cur.fetchall()

# Map back to S1
key_to_eids = {}
for k, eid in matches:
    key_to_eids.setdefault(k, []).append(eid)

res = {}
for s1_id, keys in s1_keys_map.items():
    cands = set()
    for k in keys:
        cands.update(key_to_eids.get(k, []))
    res[s1_id] = cands

elapsed = time.time() - t_query
print(f"Batch resolved 500 S1 entities in {elapsed:.3f}s ({500 / elapsed:,.0f} S1/sec)")

conn.close()
if os.path.exists(db_path):
    os.remove(db_path)
