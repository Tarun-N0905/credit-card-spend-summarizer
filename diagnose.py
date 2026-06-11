"""
python diagnose2.py
"""

import os
import msgpack
import psycopg
from psycopg.rows import dict_row

SESSION_ID = "3979c4b2-6b81-4b54-a358-56899f17ce73"
PG_CONN = "postgresql://postgres:Pass%40123@localhost:5433/creditcard_db"

conn = psycopg.connect(PG_CONN, row_factory=dict_row)

# ── 1. What _get_response_pairs actually extracts ─────────────────────────────
print("=== response pairs (query → answer[:60]) ===")
rows = conn.execute(
    "SELECT blob, version FROM checkpoint_blobs "
    "WHERE thread_id = %s AND channel = 'response' ORDER BY version ASC",
    (SESSION_ID,),
).fetchall()

pairs = []
for row in rows:
    try:
        data = msgpack.unpackb(row["blob"], raw=False)
        inner = msgpack.unpackb(data.data, raw=False)
        fields = inner[2]
        query = str(fields.get("query") or "").strip()
        answer = str(fields.get("answer") or "").strip()
        pairs.append((query, answer))
        print(f"  version={row['version'][:30]}...")
        print(f"    query  repr={repr(query)}")
        print(f"    answer[:60]={answer[:60]!r}")
    except Exception as e:
        print(f"  version={row['version'][:30]}... ERROR: {e}")

# ── 2. What human messages look like after dedup ──────────────────────────────
print("\n=== human messages from blob (after dedup) ===")
blob_row = conn.execute(
    "SELECT blob FROM checkpoint_blobs WHERE thread_id = %s AND channel = 'messages' "
    "ORDER BY version DESC LIMIT 1",
    (SESSION_ID,),
).fetchone()

data = msgpack.unpackb(blob_row["blob"], raw=False)
seen = set()
human_msgs = []
for item in data:
    if not isinstance(item, msgpack.ExtType):
        continue
    inner = msgpack.unpackb(item.data, raw=False)
    fields = inner[2]
    if (fields.get("type") or "").lower() != "human":
        continue
    content = str(fields.get("content") or "").strip()
    if content not in seen:
        seen.add(content)
        human_msgs.append(content)
        print(f"  repr={repr(content)}")

# ── 3. Match check ────────────────────────────────────────────────────────────
print("\n=== match check ===")
query_to_answer = {q: a for q, a in pairs}
for hm in human_msgs:
    match = query_to_answer.get(hm)
    print(f"  {'HIT ' if match else 'MISS'} | human={repr(hm)}")

conn.close()
