"""
run_schema.py — Execute db/schema.sql against PostgreSQL using psycopg2.
Run once to create all 3 tables (nodes, chunks, node_embeddings).

Usage:
    python run_schema.py
"""

import psycopg2
import sys
from pathlib import Path

SCHEMA_PATH = "db/schema.sql"

conn = None
try:
    conn = psycopg2.connect(
        dbname="ECF",
        user="postgres",
        password="admin",
        host="localhost",
        port="5432"
    )
    conn.autocommit = True
    cursor = conn.cursor()

    schema = Path(SCHEMA_PATH).read_text(encoding="utf-8")
    cursor.execute(schema)

    print("Schema created successfully.")
    print("Tables ready: nodes, chunks, node_embeddings")

    cursor.close()

except psycopg2.OperationalError as e:
    print(f"[error] Could not connect to PostgreSQL:\n  {e}")
    sys.exit(1)

except Exception as e:
    print(f"[error] Failed to execute schema:\n  {e}")
    sys.exit(1)

finally:
    if conn:
        conn.close()