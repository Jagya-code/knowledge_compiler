"""
db/loader.py — Load pruned_knowledge_map.json + chunks.json into PostgreSQL.

5-step process:
  Step 1: Load pruned_knowledge_map.json and chunks.json from output/
  Step 2: Flatten the pruned tree recursively (track parent_id + build path_string)
  Step 3: Insert nodes into Postgres (top-down so FK references work)
  Step 4: Insert chunks into Postgres (linked to their node via chunk_ids)
  Step 5: Generate embeddings with all-MiniLM-L6-v2 and store in node_embeddings

Usage:
    python db/loader.py
    python db/loader.py --pruned output/pruned_knowledge_map.json --chunks output/chunks.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras
# from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
PRUNED_MAP_PATH = "output/pruned_knowledge_map.json"
CHUNKS_PATH     = "output/chunks.json"

# ── DB config ─────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME",     "ECF"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "admin"),
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     os.getenv("DB_PORT",     "5432"),
}

# ── Embedding model ───────────────────────────────────────────────────────────
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE      = 64   # number of summaries to embed in one batch


# ── Step 1: Load JSON files ───────────────────────────────────────────────────

def load_json_files(pruned_path: str, chunks_path: str) -> tuple[list, list]:
    """
    Load pruned_knowledge_map.json and chunks.json.
    Returns (pruned_nodes, chunks).
    """
    p = Path(pruned_path)
    if not p.exists():
        print(f"[error] Pruned map not found: {pruned_path}")
        sys.exit(1)

    c = Path(chunks_path)
    if not c.exists():
        print(f"[error] Chunks file not found: {chunks_path}")
        sys.exit(1)

    with open(p, encoding="utf-8") as f:
        pruned_nodes = json.load(f)

    with open(c, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"  Loaded {len(pruned_nodes)} root node(s) from pruned map.")
    print(f"  Loaded {len(chunks)} chunk(s) from chunks file.")
    return pruned_nodes, chunks


# ── Step 2: Flatten tree ──────────────────────────────────────────────────────

def _flatten_tree(
    nodes:       list,
    flat:        list,
    parent_id:   int | None = None,
    path_parts:  list       = None,
) -> None:
    """
    Recursively walk the pruned tree top-down.
    Appends dicts to `flat` — each dict is one node row ready for DB insert.

    Each entry:
      {
        node_id, title, depth, parent_id (temp — db pk assigned after insert),
        summary, keywords, source_docs, user_defined, guidance_context,
        path_string, chunk_ids, _children
      }

    parent_id here is the JSON-level parent tracker (index into flat list),
    resolved to actual DB PKs during insert.
    """
    if path_parts is None:
        path_parts = []

    for node in nodes:
        current_path = path_parts + [node["title"]]
        path_string  = " > ".join(current_path)

        flat_index = len(flat)   # position this node will occupy in flat list

        flat.append({
            "node_id":          node.get("node_id", ""),
            "title":            node.get("title", ""),
            "depth":            node.get("depth", len(path_parts)),
            "parent_flat_idx":  None if parent_id is None else parent_id,  # index in flat list
            "summary":          node.get("summary", ""),
            "keywords":         node.get("keywords", []),
            "source_docs":      node.get("source_docs", []),
            "user_defined":     node.get("user_defined", False),
            "guidance_context": node.get("guidance_context", {}),
            "path_string":      path_string,
            "chunk_ids":        node.get("chunk_ids", []),
            "db_pk":            None,   # filled after INSERT
        })

        # Recurse into children
        children = node.get("nodes", [])
        if children:
            _flatten_tree(children, flat, parent_id=flat_index, path_parts=current_path)


def flatten_tree(pruned_nodes: list) -> list:
    """
    Flatten the full pruned tree into an ordered list.
    Returns flat list of node dicts (parents always before children).
    """
    flat = []
    _flatten_tree(pruned_nodes, flat, parent_id=None, path_parts=[])
    print(f"  Flattened tree: {len(flat)} total node(s).")
    return flat


# ── Step 3: Insert nodes ──────────────────────────────────────────────────────

def insert_nodes(conn, flat_nodes: list) -> None:
    """
    Insert all nodes into the `nodes` table.
    Because flat_nodes is ordered top-down, parents are always inserted
    before their children — FK constraint is always satisfied.

    After each insert, stores the returned DB PK (id) back onto the flat_node
    dict so children can reference it as parent_id.
    """
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO nodes
            (node_id, title, depth, parent_id, summary, keywords,
             source_docs, user_defined, guidance_context, path_string)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """

    print(f"  Inserting {len(flat_nodes)} node(s)...")
    for node in flat_nodes:
        # Resolve parent DB PK from flat list
        parent_db_pk = None
        if node["parent_flat_idx"] is not None:
            parent_db_pk = flat_nodes[node["parent_flat_idx"]]["db_pk"]

        cursor.execute(insert_sql, (
            node["node_id"],
            node["title"],
            node["depth"],
            parent_db_pk,
            node["summary"],
            node["keywords"],        # psycopg2 maps list -> TEXT[]
            node["source_docs"],
            node["user_defined"],
            json.dumps(node["guidance_context"]),
            node["path_string"],
        ))

        db_pk = cursor.fetchone()[0]
        node["db_pk"] = db_pk        # store for children to reference

    conn.commit()
    cursor.close()
    print(f"  Inserted {len(flat_nodes)} node(s) into `nodes`.")


# ── Step 4: Insert chunks ─────────────────────────────────────────────────────

def insert_chunks(conn, flat_nodes: list, chunks: list) -> None:
    """
    Insert chunks into the `chunks` table.

    Strategy:
      - Build a lookup: chunk_id -> chunk dict  (from chunks.json)
      - For each node, iterate its chunk_ids list
      - Insert each chunk with FK = node's db_pk
      - Skip chunk_ids not found in chunks.json (warns)
    """
    cursor = conn.cursor()

    # Build lookup from chunk_id -> chunk dict
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    insert_sql = """
        INSERT INTO chunks
            (chunk_id, node_id, doc_id, filename, heading, text, page_approx, summary)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id) DO NOTHING;
    """

    total_inserted = 0
    total_skipped  = 0

    print(f"  Inserting chunks for {len(flat_nodes)} node(s)...")
    for node in flat_nodes:
        node_db_pk = node["db_pk"]

        for chunk_id in node.get("chunk_ids", []):
            chunk = chunk_lookup.get(chunk_id)
            if chunk is None:
                print(f"  [warn] chunk_id '{chunk_id}' not found in chunks.json — skipping.")
                total_skipped += 1
                continue

            cursor.execute(insert_sql, (
                chunk.get("chunk_id", ""),
                node_db_pk,
                chunk.get("doc_id", ""),
                chunk.get("filename", ""),
                chunk.get("heading", ""),
                chunk.get("text", ""),
                chunk.get("page_approx"),
                chunk.get("summary", ""),
            ))
            total_inserted += 1

    conn.commit()
    cursor.close()
    print(f"  Inserted {total_inserted} chunk(s) into `chunks` ({total_skipped} skipped).")


# ── Step 5: Generate and store embeddings ─────────────────────────────────────

# def insert_embeddings(conn, flat_nodes: list) -> None:
#     """
#     Generate embeddings for every node summary using all-MiniLM-L6-v2,
#     then insert into node_embeddings table.

#     Processes in batches of BATCH_SIZE for efficiency.
#     Nodes with empty summaries get a zero-vector embedding.
#     """
#     print(f"  Loading embedding model: {EMBEDDING_MODEL} ...")
#     model = SentenceTransformer(EMBEDDING_MODEL)

#     cursor = conn.cursor()

#     insert_sql = """
#         INSERT INTO node_embeddings (node_id, embedding)
#         VALUES (%s, %s)
#         ON CONFLICT (node_id) DO UPDATE SET embedding = EXCLUDED.embedding;
#     """

#     # Collect (db_pk, summary) pairs
#     pairs = [(n["db_pk"], n["summary"] or n["title"]) for n in flat_nodes]

#     print(f"  Generating embeddings for {len(pairs)} node(s) in batches of {BATCH_SIZE}...")

#     total = 0
#     for batch_start in range(0, len(pairs), BATCH_SIZE):
#         batch      = pairs[batch_start: batch_start + BATCH_SIZE]
#         db_pks     = [p[0] for p in batch]
#         summaries  = [p[1] for p in batch]

#         vectors = model.encode(summaries, show_progress_bar=False)

#         rows = [
#             (db_pks[i], vectors[i].tolist())
#             for i in range(len(batch))
#         ]
#         psycopg2.extras.execute_batch(cursor, insert_sql, rows)
#         total += len(batch)
#         print(f"    Embedded {total}/{len(pairs)} nodes...", end="\r")

#     conn.commit()
#     cursor.close()
#     print(f"\n  Inserted {total} embedding(s) into `node_embeddings`.")


# ── DB connection ─────────────────────────────────────────────────────────────

def get_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"  Connected to PostgreSQL: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
        return conn
    except psycopg2.OperationalError as e:
        print(f"[error] Could not connect to PostgreSQL:\n  {e}")
        sys.exit(1)

def setup_schema(conn) -> None:
    """
    Drop and recreate all tables before every load run.
    Guarantees a clean slate — no duplicate key conflicts on re-runs.
    CASCADE handles chunks and node_embeddings automatically.
    """
    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        print(f"[error] schema.sql not found at: {schema_path}")
        sys.exit(1)
    schema_sql = schema_path.read_text(encoding="utf-8")
    cursor = conn.cursor()
    cursor.execute(schema_sql)
    conn.commit()
    cursor.close()
    print("  Schema reset — tables dropped and recreated.")
# ── Main ──────────────────────────────────────────────────────────────────────

def run_loader(pruned_path: str = PRUNED_MAP_PATH, chunks_path: str = CHUNKS_PATH) -> None:
    """
    Full 5-step load pipeline.
    """
    print("\n=== Step 0: Connect + Reset Schema ===")
    conn = get_connection()
    setup_schema(conn)
    
    print("\n=== Step 1: Load JSON files ===")
    pruned_nodes, chunks = load_json_files(pruned_path, chunks_path)

    print("\n=== Step 2: Flatten tree ===")
    flat_nodes = flatten_tree(pruned_nodes)

    print("\n=== Step 3: Connect + Insert nodes ===")
    insert_nodes(conn, flat_nodes)

    print("\n=== Step 4: Insert chunks ===")
    insert_chunks(conn, flat_nodes, chunks)

    # print("\n=== Step 5: Generate + Store embeddings ===")
    # insert_embeddings(conn, flat_nodes)

    conn.close()
    print("\n=== Loading complete ===")
    print(f"  Nodes  : {len(flat_nodes)}")
    print(f"  Chunks : {sum(len(n.get('chunk_ids', [])) for n in flat_nodes)}")
    # print(f"  Embeddings: {len(flat_nodes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load pruned knowledge map into PostgreSQL")
    parser.add_argument(
        "--pruned", default=PRUNED_MAP_PATH,
        help=f"Path to pruned_knowledge_map.json (default: {PRUNED_MAP_PATH})",
    )
    parser.add_argument(
        "--chunks", default=CHUNKS_PATH,
        help=f"Path to chunks.json (default: {CHUNKS_PATH})",
    )
    args = parser.parse_args()
    run_loader(pruned_path=args.pruned, chunks_path=args.chunks)