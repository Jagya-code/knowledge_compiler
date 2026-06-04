"""
db/retriever.py — 3 query functions for the knowledge store.

Functions:
  1. get_by_path(path_string)        — path prefix match → nodes + chunks
  2. semantic_search(query, top_k)   — embed query → cosine similarity → nodes + chunks
  3. traverse(node_title_or_id)      — parent, children, siblings, full path

Usage:
    from db.retriever import get_by_path, semantic_search, traverse
"""

import json
import os
import sys

import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

# ── DB config ─────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME",     "ECF"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "admin"),
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     os.getenv("DB_PORT",     "5432"),
}

# ── Embedding model (loaded once, reused across calls) ────────────────────────
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


# ── DB connection (simple persistent connection) ──────────────────────────────
_conn = None


def _get_connection():
    global _conn
    if _conn is None or _conn.closed:
        try:
            _conn = psycopg2.connect(**DB_CONFIG)
            _conn.autocommit = True
        except psycopg2.OperationalError as e:
            print(f"[error] Could not connect to PostgreSQL:\n  {e}")
            sys.exit(1)
    return _conn


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fetch_chunks_for_nodes(cursor, node_db_ids: list) -> list:
    """
    Given a list of node DB PKs, fetch all chunks belonging to those nodes.
    Returns list of chunk dicts.
    """
    if not node_db_ids:
        return []

    cursor.execute(
        """
        SELECT chunk_id, node_id, doc_id, filename, heading,
               text, page_approx, summary
        FROM   chunks
        WHERE  node_id = ANY(%s)
        ORDER  BY node_id, page_approx;
        """,
        (node_db_ids,),
    )
    rows = cursor.fetchall()
    cols = ["chunk_id", "node_id", "doc_id", "filename",
            "heading", "text", "page_approx", "summary"]
    return [dict(zip(cols, row)) for row in rows]


def _row_to_node(row: tuple, cols: list) -> dict:
    """Convert a DB row + column list into a node dict."""
    node = dict(zip(cols, row))
    # guidance_context comes back as dict already (psycopg2 parses JSONB)
    if isinstance(node.get("guidance_context"), str):
        try:
            node["guidance_context"] = json.loads(node["guidance_context"])
        except (json.JSONDecodeError, TypeError):
            node["guidance_context"] = {}
    return node


NODE_COLS = [
    "id", "node_id", "title", "depth", "parent_id",
    "summary", "keywords", "source_docs",
    "user_defined", "guidance_context", "path_string",
]

NODE_SELECT = f"SELECT {', '.join(NODE_COLS)} FROM nodes"


# ── Function 1: Path-based Retrieval ─────────────────────────────────────────

def get_by_path(path_string: str) -> dict:
    """
    Return all nodes whose path_string starts with `path_string`,
    plus all chunks belonging to those nodes.

    Example:
        get_by_path("Operations > Claims Management")
        → returns Claims Management node + all its descendants + their chunks

    Parameters
    ----------
    path_string : str
        Full or partial path, e.g. "Operations" or "Operations > Claims Management"

    Returns
    -------
    dict:
        {
          "nodes":  [ node_dict, ... ],
          "chunks": [ chunk_dict, ... ]
        }
    """
    conn   = _get_connection()
    cursor = conn.cursor()

    # Use LIKE prefix match — path_string is indexed with btree
    prefix = path_string.rstrip(" >") + "%"

    cursor.execute(
        f"{NODE_SELECT} WHERE path_string LIKE %s ORDER BY depth, path_string;",
        (prefix,),
    )
    rows  = cursor.fetchall()
    nodes = [_row_to_node(r, NODE_COLS) for r in rows]

    node_db_ids = [n["id"] for n in nodes]
    chunks      = _fetch_chunks_for_nodes(cursor, node_db_ids)

    cursor.close()

    print(f"[path] '{path_string}' → {len(nodes)} node(s), {len(chunks)} chunk(s)")
    return {"nodes": nodes, "chunks": chunks}


# ── Function 2: Semantic Retrieval ────────────────────────────────────────────

def semantic_search(query: str, top_k: int = 5) -> dict:
    """
    Embed the query and return the top-K most semantically similar nodes
    using cosine similarity against node_embeddings, plus their chunks.

    Example:
        semantic_search("how are fraud cases investigated?", top_k=5)

    Parameters
    ----------
    query : str
        Natural language query string.
    top_k : int
        Number of top results to return (default 5).

    Returns
    -------
    dict:
        {
          "query":  str,
          "nodes":  [ { ...node_fields, similarity_score: float }, ... ],
          "chunks": [ chunk_dict, ... ]
        }
    """
    conn   = _get_connection()
    cursor = conn.cursor()

    # Embed the query
    model        = _get_embedding_model()
    query_vector = model.encode([query])[0].tolist()

    # pgvector cosine distance operator: <=>
    # cosine similarity = 1 - cosine distance
    cursor.execute(
        f"""
        SELECT
            n.id, n.node_id, n.title, n.depth, n.parent_id,
            n.summary, n.keywords, n.source_docs,
            n.user_defined, n.guidance_context, n.path_string,
            1 - (ne.embedding <=> %s::vector) AS similarity_score
        FROM   node_embeddings ne
        JOIN   nodes n ON ne.node_id = n.id
        ORDER  BY ne.embedding <=> %s::vector
        LIMIT  %s;
        """,
        (query_vector, query_vector, top_k),
    )
    rows = cursor.fetchall()
    cols = NODE_COLS + ["similarity_score"]
    nodes = [_row_to_node(r, cols) for r in rows]

    node_db_ids = [n["id"] for n in nodes]
    chunks      = _fetch_chunks_for_nodes(cursor, node_db_ids)

    cursor.close()

    print(f"[semantic] '{query}' → top {len(nodes)} node(s), {len(chunks)} chunk(s)")
    return {"query": query, "nodes": nodes, "chunks": chunks}


# ── Function 3: Parent-Child Traversal ───────────────────────────────────────

def traverse(node_title_or_id: str | int) -> dict:
    """
    Given a node title (str) or node_id (str like "0003") or DB pk (int),
    return:
      - the node itself
      - its parent node
      - its children nodes
      - its sibling nodes (other children of the same parent)
      - its full ancestral path (root → node)
      - chunks belonging to the node itself

    Example:
        traverse("Claims Processing")
        traverse("0005")
        traverse(12)   # DB primary key

    Parameters
    ----------
    node_title_or_id : str | int
        Node title string, node_id string (e.g. "0003"), or DB pk integer.

    Returns
    -------
    dict:
        {
          "node":     node_dict | None,
          "parent":   node_dict | None,
          "children": [ node_dict, ... ],
          "siblings": [ node_dict, ... ],
          "path":     [ node_dict, ... ],   # root → node, inclusive
          "chunks":   [ chunk_dict, ... ]
        }
    """
    conn   = _get_connection()
    cursor = conn.cursor()

    # ── Find the target node ──────────────────────────────────────────────────
    if isinstance(node_title_or_id, int):
        cursor.execute(f"{NODE_SELECT} WHERE id = %s;", (node_title_or_id,))
    else:
        val = str(node_title_or_id)
        # Try node_id first (e.g. "0003"), then title
        cursor.execute(
            f"{NODE_SELECT} WHERE node_id = %s OR LOWER(title) = LOWER(%s) LIMIT 1;",
            (val, val),
        )

    row = cursor.fetchone()
    if row is None:
        cursor.close()
        print(f"[traverse] Node '{node_title_or_id}' not found.")
        return {"node": None, "parent": None, "children": [], "siblings": [], "path": [], "chunks": []}

    node = _row_to_node(row, NODE_COLS)
    print(f"[traverse] Found node: '{node['title']}' (depth={node['depth']})")

    # ── Parent ────────────────────────────────────────────────────────────────
    parent = None
    if node["parent_id"] is not None:
        cursor.execute(f"{NODE_SELECT} WHERE id = %s;", (node["parent_id"],))
        prow = cursor.fetchone()
        if prow:
            parent = _row_to_node(prow, NODE_COLS)

    # ── Children ──────────────────────────────────────────────────────────────
    cursor.execute(
        f"{NODE_SELECT} WHERE parent_id = %s ORDER BY title;",
        (node["id"],),
    )
    children = [_row_to_node(r, NODE_COLS) for r in cursor.fetchall()]

    # ── Siblings (other children of the same parent) ──────────────────────────
    siblings = []
    if node["parent_id"] is not None:
        cursor.execute(
            f"{NODE_SELECT} WHERE parent_id = %s AND id != %s ORDER BY title;",
            (node["parent_id"], node["id"]),
        )
        siblings = [_row_to_node(r, NODE_COLS) for r in cursor.fetchall()]

    # ── Full ancestral path (walk parent_id chain up to root) ─────────────────
    path       = [node]
    current_id = node["parent_id"]
    while current_id is not None:
        cursor.execute(f"{NODE_SELECT} WHERE id = %s;", (current_id,))
        anc_row = cursor.fetchone()
        if anc_row is None:
            break
        anc = _row_to_node(anc_row, NODE_COLS)
        path.insert(0, anc)          # prepend so path goes root → node
        current_id = anc["parent_id"]

    # ── Chunks for this node only ─────────────────────────────────────────────
    chunks = _fetch_chunks_for_nodes(cursor, [node["id"]])

    cursor.close()

    print(
        f"[traverse] '{node['title']}' — "
        f"parent: {'yes' if parent else 'none'}, "
        f"children: {len(children)}, "
        f"siblings: {len(siblings)}, "
        f"path depth: {len(path)}, "
        f"chunks: {len(chunks)}"
    )

    return {
        "node":     node,
        "parent":   parent,
        "children": children,
        "siblings": siblings,
        "path":     path,
        "chunks":   chunks,
    }


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the knowledge store")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path",     type=str, help="Path-based retrieval, e.g. 'Operations > Claims Management'")
    group.add_argument("--semantic", type=str, help="Semantic search query, e.g. 'how is fraud detected?'")
    group.add_argument("--traverse", type=str, help="Traverse by node title or node_id, e.g. 'Claims Processing'")
    parser.add_argument("--top-k",   type=int, default=5, help="Top-K results for semantic search (default: 5)")
    args = parser.parse_args()

    if args.path:
        result = get_by_path(args.path)
        print(json.dumps(result, indent=2, default=str))

    elif args.semantic:
        result = semantic_search(args.semantic, top_k=args.top_k)
        print(json.dumps(result, indent=2, default=str))

    elif args.traverse:
        result = traverse(args.traverse)
        print(json.dumps(result, indent=2, default=str))