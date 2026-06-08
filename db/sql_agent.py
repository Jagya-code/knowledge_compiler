"""
db/sql_agent.py — Natural language → SQL → PostgreSQL → LLM answer

Two query modes, selected automatically per question:

  KEYWORD mode  — structural / pattern questions
    "show all sub-topics under Operations"
    "which documents mention regulatory reporting?"
    LLM generates a standard SELECT with ILIKE / LIKE / JOIN.

  SEMANTIC mode — meaning-based questions
    "how is fraud investigated?"
    "what covers customer onboarding?"
    Question is embedded → pgvector cosine similarity against node_embeddings
    → top-K matching nodes → their chunks fetched → LLM summarises.

Intent is detected with a single fast LLM call before routing.
The caller (Streamlit) only ever calls chat() and gets back a plain dict.

Usage:
    from db.sql_agent import chat
    result = chat("Which nodes cover fraud detection?")
    print(result["answer"])
"""

import json
import os
import re
import sys

import psycopg2
import psycopg2.extras
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()
# ── Azure config ───────────────────────────────────────────────────────────────
AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION    = os.getenv("AZURE_API_VERSION")
AZURE_MODEL          = os.getenv("AZURE_OPENAI_MODEL")

# ── DB config ──────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
}

# ── Embedding model ────────────────────────────────────────────────────────────
# EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
# SEMANTIC_TOP_K   = 5          # how many nodes to return in semantic mode
_embedding_model = None

# def _get_embedding_model() -> SentenceTransformer:
#     global _embedding_model
#     if _embedding_model is None:
#         _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
#     return _embedding_model

# ── Safety: block all write operations ────────────────────────────────────────
_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

# ── Schema context for keyword SQL generation ──────────────────────────────────
SCHEMA_CONTEXT = """
You have access to a PostgreSQL knowledge store with these tables:

TABLE nodes
  id               SERIAL PRIMARY KEY
  node_id          VARCHAR(20)      -- short id e.g. "0001"
  title            VARCHAR(500)     -- topic/subtopic label
  depth            INT              -- 0 = root, 1 = subtopic, 2+ = deeper
  parent_id        INT FK → nodes.id  -- NULL for root nodes
  summary          TEXT             -- LLM-generated summary of the node
  keywords         TEXT[]           -- array of keywords
  source_docs      TEXT[]           -- array of source filenames
  user_defined     BOOLEAN          -- TRUE if user explicitly configured this node
  guidance_context JSONB            -- guidance metadata
  path_string      TEXT             -- e.g. "Operations > Claims Management > Claims Processing"

TABLE chunks
  id           SERIAL PRIMARY KEY
  chunk_id     VARCHAR(50)
  node_id      INT FK → nodes.id
  doc_id       VARCHAR(50)
  filename     VARCHAR(500)
  heading      TEXT
  text         TEXT                 -- actual document content
  page_approx  INT
  summary      TEXT

TABLE node_embeddings
  id        SERIAL PRIMARY KEY
  node_id   INT FK → nodes.id
  embedding vector(384)            -- do NOT query this column directly

USEFUL PATTERNS:
  -- Search by topic area (use node summary or path, not just title):
  SELECT n.title, n.summary, n.path_string FROM nodes
    WHERE n.summary ILIKE '%claims%' OR n.path_string ILIKE '%claims%' LIMIT 20;

  -- Search chunk content for a concept:
  SELECT c.heading, c.text, c.filename, c.summary FROM chunks c
    JOIN nodes n ON c.node_id = n.id
    WHERE c.text ILIKE '%claims investigation%'
       OR c.summary ILIKE '%claims investigation%'
       OR c.heading ILIKE '%claims%'
    LIMIT 20;

  -- Broad search across nodes AND chunks for a topic:
  SELECT n.title, n.path_string, c.heading, c.text FROM chunks c
    JOIN nodes n ON c.node_id = n.id
    WHERE c.text ILIKE '%fraud%' OR n.summary ILIKE '%fraud%'
    LIMIT 20;

  -- Children of a node:
  SELECT * FROM nodes WHERE parent_id =
    (SELECT id FROM nodes WHERE title ILIKE '%Operations%' LIMIT 1);

  -- Root nodes only:
  SELECT * FROM nodes WHERE depth = 0;

IMPORTANT:
  - For process/concept questions ("how is X done"), search c.text and c.summary — not just n.title.
  - Always search both nodes and chunks when looking for a concept.
  - Use OR across multiple fields (title, summary, path_string, c.text, c.heading) for better coverage.
  - If searching for a multi-word concept, also try searching individual keywords.

RULES:
  - Only SELECT statements. Never INSERT, UPDATE, DELETE, DROP, or TRUNCATE.
  - Always LIMIT to at most 20 rows.
  - Use ILIKE for case-insensitive matching.
  - Return raw SQL only — no markdown, no explanation, no semicolon at end.
"""

# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_INTENT = """You are a query router.
Classify the user question as exactly one of:

  "keyword"   — the question is structural, asks for lists, filters, or exact matches
                e.g. "show all sub-topics", "which nodes are under Operations?",
                     "what documents mention claims?"

Reply with ONLY the single word:   keyword
No punctuation. No explanation."""

SYSTEM_SQL = f"""You are a PostgreSQL expert and insurance domain assistant.
Given a user question, generate a single valid SELECT SQL query.

{SCHEMA_CONTEXT}
"""

SYSTEM_ANSWER = """You are an insurance domain assistant.
The user asked a question. Results were retrieved from a knowledge store.
Summarise the results in clear, concise prose — 2 to 5 sentences.
If results are empty, say no matching information was found.
Reference specific node titles, summaries, or chunk content where relevant.
Do not mention SQL, vectors, embeddings, or database internals in your answer.
"""


# ── LLM client ─────────────────────────────────────────────────────────────────
_client = None

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_API_VERSION,
        )
    return _client

def _call_llm(system: str, user: str, max_tokens: int = 600) -> str:
    resp = _get_client().chat.completions.create(
        model=AZURE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ── DB connection ──────────────────────────────────────────────────────────────
_conn = None

def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(**DB_CONFIG)
        _conn.autocommit = True
    else:
        try:
            _conn.cursor().execute("SELECT 1")
        except psycopg2.OperationalError:
            _conn = psycopg2.connect(**DB_CONFIG)
            _conn.autocommit = True
    return _conn

def _execute_sql(sql: str) -> tuple[list[dict], str | None]:
    if _BLOCKED.search(sql):
        return [], "Blocked: only SELECT is permitted."
    try:
        cursor = _get_conn().cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(sql)
        rows = [dict(r) for r in cursor.fetchall()]
        cursor.close()
        return rows, None
    except psycopg2.Error as e:
        return [], str(e)


# ── Intent detection ───────────────────────────────────────────────────────────

def _detect_intent(question: str) -> str:
    """
    Returns "semantic" or "keyword".
    Falls back to "keyword" if the LLM returns anything unexpected.
    """
    raw = _call_llm(SYSTEM_INTENT, question, max_tokens=5).lower().strip()
    return "semantic" if "semantic" in raw else "keyword"


# ── KEYWORD path ───────────────────────────────────────────────────────────────

def _keyword_query(question: str, history: list[dict] | None) -> tuple[list[dict], str]:
    """
    LLM generates SQL → execute → return (rows, sql).
    """
    context = ""
    if history:
        context = "\n\nConversation so far:\n"
        for turn in history[-4:]:
            role    = "User" if turn["role"] == "user" else "Assistant"
            context += f"{role}: {turn['content'][:300]}\n"
        context += "\nNow answer the latest question."

    raw = _call_llm(SYSTEM_SQL, f"Question: {question}{context}", max_tokens=400)
    sql = re.sub(r"```(?:sql)?|```", "", raw).strip()
    rows, error = _execute_sql(sql)
    if error:
        return [], sql, error
    return rows, sql, None


# ── SEMANTIC path ──────────────────────────────────────────────────────────────

# def _semantic_query(question: str) -> tuple[list[dict], str]:
#     """
#     Embed question → cosine similarity vs node_embeddings
#     → fetch top-K nodes + their chunks → return (rows, sql_description).
#     """
#     # model  = _get_embedding_model()
#     # vector = model.encode([question])[0].tolist()
#     # vector_str = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

#     sql = f"""
# SELECT
#     n.id,
#     n.title,
#     n.path_string,
#     n.summary        AS node_summary,
#     n.keywords,
#     n.source_docs,
#     1 - (ne.embedding <=> '{vector_str}'::vector) AS similarity_score,
#     c.heading        AS chunk_heading,
#     c.text           AS chunk_text,
#     c.filename       AS chunk_filename,
#     c.page_approx
# FROM node_embeddings ne
# JOIN nodes  n ON ne.node_id = n.id
# JOIN chunks c ON c.node_id  = n.id
# ORDER BY ne.embedding <=> '{vector_str}'::vector
# LIMIT {SEMANTIC_TOP_K * 3}
# """.strip()
#     # LIMIT * 3 because multiple chunks per node — we want top-K nodes, not rows

#     rows, error = _execute_sql(sql)
#     if error:
#         return [], sql, error

#     # Deduplicate to top-K unique nodes, keeping all their chunks
#     seen_nodes = {}
#     for row in rows:
#         nid = row["id"]
#         if nid not in seen_nodes:
#             if len(seen_nodes) >= SEMANTIC_TOP_K:
#                 continue
#             seen_nodes[nid] = {
#                 "id":               row["id"],
#                 "title":            row["title"],
#                 "path_string":      row["path_string"],
#                 "node_summary":     row["node_summary"],
#                 "keywords":         row["keywords"],
#                 "source_docs":      row["source_docs"],
#                 "similarity_score": round(float(row["similarity_score"]), 4),
#                 "chunks": [],
#             }
#         seen_nodes[nid]["chunks"].append({
#             "heading":    row["chunk_heading"],
#             "text":       row["chunk_text"],
#             "filename":   row["chunk_filename"],
#             "page_approx":row["page_approx"],
#         })

#     structured_rows = list(seen_nodes.values())
#     return structured_rows, sql, None


# ── Answer generation ──────────────────────────────────────────────────────────

def _generate_answer(question: str, mode: str, sql: str, rows: list[dict]) -> str:
    def _trim(row: dict) -> dict:
        out = {}
        for k, v in row.items():
            if isinstance(v, str) and len(v) > 400:
                out[k] = v[:400] + "…"
            elif isinstance(v, list) and k == "chunks":
                out[k] = [{
                    kk: (vv[:300] + "…" if isinstance(vv, str) and len(vv) > 300 else vv)
                    for kk, vv in chunk.items()
                } for chunk in v[:2]]
            else:
                out[k] = v
        return out

    trimmed  = [_trim(r) for r in rows[:15]]
    rows_str = json.dumps(trimmed, indent=2, default=str)

    user_prompt = (
        f"Question: {question}\n"
        f"Query mode: {mode}\n\n"
        f"Results ({len(rows)} item(s)):\n{rows_str}"
    )
    return _call_llm(SYSTEM_ANSWER, user_prompt, max_tokens=500)


# ── Public entry points ────────────────────────────────────────────────────────

def run_agent(
    question: str,
    history:  list[dict] | None = None,
) -> dict:
    """
    Full hybrid pipeline:
      1. Detect intent (semantic vs keyword)
      2. Route to appropriate query path
      3. Generate plain-language answer

    Returns:
        {
          "question": str,
          "mode":     str,       #  "keyword"
          "sql":      str,       # query that ran (or attempted)
          "rows":     list,      # raw results
          "answer":   str,       # plain language answer
          "error":    str|None,
        }
    """
    result = {
        "question": question,
        "mode":     "",
        "sql":      "",
        "rows":     [],
        "answer":   "",
        "error":    None,
    }

    # Step 1 — detect intent
    try:
        mode = "keyword"
        result["mode"] = mode
    except Exception as e:
        # Default to keyword on detection failure
        mode = "keyword"
        result["mode"] = mode

    # Step 2 — query
    try:
        # if mode == "semantic":
        #     rows, sql, error = _semantic_query(question)
        # else:
        rows, sql, error = _keyword_query(question, history)

        result["sql"]   = sql
        result["rows"]  = rows

        if error:
            result["error"]  = error
            result["answer"] = f"The query could not be executed: {error}"
            return result

    except Exception as e:
        result["error"]  = f"Query failed: {e}"
        result["answer"] = "I was unable to retrieve results for your question."
        return result

    # Step 3 — generate answer
    try:
        result["answer"] = _generate_answer(question, mode, sql, rows)
    except Exception as e:
        result["error"]  = f"Answer generation failed: {e}"
        result["answer"] = f"Retrieved {len(rows)} result(s) but could not summarise them."

    return result


def chat(question: str, history: list[dict] | None = None) -> dict:
    """
    Streamlit entry point — thin wrapper over run_agent().
    Returns only what the frontend needs to render.
    """
    result = run_agent(question, history=history)
    return {
        "answer": result["answer"],
        "sql":    result["sql"],
        "rows":   result["rows"],
        "mode":   result["mode"],
        "error":  result["error"],
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SQL Agent — natural language → PostgreSQL")
    parser.add_argument("question", type=str, help="Natural language question")
    args = parser.parse_args()

    res = run_agent(args.question)
    print(f"\n── Mode: {res['mode'].upper()} ───────────────────────────────")
    print(f"\n── SQL ────────────────────────────────────────────────────────")
    print(res["sql"])
    print(f"\n── Rows returned: {len(res['rows'])} ────────────────────────────────")
    if res["rows"]:
        print(json.dumps(res["rows"][:3], indent=2, default=str))
    print(f"\n── Answer ─────────────────────────────────────────────────────")
    print(res["answer"])
    if res["error"]:
        print(f"\n── Error ──────────────────────────────────────────────────────")
        print(res["error"])