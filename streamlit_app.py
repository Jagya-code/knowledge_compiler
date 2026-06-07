"""
streamlit_app.py — Knowledge Map UI (multi-page)

Pages:
  1. Knowledge Map  — upload docs, configure guidance, generate tree, auto-load to DB
  2. Chat           — SQL agent: natural language → SQL → PostgreSQL → answer

Run:
    streamlit run streamlit_app.py
"""

import json
import os
import sys
import tempfile

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Knowledge Map", page_icon="🗂️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 14px;
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #0f1117;
    }

    .sidebar-logo {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem; font-weight: 600; color: #0f1117;
        padding: 0.25rem 0 1.25rem 0; display: block;
    }

    .db-badge {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.72rem; font-weight: 600;
        padding: 4px 10px; border-radius: 20px; letter-spacing: 0.03em;
    }
    .db-connected    { background: #e6f4ea; color: #1a7c34; }
    .db-disconnected { background: #fdecea; color: #c0392b; }
    .db-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
    .db-connected .db-dot    { background: #28a745; }
    .db-disconnected .db-dot { background: #dc3545; }

    .topic-card {
        background: #f8f9fb; border: 1px solid #dde1e7;
        border-radius: 8px; padding: 1rem 1rem 0.5rem 1rem; margin-bottom: 0.75rem;
    }
    .sub-header {
        font-size: 0.72rem; font-weight: 600; color: #666;
        text-transform: uppercase; letter-spacing: 0.06em; margin: 0.5rem 0 0.25rem;
    }
    .node-path {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem; color: #999; margin-bottom: 3px;
    }

    .chat-user {
        background: #0f1117; color: #fff;
        border-radius: 14px 14px 3px 14px;
        padding: 0.7rem 1rem; margin: 0.5rem 0 0.5rem auto;
        font-size: 0.875rem; max-width: 80%; width: fit-content;
    }
    .chat-assistant {
        background: #f4f5f7; border: 1px solid #e2e4e9;
        border-radius: 14px 14px 14px 3px;
        padding: 0.8rem 1rem; margin: 0.5rem 0;
        font-size: 0.875rem; max-width: 92%;
    }
    .sql-block {
        background: #1e1e2e; color: #cdd6f4;
        border-radius: 6px; padding: 0.65rem 0.9rem;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
        margin: 0.6rem 0; white-space: pre-wrap; word-break: break-word;
    }
    .answer-text { font-size: 0.875rem; line-height: 1.65; color: #1a1a2e; margin-top: 0.4rem; }
    .rows-badge {
        display: inline-block; font-size: 0.68rem; font-weight: 600;
        padding: 2px 8px; border-radius: 4px;
        background: #e8f0fe; color: #1a56db; margin-top: 6px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .error-badge {
        display: inline-block; font-size: 0.72rem; padding: 3px 10px;
        border-radius: 4px; background: #fdecea; color: #c0392b; margin-bottom: 4px;
    }
    .stButton > button { font-size: 0.82rem; }
    .stTextInput > div > div > input { font-size: 0.875rem; }
</style>
""", unsafe_allow_html=True)

OUTPUT_PATH        = "output/output_hierarchy.json"
PRUNED_OUTPUT_PATH = "output/pruned_knowledge_map.json"


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "page":              "knowledge_map",
        "tree":              None,
        "db_loaded":         False,
        "validation_errors": [],
        "chat_history":      [],
        "topics": [{
            "topic_name": "", "topic_description": "",
            "sub_topics": [{"sub_topic_name": "", "sub_topic_description": ""}],
        }],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── DB check ──────────────────────────────────────────────────────────────────
def check_db_connection() -> bool:
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "ECF"), user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "Jagyaseni@123"),
            host=os.getenv("DB_HOST", "localhost"), port=os.getenv("DB_PORT", "5432"),
        )
        conn.close()
        return True
    except Exception:
        return False


# ── Topic mutators ────────────────────────────────────────────────────────────
def _add_topic():
    st.session_state.topics.append({
        "topic_name": "", "topic_description": "",
        "sub_topics": [{"sub_topic_name": "", "sub_topic_description": ""}],
    })
    st.session_state.validation_errors = []

def _remove_topic(ti):
    if len(st.session_state.topics) > 1:
        st.session_state.topics.pop(ti)
    st.session_state.validation_errors = []

def _add_subtopic(ti):
    st.session_state.topics[ti]["sub_topics"].append({"sub_topic_name": "", "sub_topic_description": ""})
    st.session_state.validation_errors = []

def _remove_subtopic(ti, si):
    subs = st.session_state.topics[ti]["sub_topics"]
    if len(subs) > 1:
        subs.pop(si)
    st.session_state.validation_errors = []

def _sync_form_to_state():
    for ti, topic in enumerate(st.session_state.topics):
        topic["topic_name"]        = st.session_state.get(f"tn_{ti}", topic["topic_name"])
        topic["topic_description"] = st.session_state.get(f"td_{ti}", topic["topic_description"])
        for si, sub in enumerate(topic["sub_topics"]):
            sub["sub_topic_name"]        = st.session_state.get(f"sn_{ti}_{si}", sub["sub_topic_name"])
            sub["sub_topic_description"] = st.session_state.get(f"sd_{ti}_{si}", sub["sub_topic_description"])

def _validate_form() -> list:
    errors, seen_topics = [], {}
    named = [(ti, t) for ti, t in enumerate(st.session_state.topics) if t["topic_name"].strip()]
    if not named:
        return ["Add at least one topic name before generating."]
    for ti, topic in named:
        name, norm = topic["topic_name"].strip(), topic["topic_name"].strip().lower()
        label = f"Topic {ti+1} ('{name}')"
        if norm in seen_topics:
            errors.append(f"{label}: duplicate topic name.")
        else:
            seen_topics[norm] = ti
        named_subs = [(si, s) for si, s in enumerate(topic["sub_topics"]) if s["sub_topic_name"].strip()]
        if not named_subs:
            errors.append(f"{label}: add at least one sub-topic name.")
            continue
        seen_subs = {}
        for si, sub in named_subs:
            sname, snorm = sub["sub_topic_name"].strip(), sub["sub_topic_name"].strip().lower()
            if snorm in seen_subs:
                errors.append(f"{label} > Sub-topic {si+1} ('{sname}'): duplicate.")
            else:
                seen_subs[snorm] = si
    return errors

def _build_guidance_payload() -> list:
    payload = []
    for topic in st.session_state.topics:
        tname = topic["topic_name"].strip()
        if not tname:
            continue
        payload.append({
            "topic_name": tname, "topic_description": topic["topic_description"].strip(),
            "sub_topics": [
                {"sub_topic_name": s["sub_topic_name"].strip(),
                 "sub_topic_description": s["sub_topic_description"].strip()}
                for s in topic["sub_topics"] if s["sub_topic_name"].strip()
            ],
        })
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown('<span class="sidebar-logo">🗂 KnowledgeMap</span>', unsafe_allow_html=True)

        for label, key in [("🗺️  Knowledge Map", "knowledge_map"), ("💬  Chat", "chat")]:
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if st.session_state.page == key else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")

        db_ok = check_db_connection()
        cls   = "db-connected" if db_ok else "db-disconnected"
        lbl   = "PostgreSQL connected" if db_ok else "PostgreSQL disconnected"
        st.markdown(f'<div class="db-badge {cls}"><span class="db-dot"></span>{lbl}</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Documents")
        uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt", "docx"],
                                           accept_multiple_files=True)
        st.markdown("---")
        st.markdown("#### Guidance Configuration")
        guidance_mode = st.radio("gm", ["Form", "Upload JSON", "No guidance"],
                                 horizontal=True, label_visibility="collapsed")

        if guidance_mode == "Form":
            for ti, topic in enumerate(st.session_state.topics):
                st.markdown('<div class="topic-card">', unsafe_allow_html=True)
                th_l, th_r = st.columns([6, 1])
                with th_l:
                    st.markdown(f"**Topic {ti+1}**")
                with th_r:
                    st.button("✕", key=f"del_topic_{ti}", on_click=_remove_topic,
                              args=(ti,), disabled=(len(st.session_state.topics) == 1))
                st.text_input("Topic Name *", key=f"tn_{ti}", value=topic["topic_name"],
                              placeholder="e.g. Operations")
                st.text_input("Description", key=f"td_{ti}", value=topic["topic_description"],
                              placeholder="Brief description (optional)")
                st.markdown('<p class="sub-header">Sub-topics</p>', unsafe_allow_html=True)
                for si, sub in enumerate(topic["sub_topics"]):
                    c1, c2, c3 = st.columns([3, 3, 0.5])
                    with c1:
                        st.text_input("Name *", key=f"sn_{ti}_{si}", value=sub["sub_topic_name"],
                                      placeholder="e.g. Claims Processing",
                                      label_visibility="collapsed" if si > 0 else "visible")
                    with c2:
                        st.text_input("Desc", key=f"sd_{ti}_{si}", value=sub["sub_topic_description"],
                                      placeholder="Description (optional)",
                                      label_visibility="collapsed" if si > 0 else "visible")
                    with c3:
                        if si == 0:
                            st.markdown("&nbsp;", unsafe_allow_html=True)
                        st.button("✕", key=f"del_sub_{ti}_{si}", on_click=_remove_subtopic,
                                  args=(ti, si), disabled=(len(topic["sub_topics"]) == 1))
                st.button("+ Sub-topic", key=f"add_sub_{ti}", on_click=_add_subtopic, args=(ti,))
                st.markdown("</div>", unsafe_allow_html=True)
            st.button("+ Add Topic", on_click=_add_topic, use_container_width=True)
            for err in st.session_state.validation_errors:
                st.error(err, icon="⚠️")

        elif guidance_mode == "Upload JSON":
            st.file_uploader("Upload guidance (.json)", type=["json"], key="guidance_upload")
            st.caption("Must follow input_config.json structure.")
        else:
            st.info("AI-only mode — no placement guidance.", icon="ℹ️")

        st.markdown("---")
        generate_clicked = st.button("▶ Generate", type="primary", use_container_width=True)

    return uploaded_files, guidance_mode, generate_clicked


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — KNOWLEDGE MAP
# ═══════════════════════════════════════════════════════════════════════════════
def page_knowledge_map(uploaded_files, guidance_mode, generate_clicked):
    st.markdown("### Knowledge Map")

    if not generate_clicked and st.session_state.tree is None:
        st.caption("Configure inputs in the sidebar and click ▶ Generate.")
        return

    if generate_clicked:
        _sync_form_to_state()
        guidance_path = None

        if guidance_mode == "Form":
            errors = _validate_form()
            if errors:
                st.session_state.validation_errors = errors
                st.rerun()
            st.session_state.validation_errors = []
            payload = _build_guidance_payload()
            if payload:
                tmp_g = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                json.dump(payload, tmp_g)
                tmp_g.close()
                guidance_path = tmp_g.name

        elif guidance_mode == "Upload JSON":
            ug = st.session_state.get("guidance_upload")
            if ug is not None:
                tmp_g = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
                tmp_g.write(ug.getbuffer())
                tmp_g.close()
                guidance_path = tmp_g.name
            else:
                st.warning("No guidance file uploaded — running in AI-only mode.")

        doc_paths = []
        if uploaded_files:
            tmp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                dest = os.path.join(tmp_dir, uf.name)
                with open(dest, "wb") as fh:
                    fh.write(uf.getbuffer())
                doc_paths.append(dest)

        if not doc_paths:
            st.error("No documents uploaded.", icon="⚠️")
            st.stop()

        with st.spinner("Running pipeline…"):
            log_area, log_lines = st.empty(), []
            class _Logger:
                def write(self, msg):
                    if msg.strip():
                        log_lines.append(msg.rstrip())
                        log_area.code("\n".join(log_lines[-30:]), language="bash")
                def flush(self): pass
            old = sys.stdout
            sys.stdout = _Logger()
            try:
                from main import run_pipeline, GuidanceValidationError
                tree = run_pipeline(doc_paths=doc_paths, guidance_path=guidance_path,
                                    output_path=OUTPUT_PATH)
                st.session_state.tree      = tree
                st.session_state.db_loaded = False
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}", icon="🔴")
                st.exception(exc)
                st.stop()
            finally:
                sys.stdout = old
        log_area.empty()
        st.success(f"Pipeline complete — {len(st.session_state.tree)} root node(s).", icon="✅")

        with st.spinner("Loading into PostgreSQL…"):
            db_log, db_lines = st.empty(), []
            class _DbLog:
                def write(self, msg):
                    if msg.strip():
                        db_lines.append(msg.rstrip())
                        db_log.code("\n".join(db_lines[-20:]), language="bash")
                def flush(self): pass
            old = sys.stdout
            sys.stdout = _DbLog()
            try:
                sys.path.insert(0, os.path.abspath("."))
                from db.loader import run_loader
                run_loader()
                st.session_state.db_loaded = True
            except Exception as exc:
                st.warning(f"DB load failed: {exc}", icon="⚠️")
            finally:
                sys.stdout = old
        db_log.empty()
        if st.session_state.db_loaded:
            st.success("Knowledge store loaded into PostgreSQL.", icon="🗄️")

    if st.session_state.tree:
        tree     = st.session_state.tree
        json_str = json.dumps(tree, indent=2)
        st.download_button("⬇ Download JSON", data=json_str,
                           file_name="output_hierarchy.json", mime="application/json")

        tab_tree, tab_json = st.tabs(["🌲 Tree View", "{ } Raw JSON"])

        with tab_tree:
            def render_node(node, depth=0):
                indent = "　" * depth
                tag    = "🔵" if node.get("user_defined") else "⚪"
                with st.expander(f"{indent}{tag} **{node['title']}**", expanded=(depth == 0)):
                    ps = node.get("path_string", "")
                    if ps:
                        st.markdown(f'<div class="node-path">{ps}</div>', unsafe_allow_html=True)
                    if node.get("summary"):
                        st.caption(node["summary"])
                    docs = ", ".join(node.get("source_docs", []))
                    if docs:
                        st.caption(f"📄 {docs}")
                    for child in node.get("nodes", []):
                        render_node(child, depth + 1)
            for root in tree:
                render_node(root)

        with tab_json:
            st.code(json_str, language="json")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SQL AGENT CHAT
# ═══════════════════════════════════════════════════════════════════════════════
def page_chat():
    st.markdown("### Chat — SQL Agent")

    db_ok = check_db_connection()
    if not db_ok:
        st.warning("PostgreSQL is not reachable. Generate a knowledge map first or "
                   "check your DB connection.", icon="⚠️")

    with st.expander("ℹ️ How the SQL Agent works", expanded=False):
        st.markdown("""
Every question goes through a 3-step pipeline:

1. **SQL Generation** — Azure OpenAI reads your question and writes a `SELECT` query against `nodes` and `chunks`.
2. **Execution** — The query runs against PostgreSQL. Only `SELECT` is permitted; all writes are blocked.
3. **Answer** — The LLM reads the raw rows and writes a plain-language answer.

The generated SQL is shown below each answer so you can inspect exactly what was queried.

**Example questions:**
- *"Which topics are about fraud detection?"*
- *"Show me all sub-topics under Operations"*
- *"What does the Claims Processing node cover?"*
- *"Find chunks mentioning regulatory reporting"*
- *"Which documents were used for the Risk Management topic?"*
        """)

    st.markdown("---")

    # ── Chat history ──────────────────────────────────────────────────────────
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f'<div class="chat-user">{entry["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            answer    = entry.get("answer", "")
            sql       = entry.get("sql", "")
            rows      = entry.get("rows", [])
            error     = entry.get("error")
            row_count = len(rows)

            mode      = entry.get("mode", "keyword")
            mode_label = "🔍 Semantic" if mode == "semantic" else "🗂 Keyword"
            mode_color = "#e8f0fe;color:#1a56db" if mode == "semantic" else "#fef3c7;color:#92400e"

            html = '<div class="chat-assistant">'
            html += f'<span style="display:inline-block;font-size:0.68rem;font-weight:600;padding:2px 8px;border-radius:4px;background:{mode_color};font-family:IBM Plex Mono,monospace;margin-bottom:6px;">{mode_label}</span><br>'
            if error:
                html += f'<div class="error-badge">⚠ {error}</div><br>'
            html += f'<div class="answer-text">{answer}</div>'
            html += f'<div><span class="rows-badge">{row_count} row(s) returned</span></div>'
            if sql:
                html += f'<details style="margin-top:8px;"><summary class="sql-toggle" style="cursor:pointer;font-size:0.72rem;color:#888;font-family:IBM Plex Mono,monospace;">▶ Show SQL</summary><div class="sql-block">{sql}</div></details>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    col_q, col_send, col_clear = st.columns([8, 1, 1])

    with col_q:
        query = st.text_input(
            "query", label_visibility="collapsed",
            placeholder="Ask a question about your knowledge store…",
            key="chat_query_input", disabled=not db_ok,
        )
    with col_send:
        send = st.button("Send", type="primary", use_container_width=True, disabled=not db_ok)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Run agent ─────────────────────────────────────────────────────────────
    if send and query.strip():
        st.session_state.chat_history.append({"role": "user", "content": query.strip()})

        history_context = [
            {"role": e["role"], "content": e.get("content", e.get("answer", ""))}
            for e in st.session_state.chat_history[-6:]
            if e["role"] in ("user", "assistant")
        ]

        with st.spinner("Querying knowledge store…"):
            try:
                sys.path.insert(0, os.path.abspath("."))
                from db.sql_agent import chat
                result = chat(query.strip(), history=history_context)
            except Exception as exc:
                result = {"sql": "", "rows": [], "mode": "keyword",
                          "answer": f"Agent failed: {exc}", "error": str(exc)}

        st.session_state.chat_history.append({
            "role":   "assistant",
            "answer": result["answer"],
            "sql":    result["sql"],
            "rows":   result["rows"],
            "mode":   result.get("mode", "keyword"),
            "error":  result["error"],
        })
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
uploaded_files, guidance_mode, generate_clicked = render_sidebar()

if st.session_state.page == "knowledge_map":
    page_knowledge_map(uploaded_files, guidance_mode, generate_clicked)
else:
    page_chat()