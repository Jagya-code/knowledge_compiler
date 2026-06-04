"""
streamlit_app.py — Knowledge Map UI

Run:
    streamlit run streamlit_app.py
"""

import json
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Knowledge Map", page_icon=None, layout="wide")

# ── Minimal style overrides ───────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { font-size: 1.4rem !important; font-weight: 600; letter-spacing: -0.01em; }
    .stSidebar [data-testid="stSidebarContent"] { padding-top: 1.5rem; }
    div[data-testid="stExpander"] summary { font-size: 0.9rem; }
    div[data-testid="stTextInput"] input { font-size: 0.82rem; }
    div[data-testid="stTextArea"] textarea { font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

st.title("Knowledge Map")

# ── Session state ─────────────────────────────────────────────────────────────
if "tree" not in st.session_state:
    st.session_state.tree = None

# Guidance topics state: list of {name, description, subtopics: [{name, description}]}
if "guidance_topics" not in st.session_state:
    st.session_state.guidance_topics = [
        {"name": "", "description": "", "subtopics": [{"name": "", "description": ""}]}
    ]

# ── Sidebar: Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("#### Guidance")

    guidance_source = st.radio(
        "Guidance input",
        ["Form", "Upload file", "No guidance"],
        horizontal=True,
        label_visibility="collapsed",
    )

    guidance_file = None

    if guidance_source == "Form":
        topics = st.session_state.guidance_topics
        to_delete_topic = None

        for t_idx, topic in enumerate(topics):
            st.markdown(f"**Topic {t_idx + 1}**")

            # Topic name + delete button
            col_name, col_del = st.columns([5, 1])
            with col_name:
                topic["name"] = st.text_input(
                    "Topic name",
                    value=topic["name"],
                    key=f"t_name_{t_idx}",
                    placeholder="e.g. Compliance",
                    label_visibility="collapsed",
                )
            with col_del:
                if st.button("✕", key=f"del_t_{t_idx}", help="Remove topic"):
                    to_delete_topic = t_idx

            # Topic description
            topic["description"] = st.text_area(
                "Topic description",
                value=topic["description"],
                key=f"t_desc_{t_idx}",
                placeholder="Topic description (optional)",
                height=60,
                label_visibility="collapsed",
            )

            # Subtopics header
            st.caption("Subtopics")

            # Subtopic rows
            to_delete_sub = None
            for s_idx, sub in enumerate(topic["subtopics"]):
                c1, c2, c3 = st.columns([3, 3, 1])
                with c1:
                    sub["name"] = st.text_input(
                        "Subtopic name",
                        value=sub["name"],
                        key=f"s_name_{t_idx}_{s_idx}",
                        placeholder="Name",
                        label_visibility="collapsed",
                    )
                with c2:
                    sub["description"] = st.text_input(
                        "Subtopic desc",
                        value=sub["description"],
                        key=f"s_desc_{t_idx}_{s_idx}",
                        placeholder="Description",
                        label_visibility="collapsed",
                    )
                with c3:
                    if st.button("✕", key=f"del_s_{t_idx}_{s_idx}", help="Remove subtopic"):
                        to_delete_sub = s_idx

            if to_delete_sub is not None:
                topic["subtopics"].pop(to_delete_sub)
                st.rerun()

            if st.button("＋ Add Subtopic", key=f"add_sub_{t_idx}", use_container_width=True):
                topic["subtopics"].append({"name": "", "description": ""})
                st.rerun()

            st.markdown("---")

        if to_delete_topic is not None:
            topics.pop(to_delete_topic)
            st.rerun()

        if st.button("＋ Add Topic", use_container_width=True):
            topics.append({
                "name": "",
                "description": "",
                "subtopics": [{"name": "", "description": ""}]
            })
            st.rerun()

        st.markdown("---")

    elif guidance_source == "Upload file":
        guidance_file = st.file_uploader(
            "Upload guidance (.json or .xlsx)",
            type=["json", "xlsx"],
            key="guidance_upload",
        )
        st.markdown("---")
    else:
        st.caption("AI-only mode — no guidance applied.")
        st.markdown("---")

    generate = st.button("Generate", type="primary", use_container_width=True)


# ── Main: Output ──────────────────────────────────────────────────────────────
if not generate and st.session_state.tree is None:
    st.caption("Configure inputs in the sidebar and click Generate.")

if generate:
    # ── Load fixed output JSON (demo mode — always returns the same output) ─────
    fixed_output_path = Path("output/output_hierarchy.json")
    if not fixed_output_path.exists():
        st.error(f"Fixed output file not found: {fixed_output_path}")
        st.stop()

    with open(fixed_output_path) as f:
        tree = json.load(f)

    st.session_state.tree = tree

    # No real pipeline stdout — detail lines will simply be skipped
    def find_detail(keyword):
        """No-op in demo mode — no captured stdout."""
        return None

    # ── Step definitions ──────────────────────────────────────────────────────
    # (label, detail_keyword, delay_seconds)
    STEPS = [
        (
            "== Ingesting documents — parsing text from source files...",
            "Loaded",
            1,
        ),
        (
            "== Splitting documents into semantic chunks...",
            "Produced",
            1,
        ),
        (
            "== Generating summaries for each chunk via language model...",
            "Summarised",
            2,
        ),
        (
            "== Extracting hierarchical topic paths from chunk content...",
            "raw path",
            1,
        ),
        (
            "== Canonicalising labels and verifying semantic consistency...",
            "canonical path",
            1,
        ),
        (
            "== Applying user guidance overrides to topic placement...",
            "mapping",
            2,
        ),
        (
            "== Assembling final knowledge hierarchy...",
            "root node",
            1,
        ),
    ]

    # ── Replay steps one by one with per-step delay ───────────────────────────
    log_area = st.empty()
    displayed: list[str] = []

    for i, (step_label, keyword, delay) in enumerate(STEPS):
        displayed.append(f"[Step {i+1}/7] {step_label}")
        detail = find_detail(keyword)
        if detail:
            displayed.append(f"       {detail}")
        displayed.append("")
        log_area.code("\n".join(displayed), language="bash")
        time.sleep(delay)

    # 3s pause on last step before showing Done
    time.sleep(2)
    log_area.empty()
    st.success(f"Done — {len(st.session_state.tree)} root nodes generated")


# ── Show output ───────────────────────────────────────────────────────────────
if st.session_state.tree:
    tree = st.session_state.tree
    json_str = json.dumps(tree, indent=2)

    st.download_button(
        "Download JSON",
        data=json_str,
        file_name="hierarchy_nested.json",
        mime="application/json",
    )

    tab_tree, tab_json = st.tabs(["Tree", "JSON"])

    with tab_tree:
        def render_node(node, depth=0):
            indent = "　" * depth
            tag = "🔵" if node.get("user_defined") else "⚪"
            label = f"{indent}{tag} **{node['title']}**"
            docs = ", ".join(node.get("source_docs", []))
            with st.expander(label, expanded=(depth == 0)):
                st.caption(node.get("summary", ""))
                if docs:
                    st.caption(f"Sources: {docs}")
                for child in node.get("nodes", []):
                    render_node(child, depth + 1)

        for root in tree:
            render_node(root)

    with tab_json:
        st.code(json_str, language="json")