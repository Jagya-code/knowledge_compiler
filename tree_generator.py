"""
tree_generator.py — Recursively convert the hierarchy tree dict into the
final JSON output schema. Works for any depth of nesting.

Output schema per node:
  {
    "title":        str,
    "node_id":      str,       # zero-padded 4-digit counter
    "depth":        int,       # 0 = root level
    "start_index":  int,
    "end_index":    int,
    "summary":      str,
    "keywords":     [str],
    "source_docs":  [str],
    "user_defined": bool,
    "nodes":        [...]      # child nodes, same schema
  }
"""

import os
from openai import AzureOpenAI

# Guidance index — populated by generate_tree() and used in _build_node()
_topic_index = {}
_sub_index   = {}


def _set_guidance_index(topic_index: dict, sub_index: dict) -> None:
    global _topic_index, _sub_index
    _topic_index = topic_index
    _sub_index   = sub_index


def _resolve_guidance(label: str) -> tuple[bool, dict]:
    """
    Check if label matches any topic_name or sub_topic_name in the guidance.
    Returns (user_defined, guidance_context).
    """
    norm = label.lower().strip()

    if norm in _sub_index:
        entry = _sub_index[norm]
        return True, {
            "matched_label":            label,
            "sub_topic_description":    entry["sub_topic_description"],
            "parent_topic_name":        entry["parent_topic_name"],
            "parent_topic_description": entry["parent_topic_description"],
        }

    if norm in _topic_index:
        entry = _topic_index[norm]
        return True, {
            "matched_label":     label,
            "topic_description": entry["topic_description"],
        }

    return False, {}

AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "https://bfsi-genai-demo.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
AZURE_API_VERSION    = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
AZURE_MODEL          = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-nano-for-saikat-team")

_client   = None
_counter  = [0]


def _get_client():
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_API_VERSION,
        )
    return _client


def _next_id():
    _counter[0] += 1
    return f"{_counter[0]:04d}"


def _summarize_node(title, child_summaries):
    """Roll up child summaries into a parent-level paragraph."""
    if not child_summaries:
        return f"Overview of {title}."
    bullets = "\n".join(f"- {s}" for s in child_summaries[:6])
    prompt  = (
        f"Topic: {title}\n"
        f"Sub-section summaries:\n{bullets}\n\n"
        "Write a single concise paragraph (2-3 sentences) summarising this topic "
        "for an insurance professional."
    )
    try:
        resp = _get_client().chat.completions.create(
            model=AZURE_MODEL,
            messages=[
                {"role": "system", "content": "You are an insurance domain analyst. Be concise and specific."},
                {"role": "user",   "content": prompt},
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable: {e}"


def _build_node(label, node_data, depth):
    """
    Recursively build a single node and all its descendants.

    node_data structure (from hierarchy_builder):
      {
        "children":   { label: node_data, ... },
        "chunks":     [ chunk_dict, ... ],   # only populated at leaves
        "doc_ids":    [str],
        "keywords":   [str],
        "user_defined": bool
      }
    """
    chunks   = node_data.get("chunks", [])
    children = node_data.get("children", {})

    # ── Recurse into children first ──────────────────────────────────────────
    child_nodes = []
    for child_label, child_data in sorted(children.items()):
        child_nodes.append(_build_node(child_label, child_data, depth + 1))

    # ── Collect page indices ─────────────────────────────────────────────────
    leaf_pages = [c["page_approx"] for c in chunks if c.get("page_approx")]
    child_pages = (
        [n["start_index"] for n in child_nodes] +
        [n["end_index"]   for n in child_nodes]
    )
    all_pages = leaf_pages + child_pages

    # ── Collect source docs ──────────────────────────────────────────────────
    leaf_docs  = list({c["filename"] for c in chunks})
    child_docs = list({fn for n in child_nodes for fn in n["source_docs"]})
    source_docs = list(set(leaf_docs + child_docs))

    # ── Collect chunk_ids ────────────────────────────────────────────────────
    leaf_chunk_ids  = [c["chunk_id"] for c in chunks if c.get("chunk_id")]
    child_chunk_ids = [cid for n in child_nodes for cid in n.get("chunk_ids", [])]
    chunk_ids = list(set(leaf_chunk_ids + child_chunk_ids))

    # ── Summary ──────────────────────────────────────────────────────────────
    if child_nodes:
        # Interior node: summarise from children
        child_summaries = [n["summary"] for n in child_nodes]
        indent = "  " * depth
        print(f"{indent}  Summarising: {label}")
        summary = _summarize_node(label, child_summaries)
    elif chunks:
        # Leaf node: use first chunk summary
        chunk_summaries = [c["summary"] for c in chunks if c.get("summary")]
        summary = chunk_summaries[0] if chunk_summaries else f"Details on {label}."
    else:
        summary = f"Overview of {label}."

    user_defined, guidance_context = _resolve_guidance(label)

    return {
        "title":            label,
        "node_id":          _next_id(),
        "depth":            depth,
        "start_index":      min(all_pages) if all_pages else 0,
        "end_index":        max(all_pages) if all_pages else 0,
        "summary":          summary,
        "keywords":         node_data.get("keywords", []),
        "source_docs":      source_docs,
        "chunk_ids":        chunk_ids,
        "user_defined":     user_defined,
        "guidance_context": guidance_context,
        "nodes":            child_nodes,
    }


def generate_tree(hierarchy_tree, guidance: list = None):
    """
    Convert the nested dict from hierarchy_builder into the final JSON list.
    hierarchy_tree: { label: node_data, ... }  (root level)
    guidance: validated structured list from main.load_user_guidance() ([] for AI-only)
    """
    from hierarchy_builder import _build_guidance_index
    if guidance:
        topic_index, sub_index = _build_guidance_index(guidance)
        _set_guidance_index(topic_index, sub_index)
    else:
        _set_guidance_index({}, {})

    _counter[0] = 0
    tree = []
    for label, node_data in sorted(hierarchy_tree.items()):
        tree.append(_build_node(label, node_data, depth=0))
    return tree