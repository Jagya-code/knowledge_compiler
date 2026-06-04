"""
hierarchy_builder.py — Build a recursive N-level hierarchy from canonicalised paths.

Each topic carries a 'canonical_path' list of arbitrary depth, e.g.:
  ["Operations", "Claims Management", "Claims Processing", "Triage & Assignment"]

Guidance format (input_config.json structure):
  [
    {
      "topic_name": "Operations",
      "topic_description": "Operational workflows and processes",
      "sub_topics": [
        {
          "sub_topic_name": "Claims Processing",
          "sub_topic_description": "Handling and processing insurance claims"
        }
      ]
    }
  ]

This module:
  1. Parses the structured guidance list natively — no flat-dict conversion.
  2. Builds internal lookup tables (topic index, sub-topic index) from the
     guidance for O(1) matching during path traversal.
  3. Applies placement overrides in apply_user_guidance():
       - If any label in a topic's canonical_path matches a known sub_topic_name,
         the root of that path (path[0]) is forced to the corresponding topic_name.
       - topic_description and sub_topic_description are stored on each node as
         'guidance_context' — available for downstream use but never replacing
         LLM-generated summaries.
  4. Walks each canonical_path and inserts it into a nested tree dict.
  5. Attaches chunk references to leaf nodes.

Output: recursive dict consumed by tree_generator (schema unchanged).
"""


def _normalize(s: str) -> str:
    return s.lower().strip()


# ── Guidance index builder ────────────────────────────────────────────────────

def _build_guidance_index(guidance: list) -> tuple[dict, dict]:
    """
    Build two lookup tables from the structured guidance list.

    Returns:
      topic_index:
        { normalised_topic_name -> { "topic_name": str, "topic_description": str } }

      sub_index:
        { normalised_sub_topic_name -> {
              "sub_topic_name":        str,
              "sub_topic_description": str,
              "parent_topic_name":     str,    # the topic_name this sub belongs to
              "parent_topic_description": str,
          }
        }

    Both lookups are case-insensitive (keys are normalised).
    """
    topic_index: dict[str, dict] = {}
    sub_index:   dict[str, dict] = {}

    for entry in guidance:
        topic_name = entry.get("topic_name", "").strip()
        topic_desc = entry.get("topic_description", "").strip()

        if not topic_name:
            continue

        topic_index[_normalize(topic_name)] = {
            "topic_name":        topic_name,
            "topic_description": topic_desc,
        }

        for sub in entry.get("sub_topics", []):
            sub_name = sub.get("sub_topic_name", "").strip()
            sub_desc = sub.get("sub_topic_description", "").strip()

            if not sub_name:
                continue

            sub_index[_normalize(sub_name)] = {
                "sub_topic_name":           sub_name,
                "sub_topic_description":    sub_desc,
                "parent_topic_name":        topic_name,
                "parent_topic_description": topic_desc,
            }

    return topic_index, sub_index


# ── Guidance application ──────────────────────────────────────────────────────

def apply_user_guidance(topics: list, guidance: list) -> list:
    """
    Apply placement overrides only — root correction via guidance index.

    user_defined and guidance_context are NOT set here.
    They are resolved in tree_generator._build_node by direct title match,
    so every node (at any depth) gets the correct value independently.

    Placement logic
    ---------------
    For each topic, walk its canonical_path label by label.
    On the first label that matches a known sub_topic_name (case-insensitive):
      - Force path[0] (the root) to the correct parent topic_name from guidance.
    If a label matches a known topic_name directly, the root is corrected too.
    """
    if not guidance:
        return topics

    topic_index, sub_index = _build_guidance_index(guidance)

    for topic in topics:
        path = topic["canonical_path"]

        for i, label in enumerate(path):
            norm = _normalize(label)

            # ── Match against sub_topic_name — correct root ───────────────────
            if norm in sub_index:
                entry        = sub_index[norm]
                correct_root = entry["parent_topic_name"]
                original     = path[:]
                if path[0] != correct_root:
                    path[0] = correct_root
                    print(
                        f"  [guidance] '{label}' → root forced to '{correct_root}' "
                        f"(was: {original})"
                    )
                break

            # ── Match against topic_name directly — correct root ──────────────
            if norm in topic_index and i == 0:
                entry        = topic_index[norm]
                correct_root = entry["topic_name"]
                if path[0] != correct_root:
                    path[0] = correct_root
                    print(
                        f"  [guidance] Root label '{label}' normalised to "
                        f"canonical '{correct_root}'"
                    )
                break

    return topics


# ── Tree insertion ────────────────────────────────────────────────────────────

def _insert_into_tree(
    tree:         dict,
    path:         list,
    topic_data:   dict,
    chunk_lookup: dict,
) -> None:
    """
    Recursively walk `path` and insert topic_data at the leaf.

    tree structure:
      label -> {
        "children":       { label: node, ... },
        "chunks":         [ chunk_dict, ... ],   # only at leaves
        "doc_ids":        set,
        "keywords":       set,
        "user_defined":   bool,
        "guidance_context": dict,                # contextual hints; never replaces summary
      }
    """
    label = path[0]
    if label not in tree:
        tree[label] = {
            "children":        {},
            "chunks":          [],
            "doc_ids":         set(),
            "keywords":        set(),
            "user_defined":    False,
            "guidance_context": {},
        }

    node = tree[label]
    node["doc_ids"].update(topic_data["doc_ids"])
    node["keywords"].update(topic_data.get("keywords", []))

    if len(path) == 1:
        # ── Exact matched leaf — set user_defined and guidance_context here only ──
        if topic_data.get("user_defined"):
            node["user_defined"] = True

        # Merge guidance_context — prefer the richest (non-empty) version
        if topic_data.get("guidance_context") and not node["guidance_context"]:
            node["guidance_context"] = topic_data["guidance_context"]

        # Attach chunk dicts
        for cid in topic_data["chunk_ids"]:
            if cid in chunk_lookup:
                node["chunks"].append(chunk_lookup[cid])
    else:
        _insert_into_tree(node["children"], path[1:], topic_data, chunk_lookup)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_hierarchy(topics: list, chunks: list, guidance: list) -> dict:
    """
    Build the recursive hierarchy tree.

    Steps:
      1. Apply guidance placement overrides via apply_user_guidance().
      2. Insert each topic's canonical_path into the nested tree dict.
      3. Finalise sets to lists for JSON serialisability.

    Parameters
    ----------
    topics  : canonicalised topic list from canonicaliser.py
    chunks  : flat chunk list from chunking.py / summarizer.py
    guidance: validated structured list from main.load_user_guidance()
              ([] for AI-only mode)

    Returns
    -------
    dict: { root_label -> node_dict } consumed by tree_generator.generate_tree()
    """
    topics = apply_user_guidance(topics, guidance)

    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    tree: dict = {}
    for topic in topics:
        path = topic["canonical_path"]
        if not path:
            continue
        _insert_into_tree(tree, path, topic, chunk_lookup)

    # Convert sets to lists for JSON serialisability
    def _finalise(node: dict) -> None:
        node["doc_ids"]  = list(node["doc_ids"])
        node["keywords"] = list(node["keywords"])
        for child in node["children"].values():
            _finalise(child)

    for node in tree.values():
        _finalise(node)

    return tree