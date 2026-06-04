"""
Document Hierarchy Builder — N-level Hybrid AI + User-Guided Context Tree

Usage:
    python main.py                        # real Azure LLM, sample docs + guidance
    python main.py --mock                 # mock LLM (no Azure credentials needed)
    python main.py --no-guidance          # AI-only, no user guidance
    python main.py --mock --no-guidance   # mock + AI-only
    python main.py --docs a.pdf b.docx    # custom documents
    python main.py --guidance input_config.json  # custom guidance file

Output is always saved to: output/output_hierarchy.json
"""

import argparse
import json
import sys
from pathlib import Path

# ── Fixed output locations ────────────────────────────────────────────────────
OUTPUT_PATH        = "output/output_hierarchy.json"
PRUNED_OUTPUT_PATH = "output/pruned_knowledge_map.json"
CHUNKS_OUTPUT_PATH = "output/chunks.json"


def _collect_chunk_ids(nodes: list) -> set:
    """
    Walk the pruned tree recursively and collect all chunk_ids
    from every surviving node.
    """
    ids = set()
    for node in nodes:
        ids.update(node.get("chunk_ids", []))
        ids.update(_collect_chunk_ids(node.get("nodes", [])))
    return ids


# ── Pruner ────────────────────────────────────────────────────────────────────

def _build_allowed_titles(guidance: list) -> set:
    """
    Collect every topic_name and sub_topic_name from the guidance config
    into a single case-insensitive set of allowed titles.
    """
    allowed = set()
    for entry in guidance:
        topic_name = entry.get("topic_name", "").strip()
        if topic_name:
            allowed.add(topic_name.lower())
        for sub in entry.get("sub_topics", []):
            sub_name = sub.get("sub_topic_name", "").strip()
            if sub_name:
                allowed.add(sub_name.lower())
    return allowed


def prune_knowledge_map(tree: list, guidance: list) -> list:
    """
    Post-process the full output tree and keep only nodes whose title
    exactly matches a topic_name or sub_topic_name in the guidance config.

    Rules:
      - Search the ENTIRE tree at any depth regardless of parent.
      - A node survives if its title (case-insensitive) is in the allowed set.
      - If a parent is not in config but a child is, the child is still kept.
      - Surviving nodes have their own children pruned by the same rule.

    Returns a flat list of all matching nodes (with their matched children).
    """
    if not guidance:
        return []

    allowed = _build_allowed_titles(guidance)

    def _collect_nodes(nodes: list) -> list:
        result = []
        for node in nodes:
            if node.get("title", "").lower() in allowed:
                # Keep this node, prune its children by same rule
                pruned_node = {**node, "nodes": _collect_nodes(node.get("nodes", []))}
                result.append(pruned_node)
            else:
                # Node itself not in config — but still search its children
                result.extend(_collect_nodes(node.get("nodes", [])))
        return result

    return _collect_nodes(tree)


# ── Guidance loader + validator ───────────────────────────────────────────────

class GuidanceValidationError(Exception):
    """Raised when the guidance config fails structural or semantic validation."""
    pass


def _validate_guidance(raw: list) -> None:
    """
    Validate the parsed guidance list.
    Raises GuidanceValidationError with a descriptive message on first failure.

    Rules enforced:
      - Each entry must have a non-empty topic_name (str)
      - topic_name must be unique across all entries (case-insensitive)
      - Each sub_topic must have a non-empty sub_topic_name (str)
      - sub_topic_name must be unique within its parent topic (case-insensitive)
    """
    seen_topics: dict[str, str] = {}  # normalised -> original

    for i, entry in enumerate(raw):
        # ── topic_name ────────────────────────────────────────────────────────
        topic_name = entry.get("topic_name", "")
        if not isinstance(topic_name, str) or not topic_name.strip():
            raise GuidanceValidationError(
                f"Entry at index {i}: 'topic_name' is missing or empty."
            )
        topic_name = topic_name.strip()
        norm_topic = topic_name.lower()

        if norm_topic in seen_topics:
            raise GuidanceValidationError(
                f"Duplicate topic_name '{topic_name}' at index {i} "
                f"(already defined as '{seen_topics[norm_topic]}')."
            )
        seen_topics[norm_topic] = topic_name

        # ── sub_topics ────────────────────────────────────────────────────────
        sub_topics = entry.get("sub_topics", [])
        if not isinstance(sub_topics, list):
            raise GuidanceValidationError(
                f"Topic '{topic_name}': 'sub_topics' must be a list, "
                f"got {type(sub_topics).__name__}."
            )

        seen_subs: dict[str, str] = {}  # normalised -> original (within topic)
        for j, sub in enumerate(sub_topics):
            sub_name = sub.get("sub_topic_name", "")
            if not isinstance(sub_name, str) or not sub_name.strip():
                raise GuidanceValidationError(
                    f"Topic '{topic_name}', sub-topic at index {j}: "
                    f"'sub_topic_name' is missing or empty."
                )
            sub_name = sub_name.strip()
            norm_sub = sub_name.lower()

            if norm_sub in seen_subs:
                raise GuidanceValidationError(
                    f"Topic '{topic_name}': duplicate sub_topic_name '{sub_name}' "
                    f"at index {j} (already defined as '{seen_subs[norm_sub]}')."
                )
            seen_subs[norm_sub] = sub_name


def load_user_guidance(path: str) -> list:
    """
    Parse and validate guidance from the structured input_config.json format:

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

    Returns the validated structured list as-is.
    hierarchy_builder.apply_user_guidance() consumes this structure natively —
    it owns all placement logic and description-hint handling.

    On file-not-found or unreadable file: logs a warning and returns [] (AI-only).
    On validation failure: raises GuidanceValidationError (caller handles it).
    """
    p = Path(path)

    if not p.exists():
        print(f"  [warn] Guidance file not found: '{path}'. Proceeding in AI-only mode.")
        return []

    if p.suffix.lower() != ".json":
        print(
            f"  [warn] Unsupported guidance format '{p.suffix}'. "
            "Only .json is accepted. Proceeding in AI-only mode."
        )
        return []

    with open(p, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"  [warn] Guidance file contains invalid JSON: {exc}. Proceeding in AI-only mode.")
            return []

    if not isinstance(raw, list):
        print(
            "  [warn] Guidance JSON must be a top-level list of topic objects. "
            "Proceeding in AI-only mode."
        )
        return []

    # Validate — raises GuidanceValidationError on failure
    _validate_guidance(raw)

    sub_count = sum(len(e.get("sub_topics", [])) for e in raw)
    print(f"  Loaded {len(raw)} topic(s), {sub_count} sub-topic(s) from guidance.")
    return raw


# ── Tree printer (CLI only) ───────────────────────────────────────────────────

def print_tree(nodes: list, indent: int = 0) -> None:
    for node in nodes:
        prefix = "  " * indent
        tag    = "[user]" if node.get("user_defined") else "[ai]  "
        depth  = node.get("depth", indent)
        print(f"{prefix}{tag} [d{depth}][{node['node_id']}] {node['title']}")
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 1)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    doc_paths: list,
    guidance_path: str | None = None,
    output_path: str = OUTPUT_PATH,
) -> list:
    """
    7-step pipeline:
      1. Ingest      — parse documents to clean text
      2. Chunk       — split into heading-bounded chunks with metadata
      3. Summarise   — 1-2 sentence LLM summary per chunk
      4. Extract     — LLM assigns a hierarchical path (N levels) per chunk
      4b. Canonalise — 2nd LLM pass: normalise labels + verify semantic merges
      5. Guidance    — apply user-defined placement overrides
      6. Build       — insert paths into recursive tree dict
      7. Generate    — emit final JSON schema with summaries at every node

    Output is always written to output/output_hierarchy.json.
    The output/ directory is created automatically if it does not exist.
    """
    from ingestion         import ingest_documents
    from chunking          import chunk_documents
    from summarizer        import summarize_chunks
    from topic_extractor   import extract_topics
    from canonicaliser     import canonicalise
    from hierarchy_builder import build_hierarchy
    from tree_generator    import generate_tree

    print("\n=== Step 1: Document Ingestion ===")
    documents = ingest_documents(doc_paths)
    print(f"  Loaded {len(documents)} document(s).")

    print("\n=== Step 2: Chunking ===")
    chunks = chunk_documents(documents)
    print(f"  Produced {len(chunks)} chunk(s).")

    print("\n=== Step 3: Summarisation ===")
    chunks = summarize_chunks(chunks)
    print(f"  Summarised {len(chunks)} chunk(s).")

    print("\n=== Step 4: Path Extraction (raw) ===")
    topics = extract_topics(chunks)
    print(f"  Extracted {len(topics)} raw path(s).")

    print("\n=== Step 4b: Canonicalisation (label normalise + semantic verify) ===")
    topics = canonicalise(topics)
    print(f"  {len(topics)} canonical path(s) after merge/split.")

    print("\n=== Step 5: User Guidance ===")
    # guidance is the raw validated list — [] means AI-only mode
    guidance: list = []
    if guidance_path:
        guidance = load_user_guidance(guidance_path)
        if not guidance:
            print("  No valid guidance entries — running in AI-only mode.")
    else:
        print("  No guidance provided — AI-only mode.")

    print("\n=== Step 6: Hierarchy Construction ===")
    hierarchy_tree = build_hierarchy(topics, chunks, guidance)
    print(f"  Built {len(hierarchy_tree)} root node(s).")

    print("\n=== Step 7: Tree Generation ===")
    tree = generate_tree(hierarchy_tree, guidance)

    # ── Persist full hierarchy ────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2)
    print(f"  Saved hierarchy to: {out.resolve()}")

    # ── Step 8: Prune to config-only topics/subtopics ─────────────────────────
    print("\n=== Step 8: Pruning to input_config topics/subtopics ===")
    if guidance:
        pruned = prune_knowledge_map(tree, guidance)
        pruned_out = Path(PRUNED_OUTPUT_PATH)
        pruned_out.parent.mkdir(parents=True, exist_ok=True)
        with open(pruned_out, "w", encoding="utf-8") as f:
            json.dump(pruned, f, indent=2)
        print(f"  Pruned to {len(pruned)} root node(s).")
        print(f"  Saved pruned map to: {pruned_out.resolve()}")

        # ── Save only chunks that belong to surviving pruned nodes ────────────
        surviving_ids = _collect_chunk_ids(pruned)
        chunk_lookup  = {c["chunk_id"]: c for c in chunks}
        pruned_chunks = [chunk_lookup[cid] for cid in surviving_ids if cid in chunk_lookup]
        chunks_out = Path(CHUNKS_OUTPUT_PATH)
        chunks_out.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_out, "w", encoding="utf-8") as f:
            json.dump(pruned_chunks, f, indent=2)
        print(f"  Saved {len(pruned_chunks)} pruned chunk(s) to: {chunks_out.resolve()}")
    else:
        print("  No guidance provided — skipping pruning.")

    return tree


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Hierarchy Builder")
    parser.add_argument(
        "--docs", nargs="+",
        help="Input document paths (.txt, .pdf, .docx)",
    )
    parser.add_argument(
        "--guidance", default="input_config.json",
        help="Path to structured guidance JSON (default: input_config.json)",
    )
    parser.add_argument(
        "--no-guidance", action="store_true",
        help="Skip guidance; run in AI-only mode",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM — no Azure credentials required",
    )
    args = parser.parse_args()

    if args.mock:
        import mock_llm
        mock_llm.activate()

    doc_dir   = Path("sample_docs")
    doc_paths = args.docs or [str(p) for p in sorted(doc_dir.glob("*.txt"))]

    if not doc_paths:
        print("No documents found. Place .txt/.pdf/.docx files in sample_docs/ or pass --docs.")
        sys.exit(1)

    guidance_path = None if args.no_guidance else args.guidance

    try:
        tree = run_pipeline(
            doc_paths     = doc_paths,
            guidance_path = guidance_path,
            output_path   = OUTPUT_PATH,
        )
    except GuidanceValidationError as exc:
        print(f"\n[ERROR] Guidance validation failed:\n  {exc}")
        sys.exit(1)

    print("\n=== Final Hierarchy Preview ===")
    print_tree(tree)