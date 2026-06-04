"""
canonicaliser.py — Pass 2 LLM call: normalise raw paths into a consistent
taxonomy, then verify semantic similarity using chunk summaries.

Problem this solves:
  Chunk A path: ["Operations", "Claims Management", "Claims Processing"]
  Chunk B path: ["Operations", "Claims Handling",   "Claims Processing"]

  "Claims Management" and "Claims Handling" are the same concept but the LLM
  named them differently across chunks. Without canonicalisation these become
  two separate branches in the tree.

Two-stage process:
  Stage 1 — Label canonicalisation
    Collect all unique labels at each depth level. Ask the LLM to group
    synonymous labels and pick one canonical name for each group.

  Stage 2 — Semantic verification
    For any two paths that become identical after label normalisation,
    verify using their chunk summaries that they truly cover the same
    subject matter. If the LLM judges them semantically different despite
    identical labels, they are split into distinct paths by appending a
    disambiguating suffix.
"""

import json
import os
import re
from collections import defaultdict
from openai import AzureOpenAI

# AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
# AZURE_API_VERSION    = os.getenv("AZURE_API_VERSION", "2024-02-01")
# AZURE_MODEL          = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")

AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "https://bfsi-genai-demo.openai.azure.com")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
AZURE_API_VERSION    = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
AZURE_MODEL          = os.getenv("AZURE_OPENAI_MODEL", "bfsi-genai-demo-gpt-4o") 

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_API_VERSION,
        )
    return _client


def _call_llm(system_prompt, user_prompt, max_tokens=600):
    response = _get_client().chat.completions.create(
        model=AZURE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,   # deterministic — canonicalisation must be stable
    )
    return response.choices[0].message.content.strip()


def _parse_json(text):
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


# ── Stage 1: Label canonicalisation ──────────────────────────────────────────

SYSTEM_CANONICALISE = """You are an insurance taxonomy expert.

You will receive a list of labels that appear at the SAME depth level of a
knowledge hierarchy. Your job is to group synonymous or near-identical labels
and assign ONE canonical name to each group.

Rules:
- Only merge labels that mean the same thing in an insurance context.
- Prefer the clearest, most widely-used industry term as the canonical name.
- Labels that are genuinely distinct must remain separate.
- Return ONLY valid JSON, no markdown fences.

Input format:  ["label1", "label2", ...]
Output format: {"label1": "Canonical Name", "label2": "Canonical Name", ...}
Every input label must appear as a key in the output."""


def _canonicalise_labels_at_depth(labels):
    """
    Given a list of label strings at one depth level, return a dict mapping
    each raw label -> canonical label.
    """
    if len(labels) <= 1:
        return {labels[0]: labels[0]} if labels else {}

    user_prompt = json.dumps(labels)
    raw = _call_llm(SYSTEM_CANONICALISE, user_prompt, max_tokens=400)
    mapping = _parse_json(raw)

    # Safety net: any label not returned by the LLM maps to itself
    for label in labels:
        if label not in mapping:
            mapping[label] = label

    return mapping


def build_label_mapping(topics):
    """
    Collect all unique labels at each depth level across all topic paths,
    then ask the LLM to canonicalise each level independently.

    Returns a nested dict:  label_map[depth][raw_label] = canonical_label
    """
    # Collect unique labels per depth
    labels_by_depth = defaultdict(set)
    for topic in topics:
        for depth, label in enumerate(topic["path"]):
            labels_by_depth[depth].add(label)

    label_map = {}
    max_depth = max(labels_by_depth.keys()) if labels_by_depth else 0

    for depth in range(max_depth + 1):
        labels = sorted(labels_by_depth[depth])
        print(f"  Canonicalising depth {depth+1} ({len(labels)} labels): {labels}")
        label_map[depth] = _canonicalise_labels_at_depth(labels)

    return label_map


def apply_label_mapping(topics, label_map):
    """Replace every raw label in each topic's path with its canonical form."""
    for topic in topics:
        canonical_path = []
        for depth, label in enumerate(topic["path"]):
            canon = label_map.get(depth, {}).get(label, label)
            canonical_path.append(canon)
        topic["canonical_path"] = canonical_path
    return topics


# ── Stage 2: Semantic verification ───────────────────────────────────────────

SYSTEM_VERIFY = """You are an insurance domain expert.

Two groups of document sections have been assigned the SAME position in a
knowledge hierarchy (identical path). Your job is to decide whether they
truly cover the same subject matter and belong together, or whether they
are semantically distinct topics that happen to share a label.

Respond ONLY with valid JSON, no markdown fences:
{"same_topic": true/false, "reason": "brief explanation"}

Be conservative: only return false if the sections are clearly about
different subjects (e.g. one is about premium calculation, the other about
IT system architecture). Minor differences in scope or focus are fine."""


def _verify_merge(path, summaries_a, summaries_b):
    """
    Ask the LLM whether two groups of summaries sharing the same canonical
    path are truly the same topic.
    Returns True (merge) or False (keep separate).
    """
    def fmt(summaries):
        return "\n".join(f"- {s}" for s in summaries[:3])

    user_prompt = (
        f"Path: {' > '.join(path)}\n\n"
        f"Group A summaries:\n{fmt(summaries_a)}\n\n"
        f"Group B summaries:\n{fmt(summaries_b)}"
    )
    raw = _call_llm(SYSTEM_VERIFY, user_prompt, max_tokens=150)
    result = _parse_json(raw)
    same = result.get("same_topic", True)
    reason = result.get("reason", "")
    if not same:
        print(f"  [verify] Split kept: '{' > '.join(path)}' — {reason}")
    return same


def verify_and_merge(topics):
    """
    Group topics by canonical_path. For any group with more than one entry
    (i.e. two raw paths that mapped to the same canonical path), verify
    semantically that they should actually be merged.

    If verified same  → merge into one topic (combine chunk_ids, doc_ids, summaries).
    If verified different → keep separate, append a disambiguating suffix to one.

    Returns the final, verified topic list.
    """
    # Group by canonical path
    path_groups = defaultdict(list)
    for topic in topics:
        key = tuple(topic["canonical_path"])
        path_groups[key].append(topic)

    verified = []
    for canon_path, group in path_groups.items():
        if len(group) == 1:
            verified.append(group[0])
            continue

        # Multiple raw paths collapsed to the same canonical path —
        # verify pairwise whether they should merge
        merged = group[0]
        for other in group[1:]:
            should_merge = _verify_merge(
                list(canon_path),
                merged["summaries"],
                other["summaries"],
            )
            if should_merge:
                # Merge other into merged
                merged["chunk_ids"].extend(other["chunk_ids"])
                merged["doc_ids"]  = list(set(merged["doc_ids"]) | set(other["doc_ids"]))
                merged["summaries"].extend(other["summaries"])
                merged["keywords"] = list(set(merged["keywords"]) | set(other["keywords"]))
            else:
                # Keep separate — disambiguate by appending a suffix to the other's path
                disambig = list(canon_path)
                disambig[-1] = disambig[-1] + " (II)"
                other["canonical_path"] = disambig
                verified.append(other)

        verified.append(merged)

    return verified


# ── Main entry point ──────────────────────────────────────────────────────────

def canonicalise(topics):
    """
    Full two-stage canonicalisation:
      1. Normalise labels at each depth level across all paths.
      2. Verify semantic similarity for paths that collapse to the same canonical path.

    Adds 'canonical_path' to each topic dict.
    Returns the verified, merged topic list.
    """
    print(f"  Stage 1: Label canonicalisation ({len(topics)} raw paths)")
    label_map = build_label_mapping(topics)
    topics    = apply_label_mapping(topics, label_map)

    print(f"  Stage 2: Semantic verification")
    topics = verify_and_merge(topics)

    print(f"  Canonicalisation complete: {len(topics)} final paths")
    return topics
