"""
topic_extractor.py — Extract a full hierarchical path for each chunk.

Sends chunk heading + summary (not full text) to the LLM.
Returns a path array of N levels, e.g.:
  ["Operations", "Claims Management", "Claims Processing", "Triage & Assignment"]

These raw paths are intentionally un-normalised — the canonicaliser (Step 4b)
is responsible for resolving naming inconsistencies across chunks before the
hierarchy is built.
"""

import json
import os
import re
from openai import AzureOpenAI

AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "https://bfsi-genai-demo.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
AZURE_API_VERSION    = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
AZURE_MODEL          = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-nano-for-saikat-team")


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


def _call_llm(system_prompt, user_prompt, max_tokens=300):
    response = _get_client().chat.completions.create(
        model=AZURE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    )
    return response.choices[0].message.content.strip()


def _parse_json_response(text):
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"path": ["General", "Miscellaneous"], "keywords": []}


SYSTEM_PATH = """You are an insurance domain expert building a knowledge hierarchy.

Given a document section heading and summary, return:
1. A hierarchical PATH from broad to specific — between 2 and 4 levels deep.
   Examples:
     ["Operations", "Claims Management", "Claims Processing", "Triage & Assignment"]
     ["Risk Management", "Fraud Controls", "Fraud Detection"]
     ["Policy Lifecycle", "Policy Origination"]
2. Up to 4 KEYWORDS describing the section content.

Rules:
- Level 1 is always a broad domain (e.g. Operations, Risk Management, Compliance).
- Each subsequent level is more specific than the one before it.
- Use consistent, canonical-sounding labels. Avoid abbreviations or jargon.
- Do NOT include document filenames in the path.

Respond ONLY with valid JSON, no markdown fences:
{"path": ["Level1", "Level2", ...], "keywords": ["kw1", "kw2"]}"""


def extract_path_for_chunk(chunk):
    """Return raw { path, keywords } for a single chunk."""
    user_prompt = (
        f"Section heading: {chunk['heading']}\n"
        f"Summary: {chunk.get('summary', chunk['text'][:300])}"
    )
    raw = _call_llm(SYSTEM_PATH, user_prompt)
    result = _parse_json_response(raw)

    path = result.get("path", [])
    if not isinstance(path, list) or not path:
        path = ["General", "Miscellaneous"]
    path = [str(p).strip() for p in path if str(p).strip()]
    result["path"] = path if len(path) >= 2 else path + ["Miscellaneous"]
    return result


def extract_topics(chunks):
    """
    Pass 1 — extract a raw hierarchical path + keywords per chunk.
    Attaches { path, keywords } back onto each chunk.

    Returns a deduplicated list of raw topic entries:
      { path, keywords, chunk_ids, doc_ids, summaries }

    'summaries' is passed to the canonicaliser for semantic verification.
    """
    total = len(chunks)
    topic_map = {}  # tuple(path) -> aggregated entry

    for i, chunk in enumerate(chunks):
        print(f"  Extracting path {i+1}/{total}: {chunk['heading'][:55]}...", end="\r")
        try:
            result = extract_path_for_chunk(chunk)
        except Exception as e:
            result = {"path": ["General", "Miscellaneous"], "keywords": []}
            print(f"\n  [warn] Path extraction failed for {chunk['chunk_id']}: {e}")

        chunk["path"]     = result["path"]
        chunk["keywords"] = result.get("keywords", [])

        key = tuple(chunk["path"])
        if key not in topic_map:
            topic_map[key] = {
                "path":      list(key),
                "keywords":  set(chunk["keywords"]),
                "chunk_ids": [],
                "doc_ids":   set(),
                "summaries": [],
            }
        topic_map[key]["chunk_ids"].append(chunk["chunk_id"])
        topic_map[key]["doc_ids"].add(chunk["doc_id"])
        topic_map[key]["keywords"].update(chunk["keywords"])
        if chunk.get("summary"):
            topic_map[key]["summaries"].append(chunk["summary"])

    print()

    topics = []
    for entry in topic_map.values():
        entry["keywords"] = list(entry["keywords"])
        entry["doc_ids"]  = list(entry["doc_ids"])
        topics.append(entry)

    return topics
