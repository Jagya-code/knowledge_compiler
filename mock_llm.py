"""
mock_llm.py — Drop-in mock for Azure OpenAI.
Patches summarizer, topic_extractor, canonicaliser, and tree_generator.
Activate with: python main.py --mock
"""

import re
from collections import defaultdict

# ── Domain knowledge for deterministic mock paths ────────────────────────────

PATH_MAP = {
    "claims":      ["Operations", "Claims Management", "Claims Processing"],
    "triage":      ["Operations", "Claims Management", "Claims Processing", "Triage & Assignment"],
    "settlement":  ["Operations", "Claims Management", "Claims Settlement"],
    "subrogat":    ["Operations", "Claims Management", "Subrogation"],
    "investigat":  ["Operations", "Claims Management", "Claims Investigation"],
    "fraud":       ["Risk Management", "Fraud Controls", "Fraud Detection"],
    "siu":         ["Risk Management", "Fraud Controls", "SIU Investigation"],
    "risk scor":   ["Risk Management", "Fraud Controls", "Risk Scoring"],
    "model gov":   ["Risk Management", "Fraud Controls", "Model Governance"],
    "underwrit":   ["Policy Lifecycle", "Policy Origination", "Underwriting"],
    "issuance":    ["Policy Lifecycle", "Policy Origination", "Policy Issuance"],
    "originat":    ["Policy Lifecycle", "Policy Origination"],
    "renew":       ["Policy Lifecycle", "Policy Renewal"],
    "amendment":   ["Policy Lifecycle", "Policy Amendments"],
    "terminat":    ["Policy Lifecycle", "Policy Termination"],
    "kyc":         ["Customer Management", "Customer Onboarding", "KYC Verification"],
    "onboard":     ["Customer Management", "Customer Onboarding"],
    "identity":    ["Customer Management", "Customer Onboarding", "Digital Identity Verification"],
    "customer dat":["Customer Management", "Customer Data Management"],
    "welcome":     ["Customer Management", "Customer Onboarding", "Welcome & Activation"],
    "motor":       ["Product Management", "Personal Lines", "Motor Insurance"],
    "home":        ["Product Management", "Personal Lines", "Home Insurance"],
    "commercial":  ["Product Management", "Commercial Lines", "Commercial Property"],
    "product dev": ["Product Management", "Product Development"],
    "solvency":    ["Compliance", "Regulatory Framework", "Solvency & Capital"],
    "regulat":     ["Compliance", "Regulatory Framework"],
    "compli":      ["Compliance", "Compliance Programme"],
    "reporting":   ["Compliance", "Compliance Programme", "Regulatory Reporting"],
    "consumer":    ["Compliance", "Consumer Protection"],
    "data priv":   ["Compliance", "Consumer Protection", "Data Privacy & Security"],
}


def _infer_path(text):
    lower = text.lower()
    # Longest key wins to prefer specific matches
    for kw in sorted(PATH_MAP, key=len, reverse=True):
        if kw in lower:
            return PATH_MAP[kw]
    return ["General Operations", "Miscellaneous"]


def mock_summarize(chunk):
    h = chunk.get("heading", "")
    f = chunk.get("filename", "")
    snippet = chunk.get("text", "")[:80].replace("\n", " ")
    return f"This section covers {h} from {f}. Key content: {snippet}..."


def mock_extract_path(chunk):
    combined = chunk.get("heading", "") + " " + chunk.get("summary", "") + " " + chunk.get("text", "")
    path     = _infer_path(combined)
    words    = re.findall(r"[A-Za-z]{4,}", chunk.get("heading", ""))
    keywords = list({w.lower() for w in words[:4]})
    return {"path": path, "keywords": keywords}


def mock_summarize_node(title, child_summaries):
    snippets = ", ".join(s[:45] for s in child_summaries[:3])
    return (
        f"{title} encompasses key processes within the insurance operation. "
        f"Sub-sections cover: {snippets}."
    )


# ── Canonicaliser mock ────────────────────────────────────────────────────────
# The mock returns paths that are already consistent, so canonicalisation is
# a no-op: canonical_path = path, no merges needed.

def mock_canonicalise(topics):
    print("  [mock] Canonicalisation: paths already consistent, no merges.")
    for topic in topics:
        topic["canonical_path"] = topic["path"][:]
    return topics


# ── Monkey-patch ──────────────────────────────────────────────────────────────

def activate():
    import summarizer
    import topic_extractor
    import canonicaliser
    import tree_generator

    # -- summarizer --
    def patched_summarize_chunks(chunks):
        for c in chunks:
            c["summary"] = mock_summarize(c)
        return chunks
    summarizer.summarize_chunk  = mock_summarize
    summarizer.summarize_chunks = patched_summarize_chunks

    # -- topic_extractor --
    def patched_extract_topics(chunks):
        topic_map = {}
        for chunk in chunks:
            result            = mock_extract_path(chunk)
            chunk["path"]     = result["path"]
            chunk["keywords"] = result["keywords"]
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

        topics = []
        for entry in topic_map.values():
            entry["keywords"] = list(entry["keywords"])
            entry["doc_ids"]  = list(entry["doc_ids"])
            topics.append(entry)
        print(f"  Extracted {len(topics)} raw paths")
        return topics

    topic_extractor.extract_path_for_chunk = mock_extract_path
    topic_extractor.extract_topics         = patched_extract_topics

    # -- canonicaliser --
    canonicaliser.canonicalise = mock_canonicalise

    # -- tree_generator --
    tree_generator._summarize_node = mock_summarize_node

    print("[mock] Mock LLM active — no Azure credentials required.\n")
