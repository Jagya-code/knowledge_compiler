"""
chunking.py — Split documents into logical chunks with metadata.

Strategy:
  1. Split on heading patterns (e.g. lines that look like section headers).
  2. Fall back to fixed-size page simulation if no headings found.
  3. Each chunk carries: doc_id, chunk_id, heading, text, page_approx.
"""

import re


# Heading heuristic: short lines (<= 80 chars) that are ALL CAPS, Title Case,
# or end with a colon — typical of report/policy document section headers.
HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z\s\-&/]{3,79}|[A-Z][a-z][\w\s\-&/,]{3,79}:|[\d]+[\.\)]\s+.{3,70})$"
)

CHUNK_CHAR_LIMIT = 1500  # soft cap per chunk before forcing a split


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Return list of (heading, body) pairs."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and HEADING_RE.match(stripped) and len(stripped) < 90:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = stripped.rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return [(h, b) for h, b in sections if b.strip()]


def _fixed_chunks(text: str, size: int = CHUNK_CHAR_LIMIT) -> list[tuple[str, str]]:
    """Fallback: split into fixed-size chunks labelled by position."""
    chunks = []
    for i, start in enumerate(range(0, len(text), size)):
        body = text[start: start + size].strip()
        if body:
            chunks.append((f"Section {i + 1}", body))
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk each document and return flat list of chunk dicts:
      { chunk_id, doc_id, filename, heading, text, page_approx }
    """
    all_chunks = []
    chunk_counter = 1

    for doc in documents:
        text = doc["text"]
        sections = _split_by_headings(text)
        if len(sections) < 2:
            sections = _fixed_chunks(text)

        # Further split sections that exceed the char limit
        expanded: list[tuple[str, str]] = []
        for heading, body in sections:
            if len(body) > CHUNK_CHAR_LIMIT * 1.5:
                sub = _fixed_chunks(body)
                for j, (_, sbody) in enumerate(sub):
                    expanded.append((f"{heading} (part {j+1})", sbody))
            else:
                expanded.append((heading, body))

        for page_approx, (heading, body) in enumerate(expanded, start=1):
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "heading": heading,
                "text": body,
                "page_approx": page_approx,
                "summary": None,      # filled by summarizer
                "topics": [],         # filled by topic extractor
            })
            chunk_counter += 1

    return all_chunks
