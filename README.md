# Document Hierarchy Builder

Lightweight Python pipeline that converts multiple unstructured insurance documents into a **structured, hierarchical JSON context tree** — with optional user control over topic grouping.

---

## Architecture

```
Documents (PDF/DOCX/TXT)
        │
        ▼
 ┌─────────────┐
 │  Ingestion  │  ingestion.py — parse files to clean text
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Chunking   │  chunking.py — heading-aware splitting (~1500 chars/chunk)
 └──────┬──────┘
        ▼
 ┌──────────────┐
 │ Summarizer   │  summarizer.py — 1-2 sentence summary per chunk (LLM)
 └──────┬───────┘
        ▼
 ┌──────────────────┐
 │ Topic Extractor  │  topic_extractor.py — main_topic + subtopic + keywords (LLM)
 └──────┬───────────┘
        │         ┌──────────────────────┐
        ├─────────┤  User Guidance JSON  │  sample_guidance.json
        │         └──────────────────────┘
        ▼
 ┌───────────────────┐
 │ Hierarchy Builder │  hierarchy_builder.py — merge AI topics + user overrides
 └──────┬────────────┘
        ▼
 ┌────────────────┐
 │ Tree Generator │  tree_generator.py — emit final JSON hierarchy
 └──────┬─────────┘
        ▼
  output/hierarchy.json
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set Azure OpenAI credentials (for real runs)
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_OPENAI_API_KEY=your-key-here
set AZURE_API_VERSION=2024-02-01
set AZURE_OPENAI_MODEL=gpt-4o
```

---

## Running

### Demo mode (no Azure credentials needed)
```bash
python demo.py
```
Runs both **hybrid** (AI + user guidance) and **AI-only** modes, saves results to `output/`.

### Full pipeline with real Azure LLM
```bash
python main.py
```

### Custom documents and guidance
```bash
python demo.py --docs path/to/doc1.pdf path/to/doc2.docx --guidance my_guidance.json
```

### AI-only mode (no guidance)
```bash
python demo.py --no-guidance
```

---

## User Guidance Format

**JSON** (`sample_guidance.json`):
```json
{
  "Claims Processing":    "Operations",
  "Fraud Detection":      "Risk Management",
  "Customer KYC":         "Customer Management",
  "Regulatory Reporting": "Compliance"
}
```

**Excel** (`.xlsx`): Two-column sheet — Column A: Subtopic, Column B: Parent Topic.

---

## Output Schema

```json
[
  {
    "title":        "Operations",
    "node_id":      "0001",
    "start_index":  1,
    "end_index":    12,
    "summary":      "Operations covers claims processing and settlement...",
    "source_docs":  ["claims_processing.txt"],
    "user_defined": false,
    "nodes": [
      {
        "title":        "Claims Processing",
        "node_id":      "0002",
        "start_index":  1,
        "end_index":    5,
        "summary":      "Covers FNOL through settlement and closure.",
        "keywords":     ["claims", "adjuster", "settlement"],
        "source_docs":  ["claims_processing.txt"],
        "user_defined": true,
        "nodes":        []
      }
    ]
  }
]
```

---

## Module Reference

| File | Purpose |
|------|---------|
| `main.py` | Pipeline orchestrator |
| `ingestion.py` | PDF / DOCX / TXT parsing |
| `chunking.py` | Heading-aware text splitting |
| `summarizer.py` | Per-chunk LLM summarization |
| `topic_extractor.py` | LLM topic/subtopic extraction |
| `hierarchy_builder.py` | Merge AI + user guidance |
| `tree_generator.py` | Emit final JSON tree |
| `mock_llm.py` | Mock LLM for testing |
| `demo.py` | CLI demo runner |

---

## Design Decisions

- **Chunk-level LLM calls** — never sends full documents; keeps token cost low
- **Heading-aware chunking** — preserves semantic boundaries from document structure
- **Case-insensitive guidance matching** — robust to capitalisation differences
- **`user_defined` flag** — every node records whether placement was AI or user-driven
- **Cross-document grouping** — same subtopic from multiple docs merges under one node
- **Function-based** — no classes; straightforward to read and extend
