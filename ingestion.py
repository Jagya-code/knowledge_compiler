"""
ingestion.py — Parse PDF, DOCX, and TXT files into raw text documents.
"""

from pathlib import Path


def _read_txt(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")


def _read_docx(path: str) -> str:
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")


READERS = {
    ".txt": _read_txt,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def ingest_documents(paths: list[str]) -> list[dict]:
    """
    Read each file and return a list of document dicts:
      { doc_id, filename, text }
    """
    documents = []
    for i, path in enumerate(paths):
        p = Path(path)
        suffix = p.suffix.lower()
        reader = READERS.get(suffix)
        if reader is None:
            print(f"  [skip] Unsupported format: {path}")
            continue
        try:
            text = reader(str(p))
            documents.append({
                "doc_id": f"doc_{i+1:03d}",
                "filename": p.name,
                "text": text.strip(),
            })
            print(f"  [ok] {p.name} ({len(text)} chars)")
        except Exception as e:
            print(f"  [error] {path}: {e}")
    return documents
