"""
summarizer.py — Summarize each chunk using the Azure OpenAI LLM.

One LLM call per chunk keeps token usage minimal (chunk ≤ 1500 chars).
"""

import os
from openai import AzureOpenAI

# ── Azure credentials (set as env vars or .env) ──────────────────────────────
AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "https://bfsi-genai-demo.openai.azure.com")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
AZURE_API_VERSION    = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
AZURE_MODEL          = os.getenv("AZURE_OPENAI_MODEL", "bfsi-genai-demo-gpt-4o") 



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


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=AZURE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    )
    return response.choices[0].message.content.strip()


SYSTEM_SUMMARIZE = (
    "You are an expert insurance domain analyst. "
    "Summarize the provided text in 1–2 concise sentences, "
    "capturing the key topic and purpose. Be specific."
)


def summarize_chunk(chunk: dict) -> str:
    """Return a 1-2 sentence summary for a single chunk."""
    prompt = (
        f"Document: {chunk['filename']}\n"
        f"Section: {chunk['heading']}\n\n"
        f"{chunk['text'][:1400]}"
    )
    return _call_llm(SYSTEM_SUMMARIZE, prompt)


def summarize_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add a 'summary' field to every chunk.
    Prints progress as it goes.
    """
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i+1}/{total}: {chunk['heading'][:50]}...", end="\r")
        try:
            chunk["summary"] = summarize_chunk(chunk)
        except Exception as e:
            chunk["summary"] = f"(Summary unavailable: {e})"
    print()  # newline after progress
    return chunks