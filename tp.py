import json
import os
import re
from openai import AzureOpenAI

AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "https://bfsi-genai-demo.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key")
AZURE_API_VERSION    = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
AZURE_MODEL          = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-nano-for-saikat-team")
# AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "https://bfsi-genai-demo.openai.azure.com/")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_API_VERSION    = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")
# AZURE_MODEL          = os.getenv("AZURE_OPENAI_MODEL", "bfsi-genai-demo-gpt-4o") 

_client = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_API_VERSION,
        )
    print(">>>>>>>>>>>>>>",AZURE_OPENAI_API_KEY)
    return _client


def _call_llm() -> str:
    client = _get_client()
    response = _get_client().chat.completions.create(
        model=AZURE_MODEL,
        messages=[
            {"role": "system", "content": "Your are a helpful assistant"},
            {"role": "user",   "content": "Hi How is Delhi?"},
        ]
    )
    return response.choices[0].message.content.strip()

if __name__=="__main__":
    res = _call_llm()
    print(">>>>>",res)