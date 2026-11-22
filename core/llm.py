import os
from langchain_openai import ChatOpenAI

def get_llm(temperature: float = 0.3):
    """
    Returns a LangChain LLM instance.
    Supports:
    1. Local Ollama (LLM_PROVIDER="ollama") - runs 100% locally, no API key needed.
    2. Meta AI Model API (LLM_PROVIDER="meta") - runs via cloud API.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    # ── Local Ollama Provider ──────────────────────────────────────────────────
    if provider in ["ollama", "local"]:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        model = os.getenv("OLLAMA_MODEL", "mistral:7b").strip()

        # Try native langchain_ollama if installed
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=base_url,
                model=model,
                temperature=temperature,
            )
        except ImportError:
            # Fallback to Ollama's built-in OpenAI-compatible /v1 endpoint
            openai_compat_url = base_url.rstrip("/")
            if not openai_compat_url.endswith("/v1"):
                openai_compat_url += "/v1"

            return ChatOpenAI(
                model=model,
                api_key="ollama",  # Dummy key required by OpenAI client
                base_url=openai_compat_url,
                temperature=temperature,
            )

    # ── Meta AI Cloud Provider ────────────────────────────────────────────────
    api_key = (
        os.getenv("META_AI_API_KEY")
        or os.getenv("META_API_KEY")
        or os.getenv("MODEL_API_KEY")
        or os.getenv("MISTRAL_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Meta AI API Key not found! Please set META_AI_API_KEY in your .env file, "
            "or set LLM_PROVIDER=ollama to use local Ollama without an API key."
        )

    base_url = os.getenv("META_BASE_URL", "https://api.meta.ai/v1").strip()
    model = os.getenv("META_MODEL", "muse-spark-1.3").strip()

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
