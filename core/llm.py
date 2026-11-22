import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def get_llm(temperature: float = 0.3):
    """
    Returns a LangChain LLM instance.
    Supports:
    1. Groq Free Cloud (LLM_PROVIDER="groq") - fast free open-source models.
    2. Local Ollama (LLM_PROVIDER="ollama") - runs 100% locally, no API key needed.
    """
    load_dotenv(override=True)
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    # ── Groq Free Cloud Provider (Default) ────────────────────────────────────
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API Key not found! Please set GROQ_API_KEY in your .env file or choose Ollama."
            )
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    # ── Local / Tunneled Ollama Provider ──────────────────────────────────────
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
