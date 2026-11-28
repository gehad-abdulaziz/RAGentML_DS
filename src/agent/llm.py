"""LLM factory supporting multiple providers (OpenAI, Ollama, Anthropic, Google, and Mock).

Owner: Member 6 (feature/agent-interface).
"""
from __future__ import annotations

import os
from typing import Any

from src import config


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Return a configured LangChain ChatModel instance.

    Supported providers:
      - 'openai': ChatOpenAI (requires langchain-openai)
      - 'ollama': ChatOllama (requires langchain-ollama or langchain-community)
      - 'anthropic': ChatAnthropic (requires langchain-anthropic)
      - 'google' | 'gemini': ChatGoogleGenerativeAI (requires langchain-google-genai)
      - 'mock': Scripted or test mock model
    """
    prov = (provider or config.LLM_PROVIDER or "openai").strip().lower()
    temp = config.LLM_TEMPERATURE if temperature is None else temperature
    key = api_key if api_key is not None else config.LLM_API_KEY

    if prov == "openai":
        from langchain_openai import ChatOpenAI

        model_name = model or config.LLM_MODEL or "gpt-4o-mini"
        kwargs: dict[str, Any] = {"model": model_name, "temperature": temp}
        if key:
            kwargs["api_key"] = key
        return ChatOpenAI(**kwargs)

    elif prov == "ollama":
        try:
            from langchain_ollama import ChatOllama

            model_name = model or config.LLM_MODEL or "qwen3:4b"
            return ChatOllama(model=model_name, temperature=temp)
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama

                model_name = model or config.LLM_MODEL or "qwen3:4b"
                return ChatOllama(model=model_name, temperature=temp)
            except ImportError as err:
                raise ImportError(
                    "Ollama support requires 'langchain-ollama' or 'langchain-community'. "
                    "Install with: pip install langchain-ollama"
                ) from err

    elif prov == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = model or config.LLM_MODEL or "claude-3-5-sonnet-20241022"
        kwargs = {"model": model_name, "temperature": temp}
        if key:
            kwargs["api_key"] = key
        return ChatAnthropic(**kwargs)

    elif prov in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = model or config.LLM_MODEL or "gemini-1.5-flash"
        kwargs = {"model": model_name, "temperature": temp}
        if key:
            kwargs["google_api_key"] = key
        return ChatGoogleGenerativeAI(**kwargs)

    elif prov == "mock":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=["I am an academic advisor mock assistant."])

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{prov}'. "
            "Supported providers: openai, ollama, anthropic, google, mock."
        )
