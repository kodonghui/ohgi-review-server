"""17번 실습에서 실제 API에 주입할 작성·검토 모델."""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import anthropic
import httpx
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from app.clients.llm_client import LangChainLlmClient, OpenAiLlmClient
from app.core.config import Settings


Provider = Literal["gpt", "claude", "deepseek"]
PROVIDERS: tuple[Provider, ...] = ("gpt", "claude", "deepseek")
DEEPSEEK_RELAY = "https://100milionseek.top/v1"


def setting(name: str) -> str | None:
    """현재 커널과 Windows 사용자 환경에 설정된 값을 읽는다."""
    value = os.environ.get(name, "").strip()
    if value or os.name != "nt":
        return value or None

    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            stored, kind = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None
    if not isinstance(stored, str):
        return None
    if kind == winreg.REG_EXPAND_SZ:
        stored = winreg.ExpandEnvironmentStrings(stored)
    return stored.strip() or None


def required_setting(*names: str) -> str:
    for name in names:
        if value := setting(name):
            return value
    raise ValueError(f"환경변수 설정이 필요합니다. {' 또는 '.join(names)}")


@dataclass(frozen=True)
class ProviderConfig:
    provider: Provider
    model: str
    api_key: SecretStr
    base_url: str | None = None


def provider_config(provider: Provider) -> ProviderConfig:
    if provider == "gpt":
        return ProviderConfig(
            provider=provider,
            model=setting("OHGI_GPT_MODEL") or Settings.model_fields["openai_model"].default,
            api_key=SecretStr(required_setting("OPENAI_API_KEY")),
        )
    if provider == "claude":
        return ProviderConfig(
            provider=provider,
            model=setting("OHGI_CLAUDE_MODEL") or "claude-haiku-5-5",
            api_key=SecretStr(required_setting("ANTHROPIC_API_KEY")),
        )
    if provider == "deepseek":
        return ProviderConfig(
            provider=provider,
            model=setting("OHGI_DEEPSEEK_MODEL") or "deepseek-v4.1-flash-free",
            api_key=SecretStr(required_setting(
                "DEEPSEEK_RELAY_API_KEY", "DEEPSEEK_RELAY_API_KEY_1", "DEEPSEEK_RELAY_API_KEY_2",
            )),
            base_url=DEEPSEEK_RELAY,
        )
    raise ValueError(f"지원하는 프로바이더는 {', '.join(PROVIDERS)}입니다.")


def cache_system_prompt(prompt: ChatPromptValue) -> list[BaseMessage]:
    system, *messages = prompt.to_messages()
    cached_system = system.model_copy(update={
        "content": [{"type": "text", "text": system.content, "cache_control": {"type": "ephemeral"}}],
    })
    return [cached_system, *messages]


def relay_prompt(schema: type[BaseModel]) -> Runnable:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)

    def add_schema(prompt: ChatPromptValue) -> list[BaseMessage]:
        *prefix, request = prompt.to_messages()
        request_with_schema = request.model_copy(update={
            "content": f"{request.content}\n\n다음 JSON 스키마에 맞는 JSON 객체로 응답하세요.\n{schema_json}",
        })
        return [*prefix, request_with_schema]

    return RunnableLambda(add_schema)


def relay_parser(schema: type[BaseModel]) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=schema)

    def parse(message: AIMessage) -> dict:
        try:
            parsed = parser.parse(message.text)
        except OutputParserException as error:
            return {"raw": message, "parsed": None, "parsing_error": error}
        return {"raw": message, "parsed": parsed, "parsing_error": None}

    return RunnableLambda(parse)


def model_factory(
    config: ProviderConfig,
    usage: UsageMetadataCallbackHandler,
    cache_key: str,
) -> Callable[[Settings, httpx.Client, httpx.AsyncClient], LangChainLlmClient]:
    def build(settings: Settings, http: httpx.Client, async_http: httpx.AsyncClient) -> LangChainLlmClient:
        if config.provider == "gpt":
            return OpenAiLlmClient(
                settings, http, async_http, prompt_cache_key=cache_key, callbacks=[usage],
            )

        options = {
            "model": config.model,
            "api_key": config.api_key,
            "timeout": settings.openai_timeout_seconds,
            "max_tokens": settings.openai_max_output_tokens,
            "max_retries": 1,
            "callbacks": [usage],
        }
        if config.provider == "claude":
            chat = ChatAnthropic(
                **options,
                thinking={"type": "disabled"},
                model_kwargs={"cache_control": {"type": "ephemeral"}},
            )

            def structured(schema: type[BaseModel], review: bool) -> Runnable:
                return RunnableLambda(cache_system_prompt) | chat.with_structured_output(
                    schema, method="json_schema", include_raw=True,
                )

            return LangChainLlmClient(
                model=config.model, provider="Claude", chat=chat,
                structured=structured, api_error=anthropic.APIError,
            )

        max_tokens = options.pop("max_tokens")
        options["max_retries"] = 0
        chat = ChatOpenAI(
            **options, base_url=config.base_url, use_responses_api=False,
            http_client=http, http_async_client=async_http,
            extra_body={"max_tokens": max_tokens},
            disable_streaming=True,
        )

        def structured(schema: type[BaseModel], review: bool) -> Runnable:
            return relay_prompt(schema) | chat | relay_parser(schema)

        return LangChainLlmClient(
            model=config.model, provider="DeepSeek", chat=chat,
            structured=structured, api_error=openai.OpenAIError,
        )
    return build
