"""配置。

三种后端（backend）是同一套代码的三个入口，差别只在 base_url 与鉴权头：

  stub    —— 全离线，不发起任何网络请求。默认值。面试现场断网也能演示。
  direct  —— 直连任意 OpenAI 兼容端点（厂商官方 / vLLM / Ollama / DeepSeek）。
  gateway —— 指向本仓库同族的自研网关（llm-gateway）：配额、语义缓存、
             成本账本、traceId 全链路贯通都由它提供。

direct 与 gateway 之所以能共用同一个客户端实现，是因为网关补齐了
OpenAI 兼容性（响应体 choices[].message.content、流式 delta 信封、
data: [DONE] 终止帧、/v1/embeddings）。补齐之前它只兼容请求侧，
响应侧是扁平的，客户端必须写两套解析。
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

Backend = Literal["stub", "direct", "gateway"]
RerankMode = Literal["heuristic", "llm", "off"]

GATEWAY_DEFAULT_URL = "http://127.0.0.1:8080/v1"
OPENAI_DEFAULT_URL = "https://api.openai.com/v1"

# 直连模式下的计价（元 / 1K tokens，输入与输出分开）。
# gateway 模式下这些值由网关注入（gateway.pricing），这里只用于 direct 模式的成本估算。
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.0010, 0.0030),
    "gpt-4o": (0.0200, 0.0600),
    "deepseek-flash": (0.0005, 0.0015),
    "deepseek-v4-pro": (0.0020, 0.0080),
    "fake-small": (0.0010, 0.0030),
    "fake-large": (0.0200, 0.0600),
}
DEFAULT_PRICE = (0.0010, 0.0030)


class Settings(BaseModel):
    backend: Backend = "stub"
    base_url: str = OPENAI_DEFAULT_URL
    api_key: str = ""
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    tenant: str = "agentkit"

    chat_model: str = "fake-small"
    embedding_model: str = "fake-embedding"
    temperature: float = 0.2

    # 检索
    chunk_size: int = 320
    chunk_overlap: int = 64
    recall_k: int = 12
    top_n: int = 4
    context_token_budget: int = 1200
    rerank: RerankMode = "heuristic"

    # 编排
    max_steps: int = 4
    tool_timeout_seconds: float = 10.0

    # 记忆
    memory_token_budget: int = 600
    memory_keep_recent: int = 4

    # 落盘
    index_path: str = "out/index.jsonl"

    timeout_seconds: float = 60.0

    @staticmethod
    def for_backend(backend: Backend) -> "Settings":
        """按后端给出默认 base_url 与鉴权方式。"""
        if backend == "gateway":
            return Settings(
                backend="gateway",
                base_url=GATEWAY_DEFAULT_URL,
                api_key="agentkit",
                api_key_header="X-Api-Key",
                api_key_prefix="",
                chat_model="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
            )
        if backend == "direct":
            return Settings(
                backend="direct",
                base_url=OPENAI_DEFAULT_URL,
                api_key="",
                api_key_header="Authorization",
                api_key_prefix="Bearer ",
                chat_model="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
            )
        return Settings()

    @classmethod
    def from_env(cls, backend: Backend | None = None) -> "Settings":
        """环境变量覆盖。命名前缀 AGENTKIT_，避免和别人的配置撞车。"""
        resolved: Backend = backend or os.environ.get("AGENTKIT_BACKEND", "stub")  # type: ignore[assignment]
        base = cls.for_backend(resolved)

        overrides: dict[str, str] = {}
        for field in cls.model_fields:
            env_name = "AGENTKIT_" + field.upper()
            if env_name in os.environ:
                overrides[field] = os.environ[env_name]
        if overrides:
            base = base.model_copy(update=_coerce(overrides))
        return base

    def price_of(self, model: str) -> tuple[float, float]:
        return DEFAULT_PRICING.get(model, DEFAULT_PRICE)

    def auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.api_key_header] = f"{self.api_key_prefix}{self.api_key}"
        return headers


def _coerce(raw: dict[str, str]) -> dict[str, object]:
    """环境变量都是字符串，按字段类型转一下，否则 pydantic 会拒收。"""
    from typing import get_args, get_origin

    out: dict[str, object] = {}
    for key, value in raw.items():
        field = Settings.model_fields[key]
        annotation = field.annotation
        origin = get_origin(annotation)
        if origin is Literal:
            out[key] = value
        elif annotation is int:
            out[key] = int(value)
        elif annotation is float:
            out[key] = float(value)
        elif annotation is bool:
            out[key] = value.strip().lower() in {"1", "true", "yes", "on"}
        elif get_args(annotation) and type(None) in get_args(annotation):
            out[key] = None if value == "" else value
        else:
            out[key] = value
    return out
