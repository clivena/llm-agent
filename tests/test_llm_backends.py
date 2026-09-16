"""模型客户端与传输层测试。

这里的重点是**网关兼容性**：direct 与 gateway 能共用同一个客户端实现，
靠的就是网关把响应侧补齐成标准 OpenAI 形状。这条契约必须有测试守着，
否则网关哪天改回扁平结构，客户端会静默解析出空答案。
"""

from __future__ import annotations

import pytest

from agentkit.config import Settings
from agentkit.llm import (
    OpenAiCompatibleClient,
    StubClient,
    build_client,
    build_embedder,
)
from agentkit.transport import HttpResult, JsonTransport, TransportError, join_url


# --- 配置 ---

def test_gateway_backend_uses_own_auth_header() -> None:
    settings = Settings.for_backend("gateway")
    assert settings.base_url.endswith("/v1")
    assert settings.api_key_header == "X-Api-Key"
    assert settings.api_key_prefix == ""      # 网关认裸 key，不认 Bearer
    assert settings.auth_headers()["X-Api-Key"] == "agentkit"


def test_direct_backend_uses_bearer_token() -> None:
    settings = Settings.for_backend("direct")
    settings = settings.model_copy(update={"api_key": "sk-x"})
    assert settings.auth_headers()["Authorization"] == "Bearer sk-x"


def test_auth_headers_omit_empty_key() -> None:
    headers = Settings().auth_headers()
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_from_env_overrides_and_coerces_types(monkeypatch) -> None:
    monkeypatch.setenv("AGENTKIT_RECALL_K", "20")
    monkeypatch.setenv("AGENTKIT_TEMPERATURE", "0.9")
    monkeypatch.setenv("AGENTKIT_RERANK", "off")
    monkeypatch.setenv("AGENTKIT_IGNORED", "x")
    settings = Settings.from_env("stub")
    assert settings.recall_k == 20 and isinstance(settings.recall_k, int)
    assert settings.temperature == pytest.approx(0.9)
    assert settings.rerank == "off"
    assert not hasattr(settings, "ignored")  # 未知前缀不能污染配置


def test_price_lookup_falls_back_to_default() -> None:
    settings = Settings()
    assert settings.price_of("deepseek-flash") == (0.0005, 0.0015)
    assert settings.price_of("完全不认识的模型") == (0.0010, 0.0030)


# --- 传输层 ---

def test_join_url_normalizes_slashes() -> None:
    assert join_url("http://h/v1/", "/chat/completions") == "http://h/v1/chat/completions"
    assert join_url("http://h/v1", "chat/completions") == "http://h/v1/chat/completions"


def test_transport_error_flags_rate_limit() -> None:
    assert TransportError("x", status=429).is_rate_limited() is True
    assert TransportError("x", status=502).is_rate_limited() is False


def test_transport_installs_empty_proxy_handler() -> None:
    # 不做这一步，"连本地网关"会变成"走环境代理再打回本机"
    transport = JsonTransport()
    assert transport._opener is not None


# --- OpenAI 兼容客户端 ---

def _client(monkeypatch, result: HttpResult) -> OpenAiCompatibleClient:
    settings = Settings(backend="gateway", base_url="http://gw/v1", api_key="k",
                        chat_model="fake-small", embedding_model="fake-embedding")
    client = OpenAiCompatibleClient(settings)
    monkeypatch.setattr(client._transport, "post", lambda *a, **k: result)
    return client


def _chat_body(text: str = "你好", model: str = "fake-small") -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def test_client_parses_openai_shaped_response(monkeypatch) -> None:
    client = _client(monkeypatch, HttpResult(200, _chat_body("答案是 42"), latency_ms=7))
    completion = client.complete([{"role": "user", "content": "问题"}], purpose="agent")
    assert completion.text == "答案是 42"
    assert completion.model == "fake-small"
    assert (completion.prompt_tokens, completion.completion_tokens) == (100, 50)
    assert completion.latency_ms == 7


def test_client_computes_cost_from_split_pricing(monkeypatch) -> None:
    client = _client(monkeypatch, HttpResult(200, _chat_body()))
    completion = client.complete([{"role": "user", "content": "x"}], purpose="agent")
    # fake-small = (0.0010 输入, 0.0030 输出) 元/1K
    assert completion.cost == pytest.approx(100 / 1000 * 0.0010 + 50 / 1000 * 0.0030)


def test_client_reads_cache_and_trace_from_headers(monkeypatch) -> None:
    result = HttpResult(200, _chat_body(), headers={"x-cache-hit": "true",
                                                    "x-trace-id": "t-9"}, latency_ms=1)
    completion = _client(monkeypatch, result).complete(
        [{"role": "user", "content": "x"}], purpose="agent")
    # 缓存标志放响应头而不是 body：body 要保持纯 OpenAI 形状
    assert completion.cache_hit is True
    assert completion.trace_id == "t-9"


def test_client_estimates_tokens_when_usage_missing(monkeypatch) -> None:
    body = _chat_body("一二三四五六七八")
    body.pop("usage")
    completion = _client(monkeypatch, HttpResult(200, body)).complete(
        [{"role": "user", "content": "一段比较长的提示词内容"}], purpose="agent")
    assert completion.prompt_tokens > 0 and completion.completion_tokens > 0


def test_client_raises_on_error_body(monkeypatch) -> None:
    body = {"error": {"message": "quota exceeded", "type": "insufficient_quota"}}
    client = _client(monkeypatch, HttpResult(429, body))
    with pytest.raises(TransportError) as info:
        client.complete([{"role": "user", "content": "x"}], purpose="agent")
    assert info.value.is_rate_limited() is True
    assert "quota exceeded" in str(info.value)


def test_client_raises_when_choices_missing(monkeypatch) -> None:
    client = _client(monkeypatch, HttpResult(200, {"id": "x", "usage": {}}))
    with pytest.raises(TransportError, match="choices"):
        client.complete([{"role": "user", "content": "x"}], purpose="agent")


def test_client_embeddings_are_sorted_by_index(monkeypatch) -> None:
    body = {"object": "list", "data": [
        {"object": "embedding", "index": 1, "embedding": [0.2, 0.2]},
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.1]},
    ]}
    client = _client(monkeypatch, HttpResult(200, body))
    vectors = client.embed(["第一条", "第二条"])
    # 上游可能乱序返回，必须按 index 排回来，否则向量与文本错位
    assert vectors == [[0.1, 0.1], [0.2, 0.2]]


def test_client_embeddings_reject_empty_response(monkeypatch) -> None:
    client = _client(monkeypatch, HttpResult(200, {"data": []}))
    with pytest.raises(TransportError, match="embeddings"):
        client.embed(["x"])


# --- 工厂与离线桩 ---

def test_build_client_selects_stub_offline() -> None:
    assert isinstance(build_client(Settings()), StubClient)
    assert isinstance(build_client(Settings.for_backend("gateway")), OpenAiCompatibleClient)


def test_build_embedder_matches_backend(tmp_path) -> None:
    assert build_embedder(Settings(), StubClient(Settings())).name == "hashing"
    gateway = Settings.for_backend("gateway")
    assert build_embedder(gateway, OpenAiCompatibleClient(gateway)).name.startswith("client:")


def test_stub_client_is_deterministic_and_free() -> None:
    settings = Settings()
    client = StubClient(settings)
    messages = [{"role": "user", "content": "问题：配额是怎么保证不超发的？"}]
    first = client.complete(messages, purpose="agent")
    second = client.complete(messages, purpose="agent")
    assert first.text == second.text
    assert first.cost == 0.0 and first.cache_hit is False


def test_stub_client_routes_rerank_and_summarize() -> None:
    from agentkit.llm import rerank_messages, summarize_messages

    client = StubClient(Settings())
    reranked = client.complete(rerank_messages("配额", ["配额扣减用 Lua", "无关内容"]),
                               purpose="rerank")
    assert "scores" in reranked.text
    summarized = client.complete(summarize_messages("用户：甲\n助手：乙"), purpose="summarize")
    assert summarized.text.endswith("。")
