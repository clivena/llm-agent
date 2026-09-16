"""模型客户端。

三后端里 direct 与 gateway 共用同一个实现（OpenAiCompatibleClient），
差别只有 base_url 与鉴权头 —— 这正是"网关补齐 OpenAI 兼容性"换来的直接收益：
补齐之前网关的响应是扁平结构、流式是裸文本，客户端必须写两套解析，
所谓"可切换后端"就只剩口号。

StubClient 不是 mock。它跑在同一个 ReAct 循环、同一套提示词上，
只是决策规则写死了：解析真实提示词里的标记，给出确定的动作或答案。
因此离线跑出来的 trace、引用、检索数字，和联网跑出来的形状完全一致。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Protocol

from agentkit.config import Settings
from agentkit.embedder import HashingEmbedder
from agentkit.models import Completion
from agentkit.prompts import (
    AGENT_SYSTEM,
    CANDIDATE_MARKER,
    OBSERVATION_MARKER,
    QUESTION_MARKER,
    QUERY_MARKER,
    RERANK_SYSTEM,
    SUMMARIZE_SYSTEM,
)
from agentkit.transport import JsonTransport, TransportError, join_url

CACHE_HIT_HEADER = "x-cache-hit"
TRACE_ID_HEADER = "x-trace-id"

_ARITHMETIC = re.compile(
    r"([（(]?\s*\d+(?:\.\d+)?(?:\s*[+\-*/]\s*[（(]?\s*\d+(?:\.\d+)?[）)]?)+[）)]?)"
)
_RETRIEVAL_HINTS = ("什么", "如何", "为什么", "怎么", "介绍", "说明", "原理", "哪些",
                    "是否", "区别", "好处", "缺点", "?", "？")
# 已经算过一次之后，只有出现明确的资料型疑问词才继续检索。
# 否则"帮我算 (1+2)*3 等于多少？"这种问题会因为句末问号而白检索一次。
_STRICT_HINTS = ("什么", "如何", "为什么", "怎么", "介绍", "说明", "原理", "哪些",
                 "是否", "区别")
_CALC_RESULT = re.compile(r"工具 calculator 返回：(.*)")
_BLOCK_HEADER = re.compile(r"^\[(\d+)\]\s*来源：(.+)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；.!?;])")


class LlmClient(Protocol):
    backend: str

    def complete(self, messages: list[dict[str, str]], purpose: str,
                 temperature: float | None = None) -> Completion:
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def on_completion(self, listener: Callable[[Completion], None]) -> None:
        ...


class BaseClient:
    """统一把每次调用交给监听者（通常是 UsageStats），便于统计一次问答的总开销。"""

    backend = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._listeners: list[Callable[[Completion], None]] = []

    def on_completion(self, listener: Callable[[Completion], None]) -> None:
        self._listeners.append(listener)

    def _emit(self, completion: Completion) -> Completion:
        for listener in self._listeners:
            listener(completion)
        return completion

    def _price(self) -> tuple[float, float]:
        return self.settings.price_of(self.settings.chat_model)


class StubClient(BaseClient):
    """离线确定性客户端：零网络、零成本、结果可复现。"""

    backend = "stub"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._embedder = HashingEmbedder()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def complete(self, messages: list[dict[str, str]], purpose: str,
                 temperature: float | None = None) -> Completion:
        user_text = messages[-1]["content"] if messages else ""
        if purpose == "rerank":
            text = self._rerank(user_text)
        elif purpose == "summarize":
            text = self._summarize(user_text)
        else:
            text = self._agent(user_text)

        prompt_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        return self._emit(Completion(
            purpose=purpose,  # type: ignore[arg-type]
            model=f"stub:{self.settings.chat_model}",
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=max(1, len(text) // 4),
            cost=0.0,
            latency_ms=0,
            cache_hit=False,
            trace_id=None,
            backend=self.backend,
        ))

    # --- 决策规则（写死的"模型"） ---

    def _agent(self, user_text: str) -> str:
        question = self._question_of(user_text)
        already = self._observation_kinds(user_text)

        if "calculator" not in already and _ARITHMETIC.search(question):
            expression = _ARITHMETIC.search(question).group(1)  # type: ignore[union-attr]
            return _json({"tool": "calculator", "arguments": {"expression": expression}})

        if "search" not in already:
            hints = _STRICT_HINTS if "calculator" in already else _RETRIEVAL_HINTS
            if any(hint in question for hint in hints):
                return _json({"tool": "search_knowledge_base",
                              "arguments": {"query": question, "top_k": self.settings.top_n}})

        return self._compose(user_text)

    def _question_of(self, user_text: str) -> str:
        for line in user_text.splitlines():
            if line.startswith(QUESTION_MARKER):
                return line[len(QUESTION_MARKER):].strip()
        return ""

    def _observation_kinds(self, user_text: str) -> set[str]:
        kinds = set()
        # 只有真正带 [n] 来源行的观察才算"检索过" ——
        # 否则一次失败的检索（提示里也有"观察："字样）会被误判成已检索
        if any(_BLOCK_HEADER.match(line.strip()) for line in user_text.splitlines()):
            kinds.add("search")
        if "工具 calculator" in user_text:
            kinds.add("calculator")
        return kinds

    def _compose(self, user_text: str) -> str:
        blocks = _parse_blocks(user_text)
        if not blocks:
            # 没有检索结果时，纯计算类问题不该被回答成"资料不足" ——
            # 工具已经算出了确定结果，直接采用
            calc = _CALC_RESULT.search(user_text)
            if calc:
                return _json({"answer": f"{calc.group(1).strip()}。", "used": []})
            return _json({"answer": "资料不足，无法回答该问题。", "used": []})

        question = self._question_of(user_text)
        parts: list[str] = []
        used: list[int] = []
        for index, _source, body in blocks:
            if len(parts) >= 2:
                break
            sentence = _first_sentences(body, limit=140 if not parts else 90)
            if not sentence:
                continue
            # 首段无条件采用；后续片段要求与问题存在连续 3 个字符的重合。
            # 判据看着很松，但足以滤掉"另一个话题的段落" —— 比如问账本同步阻塞时，
            # 检索回来的追踪标识回写那段。没有这道闸门，答案会变成各抄一句的拼盘：
            # 每个引用都真实存在，但整体跑题，比直接说"答不上来"更糟。
            if parts and not _shares_phrase(sentence, question, n=3):
                continue
            parts.append(f"{sentence} [{index}]")
            used.append(index)

        if not parts:
            return _json({"answer": "资料不足，无法回答该问题。", "used": []})
        # 用换行而不是空串拼接：`"".join` 会把上一段的引用标记和下一段的
        # 首句粘成 "[1]网关处理一次对话请求的顺序是……"，读起来像是引用编号
        # 出现在句首，而实际是两段被缝在了一起。
        return _json({"answer": "\n".join(parts), "used": used})

    def _rerank(self, user_text: str) -> str:
        query = ""
        for line in user_text.splitlines():
            if line.startswith(QUERY_MARKER):
                query = line[len(QUERY_MARKER):].strip()
        candidates: list[str] = []
        collecting = False
        for line in user_text.splitlines():
            if line.startswith(CANDIDATE_MARKER):
                collecting = True
                continue
            if collecting:
                candidates.append(line.split(". ", 1)[-1])
        query_grams = _grams(query)
        scores = []
        for candidate in candidates:
            candidate_grams = _grams(candidate)
            overlap = len(query_grams & candidate_grams)
            base = overlap / max(1, len(query_grams))
            scores.append(round(min(10.0, base * 14.0), 3))
        return _json({"scores": scores})

    def _summarize(self, user_text: str) -> str:
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        kept = lines[:8]
        return "；".join(ln[:60] for ln in kept) + "。"


class OpenAiCompatibleClient(BaseClient):
    """直连任意 OpenAI 兼容端点。gateway 模式只是换了 base_url 与鉴权头。"""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.backend = settings.backend
        self._transport = JsonTransport(timeout=settings.timeout_seconds)

    def complete(self, messages: list[dict[str, str]], purpose: str,
                 temperature: float | None = None) -> Completion:
        payload: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "stream": False,
        }
        result = self._transport.post(
            join_url(self.settings.base_url, "chat/completions"),
            payload,
            self.settings.auth_headers(),
        )
        body = result.body
        if "error" in body:
            raise TransportError(f"上游返回错误: {body['error']}", status=result.status)

        choices = body.get("choices") or []
        if not choices:
            raise TransportError(f"响应里没有 choices: {body}", status=result.status)
        text = (choices[0].get("message") or {}).get("content") or ""

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        # usage 缺失时退回粗估：宁可估算，也不要让成本统计变成 0
        if prompt_tokens == 0 and completion_tokens == 0:
            prompt_tokens = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
            completion_tokens = max(1, len(text) // 4)

        input_price, output_price = self._price()
        cost = prompt_tokens / 1000.0 * input_price + completion_tokens / 1000.0 * output_price

        return self._emit(Completion(
            purpose=purpose,  # type: ignore[arg-type]
            model=str(body.get("model") or self.settings.chat_model),
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency_ms=result.latency_ms,
            # 缓存命中由网关在响应头里告知（body 保持纯 OpenAI 形状，所以不放 body 里）
            cache_hit=result.headers.get(CACHE_HIT_HEADER, "false").lower() == "true",
            trace_id=result.headers.get(TRACE_ID_HEADER),
            backend=self.backend,
        ))

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._transport.post(
            join_url(self.settings.base_url, "embeddings"),
            {"model": self.settings.embedding_model, "input": texts},
            self.settings.auth_headers(),
        )
        data = result.body.get("data") or []
        if not data:
            raise TransportError(f"embeddings 响应为空: {result.body}", status=result.status)
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [[float(x) for x in item["embedding"]] for item in ordered]


def build_client(settings: Settings) -> LlmClient:
    if settings.backend == "stub":
        return StubClient(settings)
    return OpenAiCompatibleClient(settings)


def build_embedder(settings: Settings, client: LlmClient):
    """stub 用本地哈希向量；其余后端走远程 embedding（带缓存）。"""
    from agentkit.embedder import ClientEmbedder

    if settings.backend == "stub":
        return HashingEmbedder()
    return ClientEmbedder(client, settings.embedding_model)


def agent_messages(question: str, observations: list[str], tool_specs: str,
                   memory_block: str = "") -> list[dict[str, str]]:
    from agentkit.prompts import render_agent_user

    messages = [{"role": "system", "content": AGENT_SYSTEM.format(tools=tool_specs)}]
    if memory_block:
        messages.append({"role": "system", "content": memory_block})
    messages.append({"role": "user", "content": render_agent_user(question, observations)})
    return messages


def rerank_messages(query: str, candidates: list[str]) -> list[dict[str, str]]:
    from agentkit.prompts import render_candidates

    return [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": render_candidates(query, candidates)},
    ]


def summarize_messages(transcript: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": transcript},
    ]


# --- 内部工具 ---

def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _parse_blocks(user_text: str) -> list[tuple[int, str, str]]:
    """从观察文本里抽出 [n] 来源：... 块。返回 [(index, source, body)]。"""
    blocks: list[tuple[int, str, str]] = []
    current: tuple[int, str, list[str]] | None = None
    for line in user_text.splitlines():
        match = _BLOCK_HEADER.match(line.strip())
        if match:
            if current:
                blocks.append((current[0], current[1], "\n".join(current[2]).strip()))
            current = (int(match.group(1)), match.group(2).strip(), [])
            continue
        if current is not None:
            if line.startswith(OBSERVATION_MARKER):
                continue
            stripped = line.strip()
            if stripped.startswith("["):
                continue
            if stripped:
                current[2].append(stripped)
    if current:
        blocks.append((current[0], current[1], "\n".join(current[2]).strip()))
    return blocks


def _first_sentences(text: str, limit: int = 110) -> str:
    """从片段里取整句。

    关键是**只取整句**：早先的实现按字符数截断，会把片段切成半个句子，
    再和下一段的半句拼起来，于是答案里出现 "...这和真实" + "用量”三步放进..." 这种
    接缝断裂。对用户来说这比"短一点"糟糕得多 —— 看起来像模型在胡说。

    标题行（# 开头）在这里丢掉：它是导航信息，不是答案内容。
    """
    cleaned = " ".join(
        line.strip() for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    if not cleaned:
        return ""

    sentences = [s for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]
    out = ""
    for sentence in sentences:
        if out and len(out) + len(sentence) > limit:
            break
        out += sentence
        if len(out) >= limit:
            break
    if not out:
        # 整段没有句末标点：按上限截断，并退到最后一个分隔符，避免切在词中间
        out = cleaned[:limit]
        for sep in ("，", "、", ",", " "):
            cut = out.rfind(sep)
            if cut > limit // 2:
                return out[:cut + 1]
        return out
    # 长句兜底：单句就超过两倍上限时，退到最后一个句末标点，宁可短也不切碎
    if len(out) > limit * 2:
        cut = max(out.rfind(ch) for ch in "。！？；")
        if cut > 0:
            out = out[:cut + 1]
    return out


def _grams(text: str, n: int = 2) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[i:i + n] for i in range(max(0, len(compact) - n + 1))}


def _shares_phrase(text: str, question: str, n: int = 3) -> bool:
    """文本与问题是否存在连续 n 个字符的重合（忽略空白）。"""
    compact_question = re.sub(r"\s+", "", question).lower()
    compact_text = re.sub(r"\s+", "", text).lower()
    if len(compact_question) < n:
        return True
    return any(
        compact_question[i:i + n] in compact_text
        for i in range(len(compact_question) - n + 1)
    )
