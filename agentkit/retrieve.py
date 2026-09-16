"""检索编排：召回 → 重排 → 去重 → 上下文组装。

每一步都可以独立解释，也是面试里最容易被追问的三处：
  召回为什么取 k=12 而不是 4？  —— 重排需要候选池，池子太小重排没有腾挪空间；
                                   池子太大则重排成本线性上升。12 是这两者的折中。
  重排为什么失败要降级而不是报错？—— 检索降级只影响答案质量，报错会让整个问答失败，
                                   可用性优先。但降级必须**记录在案**（notes），
                                   否则就成了"悄悄变差"。
  上下文为什么要有预算？         —— Agent 一次问答会把上下文塞进多轮提示，
                                   不加预算会在多轮里重复计费，成本指数上升。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentkit.config import Settings
from agentkit.embedder import Embedder
from agentkit.models import Retrieved
from agentkit.rerank import RerankError, Reranker
from agentkit.store import VectorStore


@dataclass
class ContextItem:
    """进入上下文的一条，index 就是正文里的 [n]。"""

    index: int
    retrieved: Retrieved

    @property
    def chunk(self):
        return self.retrieved.chunk


@dataclass
class RetrievalOutcome:
    query: str
    recalled: list[Retrieved]
    items: list[ContextItem] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rerank_mode: str = "off"

    def sources(self) -> list[str]:
        return [item.chunk.source for item in self.items]

    def rank_changes(self) -> int:
        """重排后位次发生变化的条数。这个数字是"重排到底有没有用"的直接证据。"""
        return sum(1 for r in self.recalled if r.rank_before != r.rank_after)

    def dropped_for_budget(self) -> int:
        return max(0, len(self.recalled) - len(self.items))


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings,
                 reranker: Reranker | None = None, fallback: Reranker | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker
        self.fallback = fallback

    def search(self, query: str, top_n: int | None = None) -> RetrievalOutcome:
        if not query.strip():
            return RetrievalOutcome(query=query, recalled=[], notes=["查询为空"])

        vector = self.embedder.embed([query])[0]
        hits = self.store.search(vector, self.settings.recall_k)
        recalled: list[Retrieved] = []
        for position, (chunk, score) in enumerate(hits, start=1):
            recalled.append(Retrieved(chunk=chunk, recall_score=score,
                                      rank_before=position, rank_after=position))

        outcome = RetrievalOutcome(query=query, recalled=recalled,
                                   rerank_mode=self.reranker.name if self.reranker else "off")
        ordered = self._rerank(query, recalled, outcome)
        ordered = _dedupe(ordered)
        outcome.items, outcome.blocks = assemble_context(
            ordered, self.settings.context_token_budget, top_n or self.settings.top_n)
        return outcome

    def _rerank(self, query: str, recalled: list[Retrieved],
                outcome: RetrievalOutcome) -> list[Retrieved]:
        if self.reranker is None:
            return recalled
        try:
            return self.reranker.rerank(query, recalled)
        except RerankError as e:
            # 降级但留痕：静默降级等于把"质量下降"藏起来
            outcome.notes.append(f"{self.reranker.name} 重排失败，已回退启发式：{e}")
            outcome.rerank_mode = "heuristic(fallback)"
            if self.fallback is None:
                return recalled
            return self.fallback.rerank(query, recalled)


def _dedupe(items: list[Retrieved]) -> list[Retrieved]:
    """同一段文本被多个块覆盖时只保留最高分那一条，避免上下文被复制粘贴挤满。"""
    seen: set[str] = set()
    out: list[Retrieved] = []
    for item in items:
        fingerprint = " ".join(item.chunk.text.split())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(item)
    return out


def assemble_context(items: list[Retrieved], budget_tokens: int,
                     top_n: int) -> tuple[list[ContextItem], list[str]]:
    """按预算组装上下文。超出预算的尾部条目直接丢弃（而不是截断到半个句子）。"""
    context: list[ContextItem] = []
    blocks: list[str] = []
    used_tokens = 0
    for item in items[:top_n]:
        block = render_block(len(context) + 1, item)
        tokens = estimate_tokens(block)
        if context and used_tokens + tokens > budget_tokens:
            break
        context.append(ContextItem(index=len(context) + 1, retrieved=item))
        blocks.append(block)
        used_tokens += tokens
    return context, blocks


def render_block(index: int, item: Retrieved) -> str:
    chunk = item.chunk
    location = chunk.source
    if chunk.section:
        location = f"{location} §{chunk.section}"
    return f"[{index}] 来源：{location}\n    {chunk.text}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
