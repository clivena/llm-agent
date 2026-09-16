"""数据模型。

两个刻意的设计：
  ① 引用（Citation）带字符区间 span，而不只是"来自某文档"。
     只记文档名的引用是无法验证的 —— 表面像溯源，实际没法反查，
     这是 RAG 项目最常见的自欺。带 span 才能做「引用必须能在原文里命中」的断言。
  ② UsageStats 记的是**一次问答的全部开销**，不是单次调用。
     Agent 和聊天应用的本质差别就在这里：一次问答会触发 1 + N 轮编排、
     N 次重排、若干次 embedding，成本是叠加的。只统计最后一次调用等于没统计。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

Backend = Literal["stub", "direct", "gateway"]
StepKind = Literal["thought", "action", "observation", "answer"]
Purpose = Literal["agent", "rerank", "summarize", "embed", "eval"]

_MARKER = re.compile(r"\[(\d+)\]")


def markers_in(text: str) -> list[int]:
    """抽出正文里的 [n] 引用标记（去重、保持出现顺序）。"""
    seen: list[int] = []
    for match in _MARKER.finditer(text):
        value = int(match.group(1))
        if value not in seen:
            seen.append(value)
    return seen


_SENTENCE_END_CHARS = "。！？；.!?;"


def snippet_of(text: str, limit: int = 160) -> str:
    """截取用于展示的片段。优先在句末截断，实在没有句末标点才硬截 ——
    引用片段被切在词中间会让人以为原文就是这样，比截短更难排查。"""
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(ch) for ch in _SENTENCE_END_CHARS)
    if cut >= limit // 3:
        return head[:cut + 1]
    return head + "…"


class Chunk(BaseModel):
    """文档切片。start/end 是相对于原文的字符区间，引用高亮与反查都靠它。"""

    id: str
    doc_id: str
    source: str
    title: str = ""
    section: str = ""
    ordinal: int = 0
    text: str
    start: int = 0
    end: int = 0


class Document(BaseModel):
    id: str
    source: str
    title: str = ""
    text: str
    chunks: list[Chunk] = Field(default_factory=list)


class Retrieved(BaseModel):
    """一次召回结果。同时保留召回位次与重排后位次，才能解释重排到底改了什么。"""

    chunk: Chunk
    recall_score: float = 0.0
    rerank_score: float | None = None
    rank_before: int = 0
    rank_after: int = 0

    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.recall_score


class Citation(BaseModel):
    """正文里的 [index] 对应到这里。span 用来反查原文，snippet 用来给人看。"""

    index: int
    chunk_id: str
    doc_id: str
    source: str
    start: int
    end: int
    snippet: str

    def verify(self, docs: dict[str, Document]) -> bool:
        """引用的强校验：chunk 必须真实存在，且 span 落在原文范围之内。"""
        doc = docs.get(self.doc_id)
        if doc is None:
            return False
        chunk = next((c for c in doc.chunks if c.id == self.chunk_id), None)
        if chunk is None:
            return False
        return 0 <= self.start < self.end <= len(doc.text)


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    output: str = ""
    error: str = ""


class AgentStep(BaseModel):
    """ReAct 的一步。留 full trace 是为了能回答"你为什么给出这个答案"。"""

    kind: StepKind
    text: str = ""
    tool: ToolCall | None = None
    step_index: int = 0


class Completion(BaseModel):
    """一次模型调用的完整记录（含成本与缓存命中）。"""

    purpose: Purpose
    model: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    cache_hit: bool = False
    trace_id: str | None = None
    backend: str = "stub"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class UsageStats(BaseModel):
    """一次问答的累计开销。按 purpose 分组，才能看出钱花在哪一步。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    cache_hits: int = 0
    by_purpose: dict[str, int] = Field(default_factory=dict)
    trace_ids: list[str] = Field(default_factory=list)
    completions: list[Completion] = Field(default_factory=list)

    def hit_rate(self) -> float:
        return self.cache_hits / self.calls if self.calls else 0.0

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, completion: Completion) -> None:
        self.calls += 1
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        self.cost += completion.cost
        if completion.cache_hit:
            self.cache_hits += 1
        self.by_purpose[completion.purpose] = self.by_purpose.get(completion.purpose, 0) + 1
        if completion.trace_id and completion.trace_id not in self.trace_ids:
            self.trace_ids.append(completion.trace_id)
        self.completions.append(completion)

    def merge(self, other: "UsageStats") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost += other.cost
        self.cache_hits += other.cache_hits
        for k, v in other.by_purpose.items():
            self.by_purpose[k] = self.by_purpose.get(k, 0) + v
        for t in other.trace_ids:
            if t not in self.trace_ids:
                self.trace_ids.append(t)
        self.completions.extend(other.completions)


class Answer(BaseModel):
    """最终回答。citations 与正文里的 [n] 一一对应。

    unresolved_citations 记录的是一类**必须暴露而不是吞掉**的失败：
    模型在正文里写了 [7]，但检索结果里根本没有第 7 条。这说明它在编引用 ——
    比编事实更隐蔽，因为看起来"有出处"。留一个字段如实记下来，
    再配一条测试断言理想语料下它为空。
    """

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)
    backend: str = "stub"
    unresolved_citations: list[int] = Field(default_factory=list)

    def citation_for(self, index: int) -> Citation | None:
        return next((c for c in self.citations if c.index == index), None)

    def used_indices(self) -> list[int]:
        """从正文里抽出实际用到的 [n]（去重、保序）。"""
        return markers_in(self.text)


class QueryResult(BaseModel):
    """检索评测的一行。"""

    query: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    rank_of_first_hit: int | None = None


class EvalReport(BaseModel):
    label: str
    n: int
    recall_at_k: float
    mrr: float
    hit_rate: float
    k: int
