"""检索评测。

没有评测的 RAG 只能靠"感觉答得还行"。这里把检索质量做成可比较的数字，
做法是**同一次检索内部的前后对比**：同一次召回结果，分别按召回位次和
重排位次取 top-k 计算指标，中间没有任何其他变量 —— 因此两个数字之差
可以干净地归因到重排策略上。

指标口径（按"来源文件"做 ground truth，避免块编号变动导致评测集失效）：
  recall@k —— top-k 中命中的来源数 / 期望来源数，取查询平均
  hit_rate —— 至少命中一个来源的查询占比
  MRR      —— 首个命中来源位次的倒数均值（没命中记 0）

只按来源匹配是不够的：同一个文件里可能有多个块，检索到"对的文件"
不等于检索到"对的那一段"。因此评测集可以额外给 expected_keywords，
块文本里至少要出现其中一个才算相关 —— 这是把评测从"文件级"收紧到"段落级"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from agentkit.models import EvalReport, QueryResult, Retrieved
from agentkit.retrieve import Retriever


class EvalCase(BaseModel):
    query: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)


@dataclass
class EvalOutcome:
    k: int
    before: EvalReport
    after: EvalReport
    results: list[QueryResult] = field(default_factory=list)
    rank_changes: int = 0
    notes: list[str] = field(default_factory=list)

    def delta_recall(self) -> float:
        return self.after.recall_at_k - self.before.recall_at_k

    def delta_mrr(self) -> float:
        return self.after.mrr - self.before.mrr


def load_cases(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = data["cases"] if isinstance(data, dict) else data
    return [EvalCase.model_validate(item) for item in raw_cases]


def is_relevant(item: Retrieved, case: EvalCase) -> bool:
    if case.expected_sources and item.chunk.source not in case.expected_sources:
        return False
    if case.expected_keywords:
        return any(keyword in item.chunk.text for keyword in case.expected_keywords)
    return True


def evaluate_retrieval(retriever: Retriever, cases: list[EvalCase], k: int = 4) -> EvalOutcome:
    if k <= 0:
        raise ValueError("k must be positive")
    if retriever.settings.recall_k < k:
        # 召回池小于 k 时，重排没有腾挪空间，前后对比会失去意义（两边都被截断）
        raise ValueError(f"recall_k({retriever.settings.recall_k}) 必须 >= k({k})")

    before_rows: list[tuple[float, int | None]] = []
    after_rows: list[tuple[float, int | None]] = []
    results: list[QueryResult] = []
    rank_changes = 0
    notes: list[str] = []

    for case in cases:
        outcome = retriever.search(case.query, top_n=k)
        rank_changes += outcome.rank_changes()
        notes.extend(outcome.notes)

        ordered_before = sorted(outcome.recalled, key=lambda r: r.rank_before)[:k]
        ordered_after = sorted(outcome.recalled, key=lambda r: r.rank_after)[:k]
        before_rows.append(_measure(case, ordered_before))
        recall, first_rank = _measure(case, ordered_after)
        after_rows.append((recall, first_rank))

        results.append(QueryResult(
            query=case.query,
            expected_sources=case.expected_sources,
            retrieved_sources=[r.chunk.source for r in ordered_after],
            rank_of_first_hit=first_rank,
        ))

    return EvalOutcome(
        k=k,
        before=_report("recall-only", before_rows, k),
        after=_report(f"rerank({outcome.rerank_mode})", after_rows, k),
        results=results,
        rank_changes=rank_changes,
        notes=sorted(set(notes)),
    )


def _measure(case: EvalCase, ordered: list[Retrieved]) -> tuple[float, int | None]:
    """返回 (该查询的 recall, 首个相关项位次或 None)。"""
    if not case.expected_sources:
        # 没有期望来源的用例无法判定对错，按"不参与惩罚"处理：记为 1.0 且不贡献 MRR
        return 1.0, None
    hit_sources = {item.chunk.source for item in ordered if is_relevant(item, case)}
    recall = len(hit_sources & set(case.expected_sources)) / len(case.expected_sources)
    first_rank = next((i for i, item in enumerate(ordered, start=1) if is_relevant(item, case)), None)
    return recall, first_rank


def _report(label: str, rows: list[tuple[float, int | None]], k: int) -> EvalReport:
    n = len(rows)
    if n == 0:
        return EvalReport(label=label, n=0, recall_at_k=0.0, mrr=0.0, hit_rate=0.0, k=k)
    recall = sum(r for r, _ in rows) / n
    hit = sum(1 for r, _ in rows if r > 0) / n
    rr = sum((1.0 / rank) for _, rank in rows if rank) / n
    return EvalReport(
        label=label, n=n, recall_at_k=round(recall, 4), mrr=round(rr, 4),
        hit_rate=round(hit, 4), k=k,
    )


def render_report(outcome: EvalOutcome) -> str:
    header = "| 排序策略 | recall@k | MRR | hit_rate | n |"
    divider = "|---|---|---|---|---|"
    rows = [
        f"| {outcome.before.label} | {outcome.before.recall_at_k:.4f} | "
        f"{outcome.before.mrr:.4f} | {outcome.before.hit_rate:.2%} | {outcome.before.n} |",
        f"| {outcome.after.label} | {outcome.after.recall_at_k:.4f} | "
        f"{outcome.after.mrr:.4f} | {outcome.after.hit_rate:.2%} | {outcome.after.n} |",
    ]
    lines = [header, divider, *rows, "",
             f"重排后位次发生变化的条目数：{outcome.rank_changes}（k={outcome.k}）"]
    if outcome.notes:
        lines.append("降级记录：" + "；".join(outcome.notes))
    return "\n".join(lines)
