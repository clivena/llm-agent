"""观测：把一次（或一批）问答的开销算清楚。

这个模块存在的唯一理由是回答一个具体问题：**Agent 一次问答到底花了多少钱、几次调用。**

普通聊天应用一次问答 = 1 次模型调用，token 数从响应里读出来就行。
Agent 不是：1 次规划 + N 轮工具调用后的再规划 + N 次重排 + 若干次 embedding 摘要，
调用次数是**乘数**放大的。只统计最后一次调用，会把成本低估到没有参考价值。

p95 用最近秩法（nearest-rank）而不是 sorted[int(n*0.95)]：
后者在 n 较小时会直接返回最大值，把小样本的 p95 报成大数。
这个错误在 llm-eval 项目里踩过一次，这里保持同一口径，避免两个项目指标不可比。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from agentkit.models import UsageStats

BACKEND_LABELS = {
    "stub": "stub（离线）",
    "direct": "direct（直连）",
    "gateway": "gateway（自研网关）",
}


def percentile(values: list[float], p: float) -> float:
    """最近秩法。n=1 时返回该值；p=95 在 n=20 时取第 19 个（升序），不是最大值。"""
    if not values:
        return 0.0
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-len(ordered) * p // 100))))
    return ordered[rank - 1]


@dataclass
class RunReport:
    label: str
    backend: str
    questions: int
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost: float
    cache_hits: int
    cache_hit_rate: float
    by_purpose: dict[str, int] = field(default_factory=dict)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    trace_ids: list[str] = field(default_factory=list)
    unresolved_citations: int = 0

    def calls_per_question(self) -> float:
        return self.calls / self.questions if self.questions else 0.0

    def tokens_per_question(self) -> float:
        return (self.prompt_tokens + self.completion_tokens) / self.questions if self.questions else 0.0

    def cost_per_question(self) -> float:
        return self.cost / self.questions if self.questions else 0.0

    def as_dict(self) -> dict:
        data = asdict(self)
        data["calls_per_question"] = round(self.calls_per_question(), 2)
        data["tokens_per_question"] = round(self.tokens_per_question(), 1)
        data["cost_per_question"] = round(self.cost_per_question(), 6)
        return data


def build_report(label: str, backend: str, usages: list[UsageStats],
                 unresolved_citations: int = 0) -> RunReport:
    total = UsageStats()
    for usage in usages:
        total.merge(usage)

    latencies = [c.latency_ms for c in total.completions if c.latency_ms]
    return RunReport(
        label=label,
        backend=backend,
        questions=len(usages),
        calls=total.calls,
        prompt_tokens=total.prompt_tokens,
        completion_tokens=total.completion_tokens,
        cost=round(total.cost, 6),
        cache_hits=total.cache_hits,
        cache_hit_rate=round(total.hit_rate(), 4),
        by_purpose=dict(sorted(total.by_purpose.items())),
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        trace_ids=total.trace_ids,
        unresolved_citations=unresolved_citations,
    )


COLUMNS = [
    ("指标", "label"),
    ("后端", "backend"),
    ("问题数", "questions"),
    ("模型调用次数", "calls"),
    ("次/问", "calls_per_question"),
    ("token/问", "tokens_per_question"),
    ("成本/问", "cost_per_question"),
    ("缓存命中", "cache_hits"),
    ("命中率", "cache_hit_rate"),
    ("p95 延迟(ms)", "p95_latency_ms"),
]


def render_comparison(reports: list[RunReport]) -> str:
    """Markdown 对比表。列固定、行固定，方便直接贴进 README 和聊天记录。"""
    header = "| " + " | ".join(name for name, _ in COLUMNS) + " |"
    divider = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, divider]
    for report in reports:
        data = report.as_dict()
        cells = []
        for _, key in COLUMNS:
            value = data.get(key, "")
            if key == "cost_per_question":
                cells.append(f"{value:.6f}")
            elif key == "cache_hit_rate":
                cells.append(f"{value:.2%}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_purpose_breakdown(report: RunReport) -> str:
    if not report.by_purpose:
        return "（没有调用记录）"
    return "，".join(f"{name}×{count}" for name, count in report.by_purpose.items())
