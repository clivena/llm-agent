"""观测与引用模型的测试。

观测这块只回答一个问题：一次问答到底调用了几次、花了多少钱。
Agent 的调用次数是乘数放大的，所以统计口径错一点，结论就完全不可用。
"""

from __future__ import annotations

import pytest

from agentkit.models import Answer, Citation, Completion, UsageStats, markers_in, snippet_of
from agentkit.observability import (
    build_report,
    percentile,
    render_comparison,
    render_purpose_breakdown,
)


# --- 最近秩法 ---

def test_percentile_uses_nearest_rank() -> None:
    values = [float(i) for i in range(1, 21)]  # 1..20
    # 20 个样本的 p95 取第 19 个，而不是最大值 —— 小样本下"取最大值"会把 p95 报虚高
    assert percentile(values, 95) == 19.0
    assert percentile(values, 50) == 10.0
    assert percentile(values, 100) == 20.0


def test_percentile_edges() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        percentile([1.0], 0)


# --- 聚合 ---

def _completion(purpose: str, prompt: int, output: int, cost: float,
                latency: int = 10, cache: bool = False) -> Completion:
    return Completion(purpose=purpose, model="m", text="t", prompt_tokens=prompt,
                      completion_tokens=output, cost=cost, latency_ms=latency,
                      cache_hit=cache)


def test_usage_stats_aggregate_by_purpose() -> None:
    stats = UsageStats()
    stats.add(_completion("agent", 100, 20, 0.001))
    stats.add(_completion("rerank", 50, 5, 0.0001))
    stats.add(_completion("agent", 80, 10, 0.0008, cache=True))
    assert stats.calls == 3
    assert stats.by_purpose == {"agent": 2, "rerank": 1}
    assert stats.total_tokens() == 265
    assert stats.hit_rate() == pytest.approx(1 / 3)


def test_usage_stats_merge_is_additive() -> None:
    left, right = UsageStats(), UsageStats()
    left.add(_completion("agent", 10, 1, 0.01, cache=True))
    right.add(_completion("agent", 20, 2, 0.02))
    left.merge(right)
    assert left.calls == 2
    assert left.total_tokens() == 33
    assert left.cost == pytest.approx(0.03)
    assert left.hit_rate() == 0.5


def test_build_report_computes_per_question_metrics() -> None:
    first, second = UsageStats(), UsageStats()
    first.add(_completion("agent", 100, 20, 0.002, latency=100))
    first.add(_completion("rerank", 50, 5, 0.001, latency=200, cache=True))
    second.add(_completion("agent", 100, 20, 0.002, latency=300))
    report = build_report("stub（离线）", "stub", [first, second], unresolved_citations=2)
    assert report.questions == 2
    assert report.calls == 3
    assert report.calls_per_question() == pytest.approx(1.5)
    assert report.tokens_per_question() == pytest.approx(147.5)
    assert report.cost_per_question() == pytest.approx(0.0025)
    assert report.cache_hits == 1
    assert report.unresolved_citations == 2
    assert report.by_purpose == {"agent": 2, "rerank": 1}


def test_build_report_handles_no_data() -> None:
    report = build_report("空", "stub", [])
    assert report.questions == 0
    assert report.calls_per_question() == 0.0
    assert report.p95_latency_ms == 0.0


def test_render_comparison_is_markdown_table() -> None:
    stats = UsageStats()
    stats.add(_completion("agent", 10, 5, 0.001))
    table = render_comparison([build_report("stub（离线）", "stub", [stats])])
    lines = table.splitlines()
    assert lines[0].startswith("| 指标 |")
    assert len(lines) == 3  # 表头 + 分隔 + 1 行
    assert "stub（离线）" in lines[2]
    assert lines[2].count("|") == lines[0].count("|")


def test_render_purpose_breakdown() -> None:
    stats = UsageStats()
    stats.add(_completion("agent", 1, 1, 0.0))
    stats.add(_completion("rerank", 1, 1, 0.0))
    assert render_purpose_breakdown(build_report("x", "stub", [stats])) == "agent×1，rerank×1"
    assert render_purpose_breakdown(build_report("x", "stub", [])) == "（没有调用记录）"


# --- 引用与片段 ---

def test_markers_in_dedupes_and_keeps_order() -> None:
    assert markers_in("见 [3] 和 [1]，还有 [3]。") == [3, 1]
    assert markers_in("没有引用") == []


def test_snippet_of_breaks_at_sentence_boundary() -> None:
    text = "第一句话足够长，用来把长度撑过阈值。" * 10
    snippet = snippet_of(text, limit=30)
    assert len(snippet) <= 31
    assert snippet.endswith("。")


def test_snippet_of_marks_hard_cut_when_no_sentence_end() -> None:
    assert snippet_of("没有任何标点的长文本" * 20, limit=20).endswith("…")


def test_citation_verify_checks_chunk_membership() -> None:
    from agentkit.chunker import chunk_document

    document = chunk_document("a.md", "# 标题\n\n正文内容足够长，用来产生一个片段。" * 3)
    docs = {document.id: document}
    chunk = document.chunks[0]
    ok = Citation(index=1, chunk_id=chunk.id, doc_id=document.id, source="a.md",
                  start=chunk.start, end=chunk.end, snippet=chunk.text[:10])
    assert ok.verify(docs) is True

    wrong_chunk = ok.model_copy(update={"chunk_id": "不存在的块"})
    assert wrong_chunk.verify(docs) is False
    out_of_range = ok.model_copy(update={"end": len(document.text) + 10})
    assert out_of_range.verify(docs) is False
    assert ok.model_copy(update={"doc_id": "别的文档"}).verify(docs) is False


def test_answer_used_indices_follows_text_not_field() -> None:
    answer = Answer(question="q", text="甲 [2]，乙 [1]。", citations=[])
    assert answer.used_indices() == [2, 1]
    assert answer.citation_for(1) is None
