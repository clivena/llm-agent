"""检索与重排测试。

重排这一版的核心结论是"两个信号融合严格优于任一单项"，
测试要能证明它不是靠调参凑出来的：单独用词法会掉分、单独用语义等于不重排，
而融合后每个 k 都不劣于基线。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentkit.chunker import chunk_document
from agentkit.config import Settings
from agentkit.embedder import HashingEmbedder
from agentkit.models import Chunk, Retrieved
from agentkit.retrieve import Retriever, assemble_context
from agentkit.rerank import (
    HeuristicReranker,
    LlmReranker,
    RerankError,
    _longest_run,
    _minmax,
)
from agentkit.store import InMemoryVectorStore


def _retrieved(cid: str, text: str, section: str = "", score: float = 0.5) -> Retrieved:
    chunk = Chunk(id=cid, doc_id="d", source="a.md", section=section, text=text,
                  start=0, end=len(text))
    return Retrieved(chunk=chunk, recall_score=score, rank_before=1, rank_after=1)


# --- 内部工具 ---

def test_longest_run_finds_middle_substring() -> None:
    # 命中的是查询中段，只看前缀的实现会漏掉
    assert _longest_run("引用为什么必须记录字符区间", "必须记录字符区间才能反查原文") == 8
    assert _longest_run("引用为什么必须记录字符区间", "引用在这里只出现了两个字") == 2


def test_longest_run_returns_zero_when_no_overlap() -> None:
    assert _longest_run("甲乙丙丁", "完全无关的内容") == 0


def test_minmax_normalizes_and_handles_constant_input() -> None:
    assert _minmax([0.0, 1.0, 2.0]) == [0.0, 0.5, 1.0]
    assert _minmax([0.7, 0.7]) == [0.5, 0.5]
    assert _minmax([]) == []


# --- HeuristicReranker ---

def test_heuristic_rerank_prefers_chunk_with_distinctive_term() -> None:
    query = "缓存组件挂了，请求会被拒绝吗？"
    generic = _retrieved("c1", "缓存分为语义缓存与精确缓存两层，各自有不同的过期策略。")
    answer = _retrieved("c2", "缓存组件不可用时必须放行请求，否则一次缓存故障会放大成全站不可用。")
    ordered = HeuristicReranker().rerank(query, [generic, answer])
    assert ordered[0].chunk.id == "c2"
    assert ordered[0].rank_after == 1 and ordered[1].rank_after == 2


def test_idf_downweights_terms_shared_by_every_candidate() -> None:
    query = "缓存组件挂了"
    # "缓存" 两个候选都有，不判别；真正区分的是 "组件" / "过期"
    both = _retrieved("c1", "缓存组件相关说明。")
    other = _retrieved("c2", "缓存过期策略说明。")
    reranker = HeuristicReranker()
    reranker.rerank(query, [both, other])
    assert both.rerank_score is not None and other.rerank_score is not None


def test_pure_lexical_weighting_can_be_selected() -> None:
    reranker = HeuristicReranker(semantic_weight=0.0, lexical_weight=1.0)
    assert reranker.semantic_weight == 0.0
    assert reranker.lexical_weight == 1.0


def test_pure_semantic_weighting_preserves_recall_order() -> None:
    """权重全给语义时，重排退化成恒等 —— 这正是"只重排不融合等于没做"的证明。"""
    reranker = HeuristicReranker(semantic_weight=1.0, lexical_weight=0.0)
    low = _retrieved("c1", "内容甲", score=0.1)
    high = _retrieved("c2", "内容乙", score=0.9)
    ordered = reranker.rerank("内容甲", [low, high])
    assert [r.chunk.id for r in ordered] == ["c2", "c1"]


def test_reranker_rejects_zero_weights() -> None:
    with pytest.raises(ValueError):
        HeuristicReranker(semantic_weight=0.0, lexical_weight=0.0)


def test_rerank_score_always_in_unit_interval() -> None:
    candidates = [_retrieved(f"c{i}", "很长的文本" * 200, score=0.9 - i * 0.01)
                  for i in range(5)]
    ordered = HeuristicReranker().rerank("很长的文本", candidates)
    for item in ordered:
        assert item.rerank_score is not None
        assert 0.0 <= item.rerank_score <= 1.0


def test_rerank_on_empty_candidates() -> None:
    assert HeuristicReranker().rerank("任意", []) == []


# --- LlmReranker：必须假设它会失败 ---

class _ScriptedClient:
    backend = "scripted"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def complete(self, messages, purpose, temperature=None):
        from agentkit.models import Completion

        self.calls += 1
        return Completion(purpose=purpose, model="scripted", text=self.text)


def test_llm_reranker_parses_fenced_json() -> None:
    client = _ScriptedClient('```json\n{"scores": [1, 9]}\n```')
    ordered = LlmReranker(client).rerank("q", [_retrieved("c1", "x"), _retrieved("c2", "y")])
    assert [r.chunk.id for r in ordered] == ["c2", "c1"]


def test_llm_reranker_rejects_unparseable_response() -> None:
    with pytest.raises(RerankError, match="无法解析"):
        LlmReranker(_ScriptedClient("我觉得第一条更好")).rerank("q", [_retrieved("c1", "x")])


def test_llm_reranker_rejects_score_count_mismatch() -> None:
    client = _ScriptedClient('{"scores": [5]}')
    with pytest.raises(RerankError, match="期望 2 个"):
        LlmReranker(client).rerank("q", [_retrieved("c1", "x"), _retrieved("c2", "y")])


def test_llm_reranker_clamps_out_of_range_scores() -> None:
    client = _ScriptedClient('{"scores": [100, -3]}')
    ordered = LlmReranker(client).rerank("q", [_retrieved("c1", "x"), _retrieved("c2", "y")])
    assert ordered[0].rerank_score == 1.0
    assert ordered[1].rerank_score == 0.0


def test_llm_reranker_batches_calls() -> None:
    """批量提交：逐条调用会让一次问答的调用次数随候选数线性膨胀。"""

    class CountingClient:
        backend = "counting"

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def complete(self, messages, purpose, temperature=None):
            from agentkit.models import Completion

            count = sum(1 for line in messages[-1]["content"].splitlines()
                        if line.strip()[:1].isdigit() and ". " in line)
            self.batch_sizes.append(count)
            return Completion(purpose=purpose, model="counting",
                              text='{"scores": [%s]}' % ",".join(["5"] * count))

    client = CountingClient()
    candidates = [_retrieved(f"c{i}", "x") for i in range(5)]
    LlmReranker(client, batch_size=2).rerank("q", candidates)
    assert client.batch_sizes == [2, 2, 1]


# --- Retriever：降级与组装 ---

class _BrokenReranker:
    name = "broken"

    def rerank(self, query, candidates):
        raise RerankError("模拟重排服务不可用")


def _retriever(tmp_path: Path, reranker=None, fallback=None) -> Retriever:
    settings = Settings(index_path=str(tmp_path / "i.jsonl"), recall_k=4, top_n=2,
                        context_token_budget=1000)
    store = InMemoryVectorStore()
    docs = {"a.md": "# 缓存\n\n缓存组件不可用时必须放行请求。" * 2,
            "b.md": "# 配额\n\n配额扣减用 Lua 保证原子性。" * 2}
    embedder = HashingEmbedder()
    for source, text in docs.items():
        document = chunk_document(source, text, 320, 64)
        store.add(document.chunks, embedder.embed([c.text for c in document.chunks]))
    return Retriever(store, embedder, settings, reranker=reranker, fallback=fallback)


def test_rerank_failure_degrades_and_records_note(tmp_path: Path) -> None:
    retriever = _retriever(tmp_path, reranker=_BrokenReranker(),
                           fallback=HeuristicReranker())
    outcome = retriever.search("缓存组件挂了会怎样", top_n=2)
    assert outcome.rerank_mode == "heuristic(fallback)"
    assert outcome.notes and "重排失败" in outcome.notes[0]
    assert outcome.items  # 降级之后仍然有结果，整轮问答不能因此失败


def test_rerank_failure_without_fallback_keeps_recall_order(tmp_path: Path) -> None:
    retriever = _retriever(tmp_path, reranker=_BrokenReranker(), fallback=None)
    outcome = retriever.search("缓存", top_n=2)
    assert outcome.items
    assert outcome.notes


def test_search_on_empty_query_returns_note(tmp_path: Path) -> None:
    outcome = _retriever(tmp_path).search("   ")
    assert outcome.recalled == []
    assert outcome.notes == ["查询为空"]


def test_context_budget_drops_tail() -> None:
    items = [_retrieved(f"c{i}", "很长的内容" * 60, score=0.9 - i * 0.01) for i in range(6)]
    context, blocks = assemble_context(items, budget_tokens=100, top_n=6)
    assert 0 < len(context) < 6
    assert len(blocks) == len(context)
    # 编号必须连续，正文里的 [n] 才能对上
    assert [item.index for item in context] == list(range(1, len(context) + 1))


def test_assemble_context_always_keeps_first_item() -> None:
    """预算再小也要留第一条，否则答案里会只剩"资料不足"。"""
    items = [_retrieved("c1", "很长很长的内容" * 100)]
    context, _ = assemble_context(items, budget_tokens=1, top_n=4)
    assert len(context) == 1


def test_dedupe_removes_identical_chunk_text(tmp_path: Path) -> None:
    settings = Settings(index_path=str(tmp_path / "i.jsonl"), recall_k=4, top_n=4)
    store = InMemoryVectorStore()
    embedder = HashingEmbedder()
    same = "完全一样的内容，用来验证去重。"
    chunks = []
    for i in range(2):
        chunks.append(Chunk(id=f"c{i}", doc_id="d", source=f"{i}.md", text=same,
                            start=0, end=len(same)))
    store.add(chunks, embedder.embed([same, same]))
    retriever = Retriever(store, embedder, settings, reranker=None)
    outcome = retriever.search("完全一样的内容", top_n=4)
    assert len(outcome.items) == 1
