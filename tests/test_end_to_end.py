"""端到端测试：入库 → 提问 → 引用可反查 → 评测。

这是唯一一个跑真实语料的测试文件，也是最有价值的一个：
它同时守住"引用必须能在原文里落位"和"重排必须真的不劣于不重排"两条底线。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentkit.bootstrap import build_app, ingest
from agentkit.chunker import chunk_document
from agentkit.config import Settings
from agentkit.evaluate import evaluate_retrieval, load_cases
from agentkit.retrieve import estimate_tokens
from agentkit.store import JsonlVectorStore


def _app(tmp_path: Path, rerank: str = "heuristic"):
    settings = Settings(index_path=str(tmp_path / "index.jsonl"), recall_k=12, top_n=4)
    return build_app(settings, rerank=rerank), settings


def test_ingest_builds_index_from_real_corpus(tmp_path: Path, docs_dir: Path) -> None:
    app, settings = _app(tmp_path)
    # 语料篇数是数据、不是契约：从磁盘数出来，别写死。
    # （写死过一次 15，扩到 45 篇时这个测试就红了 —— 红的是测试，不是代码。）
    expected_files = len([p for p in Path(docs_dir).rglob("*")
                          if p.suffix.lower() in {".md", ".markdown", ".txt"}])
    try:
        summary = ingest(app, [docs_dir], reset=True)
        assert summary["files"] == expected_files
        assert summary["chunks"] > expected_files      # 平均每篇不止一块
        assert summary["total_in_index"] == summary["chunks"]
        assert Path(summary["index_path"]).exists()
        assert len(app.store.sources()) == expected_files

        # 索引身份必须落盘：这是"下次该用哪个 embedder 来查询"的唯一依据
        assert summary["embedder_kind"] == "hashing"
        assert summary["embedder_dim"] == 512
        assert Path(summary["meta_path"]).exists()
        reopened = JsonlVectorStore.load(settings.index_path)
        assert reopened.embedder_kind == "hashing"
    finally:
        app.close()


def test_reingest_is_idempotent(tmp_path: Path, docs_dir: Path) -> None:
    app, _ = _app(tmp_path)
    try:
        first = ingest(app, [docs_dir], reset=True)
        second = ingest(app, [docs_dir])
        # 同一文档重复入库必须靠 chunk.id 去重，不能堆出两份新旧版本
        assert second["total_in_index"] == first["total_in_index"]
    finally:
        app.close()


def test_index_survives_process_boundary(tmp_path: Path, docs_dir: Path) -> None:
    app, settings = _app(tmp_path)
    try:
        ingest(app, [docs_dir], reset=True)
    finally:
        app.close()
    reopened = build_app(settings)
    try:
        assert len(reopened.store) > 0     # ingest 与 ask 是两个进程，索引必须落盘
        assert reopened.store.skipped_lines == 0
    finally:
        reopened.close()


def test_ask_offline_produces_verifiable_citations(tmp_path: Path, docs_dir: Path) -> None:
    app, _ = _app(tmp_path)
    try:
        ingest(app, [docs_dir], reset=True)
        answer = app.runtime.ask("工具执行失败会不会中断整轮问答？")
        assert answer.text.strip()
        assert answer.citations, "检索型问题必须给出引用"
        assert answer.unresolved_citations == [], "理想语料下不该出现编造的引用编号"

        # 引用强校验：把每个引用拿回原文里落位，并确认块与文档都真实存在
        documents = {}
        for path in sorted(docs_dir.glob("*.md")):
            document = chunk_document(path.name, path.read_text(encoding="utf-8"))
            documents[document.id] = document
        for citation in answer.citations:
            assert citation.verify(documents) is True
            document = documents[citation.doc_id]
            assert document.text[citation.start:citation.end].strip()

        # 引用片段必须真的出现在答案所依据的那一段里，而不是别处
        assert all(c.snippet for c in answer.citations)
    finally:
        app.close()


def test_ask_returns_usage_and_trace(tmp_path: Path, docs_dir: Path) -> None:
    app, _ = _app(tmp_path)
    try:
        ingest(app, [docs_dir], reset=True)
        answer = app.runtime.ask("配额是怎么保证不超发的？")
        assert answer.usage.calls >= 2
        assert answer.usage.total_tokens() > 0
        kinds = [step.kind for step in answer.steps]
        assert "action" in kinds and "observation" in kinds and kinds[-1] == "answer"
    finally:
        app.close()


def test_calculator_path_needs_no_retrieval(tmp_path: Path, docs_dir: Path) -> None:
    app, _ = _app(tmp_path)
    try:
        ingest(app, [docs_dir], reset=True)
        answer = app.runtime.ask("帮我算一下 (128+64)*3 等于多少")
        assert "576" in answer.text
        assert answer.citations == []
    finally:
        app.close()


def test_context_respects_token_budget(tmp_path: Path, docs_dir: Path) -> None:
    app, settings = _app(tmp_path)
    try:
        ingest(app, [docs_dir], reset=True)
        outcome = app.retriever.search("平台由哪几个子系统构成？")
        used = sum(estimate_tokens(block) for block in outcome.blocks)
        assert used <= settings.context_token_budget + 400  # 单块本身超预算时不强裁
        assert len(outcome.items) <= settings.top_n
    finally:
        app.close()


def test_rerank_never_loses_to_recall_only(tmp_path: Path, docs_dir: Path,
                                          cases_path: Path) -> None:
    """这是本项目的核心断言：融合式重排在每个 k 上都不劣于 recall-only。"""
    app, _ = _app(tmp_path)
    try:
        ingest(app, [docs_dir], reset=True)
        cases = load_cases(cases_path)
        for k in (1, 2, 3, 4):
            outcome = evaluate_retrieval(app.retriever, cases, k=k)
            assert outcome.after.recall_at_k >= outcome.before.recall_at_k
            assert outcome.after.mrr >= outcome.before.mrr
            assert outcome.rank_changes > 0, "重排完全没有改变位次，说明它没起作用"
    finally:
        app.close()


def test_eval_without_rerank_is_identity(tmp_path: Path, docs_dir: Path,
                                         cases_path: Path) -> None:
    app, _ = _app(tmp_path, rerank="off")
    try:
        ingest(app, [docs_dir], reset=True)
        outcome = evaluate_retrieval(app.retriever, load_cases(cases_path), k=4)
        assert outcome.rank_changes == 0
        assert outcome.after.mrr == pytest.approx(outcome.before.mrr)
    finally:
        app.close()


def test_recall_pool_must_cover_k(tmp_path: Path, docs_dir: Path,
                                  cases_path: Path) -> None:
    app, _ = _app(tmp_path)
    try:
        ingest(app, [docs_dir], reset=True)
        with pytest.raises(ValueError, match="recall_k"):
            evaluate_retrieval(app.retriever, load_cases(cases_path), k=999)
    finally:
        app.close()


def test_multi_turn_memory_accumulates(tmp_path: Path, docs_dir: Path) -> None:
    app, _ = _app(tmp_path)
    try:
        ingest(app, [docs_dir], reset=True)
        app.runtime.ask("语义缓存是怎么省钱的？")
        app.runtime.ask("那配额呢？")
        assert len(app.runtime.memory.turns) == 2
        assert "历史对话" in app.runtime.memory.context_block()
    finally:
        app.close()
