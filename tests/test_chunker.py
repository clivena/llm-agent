"""切块测试。

这一层是整个 RAG 的天花板，所以测得比别处细：
块的字符区间必须能精确还原原文、不能横跨章节、不能以半句开头。
"""

from __future__ import annotations

import pytest

from agentkit.chunker import (
    chunk_document,
    chunk_id_for,
    chunk_text,
    doc_id_for,
    normalize_text,
)


def test_normalize_joins_hard_wrapped_lines(hard_wrapped_markdown: str) -> None:
    normalized = normalize_text(hard_wrapped_markdown)
    # 句中的硬折行必须被接回去，否则切块会从半句开始
    assert "候选数不一致、返回内容不是合法 JSON、分数超出范围" in normalized
    assert "记录降级原因，而不是继续使用未重排的结果" in normalized
    # 段落之间仍然保留空行
    assert "\n\n## 重排序" in normalized


def test_normalize_keeps_space_between_latin_lines() -> None:
    text = "the quick brown\nfox jumps over\n\n下一个段落"
    assert "the quick brown fox jumps over" in normalize_text(text)


def test_normalize_is_idempotent(hard_wrapped_markdown: str) -> None:
    once = normalize_text(hard_wrapped_markdown)
    assert normalize_text(once) == once


def test_chunk_span_exactly_matches_chunk_text(hard_wrapped_markdown: str) -> None:
    doc = chunk_document("a.md", hard_wrapped_markdown, size=120, overlap=24)
    assert doc.chunks
    for chunk in doc.chunks:
        assert doc.text[chunk.start:chunk.end] == chunk.text


def test_chunk_does_not_start_mid_sentence(hard_wrapped_markdown: str) -> None:
    doc = chunk_document("a.md", hard_wrapped_markdown, size=120, overlap=24)
    for chunk in doc.chunks[1:]:
        # 前一个字符应当是句末标点或换行，而不是逗号/顿号这种句中符号
        previous = doc.text[:chunk.start].rstrip()
        assert previous.endswith(("。", "！", "？", "；")), f"块从半句开始: {chunk.text[:20]!r}"


def test_chunk_never_crosses_section(hard_wrapped_markdown: str) -> None:
    doc = chunk_document("a.md", hard_wrapped_markdown, size=200, overlap=40)
    for chunk in doc.chunks:
        # 一个块里最多只能出现一个二级标题（自己的）
        assert chunk.text.count("## ") <= 1


def test_large_section_still_gets_split_with_overlap() -> None:
    body = "这是第%d句话，用来把段落撑长。" % 0
    text = "# 标题\n\n" + "".join(f"这是第{i}句话，用来把段落撑长。" for i in range(60))
    doc = chunk_document("big.md", text, size=100, overlap=30)
    assert len(doc.chunks) > 1
    # 相邻块必须有重叠，否则答案恰好落在接缝处就丢了
    for left, right in zip(doc.chunks, doc.chunks[1:]):
        assert right.start < left.end


def test_title_only_section_is_merged_forward() -> None:
    text = "# 大标题\n\n## 小节\n\n正文内容在这里，长度够长不会被当成纯标题。"
    doc = chunk_document("a.md", text, size=400, overlap=64)
    assert len(doc.chunks) == 1
    assert "大标题" in doc.chunks[0].text


def test_document_title_falls_back_to_first_heading() -> None:
    doc = chunk_document("a.md", "# 我是标题\n\n正文内容足够长，用来占位。" * 3)
    assert doc.title == "我是标题"


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_document("a.md", "   \n\n  ").chunks == []


def test_chunk_text_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=-1)


def test_chunk_text_terminates_on_fragmented_input() -> None:
    # 回归：曾经因为章节尾部的空白没被裁掉，循环每次只前进 1 个字符，产出上百个碎片块
    text = "# 标题\n\n" + "句子。" * 200
    spans = chunk_text(text, size=60, overlap=20)
    assert len(spans) < 60
    assert all(end > start for start, end, _ in spans)


def test_ids_are_deterministic_and_content_addressed() -> None:
    assert doc_id_for("a.md") == doc_id_for("a.md")
    assert doc_id_for("a.md") != doc_id_for("b.md")
    first = chunk_id_for("d1", 0, "同样的内容")
    second = chunk_id_for("d1", 0, "同样的内容")
    assert first == second
    assert chunk_id_for("d1", 0, "同样的内容") != chunk_id_for("d1", 1, "同样的内容")
