"""向量化与存储测试。

关注点不是"向量准不准"（离线哈希向量本来就不追求语义质量），
而是**确定性**与**失败时的行为**：条数错位必须报错、坏行必须计数、
重复入库必须去重。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agentkit.embedder import CachedEmbedder, ClientEmbedder, HashingEmbedder
from agentkit.models import Chunk
from agentkit.store import InMemoryVectorStore, JsonlVectorStore, cosine


def _chunk(cid: str, text: str, source: str = "a.md", ordinal: int = 0) -> Chunk:
    return Chunk(id=cid, doc_id="d1", source=source, ordinal=ordinal, text=text,
                 start=0, end=len(text))


# --- embedder ---

def test_hashing_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder()
    assert embedder.embed(["语义缓存"]) == embedder.embed(["语义缓存"])


def test_hashing_embedder_is_l2_normalized() -> None:
    vector = HashingEmbedder().embed(["配额扣减与语义缓存，长度足够产生多个 n-gram"])[0]
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_hashing_embedder_handles_empty_text() -> None:
    vector = HashingEmbedder(dim=16).embed([""])[0]
    assert len(vector) == 16
    assert any(v != 0 for v in vector)  # 不能返回零向量，否则下游除零


def test_hashing_embedder_dimension_is_configurable() -> None:
    assert len(HashingEmbedder(dim=64).embed(["abc"])[0]) == 64
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)


def test_hashing_embedder_batches() -> None:
    vectors = HashingEmbedder().embed(["甲", "乙", "丙"])
    assert len(vectors) == 3


def test_client_embedder_caches_and_reports_stats() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return [[float(len(t))] for t in texts]

    client = FakeClient()
    embedder = ClientEmbedder(client, "fake")
    embedder.embed(["甲", "乙"])
    embedder.embed(["甲", "乙"])
    assert client.calls == 1  # 第二次全部命中缓存
    assert embedder.cache_stats()["cached_texts"] == 2


def test_client_embedder_rejects_count_mismatch() -> None:
    class BadClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0]]  # 无论请求几条都只回一条

    with pytest.raises(RuntimeError, match="返回 1 条"):
        ClientEmbedder(BadClient()).embed(["甲", "乙"])


def test_cached_embedder_wraps_any_embedder() -> None:
    inner = HashingEmbedder(dim=32)
    cached = CachedEmbedder(inner)
    first = cached.embed(["同一段文本"])
    second = cached.embed(["同一段文本"])
    assert first == second
    assert cached.name == "cached:hashing"


# --- store ---

def test_cosine_edge_cases() -> None:
    assert cosine([], []) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # 零向量不能除零
    assert math.isclose(cosine([1.0, 0.0], [2.0, 0.0]), 1.0)


def test_in_memory_store_upserts_by_chunk_id(store: InMemoryVectorStore) -> None:
    chunk = _chunk("c1", "第一版")
    store.add([chunk], [[1.0, 0.0]])
    store.add([chunk], [[0.0, 1.0]])
    assert len(store) == 1


def test_in_memory_store_rejects_count_mismatch(store: InMemoryVectorStore) -> None:
    with pytest.raises(ValueError, match="数量不一致"):
        store.add([_chunk("c1", "x")], [])


def test_in_memory_store_search_orders_by_similarity(store: InMemoryVectorStore) -> None:
    store.add([_chunk("c1", "a"), _chunk("c2", "b")], [[1.0, 0.0], [0.0, 1.0]])
    hits = store.search([1.0, 0.0], k=2)
    assert [c.id for c, _ in hits] == ["c1", "c2"]


def test_in_memory_store_search_rejects_bad_k(store: InMemoryVectorStore) -> None:
    with pytest.raises(ValueError):
        store.search([1.0], k=0)


def test_store_sources_are_sorted_unique(store: InMemoryVectorStore) -> None:
    store.add([_chunk("c1", "a", "b.md"), _chunk("c2", "b", "a.md"),
               _chunk("c3", "c", "a.md")], [[1.0], [1.0], [1.0]])
    assert store.sources() == ["a.md", "b.md"]


def test_jsonl_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    store = JsonlVectorStore(path)
    store.add([_chunk("c1", "内容", "a.md")], [[0.5, 0.5]])
    store.save()

    reloaded = JsonlVectorStore.load(path)
    assert len(reloaded) == 1
    assert reloaded.chunks()[0].text == "内容"


def test_jsonl_store_counts_corrupted_lines(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text(
        json.dumps({"chunk": _chunk("c1", "好的").model_dump(), "vector": [1.0]},
                   ensure_ascii=False) + "\n"
        + "{ 这不是合法 json\n",
        encoding="utf-8",
    )
    store = JsonlVectorStore.load(path)
    assert len(store) == 1
    assert store.skipped_lines == 1  # 坏行必须计数，不能静默丢


def test_jsonl_store_missing_file_is_empty(tmp_path: Path) -> None:
    store = JsonlVectorStore.load(tmp_path / "nope.jsonl")
    assert len(store) == 0


def test_drop_sources_removes_only_targets(tmp_path: Path) -> None:
    store = JsonlVectorStore(tmp_path / "i.jsonl")
    store.add([_chunk("c1", "a", "a.md"), _chunk("c2", "b", "b.md")], [[1.0], [1.0]])
    assert store.drop_sources(["a.md"]) == 1
    assert store.sources() == ["b.md"]


def test_clear_empties_store(tmp_path: Path) -> None:
    store = JsonlVectorStore(tmp_path / "i.jsonl")
    store.add([_chunk("c1", "a")], [[1.0]])
    store.clear()
    assert len(store) == 0
