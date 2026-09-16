"""向量存储。

默认实现是 JSONL 落盘 + 内存检索，这是一个刻意的取舍：

  为什么需要落盘：`ingest` 和 `ask` 是两次独立的进程调用，
  向量必须持久化，否则每次提问都要重算全量 embedding —— 又慢又花钱。

  为什么不用 pgvector / FAISS：本项目的目标是"一个团队规模的知识库"
  （数百到数千块）。在这个量级上，暴力余弦的延迟在毫秒级，
  引入外部服务只会增加"演示时跑不起来"的风险。JSONL 还能直接 diff 和肉眼检查。
  超过数千块之后应当换实现 —— 换的是 VectorStore 这一个类，接口不变。

  已知边界：检索是 O(n) 暴力扫描，没有倒排/量化索引。这是刻意的，
  并且评测里的延迟数字会如实反映出来，不做隐瞒。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Protocol

from agentkit.models import Chunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        ...

    def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        ...

    def chunks(self) -> list[Chunk]:
        ...

    def sources(self) -> list[str]:
        ...

    def __len__(self) -> int:
        ...


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class InMemoryVectorStore:
    """按 chunk.id 去重（upsert）。重复 ingest 同一文档不会产生重复片段。"""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            # 段落与向量数量不一致时必须报错。静默截断会让"检索不到"变成
            # 一个查不出原因的现象 —— 这比直接失败糟糕得多。
            raise ValueError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vector
        return len(chunks)

    def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        scored = [(self._chunks[cid], cosine(vector, vec)) for cid, vec in self._vectors.items()]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def chunks(self) -> list[Chunk]:
        return sorted(self._chunks.values(), key=lambda c: (c.source, c.ordinal))

    def sources(self) -> list[str]:
        return sorted({c.source for c in self._chunks.values()})

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks


class JsonlVectorStore(InMemoryVectorStore):
    """每行 {"chunk": {...}, "vector": [...]}。逐行读取，坏行跳过并计数。"""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.skipped_lines = 0

    @classmethod
    def load(cls, path: str | Path) -> "JsonlVectorStore":
        store = cls(path)
        if not store.path.exists():
            return store
        with store.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    chunk = Chunk.model_validate(record["chunk"])
                    vector = [float(x) for x in record["vector"]]
                except Exception:
                    # 索引文件损坏时保留可用的部分，并把跳过行数记下来 ——
                    # 静默丢数据比崩溃更难排查。
                    store.skipped_lines += 1
                    continue
                store._chunks[chunk.id] = chunk
                store._vectors[chunk.id] = vector
        return store

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks():
                record = {"chunk": chunk.model_dump(), "vector": self._vectors[chunk.id]}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()

    def drop_sources(self, sources: Iterable[str]) -> int:
        """删除指定来源的片段（重新 ingest 某个文档时先清旧版本）。"""
        targets = set(sources)
        doomed = [cid for cid, chunk in self._chunks.items() if chunk.source in targets]
        for cid in doomed:
            self._chunks.pop(cid, None)
            self._vectors.pop(cid, None)
        return len(doomed)
