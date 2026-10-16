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
    """余弦相似度。**维度不一致直接报错，不做截断。**

    旧实现是 `n = min(len(a), len(b))` —— 拿 512 维的查询向量去比 1536 维的索引向量时，
    它会安安静静地只比前 512 个分量，返回一个**看起来有分、其实毫无意义**的数。
    跨向量空间比余弦是这个项目里最危险的一类 bug：不报错、不崩溃，
    只是检索结果整体错乱，然后所有下游指标一起失真。
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(
            f"向量维度不一致：查询 {len(a)} 维 vs 索引 {len(b)} 维。"
            "这几乎总是意味着建索引和查询用了不同的 embedder。"
        )
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
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
        # 先集中校验一次维度，给出可操作的信息；否则会在 cosine 里对每一行各报一次。
        dims = {len(v) for v in self._vectors.values()}
        if dims and len(vector) not in dims:
            raise ValueError(
                f"查询向量 {len(vector)} 维，索引是 {sorted(dims)} 维 —— "
                "建索引和查询用的不是同一个 embedder。"
                "换 embedder 必须整个索引重建（agentkit ingest --reset），"
                "不能跨向量空间比余弦。"
            )
        scored = [(self._chunks[cid], cosine(vector, vec)) for cid, vec in self._vectors.items()]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def vector_dim(self) -> int | None:
        """索引里向量的维度（空索引返回 None）。用于和当前 embedder 对账。"""
        for vec in self._vectors.values():
            return len(vec)
        return None

    def chunks(self) -> list[Chunk]:
        return sorted(self._chunks.values(), key=lambda c: (c.source, c.ordinal))

    def sources(self) -> list[str]:
        return sorted({c.source for c in self._chunks.values()})

    def __len__(self) -> int:
        return len(self._chunks)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks


class JsonlVectorStore(InMemoryVectorStore):
    """每行 {"chunk": {...}, "vector": [...]}。逐行读取，坏行跳过并计数。

    索引身份（谁建的、什么 embedder、多少维）写在旁边一个 `<stem>.meta.json` 里，
    **不混进 JSONL**：那个文件的格式是纯数据行，插一行元数据会让所有
    逐行读取的地方（含外部脚本）都得先判断"这行是不是数据"。
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.skipped_lines = 0
        # 索引的"身份证"，由 ingest 写入、由 build_app 对账
        self.embedder_name: str | None = None
        self.embedder_kind: str | None = None

    @property
    def meta_path(self) -> Path:
        return self.path.with_name(self.path.stem + ".meta.json")

    @classmethod
    def load(cls, path: str | Path) -> "JsonlVectorStore":
        store = cls(path)
        meta = read_index_meta(store.path)
        if meta:
            store.embedder_name = meta.get("embedder")
            store.embedder_kind = meta.get("kind")
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
        if self.embedder_name:
            write_index_meta(self.path, name=self.embedder_name,
                             kind=self.embedder_kind or "", dim=self.vector_dim(),
                             chunks=len(self))

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


def read_index_meta(index_path: str | Path) -> dict | None:
    """读索引的元数据；不存在或损坏都返回 None（"不知道"，而不是"猜一个"）。"""
    path = Path(index_path).with_name(Path(index_path).stem + ".meta.json")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_index_meta(index_path: str | Path, *, name: str, kind: str,
                     dim: int | None, chunks: int) -> Path:
    """写索引身份。这笔信息是"能不能安全复用这个索引"的唯一依据，必须落盘。"""
    from datetime import datetime

    path = Path(index_path).with_name(Path(index_path).stem + ".meta.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedder": name,
        "kind": kind,
        "dim": dim,
        "chunks": chunks,
        "builtAt": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
