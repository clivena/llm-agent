"""Embedding 提供方。

两种实现，对应"能不能联网"这一个现实约束：

  HashingEmbedder —— 纯本地的确定性向量化（字符 n-gram 哈希 + L2 归一）。
      不追求语义质量，只保证：同样的输入永远得到同样的向量、断网可用、
      零成本。它的存在让整条链路（入库 / 召回 / 评测）在没有 API Key 的
      机器上也能真实跑起来 —— 面试现场演示不依赖网络，这一点比向量质量重要。

  ClientEmbedder —— 走 HTTP 的远程 embedding（直连厂商或走自研网关），
      带进程内缓存：入库和评测会对同一批文本反复请求，缓存是实打实的省钱。

为什么把 HashingEmbedder 做成一等公民而不是"测试桩"：
  RAG 的效果天花板由检索质量决定，而检索质量的验证需要一个**稳定的基线**。
  离线向量就是那个基线 —— 换了远程 embedding 之后，同一套评测集上的
  Recall@k 提升多少，是可以归因的，而不是"感觉变好了"。

⚠️ **embedder 与 backend 是两件独立的事，别再绑在一起。**
  曾经 `build_embedder` 写的是 `backend == "stub" ? hashing : client`，
  但这个耦合是错的：索引里的向量和查询向量**必须来自同一个 embedder**，
  否则余弦是在两个不同的向量空间之间比，得到的分数没有意义（而且不报错）。
  现在 embedder 由独立配置项 `Settings.embedder` 决定，默认 `auto`
  —— 沿用索引自己记录的那个 embedder；要换远程 embedding 就必须 `ingest --reset` 整库重建。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from agentkit.transport import TransportError

_WHITESPACE = re.compile(r"\s+")


class Embedder(Protocol):
    name: str
    #: embedder 的"种类"。用于和索引对账：只有同一种类才共享向量空间。
    #: name 是给人看的（含模型名），kind 是给程序比的。
    kind: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashingEmbedder:
    """字符 n-gram 哈希向量。中文按字切，2-gram 已经能较好地区分主题。"""

    name = "hashing"
    kind = "hashing"

    def __init__(self, dim: int = 512, weights: dict[int, float] | None = None) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.weights = weights or {1: 1.0, 2: 1.6, 3: 0.7}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        normalized = _WHITESPACE.sub("", text.lower())
        for n, weight in self.weights.items():
            if n > len(normalized):
                continue
            for i in range(len(normalized) - n + 1):
                gram = normalized[i:i + n]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                vec[int.from_bytes(digest, "big") % self.dim] += weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            # 空串或纯符号：返回一个固定方向的单位向量，避免下游除零
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]


class ClientEmbedder:
    """远程 embedding，带进程内缓存（同一段文本只付一次钱）。

    维度要等第一次真正拿到向量才知道（不同厂商/模型不一样），所以 `dim` 初始为 None、
    由第一次调用回填。它存在 session 里的唯一用途是和索引的维度对账。
    """

    kind = "client"

    def __init__(self, client, model: str = "", cache_size: int = 4096) -> None:
        self._client = client
        self._model = model
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size
        self.name = f"client:{model or 'default'}"
        self.dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            try:
                vectors = self._client.embed(missing)
            except TransportError as e:
                raise RuntimeError(f"embedding 请求失败（{self.name}）: {e}") from e
            if len(vectors) != len(missing):
                # 上游返回条数对不上时必须报错，不能静默错位：
                # 向量与文本错位会让检索结果整体错乱，而且很难察觉。
                raise RuntimeError(
                    f"embedding 返回 {len(vectors)} 条，请求了 {len(missing)} 条（{self.name}）"
                )
            if vectors and self.dim is None:
                self.dim = len(vectors[0])
            for text, vector in zip(missing, vectors):
                if len(self._cache) >= self._cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[text] = vector
        return [self._cache[t] for t in texts]

    def cache_stats(self) -> dict[str, int]:
        return {"cached_texts": len(self._cache)}


class CachedEmbedder:
    """给任意 Embedder 套一层缓存（HashingEmbedder 用不上，远程 embedder 用得上）。"""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self.name = f"cached:{inner.name}"
        self.kind = inner.kind

    @property
    def dim(self) -> int | None:
        return getattr(self._inner, "dim", None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self._inner.embed(missing)):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]
