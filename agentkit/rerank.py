"""重排序（rerank）。

为什么必须有这一步：向量召回按"整体语义接近"排序，而用户要的是
"能回答这个问题的片段"。两者经常不一致 —— 一个泛泛而谈的概述段
可能比真正包含答案的窄段余弦分更高。重排就是把这个偏差纠回来。

两档实现，成本差三个数量级：
  HeuristicReranker —— 把「召回余弦」与「词法匹配分」加权融合。
      零成本、零延迟，作为默认档，也是 LLM 档失败时的兜底。
  LlmReranker —— 让模型逐条打分。质量更好但要花钱、有延迟，
      而且**必须假设它会失败**：返回条数对不上、JSON 不合法、
      分数越界都要能识别并降级，不能把异常吞掉当成"排好了"。

这里有两个踩过坑才明白的点：

① **重排不能覆盖召回，只能与召回融合。** 第一版直接用词法分重排，
   等于把已经算好的语义分扔掉 —— 结果是修好一个查询就弄坏另一个：
   recall@2 涨了，MRR 反而跌。语义分管"说得像"，词法分管"真的写着"，
   各自的强项恰好是对方的盲区，融合起来才互为兜底。

② **词法分必须按候选池 IDF 加权。** 朴素做法是数查询与片段共有几个 2-gram。
   中文里 2-gram 噪声极大：问"缓存组件挂了，请求会被拒绝吗"，
   "缓存"这个词几乎所有候选都有，谁命中它谁得分，于是"语义缓存与路由"
   这种泛泛的概述段会压过真正写着"必须放行"的那一段。
   按文档频率把"人手一个"的 gram 权重压下去，判别力立刻回来 ——
   这是纯启发式最接近 IDF 的做法：不需要全语料，只看这一轮候选池。
"""

from __future__ import annotations

import json
import math
import re
from typing import Protocol

from agentkit.llm import LlmClient, rerank_messages
from agentkit.models import Retrieved

_GRAMS = re.compile(r"\s+")
# 参与打分的 gram 长度。2-gram 管召回，3/4-gram 管判别，后者权重更高。
_GRAM_WEIGHTS = {2: 1.0, 3: 2.0, 4: 3.0}
_MAX_RUN = 20


class RerankError(RuntimeError):
    """重排失败。调用方应当降级到启发式，而不是继续用未重排的结果装作已完成。"""


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        ...


def _compact(text: str) -> str:
    return _GRAMS.sub("", text.lower())


def _ngrams(compact: str, n: int) -> set[str]:
    return {compact[i:i + n] for i in range(max(0, len(compact) - n + 1))}


def _query_terms(query: str) -> dict[int, set[str]]:
    compact = _compact(query)
    return {n: _ngrams(compact, n) for n in _GRAM_WEIGHTS}


def _longest_run(query: str, haystack: str, cap: int = _MAX_RUN) -> int:
    """查询里出现在片段中的最长连续子串长度。

    取"任意子串"而不是"查询前缀"：问"引用为什么必须记录字符区间"，
    答案段可能写"必须记录字符区间才能反查" —— 命中的是查询中段，
    只看前缀会漏掉。
    """
    limit = min(len(query), cap)
    for length in range(limit, 1, -1):
        for i in range(len(query) - length + 1):
            if query[i:i + length] in haystack:
                return length
    return 0


def _minmax(values: list[float]) -> list[float]:
    """把召回分数归一化到 [0,1]。候选池内相对比较，绝对值没有意义。"""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [0.5] * len(values)
    return [(v - low) / (high - low) for v in values]


class HeuristicReranker:
    """语义分与词法分融合：召回余弦 + IDF 加权词项覆盖 + 最长连续命中。

    权重不是拍出来的：`scripts/tune_rerank.py` 在评测集上扫过 0.0~1.0。
    结论很干净 —— 只用词法（sem=0）每个 k 都变差；只用语义（sem=1）等于不重排；
    0.30/0.70 在 k=1/2/3/4 上严格不劣于 recall-only，且 MRR 全面提升。
    两个信号各自都有翻车的场景，融合才互相当对方的兜底。
    """

    name = "heuristic"

    def __init__(self, semantic_weight: float = 0.30, lexical_weight: float = 0.70,
                 length_penalty_after: int = 700) -> None:
        total = semantic_weight + lexical_weight
        if total <= 0:
            raise ValueError("两项权重不能同时为 0")
        self.semantic_weight = semantic_weight / total
        self.lexical_weight = lexical_weight / total
        self.length_penalty_after = length_penalty_after

    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        if not candidates:
            return []
        terms = _query_terms(query)
        compacts = [_compact(c.chunk.text) for c in candidates]
        idf = self._idf(terms, compacts)
        denominator = sum(
            _GRAM_WEIGHTS[n] * idf[gram]
            for n, gram_set in terms.items() for gram in gram_set
        ) or 1.0

        query_compact = _compact(query)
        semantic = _minmax([c.recall_score for c in candidates])
        for candidate, compact, sem in zip(candidates, compacts, semantic):
            lexical = self._lexical(terms, idf, compact, query_compact, denominator)
            candidate.rerank_score = self._fuse(candidate, query_compact, sem, lexical)
        ordered = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return _renumber(ordered)

    @staticmethod
    def _idf(terms: dict[int, set[str]], compacts: list[str]) -> dict[str, float]:
        """文档频率只在本轮候选池里统计 —— 不需要全语料，判别力已经足够。"""
        total = len(compacts)
        idf: dict[str, float] = {}
        for gram_set in terms.values():
            for gram in gram_set:
                if gram not in idf:
                    df = sum(1 for compact in compacts if gram in compact)
                    idf[gram] = math.log((total + 1) / (df + 0.5)) + 1e-6
        return idf

    @staticmethod
    def _lexical(terms: dict[int, set[str]], idf: dict[str, float], compact: str,
                 query_compact: str, denominator: float) -> float:
        numerator = 0.0
        for n, gram_set in terms.items():
            weight = _GRAM_WEIGHTS[n]
            for gram in gram_set:
                if gram in compact:
                    numerator += weight * idf[gram]
        coverage = numerator / denominator
        run = _longest_run(query_compact, compact) / max(1, len(query_compact))
        return 0.68 * coverage + 0.32 * run

    def _fuse(self, candidate: Retrieved, query_compact: str,
              semantic: float, lexical: float) -> float:
        """融合语义分与词法分，而不是用词法分覆盖语义分。

        这是这一版最重要的修正。早先的实现直接按词法分重排，等于把
        "整体语义接近"这个已经算出来的信号扔掉了 —— 结果是每修一个查询
        就弄坏另一个。语义分管"说得像"，词法分管"真的写着"，两者互补：
        只保留任意一个，都会在另一个的强项上翻车。
        """
        chunk = candidate.chunk
        # 章节标题是作者给这段内容起的名字。标题与查询有较长连续命中时给一点加成，
        # 相当于"这段就是专门讲这个的"；权重必须小，否则会盖过正文证据。
        heading = _compact(f"{chunk.title}{chunk.section}")
        section_bonus = min(0.06, 0.015 * _longest_run(query_compact, heading, cap=8)) if heading else 0.0
        length_penalty = 0.0
        if len(chunk.text) > self.length_penalty_after:
            length_penalty = min(0.06, (len(chunk.text) - self.length_penalty_after) / 8000.0)
        score = (self.semantic_weight * semantic
                 + self.lexical_weight * lexical
                 + section_bonus - length_penalty)
        return round(max(0.0, min(1.0, score)), 6)


class LlmReranker:
    """让模型逐条打分。批量提交以控制调用次数（一次问答的调用数是成本主因）。"""

    name = "llm"

    def __init__(self, client: LlmClient, batch_size: int = 8) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.client = client
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        if not candidates:
            return []
        scores: list[float] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start:start + self.batch_size]
            scores.extend(self._score_batch(query, batch))
        if len(scores) != len(candidates):
            raise RerankError(f"打分条数 {len(scores)} 与候选数 {len(candidates)} 不一致")
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = score
        ordered = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return _renumber(ordered)

    def _score_batch(self, query: str, batch: list[Retrieved]) -> list[float]:
        texts = [c.chunk.text[:400] for c in batch]
        completion = self.client.complete(rerank_messages(query, texts), purpose="rerank")
        try:
            payload = json.loads(_strip_fence(completion.text))
            raw = payload["scores"]
            values = [float(x) for x in raw]
        except Exception as e:
            raise RerankError(f"重排响应无法解析: {completion.text[:160]!r} ({e})") from None
        if len(values) != len(batch):
            raise RerankError(f"重排返回 {len(values)} 个分数，期望 {len(batch)} 个")
        # 模型可能给出越界分数（0-10 之外的夸张值），必须夹紧而不是信它
        return [round(max(0.0, min(10.0, v)) / 10.0, 6) for v in values]


def _renumber(items: list[Retrieved]) -> list[Retrieved]:
    for position, item in enumerate(items, start=1):
        item.rank_after = position
    return items


def _strip_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def build_reranker(mode: str, client: LlmClient | None = None) -> Reranker | None:
    if mode == "off":
        return None
    if mode == "llm":
        if client is None:
            raise ValueError("llm 重排需要 client")
        return LlmReranker(client)
    return HeuristicReranker()
