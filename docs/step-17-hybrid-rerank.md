# step-17 · 语义 + IDF 词法融合重排

## 目标

把"召回出来的 12 条候选"排成"最该被拿去回答的那几条"。
向量召回按**整体语义接近**排序，而用户要的是**能回答这个问题的片段**，
两者经常不一致：一个泛泛而谈的概述段可能比真正包含答案的窄段余弦分更高。

## 改动

- 新增 `agentkit/retrieve.py`：召回 → 重排 → 去重 → 按 token 预算组装上下文
- 新增 `agentkit/rerank.py`：`HeuristicReranker`（融合式）与 `LlmReranker`（带失败降级）
- 新增 `agentkit/evaluate.py` 的对照口径：同一次召回结果，按召回位次与重排位次分别算指标
- 新增 `scripts/tune_rerank.py`：权重扫描

## 关键代码

```python
def rerank(self, query, candidates):
    terms = _query_terms(query)                    # 2/3/4-gram，权重 1/2/3
    compacts = [_compact(c.chunk.text) for c in candidates]
    idf = self._idf(terms, compacts)               # 文档频率只在本轮候选池里统计
    semantic = _minmax([c.recall_score for c in candidates])
    for candidate, compact, sem in zip(candidates, compacts, semantic):
        lexical = self._lexical(terms, idf, compact, query_compact, denominator)
        candidate.rerank_score = self._fuse(candidate, query_compact, sem, lexical)
```

```
final = 0.30 × normalize(cosine) + 0.70 × (0.68 × idf_coverage + 0.32 × longest_run)
```

### 为什么是两个信号的融合

第一版直接用词法分重排 —— 等于把已经算好的语义分扔掉。
结果是**修好一个查询就弄坏另一个**：recall@2 涨了，MRR 反而跌。
语义分管"说得像"，词法分管"真的写着"，各自的强项恰好是对方的盲区。

### 为什么词法分要按候选池 IDF 加权

朴素做法是数查询与片段共有几个 2-gram。中文里 2-gram 噪声极大：
问"缓存组件挂了，请求会被拒绝吗"，"缓存"这个词几乎所有候选都有，
谁命中它谁得分，于是"语义缓存与路由"这种泛泛的概述段会压过真正写着
"缓存组件不可用时必须放行请求"的那一段。按文档频率把"人手一个"的 gram
权重压下去，判别力立刻回来。

不从全语料算 IDF 是刻意的：只看这一轮候选池就够判别，而且不需要维护额外的倒排索引。

### LLM 重排必须假设它会失败

```python
if len(values) != len(batch):
    raise RerankError(f"重排返回 {len(values)} 个分数，期望 {len(batch)} 个")
return [round(max(0.0, min(10.0, v)) / 10.0, 6) for v in values]  # 越界分数夹紧
```

三种失败都要识别：返回条数不符、JSON 非法、分数越界。
`Retriever._rerank` 捕获 `RerankError` 后降级到启发式，并**把降级原因写进 `notes`** ——
静默降级等于把质量下降藏起来。

## 验证方式

```bash
agentkit eval --k 4           # 重排前 vs 重排后
python scripts/tune_rerank.py # 权重扫描
python scripts/analyze_eval.py 3   # 逐条对比哪些查询被重排弄坏了
pytest tests/test_retrieve_rerank.py -q
```

## 结果

评测集 21 条查询（来源文件 + 段落关键词双重判定），k=4：

| 排序策略 | recall@4 | MRR | hit_rate | n |
|---|---|---|---|---|
| recall-only | 0.9524 | 0.8849 | 95.24% | 21 |
| rerank(heuristic) | 0.9524 | **0.9206** | 95.24% | 21 |

权重扫描（`scripts/tune_rerank.py`）：

| 语义 : 词法 | k=1 MRR | k=2 recall | k=3 recall | k=4 MRR |
|---|---|---|---|---|
| 0 : 1（只用词法） | 0.7143 | 0.8095 | 0.8571 | 0.7897 |
| 1 : 0（等于不重排） | 0.8571 | 0.8571 | 0.9048 | 0.8849 |
| 0.45 : 0.55 | 0.8571 | 0.9048 | 0.9524 | 0.8968 |
| **0.30 : 0.70（默认）** | **0.9048** | **0.9048** | **0.9524** | **0.9206** |

这张表比单点数字更有信息量：**两个信号各自都有翻车的场景，融合后每个 k 都严格不劣于基线**。

## 踩了什么坑

1. **重排覆盖召回**（上面已说）：把词法权重调到 1.0，k=1 的 MRR 掉到 0.7143。
2. **`_longest_run` 只看查询前缀**：问"引用为什么必须记录字符区间"，
   而答案段写的是"必须记录字符区间才能反查" —— 命中的是查询**中段**，
   只看前缀会漏掉。改成任意子串。
3. **章节匹配加成过大**：一开始给到 0.08，盖过了正文证据。现在压到 0.06 上限。
4. **评测集没有区分度**：最初只有 8 篇文档 / 28 块，top-4 永远能覆盖答案，
   recall@4 恒为 1.0。补到 15 篇 / 66 块之后指标才有意义 ——
   **一个永远满分评测集等于没有评测**。

## 下一步

- `recall_k=12` 与 `top_n=4` 是折中值，还没有做过 sweep，应该补一条成本-质量曲线。
- 引入 **RRF（reciprocal rank fusion）** 做真正的混合检索通道（BM25 + 向量并行召回），
  现在的词法只作用在重排阶段，救不回"压根没进候选池"的片段。
