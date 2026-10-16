#!/usr/bin/env python3
"""把检索评测结果导出成 dashboard 直接消费的 JSON。

为什么是静态快照而不是实时接口
--------------------------------
agentkit 控制台只暴露 `/api/ask`（完整问答，每问要花真实 token），
**没有"只检索不生成"的端点**。而检索评测本身是纯离线计算：
`HashingEmbedder` + 本地索引，零网络、零成本、结果完全确定。
所以固化成快照反而更可靠 —— 面试现场不起任何服务也能看到与
`DEMO-REPORT.md` 完全一致的数字，且不会因为"忘了起服务"而降级成样例。

产出的 JSON 有两层：
  · `metrics` —— 汇总指标，直接来自 `agentkit.evaluate.evaluate_retrieval`，
    与 CLI `agentkit eval` 是同一份实现，因此两边数字不可能对不上；
  · `cases[]` —— 逐条明细（召回了哪些块、重排前后位次怎么变、哪些算命中），
    这是 dashboard "检索浏览器"的数据源。汇总指标看不出"重排到底改了什么"，
    只有逐条位次变化才是证据。

用法：
    python scripts/export-retrieval.py                     # 落 dashboard/public/data/retrieval.json
    python scripts/export-retrieval.py --out /tmp/r.json    # 换输出位置
    python scripts/export-retrieval.py --k 4 --detail 4     # 调指标 k 与逐条明细条数
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agentkit.bootstrap import build_app  # noqa: E402
from agentkit.config import Settings  # noqa: E402
from agentkit.evaluate import (  # noqa: E402
    _measure,
    evaluate_retrieval,
    is_relevant,
    load_cases,
)
from agentkit.models import snippet_of  # noqa: E402
from agentkit.store import read_index_meta  # noqa: E402

DEFAULT_CASES = REPO / "data" / "eval" / "queries.json"
DEFAULT_INDEX = REPO / "out" / "index.jsonl"
DEFAULT_OUT = REPO.parent / "dashboard" / "public" / "data" / "retrieval.json"

# 片段截断长度。取 120 而不是 200 是**有意的取舍**：134 条用例 × 每条 5~8 个块
# 会让 JSON 涨到近 500 KB，而 120 字已经足够让人认出"这是哪一段"。
SNIPPET_LIMIT = 120


def _rel(path: str | Path) -> str:
    """把绝对路径压成仓库相对路径。

    产物会被浏览器直接加载，**绝不能把本机绝对路径（含用户名）写进去** ——
    这类泄漏在简历项目里尤其致命，看一眼就知道是本地机器上跑出来的。
    """
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(path).name


def _corpus_stats(store) -> dict:
    chunks = store.chunks()
    sources = sorted({c.source for c in chunks})
    sections = {(c.source, c.section) for c in chunks if c.section}
    chars = sum(len(c.text) for c in chunks)
    return {
        # 切片字符总量含重叠部分，所以**不等于**语料原文总字数，别拿来当"语料规模"报。
        "chunkChars": chars,
        "chunks": len(chunks),
        "docs": len(sources),
        "sections": len(sections),
        "avgChunkChars": round(chars / len(chunks)) if chunks else 0,
        "sampleSources": sources[:8],
    }


def _candidates(outcome, case, k: int, detail: int) -> list[dict]:
    """挑出要展示的候选块。

    取「重排后前 detail 条」∪「重排前前 k 条」的并集。为什么必须带上后者：
    只看重排后的结果，就看不到**被重排挤出前 k 的块**，而"挤出"恰恰是重排
    最该被解释的行为（可能是纠正，也可能是误伤）。只取并集能把两种都呈现出来。
    """
    keep: dict[str, object] = {}
    for item in sorted(outcome.recalled, key=lambda r: r.rank_after)[:detail]:
        keep.setdefault(item.chunk.id, item)
    for item in sorted(outcome.recalled, key=lambda r: r.rank_before)[:k]:
        keep.setdefault(item.chunk.id, item)

    ordered = sorted(keep.values(), key=lambda r: (r.rank_after, r.rank_before))  # type: ignore[attr-defined]
    return [
        {
            "chunkId": r.chunk.id,
            "source": r.chunk.source,
            "section": r.chunk.section,
            "ordinal": r.chunk.ordinal,
            "start": r.chunk.start,
            "end": r.chunk.end,
            "text": snippet_of(r.chunk.text, SNIPPET_LIMIT),
            "recallScore": round(r.recall_score, 4),
            "rerankScore": None if r.rerank_score is None else round(r.rerank_score, 4),
            "rankBefore": r.rank_before,
            "rankAfter": r.rank_after,
            "relevant": is_relevant(r, case),
        }
        for r in ordered
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="导出检索评测快照给 dashboard")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="评测集路径")
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="索引文件路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出 JSON 路径")
    parser.add_argument("--k", type=int, default=4, help="计算指标时取前 k 条")
    parser.add_argument("--detail", type=int, default=4, help="每条用例导出多少个候选块")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    index_path = Path(args.index)
    if not cases_path.exists():
        print(f"[x] 评测集不存在：{cases_path}", file=sys.stderr)
        return 2
    if not index_path.exists():
        print(f"[x] 索引不存在：{index_path}\n    先运行：python -m agentkit.cli ingest", file=sys.stderr)
        return 2

    # backend 固定取 stub：检索全程只用本地哈希向量，不碰任何远程模型 ——
    # 这样导出是零成本、离线可复现的，不会因为没配 key 就失败。
    settings = Settings.from_env("stub").model_copy(
        update={"index_path": str(index_path), "embedder": "auto"}
    )
    app = build_app(settings)
    try:
        store = app.store
        if len(store) == 0:
            print("[x] 索引为空。先运行：python -m agentkit.cli ingest", file=sys.stderr)
            return 2

        cases = load_cases(cases_path)
        outcome = evaluate_retrieval(app.retriever, cases, k=args.k)

        rows = []
        for case in cases:
            result = app.retriever.search(case.query, top_n=args.k)
            # 刻意复用 evaluate 的私有 _measure：逐条 recall 必须与汇总指标出自
            # 同一份实现，否则面板上的分项加总对不上 CLI 报的总数。
            after_top = sorted(result.recalled, key=lambda r: r.rank_after)[: args.k]
            recall, first_rank = _measure(case, after_top)
            rows.append(
                {
                    "query": case.query,
                    "expectedSources": case.expected_sources,
                    "expectedKeywords": case.expected_keywords,
                    "recall": round(recall, 4),
                    "firstHitRank": first_rank,
                    "recalledCount": len(result.recalled),
                    "rankChanges": result.rank_changes(),
                    "rerankMode": result.rerank_mode,
                    "notes": result.notes,
                    "candidates": _candidates(result, case, args.k, args.detail),
                }
            )

        payload = {
            "generatedAt": datetime.now().isoformat(timespec="seconds"),
            "k": args.k,
            "casesFile": _rel(cases_path),
            "index": {
                "path": _rel(index_path),
                "embedder": app.embedder.name,
                "kind": app.embedder.kind,
                "dim": store.vector_dim(),
                "builtAt": (read_index_meta(index_path) or {}).get("builtAt"),
                # 远程 embedding 当前不可用（DeepSeek 无 /embeddings），
                # 所以检索跑在离线哈希向量上 —— 这条边界必须跟着数据一起走，
                # 否则面板上 0.9179 会被误读成"语义检索效果好"。
                "semanticGeneralization": False,
            },
            "corpus": _corpus_stats(store),
            "metrics": {
                "before": outcome.before.model_dump(),
                "after": outcome.after.model_dump(),
                "deltaRecall": round(outcome.delta_recall(), 4),
                "deltaMrr": round(outcome.delta_mrr(), 4),
                "rankChanges": outcome.rank_changes,
                "notes": outcome.notes,
            },
            "distribution": {
                "full": sum(1 for r in rows if r["recall"] >= 1.0),
                "partial": sum(1 for r in rows if 0 < r["recall"] < 1.0),
                "miss": sum(1 for r in rows if r["recall"] == 0.0),
            },
            "cases": rows,
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 紧凑输出而不是 indent：这是给浏览器直接加载的数据源，不是给人读的文件，
        # 缩进会让产物凭空胖一大截（实测 476 KB → 三百多 KB）。
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

        m = payload["metrics"]
        print(f"[+] 已写出：{out_path}（{out_path.stat().st_size / 1024:.0f} KB）")
        print(
            f"    语料 {payload['corpus']['docs']} 篇 / {payload['corpus']['chunks']} 切片"
            f" · 用例 {len(rows)} 条 · k={args.k}"
        )
        print(
            f"    recall@{args.k} {m['before']['recall_at_k']:.4f} → {m['after']['recall_at_k']:.4f}"
            f"（{m['deltaRecall']:+.4f}）"
        )
        print(f"    MRR {m['before']['mrr']:.4f} → {m['after']['mrr']:.4f}（{m['deltaMrr']:+.4f}）")
        print(
            f"    全命中 {payload['distribution']['full']} · 部分 {payload['distribution']['partial']}"
            f" · 未命中 {payload['distribution']['miss']}"
        )
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
