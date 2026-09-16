"""检索质量门禁。

把 `agentkit eval` 的数字变成 CI 的退出码：低于阈值就红灯。
和 llm-eval 的 `gate` 是同一个思路 —— 质量指标不进门禁，就只是报表。

阈值刻意设得比当前实测值低一点（留出噪声余量），但保证两件事：
  1. 召回质量不能悄悄退化（recall@k 有下界）；
  2. 重排不能变成负收益（rerank 必须不劣于 recall-only）。

用法：
    python scripts/quality_gate.py [--k 4] [--min-recall 0.90] [--min-mrr 0.80]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit.bootstrap import build_app, ingest  # noqa: E402
from agentkit.config import Settings  # noqa: E402
from agentkit.evaluate import evaluate_retrieval, load_cases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量门禁")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--min-mrr", type=float, default=0.80)
    args = parser.parse_args()

    settings = Settings(index_path=str(ROOT / "out" / "gate-index.jsonl"),
                        recall_k=12, top_n=4)
    app = build_app(settings, rerank="heuristic")
    try:
        ingest(app, [str(ROOT / "data" / "docs")], reset=True)
        outcome = evaluate_retrieval(app.retriever, load_cases(ROOT / "data" / "eval" / "queries.json"),
                                     k=args.k)
    finally:
        app.close()

    after, before = outcome.after, outcome.before
    print(f"recall@{args.k}: {after.recall_at_k:.4f}  (基线 {before.recall_at_k:.4f})")
    print(f"MRR:       {after.mrr:.4f}  (基线 {before.mrr:.4f})")
    print(f"hit_rate:  {after.hit_rate:.4f}  n={after.n}")

    failures: list[str] = []
    if after.recall_at_k < args.min_recall:
        failures.append(f"recall@{args.k} {after.recall_at_k:.4f} < 阈值 {args.min_recall}")
    if after.mrr < args.min_mrr:
        failures.append(f"MRR {after.mrr:.4f} < 阈值 {args.min_mrr}")
    if after.recall_at_k < before.recall_at_k:
        failures.append("重排把 recall 弄差了")
    if after.mrr < before.mrr:
        failures.append("重排把 MRR 弄差了")

    if failures:
        for item in failures:
            print(f"::error::{item}")
        return 1
    print("质量门禁通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
