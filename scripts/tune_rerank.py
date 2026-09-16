"""扫描启发式重排的两项权重，挑一组在 recall 与 MRR 上都不劣于 recall-only 的参数。

用法：python scripts/tune_rerank.py
只是选参工具，不参与线上链路。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit.bootstrap import build_app  # noqa: E402
from agentkit.config import Settings  # noqa: E402
from agentkit.evaluate import evaluate_retrieval, load_cases  # noqa: E402
from agentkit.rerank import HeuristicReranker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    instance = build_app(Settings.from_env("stub"), rerank="heuristic")
    try:
        cases = load_cases(ROOT / "data" / "eval" / "queries.json")
        print(f"{'sem/lex':>10} | {'k':>2} | {'recall-':>8} {'recall+':>8} | {'mrr-':>7} {'mrr+':>7}")
        for sem in (0.0, 0.3, 0.45, 0.55, 0.62, 0.7, 0.8, 1.0):
            instance.retriever.reranker = HeuristicReranker(sem, 1.0 - sem)
            for k in (1, 2, 3, 4):
                out = evaluate_retrieval(instance.retriever, cases, k=k)
                print(f"{sem:>4.2f}/{1 - sem:>4.2f} | {k:>2} | "
                      f"{out.before.recall_at_k:>8.4f} {out.after.recall_at_k:>8.4f} | "
                      f"{out.before.mrr:>7.4f} {out.after.mrr:>7.4f}")
            print()
    finally:
        instance.close()


if __name__ == "__main__":
    main()
