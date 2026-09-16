"""逐条对比重排前后的相关性位次，定位哪些查询被重排弄坏了。

用法：python scripts/analyze_eval.py [k]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit.bootstrap import build_app  # noqa: E402
from agentkit.config import Settings  # noqa: E402
from agentkit.evaluate import is_relevant, load_cases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def first_relevant(items, case):
    for i, it in enumerate(items, start=1):
        if is_relevant(it, case):
            return i
    return None


def main() -> None:
    instance = build_app(Settings.from_env("stub"), rerank="heuristic")
    try:
        cases = load_cases(ROOT / "data" / "eval" / "queries.json")
        for case in cases:
            outcome = instance.retriever.search(case.query, top_n=K)
            before = sorted(outcome.recalled, key=lambda r: r.rank_before)[:K]
            after = sorted(outcome.recalled, key=lambda r: r.rank_after)[:K]
            b = first_relevant(before, case)
            a = first_relevant(after, case)
            if a == b:
                continue
            flag = "↓ 变差" if (a or 99) > (b or 99) else "↑ 变好"
            print(f"{flag}  before={b} after={a}  {case.query}")
            for label, items in (("before", before), ("after", after)):
                for rank, it in enumerate(items, start=1):
                    mark = "*" if is_relevant(it, case) else " "
                    print(f"      {label} {rank}{mark} cos={it.recall_score:.4f} "
                          f"rr={it.rerank_score} {it.chunk.source} §{it.chunk.section}")
            print()
    finally:
        instance.close()


if __name__ == "__main__":
    main()
