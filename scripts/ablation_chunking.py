"""切块两项改动的消融实验。

README 与文档里有一句话："切块是检索的天花板"。这句话需要能被拆开验证，
否则就是一句正确的废话。这里把两项改动做成 2×2：

  A 硬折行归一化（normalize_text）   —— 不还原折行，块会从半句开始
  B 章节作为硬边界（_split_sections）—— 不设边界，块会横跨两三个话题

四个组合各跑一次 ingest + eval，看召回侧（recall-only）的指标怎么变。

用法：python scripts/ablation_chunking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agentkit.chunker as chunker  # noqa: E402
from agentkit.bootstrap import build_app, ingest  # noqa: E402
from agentkit.config import Settings  # noqa: E402
from agentkit.evaluate import evaluate_retrieval, load_cases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "docs"
CASES = ROOT / "data" / "eval" / "queries.json"

REAL_NORMALIZE = chunker.normalize_text
REAL_SPLIT = chunker._split_sections


def main() -> None:
    cases = load_cases(CASES)
    print(f"{'归一化':>6} {'章节边界':>8} | {'块数':>4} | "
          f"{'k=4 recall':>10} {'k=4 MRR':>8} | {'k=1 rec':>8} {'k=3 rec':>8}")
    for normalize in (True, False):
        for sectioned in (True, False):
            chunker.normalize_text = REAL_NORMALIZE if normalize else (lambda t: t)
            chunker._split_sections = REAL_SPLIT if sectioned else (
                lambda text: [(0, len(text))] if text.strip() else []
            )
            settings = Settings(index_path=str(ROOT / "out" / "_ablation.jsonl"))
            app = build_app(settings, rerank="heuristic")
            try:
                summary = ingest(app, [DOCS], reset=True)
                out4 = evaluate_retrieval(app.retriever, cases, k=4)
                out1 = evaluate_retrieval(app.retriever, cases, k=1)
                out3 = evaluate_retrieval(app.retriever, cases, k=3)
                print(f"{'是' if normalize else '否':>6} {'是' if sectioned else '否':>8} | "
                      f"{summary['chunks']:>4} | "
                      f"{out4.before.recall_at_k:>10.4f} {out4.before.mrr:>8.4f} | "
                      f"{out1.before.recall_at_k:>8.4f} {out3.before.recall_at_k:>8.4f}")
            finally:
                app.close()
    chunker.normalize_text = REAL_NORMALIZE
    chunker._split_sections = REAL_SPLIT


if __name__ == "__main__":
    main()
