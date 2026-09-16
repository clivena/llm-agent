"""扫描离线哈希向量的 n-gram 权重组合，看召回质量对权重有多敏感。

用法：python scripts/tune_embedder.py
结论用来给 HashingEmbedder 选默认权重。不参与线上链路。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit.bootstrap import build_app  # noqa: E402
from agentkit.config import Settings  # noqa: E402
from agentkit.embedder import HashingEmbedder  # noqa: E402
from agentkit.evaluate import evaluate_retrieval, load_cases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    {1: 1.0, 2: 1.6, 3: 0.7},      # 当前默认
    {1: 0.0, 2: 1.0, 3: 0.6},
    {1: 0.3, 2: 1.0, 3: 0.8},
    {1: 0.5, 2: 1.0, 3: 0.5},
    {2: 1.0, 3: 1.0},
    {1: 0.2, 2: 1.0, 3: 1.0, 4: 0.4},
    {1: 1.0, 2: 2.0, 3: 1.0, 4: 0.3},
]


def main() -> None:
    instance = build_app(Settings.from_env("stub"), rerank="heuristic")
    try:
        cases = load_cases(ROOT / "data" / "eval" / "queries.json")
        for weights in CANDIDATES:
            instance.retriever.embedder = HashingEmbedder(weights=weights)
            # 索引是按旧向量入库的，换 embedder 必须重算索引
            from agentkit.bootstrap import ingest
            ingest(instance, [str(ROOT / "data" / "docs")], reset=True)
            out3 = evaluate_retrieval(instance.retriever, cases, k=3)
            out4 = evaluate_retrieval(instance.retriever, cases, k=4)
            print(f"{weights}  k3: recall {out3.after.recall_at_k:.4f} mrr {out3.after.mrr:.4f}"
                  f"  | k4: recall {out4.after.recall_at_k:.4f} mrr {out4.after.mrr:.4f}")
    finally:
        instance.close()


if __name__ == "__main__":
    main()
