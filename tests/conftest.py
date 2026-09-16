"""共享测试夹具。

原则：测试不依赖真实网络、不依赖仓库里的 out/ 索引 ——
每个用例自带数据，跑完即弃（tmp_path）。这样 CI 上 pytest 永远是绿的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentkit.config import Settings  # noqa: E402
from agentkit.store import InMemoryVectorStore  # noqa: E402

DOCS = ROOT / "data" / "docs"
CASES = ROOT / "data" / "eval" / "queries.json"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(index_path=str(tmp_path / "index.jsonl"))


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def docs_dir() -> Path:
    return DOCS


@pytest.fixture
def cases_path() -> Path:
    return CASES


@pytest.fixture
def hard_wrapped_markdown() -> str:
    """模拟排版工具在句子中间硬折行的文档 —— 真实语料里非常常见。"""
    return (
        "# 检索链路\n"
        "\n"
        "文档先切块再向量化。切块的策略是段落优先：段落是天然的语义单元，\n"
        "优先在段落边界断开；单个段落超过长度上限时，回退到句末标点断开，\n"
        "避免切在句子中间。\n"
        "\n"
        "## 重排序\n"
        "\n"
        "模型打分必须假设它会失败。返回分数条数与候选数不一致、返回内容不是合法 JSON、\n"
        "分数超出范围，这三种情况都要能识别。识别后的处理是降级到启发式并记录降级原因，\n"
        "而不是继续使用未重排的结果冒充已完成。\n"
    )
