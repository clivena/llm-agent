"""打印切块结果，用于人工核对块边界是否落在自然语义单元上。

用法：python scripts/dump_chunks.py [文档名 ...]
不带参数时默认检查两篇最容易暴露接缝问题的文档。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentkit.chunker import chunk_document  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data" / "docs"
DEFAULT = ("05-agent-orchestration.md", "08-deployment.md")


def dump(name: str) -> None:
    raw = (ROOT / name).read_text(encoding="utf-8")
    doc = chunk_document(name, raw, 320, 64)
    print(f"=== {name}  raw={len(raw)}  norm={len(doc.text)}  chunks={len(doc.chunks)}")
    for c in doc.chunks:
        # 用归一化后的文本取上下文，start/end 属于归一化文本
        prev = doc.text[max(0, c.start - 12):c.start].replace("\n", "\\n")
        head = c.text[:24].replace("\n", "\\n")
        tail = c.text[-24:].replace("\n", "\\n")
        print(f"  #{c.ordinal} [{c.start:>4}-{c.end:>4}] §{c.section}")
        print(f"      prev=…{prev}")
        print(f"      head={head}")
        print(f"      tail=…{tail}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(DEFAULT)
    for n in names:
        dump(n)
