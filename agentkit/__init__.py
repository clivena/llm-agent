"""agentkit —— 可溯源的检索增强智能体。

对外只暴露四件事：数据模型、配置、运行时、CLI。
内部实现按链路分层（chunker → embedder → store → retrieve → rerank → tools → runtime），
每层都依赖 Protocol 而不是具体实现，因此可以整体替换（离线 stub / 直连厂商 / 走自研网关）。
"""

from agentkit.config import Settings
from agentkit.models import (
    AgentStep,
    Answer,
    Chunk,
    Citation,
    Completion,
    Document,
    Retrieved,
    ToolCall,
    UsageStats,
)
from agentkit.runtime import AgentRuntime

__all__ = [
    "Settings",
    "AgentRuntime",
    "AgentStep",
    "Answer",
    "Chunk",
    "Citation",
    "Completion",
    "Document",
    "Retrieved",
    "ToolCall",
    "UsageStats",
]

__version__ = "0.1.0"
