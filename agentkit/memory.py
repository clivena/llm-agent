"""多轮会话记忆。

对话变长有两条路可走：
  ① 把全部历史塞进提示词 —— 上下文窗口迟早撞墙，而且每轮都在为同一段历史重复付费；
  ② 丢弃旧轮次 —— 省钱了，但用户三句话前提到的约束会凭空消失。

这里走第三条：**滚动摘要**。超出预算时把较早的轮次压缩成一段摘要，
只保留最近 N 轮原文。压缩本身是一次模型调用，所以要注意两件事：
  · 压缩必须节流（只在超预算时触发），否则每轮都摘一次，成本反而更高；
  · 压缩失败不能丢历史（降级为"保留最近轮次 + 保留旧原文的截断版"），
    宁可贵一点，也不能让用户觉得"你怎么忘了"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentkit.config import Settings
from agentkit.llm import LlmClient, summarize_messages
from agentkit.retrieve import estimate_tokens


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    settings: Settings
    client: LlmClient | None = None
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    compressions: int = 0
    last_compression_degraded: bool = False

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))

    def transcript(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"（更早对话摘要）{self.summary}")
        for turn in self.turns:
            parts.append(f"用户：{turn.question}")
            parts.append(f"助手：{turn.answer}")
        return "\n".join(parts)

    def context_block(self) -> str:
        """拼进提示词的记忆块。没有历史时返回空串，避免提示词里出现空段落。"""
        transcript = self.transcript()
        return f"（历史对话）\n{transcript}" if transcript else ""

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.transcript())

    def needs_compression(self) -> bool:
        return (self.estimated_tokens() > self.settings.memory_token_budget
                and len(self.turns) > self.settings.memory_keep_recent)

    def maybe_compress(self) -> bool:
        """超预算时压缩较早轮次。返回是否真的发生了压缩。"""
        if not self.needs_compression():
            return False
        keep = self.settings.memory_keep_recent
        older, recent = self.turns[:-keep], self.turns[-keep:]
        if not older:
            return False

        older_text = "\n".join(
            f"用户：{t.question}\n助手：{t.answer}" for t in older
        )
        compressed = ""
        self.last_compression_degraded = False
        if self.client is not None:
            try:
                completion = self.client.complete(summarize_messages(older_text), purpose="summarize")
                compressed = completion.text.strip()
            except Exception:
                # 压缩失败不丢历史：退回机械截断，并标记降级以留痕
                compressed = older_text[: self.settings.memory_token_budget]
                self.last_compression_degraded = True
        else:
            compressed = older_text[: self.settings.memory_token_budget]
            self.last_compression_degraded = True

        self.summary = (self.summary + " " + compressed).strip() if self.summary else compressed
        self.turns = recent
        self.compressions += 1
        return True

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""
        self.compressions = 0
