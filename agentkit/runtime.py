"""ReAct 运行时。

循环本身很短，难点全在"出错时怎么办"。这里固化了四条处理方式：

  1. 模型输出不是合法 JSON → 回灌一条格式纠正提示，而不是直接失败。
     模型偶尔会带上 ``` 或解释性文字，这属于可自愈的抖动。
  2. 工具失败 → 把失败信息作为观察回灌（见 tools.ToolRegistry.call）。
  3. 步数用尽 → 再做一次"禁止调工具"的收尾调用，逼出一个基于已有资料的答案。
     直接把"我没想出来"丢给用户是最差的一档，能兜就必须兜。
  4. 引用校验 → 正文里的 [n] 必须在检索池里有对应条目。
     解析不到的编号记入 unresolved_citations（严格模式直接抛 CitationError）。
     编引用比编事实更危险，因为它看起来"有出处"。

离线桩走的是同一个循环、同一套提示词，所以上面四条在断网时同样被真实执行。
"""

from __future__ import annotations

import json
from typing import Any

from agentkit.config import Settings
from agentkit.llm import LlmClient, agent_messages
from agentkit.memory import ConversationMemory
from agentkit.models import (
    AgentStep,
    Answer,
    Citation,
    ToolCall,
    UsageStats,
    markers_in,
    snippet_of,
)
from agentkit.prompts import OBSERVATION_MARKER, render_observation
from agentkit.retrieve import ContextItem, render_block
from agentkit.tools import ToolRegistry

NO_TOOLS_HINT = "（本轮不允许再调用工具，必须直接给出最终答案）"
FORMAT_HINT = (
    f"{OBSERVATION_MARKER}你的上一条回复不是合法 JSON。"
    "请只输出一行 JSON，不要加解释文字、不要用 markdown 代码块包裹。"
)
BUDGET_HINT = f"{OBSERVATION_MARKER}已达最大步数限制，请立即基于已有观察给出最终答案。"


class CitationError(RuntimeError):
    """正文引用了不存在的资料编号。"""


def parse_action(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出 JSON 对象。容忍代码块包裹与前后解释文字。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start:end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


class AgentRuntime:
    def __init__(self, settings: Settings, client: LlmClient, registry: ToolRegistry,
                 memory: ConversationMemory | None = None) -> None:
        self.settings = settings
        self.client = client
        self.registry = registry
        self.memory = memory or ConversationMemory(settings, client)
        self._stats = UsageStats()
        # 只注册一次：client 被检索器（重排）、记忆（压缩）、运行时（编排）共用，
        # 全部开销都要落到当前这一次问答的账上。
        #
        # 注意这里必须注册 self._record 而不是 self._stats.add ——
        # 后者会把方法绑在**当时那个** UsageStats 实例上，而 ask() 每次都会新建
        # 一个 _stats。结果是所有调用都记进了第一次问答的对象里，
        # 表现为"跑了 3 次调用，输出却写着 0 次"。
        self.client.on_completion(self._record)

    def _record(self, completion) -> None:
        self._stats.add(completion)

    @property
    def usage(self) -> UsageStats:
        return self._stats

    def ask(self, question: str, max_steps: int | None = None,
            strict_citations: bool = False) -> Answer:
        self._stats = UsageStats()
        steps: list[AgentStep] = []
        observations: list[str] = []
        pool: dict[int, ContextItem] = {}
        limit = max_steps or self.settings.max_steps
        memory_block = self.memory.context_block()

        for step_index in range(1, limit + 1):
            messages = agent_messages(question, observations, self.registry.specs(), memory_block)
            completion = self.client.complete(messages, purpose="agent")
            payload = parse_action(completion.text)

            if payload is None:
                steps.append(AgentStep(kind="observation", step_index=step_index,
                                       text="模型输出不是合法 JSON，已回灌格式化提示"))
                observations.append(FORMAT_HINT)
                continue

            if "tool" in payload:
                self._execute_tool(payload, step_index, steps, observations, pool)
                continue

            if "answer" in payload:
                return self._finalize(question, str(payload.get("answer") or ""),
                                      payload.get("used"), steps, step_index, pool,
                                      strict_citations)

        # 步数用尽：再要一次答案，不允许继续调工具
        steps.append(AgentStep(kind="observation", step_index=limit,
                               text=f"已达最大步数 {limit}，转入收尾"))
        messages = agent_messages(question, observations + [BUDGET_HINT],
                                  NO_TOOLS_HINT, memory_block)
        completion = self.client.complete(messages, purpose="agent")
        payload = parse_action(completion.text) or {}
        fallback = str(payload.get("answer") or "已达到最大推理步数，未能给出确定结论。")
        return self._finalize(question, fallback, payload.get("used"), steps, limit + 1,
                              pool, strict_citations)

    # --- 内部 ---

    def _execute_tool(self, payload: dict[str, Any], step_index: int, steps: list[AgentStep],
                      observations: list[str], pool: dict[int, ContextItem]) -> None:
        name = str(payload.get("tool") or "")
        arguments = payload.get("arguments") or {}
        if not isinstance(arguments, dict):
            steps.append(AgentStep(kind="observation", step_index=step_index,
                                   text="arguments 必须是 JSON 对象，已回灌"))
            observations.append(f"{OBSERVATION_MARKER}arguments 必须是一个 JSON 对象。")
            return

        result = self.registry.call(name, arguments)
        steps.append(AgentStep(
            kind="action", step_index=step_index, text=f"调用工具 {name}",
            tool=ToolCall(name=name, arguments=arguments, ok=result.ok,
                          output=result.text[:400],
                          error="" if result.ok else result.text[:200])))

        if result.contexts:
            renumbered = self._renumber(result.contexts, pool)
            observation = render_observation(
                name, [render_block(item.index, item.retrieved) for item in renumbered])
        else:
            observation = f"{OBSERVATION_MARKER}工具 {name} 返回：{result.text}"

        steps.append(AgentStep(kind="observation", step_index=step_index, text=observation))
        observations.append(observation)

    def _renumber(self, contexts: list[ContextItem],
                  pool: dict[int, ContextItem]) -> list[ContextItem]:
        """把工具内部的编号改写成全局编号。

        检索工具可能被调用多次，各自的 [1] 指的是不同片段。
        不做重编号就会出现"同一编号指向两个来源"，引用直接失效。
        """
        out: list[ContextItem] = []
        for item in contexts:
            index = len(pool) + 1
            renumbered = ContextItem(index=index, retrieved=item.retrieved)
            pool[index] = renumbered
            out.append(renumbered)
        return out

    def _finalize(self, question: str, text: str, used: Any, steps: list[AgentStep],
                  step_index: int, pool: dict[int, ContextItem],
                  strict_citations: bool) -> Answer:
        markers = markers_in(text)
        wanted: set[int] = set(markers)
        for raw in used or []:
            if isinstance(raw, bool):
                continue
            if isinstance(raw, int):
                wanted.add(raw)
            elif isinstance(raw, str) and raw.strip().isdigit():
                wanted.add(int(raw.strip()))

        citations: list[Citation] = []
        for index in sorted(wanted):
            item = pool.get(index)
            if item is None:
                continue
            chunk = item.chunk
            citations.append(Citation(
                index=index, chunk_id=chunk.id, doc_id=chunk.doc_id, source=chunk.source,
                start=chunk.start, end=chunk.end, snippet=snippet_of(chunk.text)))

        unresolved = sorted(index for index in wanted if index not in pool)
        if strict_citations and unresolved:
            raise CitationError(f"答案引用了不存在的资料编号: {unresolved}")

        steps.append(AgentStep(kind="answer", step_index=step_index, text=text))
        self.memory.add_turn(question, text)
        self.memory.maybe_compress()

        return Answer(
            question=question,
            text=text,
            citations=citations,
            steps=steps,
            usage=self._stats.model_copy(deep=True),
            backend=self.client.backend,
            unresolved_citations=unresolved,
        )

    def close(self) -> None:
        self.registry.close()
