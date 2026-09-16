"""运行时、工具与记忆测试。

这是"Agent 到底靠不靠谱"的核心：模型会输出垃圾、工具会挂、步数会用尽。
每一条都必须有确定的、可断言的行为，而不是"一般不会出问题"。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agentkit.chunker import chunk_document
from agentkit.config import Settings
from agentkit.embedder import HashingEmbedder
from agentkit.llm import StubClient
from agentkit.memory import ConversationMemory
from agentkit.models import Completion
from agentkit.retrieve import Retriever
from agentkit.rerank import HeuristicReranker
from agentkit.runtime import AgentRuntime, CitationError, parse_action
from agentkit.store import InMemoryVectorStore
from agentkit.tools import (
    CalculatorTool,
    NowTool,
    ToolRegistry,
    build_registry,
    safe_eval,
)


# --- safe_eval ---

def test_safe_eval_basic_arithmetic() -> None:
    assert safe_eval("(128+64)*3") == 576
    assert safe_eval("7//2") == 3
    assert safe_eval("2**10") == 1024
    assert safe_eval("-5 + 2") == -3


def test_safe_eval_normalizes_fullwidth_symbols() -> None:
    assert safe_eval("（12＋8）×2") == 40
    assert safe_eval("9÷3") == 3


@pytest.mark.parametrize("expression", [
    "__import__('os').system('ls')",
    "open('/etc/passwd').read()",
    "a + 1",
    "1 if True else 2",
])
def test_safe_eval_rejects_non_arithmetic(expression: str) -> None:
    # Agent 的输入来自模型，等于用户可控；绝不能走 eval
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_safe_eval_reports_syntax_errors() -> None:
    with pytest.raises(ValueError, match="语法错误"):
        safe_eval("1 +")


# --- 工具 ---

def test_calculator_renders_integer_without_decimal() -> None:
    assert CalculatorTool().run(expression="(128+64)*3").text == "(128+64)*3 = 576"
    assert CalculatorTool().run(expression="10/4").text == "10/4 = 2.5"


def test_calculator_rejects_empty_expression() -> None:
    with pytest.raises(ValueError):
        CalculatorTool().run(expression="  ")


def test_now_tool_returns_iso_time() -> None:
    assert "T" in NowTool().run().text


def test_registry_specs_lists_tools_and_params() -> None:
    specs = ToolRegistry([CalculatorTool(), NowTool()]).specs()
    assert "calculator(expression:string)" in specs
    assert "now()" in specs


def test_registry_reports_unknown_tool_instead_of_raising() -> None:
    result = ToolRegistry([CalculatorTool()]).call("nope", {})
    assert result.ok is False
    assert "不存在" in result.text


def test_registry_wraps_tool_type_errors() -> None:
    result = ToolRegistry([CalculatorTool()]).call("calculator", {"unknown_arg": 1})
    # CalculatorTool 声明了 **_ 兜底，所以未知参数不会触发 TypeError；
    # expression 缺失才会走到工具自身的校验
    assert result.ok is False


def test_registry_wraps_bad_arguments() -> None:
    class Strict:
        name = "strict"
        description = "只接受 query"
        parameters = {"type": "object", "properties": {"query": {"type": "string"}}}

        def run(self, query: str) -> object:
            from agentkit.tools import ToolResult

            return ToolResult(text=query)

    result = ToolRegistry([Strict()]).call("strict", {"wrong": 1})
    assert result.ok is False
    assert "参数不合法" in result.text


def test_registry_wraps_tool_exception() -> None:
    class Boom:
        name = "boom"
        description = "总是失败"
        parameters = {"type": "object", "properties": {}}

        def run(self, **_):
            raise RuntimeError("内部炸了")

    result = ToolRegistry([Boom()]).call("boom", {})
    assert result.ok is False
    assert "执行失败" in result.text and "内部炸了" in result.text


def test_registry_times_out_instead_of_hanging() -> None:
    class Slow:
        name = "slow"
        description = "睡很久"
        parameters = {"type": "object", "properties": {}}

        def run(self, **_):
            time.sleep(1.0)
            return None

    registry = ToolRegistry([Slow()], timeout_seconds=0.05)
    started = time.monotonic()
    result = registry.call("slow", {})
    assert result.ok is False
    assert "超时" in result.text
    # 必须真的提前返回，而不是等工具跑完
    assert time.monotonic() - started < 0.5


def test_registry_builds_three_default_tools(tmp_path: Path) -> None:
    registry = build_registry(_retriever(tmp_path))
    assert registry.names() == ["calculator", "now", "search_knowledge_base"]
    registry.close()


# --- parse_action ---

def test_parse_action_accepts_plain_json() -> None:
    assert parse_action('{"tool": "calculator"}') == {"tool": "calculator"}


def test_parse_action_tolerates_fences_and_prose() -> None:
    text = '好的，我先查询。\n```json\n{"answer": "在", "used": [1]}\n```\n以上。'
    assert parse_action(text) == {"answer": "在", "used": [1]}


@pytest.mark.parametrize("text", ["完全不是 JSON", "", "{ 坏的 }", "[1,2,3]"])
def test_parse_action_rejects_garbage(text: str) -> None:
    assert parse_action(text) is None


# --- 运行时 ---

class ScriptedClient:
    """按剧本逐条返回的客户端。用来构造那些真实模型很难稳定复现的坏情况。"""

    backend = "scripted"

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.prompts: list[str] = []
        self._listeners = []

    def on_completion(self, listener) -> None:
        self._listeners.append(listener)

    def complete(self, messages, purpose, temperature=None) -> Completion:
        text = self.script.pop(0) if self.script else '{"answer": "剧本用完了", "used": []}'
        self.prompts.append(messages[-1]["content"])
        completion = Completion(
            purpose=purpose,  # type: ignore[arg-type]
            model="scripted", text=text, prompt_tokens=10, completion_tokens=5,
            cost=0.0001, latency_ms=3, backend=self.backend,
        )
        for listener in self._listeners:
            listener(completion)
        return completion

    def embed(self, texts: list[str]) -> list[list[float]]:
        return HashingEmbedder().embed(texts)


def _retriever(tmp_path: Path) -> Retriever:
    settings = Settings(index_path=str(tmp_path / "i.jsonl"), recall_k=4, top_n=3)
    store = InMemoryVectorStore()
    text = "# 配额是怎么做到不超发的\n\n配额扣减走 Lua 脚本，保证原子性，不会超发。\n\n" \
           "## 上游不可用时怎么办\n\n缓存组件不可用时必须放行请求，否则一次缓存故障会放大。"
    document = chunk_document("a.md", text, 320, 64)
    embedder = HashingEmbedder()
    store.add(document.chunks, embedder.embed([c.text for c in document.chunks]))
    return Retriever(store, embedder, settings, reranker=HeuristicReranker())


def _runtime(tmp_path: Path, script: list[str]) -> AgentRuntime:
    settings = Settings(index_path=str(tmp_path / "i.jsonl"), recall_k=4, top_n=3,
                        max_steps=4)
    client = ScriptedClient(script)
    return AgentRuntime(settings, client, build_registry(_retriever(tmp_path)))


def test_runtime_records_usage_of_every_call(tmp_path: Path) -> None:
    """回归：曾经把 self._stats.add 注册成监听器，导致每次问答都报"调用 0 次"。"""
    runtime = _runtime(tmp_path, [
        '{"tool": "search_knowledge_base", "arguments": {"query": "配额"}}',
        '{"answer": "配额扣减走 Lua 脚本保证原子性 [1]。", "used": [1]}',
    ])
    answer = runtime.ask("配额是怎么保证不超发的？")
    assert answer.usage.calls == 2
    assert answer.usage.by_purpose == {"agent": 2}
    assert answer.usage.total_tokens() > 0
    runtime.close()


def test_runtime_resolves_citations_to_real_chunks(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, [
        '{"tool": "search_knowledge_base", "arguments": {"query": "配额"}}',
        '{"answer": "用 Lua 保证原子扣减 [1]。", "used": [1]}',
    ])
    answer = runtime.ask("配额是怎么保证不超发的？")
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.source == "a.md"
    assert citation.start < citation.end
    assert answer.unresolved_citations == []
    assert answer.used_indices() == [1]
    runtime.close()


def test_runtime_renumbers_citations_across_two_searches(tmp_path: Path) -> None:
    """两次检索各自的 [1] 指向不同片段，不重编号就会出现同一编号指两个来源。

    语料只有两个章节，所以第一次检索占 [1][2]，第二次从 [3] 开始。
    """
    runtime = _runtime(tmp_path, [
        '{"tool": "search_knowledge_base", "arguments": {"query": "配额"}}',
        '{"tool": "search_knowledge_base", "arguments": {"query": "缓存"}}',
        '{"answer": "配额见 [1]，缓存见 [3]。", "used": [1, 3]}',
    ])
    answer = runtime.ask("配额和缓存分别怎么处理？")
    indices = [c.index for c in answer.citations]
    assert indices == [1, 3]
    chunk_ids = {c.chunk_id for c in answer.citations}
    assert len(chunk_ids) == 2, "编号必须指向两个不同的片段"
    assert answer.unresolved_citations == []
    runtime.close()


def test_runtime_renumbering_keeps_tool_local_numbering_stable(tmp_path: Path) -> None:
    """工具内部永远从 [1] 开始；全局编号只在运行时的池子里递增。"""
    runtime = _runtime(tmp_path, [
        '{"tool": "search_knowledge_base", "arguments": {"query": "配额"}}',
        '{"tool": "search_knowledge_base", "arguments": {"query": "缓存"}}',
        '{"answer": "缓存见 [3]。", "used": [3]}',
    ])
    answer = runtime.ask("缓存怎么处理？")
    observations = [s.text for s in answer.steps if s.kind == "observation"]
    joined = "\n".join(observations)
    assert "[1] 来源：" in joined and "[3] 来源：" in joined
    runtime.close()


def test_runtime_records_fabricated_citation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, [
        '{"tool": "search_knowledge_base", "arguments": {"query": "配额"}}',
        '{"answer": "编造的内容 [99]。", "used": [99]}',
    ])
    answer = runtime.ask("配额？")
    assert answer.unresolved_citations == [99]
    assert answer.citations == []
    runtime.close()


def test_runtime_strict_mode_raises_on_fabricated_citation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, [
        '{"tool": "search_knowledge_base", "arguments": {"query": "配额"}}',
        '{"answer": "编造的内容 [99]。", "used": [99]}',
    ])
    with pytest.raises(CitationError, match="99"):
        runtime.ask("配额？", strict_citations=True)
    runtime.close()


def test_runtime_falls_back_after_bad_json_then_recovers(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, [
        "这不是 JSON",
        '{"tool": "search_knowledge_base", "arguments": {"query": "配额"}}',
        '{"answer": "配额用 Lua 保证原子 [1]。", "used": [1]}',
    ])
    answer = runtime.ask("配额？")
    assert answer.citations
    assert any("不是合法 JSON" in step.text for step in answer.steps)
    runtime.close()


def test_runtime_answers_calculator_question_without_citation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, [
        '{"tool": "calculator", "arguments": {"expression": "(128+64)*3"}}',
        '{"answer": "(128+64)*3 = 576。", "used": []}',
    ])
    answer = runtime.ask("帮我算一下 (128+64)*3")
    assert "576" in answer.text
    assert answer.citations == []
    runtime.close()


def test_runtime_survives_tool_failure(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, [
        '{"tool": "calculator", "arguments": {"expression": "1 plus 2"}}',
        '{"answer": "表达式无法解析 [1]。", "used": [1]}',
    ])
    answer = runtime.ask("帮我算 1 plus 2")
    failed = [s for s in answer.steps if s.kind == "action" and s.tool and not s.tool.ok]
    assert failed, "工具失败必须留下痕迹，而不是被吞掉"
    runtime.close()


def test_runtime_uses_final_call_when_steps_run_out(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, [
        '{"tool": "search_knowledge_base", "arguments": {"query": "a"}}',
        '{"tool": "search_knowledge_base", "arguments": {"query": "b"}}',
        '{"tool": "search_knowledge_base", "arguments": {"query": "c"}}',
        '{"tool": "search_knowledge_base", "arguments": {"query": "d"}}',
        '{"answer": "基于已有资料的回答 [1]。", "used": [1]}',
    ])
    answer = runtime.ask("一直检索不出结论的问题")
    assert "基于已有资料的回答" in answer.text
    assert any("最大步数" in step.text for step in answer.steps)
    runtime.close()


def test_runtime_rejects_non_object_arguments(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, [
        '{"tool": "calculator", "arguments": [1, 2]}',
        '{"answer": "无法调用 [1]。", "used": [1]}',
    ])
    answer = runtime.ask("随便问问")
    assert any("JSON 对象" in step.text for step in answer.steps)
    runtime.close()


def test_runtime_with_stub_client_works_offline(tmp_path: Path) -> None:
    settings = Settings(index_path=str(tmp_path / "i.jsonl"), recall_k=4, top_n=3)
    runtime = AgentRuntime(settings, StubClient(settings), build_registry(_retriever(tmp_path)))
    answer = runtime.ask("配额是怎么保证不超发的？")
    assert answer.citations
    assert answer.unresolved_citations == []
    assert answer.usage.calls >= 2
    runtime.close()


# --- 记忆 ---

def _memory(tmp_path: Path, client=None, budget: int = 60, keep: int = 2) -> ConversationMemory:
    settings = Settings(index_path=str(tmp_path / "i.jsonl"),
                        memory_token_budget=budget, memory_keep_recent=keep)
    return ConversationMemory(settings, client)


def test_memory_does_not_compress_under_budget(tmp_path: Path) -> None:
    memory = _memory(tmp_path, ScriptedClient([]))
    memory.add_turn("短问题", "短答案")
    assert memory.maybe_compress() is False
    assert memory.compressions == 0


def test_memory_compresses_and_keeps_recent_turns(tmp_path: Path) -> None:
    memory = _memory(tmp_path, ScriptedClient(['{"answer": "没用到"}', "摘要内容"]))
    for i in range(6):
        memory.add_turn(f"第{i}轮问题", "答案" * 20)
    assert memory.needs_compression()
    assert memory.maybe_compress() is True
    assert memory.compressions == 1
    assert len(memory.turns) == 2          # 保留最近 N 轮原文
    assert memory.summary                 # 更早的轮次被压成摘要
    assert "（更早对话摘要）" in memory.transcript()


def test_memory_degrades_when_summarization_fails(tmp_path: Path) -> None:
    class Boom:
        backend = "boom"

        def on_completion(self, listener):
            pass

        def complete(self, messages, purpose, temperature=None):
            raise RuntimeError("摘要服务挂了")

        def embed(self, texts):
            return [[0.0] for _ in texts]

    memory = _memory(tmp_path, Boom())
    for i in range(6):
        memory.add_turn(f"第{i}轮问题", "答案" * 20)
    assert memory.maybe_compress() is True
    assert memory.last_compression_degraded is True
    assert memory.summary  # 降级也要留下东西，不能把历史丢掉
    assert len(memory.turns) == 2


def test_memory_context_block_is_empty_without_history(tmp_path: Path) -> None:
    assert _memory(tmp_path).context_block() == ""


def test_memory_clear_resets_everything(tmp_path: Path) -> None:
    memory = _memory(tmp_path, ScriptedClient(["摘要"]))
    memory.add_turn("q", "a")
    memory.clear()
    assert memory.turns == [] and memory.summary == "" and memory.compressions == 0
