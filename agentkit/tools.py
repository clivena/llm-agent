"""工具层。

三个设计约束，都是被真实的 Agent 事故逼出来的：

① 工具失败要**回灌给模型**，而不是抛穿整个问答。
   模型很可能只是参数写错了（top_k 传了字符串、表达式带了单位），
   把错误信息作为观察回灌，它下一步就能自我纠正；直接抛异常则整个问答失败。

② 工具必须有超时。
   Agent 的死循环大多数不是模型在想，而是某个工具卡住了。
   超时后返回一个"超时"的观察，让模型有机会换条路，也保证请求一定会结束。

③ 工具返回的引用块要带上来源编号映射。
   检索工具返回的 [n] 必须能被运行时装回真实 chunk —— 否则最终答案里的
   引用就成了装饰品，无法验证。这是 Citation 能强校验的前提。
"""

from __future__ import annotations

import ast
import operator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentkit.retrieve import ContextItem, Retriever, render_block


@dataclass
class ToolResult:
    text: str
    contexts: list[ContextItem] = field(default_factory=list)
    ok: bool = True
    elapsed_ms: int = 0


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    def run(self, **kwargs: Any) -> ToolResult:
        ...


class SearchKnowledgeBaseTool:
    name = "search_knowledge_base"
    description = "在本地知识库中做语义检索，返回最相关的文档片段（带来源编号）。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索用的自然语言查询"},
            "top_k": {"type": "integer", "description": "返回片段数，默认取配置值"},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        # 最近一次检索的完整结果，供评测与解释使用（如重排前后的位次变化）
        self.last_outcome = None

    def run(self, query: str = "", top_k: int | None = None, **_: Any) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if top_k is not None and not isinstance(top_k, int):
            raise ValueError("top_k 必须是整数")
        outcome = self.retriever.search(query, top_n=top_k)
        self.last_outcome = outcome
        text = "\n".join(outcome.blocks) if outcome.blocks else "（没有检索到任何资料）"
        return ToolResult(text=text, contexts=outcome.items, ok=True)


_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """只允许数字与算术运算的求值器。

    不能用 eval()：Agent 的输入是模型生成的，等于把用户可控字符串送进解释器。
    这里用 ast 白名单，出现任何 Name/Call/Attribute 直接拒绝。
    """
    # 全角括号与全角运算符在中文输入里很常见，先归一化，否则会以"语法错误"失败
    normalized = (expression
                  .replace("（", "(").replace("）", ")")
                  .replace("＋", "+").replace("－", "-")
                  .replace("×", "*").replace("÷", "/"))
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {expression!r}") from e

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"表达式包含不允许的语法: {type(node).__name__}")

    return evaluate(tree)


class CalculatorTool:
    name = "calculator"
    description = "对纯算术表达式求值，支持 + - * / // % ** 与括号。"
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "如 (12+8)*3"}},
        "required": ["expression"],
    }

    def run(self, expression: str = "", **_: Any) -> ToolResult:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression 必须是非空字符串")
        value = safe_eval(expression)
        rendered = int(value) if value == int(value) else value
        return ToolResult(text=f"{expression} = {rendered}", ok=True)


class NowTool:
    name = "now"
    description = "返回当前时间（ISO 8601）。"
    parameters = {"type": "object", "properties": {}}

    def run(self, **_: Any) -> ToolResult:
        from datetime import datetime

        return ToolResult(text=datetime.now().isoformat(timespec="seconds"), ok=True)


class ToolRegistry:
    """工具的注册、描述渲染与调用（含超时与错误回灌）。"""

    def __init__(self, tools: list[Tool] | None = None, timeout_seconds: float = 10.0) -> None:
        self._tools: dict[str, Tool] = {}
        self.timeout_seconds = timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agentkit-tool")
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> str:
        """渲染给模型看的工具清单（名称 + 说明 + 参数）。"""
        lines = []
        for name in self.names():
            tool = self._tools[name]
            params = ", ".join(
                f"{key}:{value.get('type', 'any')}"
                for key, value in (tool.parameters.get("properties") or {}).items()
            )
            lines.append(f"- {name}({params}): {tool.description}")
        return "\n".join(lines) if lines else "（无可用工具）"

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """永不抛异常：失败一律包成 ok=False 的结果，交给模型自我纠正。"""
        arguments = dict(arguments or {})
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(text=f"工具 {name} 不存在。可用工具：{', '.join(self.names())}", ok=False)

        future = self._executor.submit(tool.run, **arguments)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeout:
            future.cancel()
            # 超时是 Agent 最常见的卡死原因：必须让循环能继续，并明确告知模型
            return ToolResult(text=f"工具 {name} 超时（>{self.timeout_seconds}s）", ok=False)
        except TypeError as e:
            return ToolResult(text=f"工具 {name} 参数不合法：{e}", ok=False)
        except Exception as e:  # noqa: BLE001 — 工具的错误必须被兜住，不能中断问答
            return ToolResult(text=f"工具 {name} 执行失败：{e}", ok=False)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def build_registry(retriever: Retriever, timeout_seconds: float = 10.0) -> ToolRegistry:
    return ToolRegistry(
        [SearchKnowledgeBaseTool(retriever), CalculatorTool(), NowTool()],
        timeout_seconds=timeout_seconds,
    )


def render_context_observation(contexts: list[ContextItem]) -> str:
    """把工具返回的引用块重新渲染成规范的观察文本（编号与上下文一致）。"""
    return "\n".join(render_block(item.index, item.retrieved) for item in contexts)
