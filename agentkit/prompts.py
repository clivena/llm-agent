"""提示词模板与标记。

标记常量单独放在这里，是为了让离线桩（StubClient）能**解析真实提示词**，
而不是靠"另写一条捷径"。这样离线模式跑的是同一个 ReAct 循环、
同一套提示词，只是把决策者换成了一个确定的、很笨的模型 ——
循环代码零分支，离线验证才有意义。
"""

from __future__ import annotations

QUESTION_MARKER = "问题："
OBSERVATION_MARKER = "观察："
CANDIDATE_MARKER = "候选："
QUERY_MARKER = "查询："

AGENT_SYSTEM = """你是一个检索增强问答智能体。你只能依据工具返回的资料作答，不得编造。

可用工具：
{tools}

工作方式（ReAct）：
1. 需要资料时，先调用工具。严格输出一行 JSON：
   {{"tool": "<工具名>", "arguments": {{...}}}}
2. 拿到"观察"后，如果资料已足够，输出最终答案。严格输出一行 JSON：
   {{"answer": "<答案正文>", "used": [<用到的资料编号>]}}
3. 正文中的每一处事实后面必须紧跟引用编号，例如：网关用 Lua 保证原子扣减 [2]。
4. 如果观察里的资料不足以回答，直接说明"资料不足"，不要猜测。

只输出 JSON，不要输出解释、不要用 markdown 代码块包裹。"""

RERANK_SYSTEM = """你是一个检索结果重排器。给定一个查询和若干候选片段，
按"该片段对回答这个查询的有用程度"给每个候选打 0-10 分（10 分最有用）。
严格输出一行 JSON：{"scores": [<按候选顺序的分数>]}。不要输出其他内容。"""

SUMMARIZE_SYSTEM = """把下面的多轮对话压缩成一段要点摘要，保留用户提到的事实、约束与偏好，
丢弃寒暄与重复内容。只输出摘要正文。"""


def render_agent_user(question: str, observations: list[str]) -> str:
    """问题 + 累积的观察。没有观察时模型应当先调工具。"""
    parts = [f"{QUESTION_MARKER}{question}"]
    parts.extend(observations)
    return "\n\n".join(parts)


def render_observation(tool_name: str, blocks: list[str]) -> str:
    """观察文本里必须带**工具名**。

    早期实现只写"观察：检索到 N 个片段"，于是离线桩无从判断某个工具是否已经调用过，
    计算器类工具会被反复调用直到步数用尽 —— 表现为"帮算个数"直接返回
    "已达到最大推理步数"。带工具名同时也是真实 ReAct 轨迹的写法，
    模型能据此对齐"我上一步做了什么"。
    """
    body = "\n".join(blocks)
    return f"{OBSERVATION_MARKER}工具 {tool_name} 返回 {len(blocks)} 个片段\n{body}"


def render_candidates(query: str, candidates: list[str]) -> str:
    lines = [f"{QUERY_MARKER}{query}", f"{CANDIDATE_MARKER}"]
    for i, text in enumerate(candidates, start=1):
        lines.append(f"{i}. {text}")
    return "\n".join(lines)


def render_turns(turns: list[str]) -> str:
    return "\n".join(turns)
