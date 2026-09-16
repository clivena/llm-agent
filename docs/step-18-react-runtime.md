# step-18 · ReAct 运行时、工具容错与引用验证

## 目标

把检索能力装进一个能**自己决定要不要调工具、调用失败也不会崩**的循环里，
并且让最终答案里的每一处事实都能反查回原文。

## 改动

| 文件 | 职责 |
|---|---|
| `agentkit/runtime.py` | ReAct 循环、引用全局重编号、严格引用校验 |
| `agentkit/tools.py` | 工具注册表（超时 / 错误回灌 / 永不抛异常）、`safe_eval` |
| `agentkit/memory.py` | 滚动摘要记忆 |
| `agentkit/prompts.py` | 提示词模板与标记常量 |

## 关键代码

### 循环本身很短，难点全在"出错时怎么办"

```python
for step_index in range(1, limit + 1):
    completion = self.client.complete(messages, purpose="agent")
    payload = parse_action(completion.text)

    if payload is None:                 # 模型输出不是 JSON
        observations.append(FORMAT_HINT)  # 回灌格式纠正，继续循环
        continue

    if "tool" in payload:
        self._execute_tool(...)         # 失败也回灌，不抛穿
        continue

    if "answer" in payload:
        return self._finalize(...)
```

步数用尽时**再要一次答案**，并且明确禁止调工具：

```python
messages = agent_messages(question, observations + [BUDGET_HINT],
                          NO_TOOLS_HINT, memory_block)
```

直接把"我没想出来"丢给用户是最差的一档，能兜就必须兜。

### 引用必须全局重编号

```python
def _renumber(self, contexts, pool):
    out = []
    for item in contexts:
        index = len(pool) + 1
        renumbered = ContextItem(index=index, retrieved=item.retrieved)
        pool[index] = renumbered
        out.append(renumbered)
    return out
```

检索工具可能被调用多次，每次的 `[1]` 指向不同片段。
不做重编号就会出现"同一编号指向两个来源"，引用直接失效。

### 编引用要单独暴露，而不是悄悄删掉

```python
unresolved = sorted(index for index in wanted if index not in pool)
if strict_citations and unresolved:
    raise CitationError(f"答案引用了不存在的资料编号: {unresolved}")
```

正文写了 `[7]` 但检索池里没有第 7 条 —— 这说明它在编引用，
比编事实更隐蔽，因为看起来"有出处"。如实记进 `unresolved_citations`。

### 工具层三条约束

```python
fut = self._executor.submit(tool.run, **arguments)
try:
    return fut.result(timeout=self.timeout_seconds)
except FutureTimeout:
    future.cancel()
    return ToolResult(text=f"工具 {name} 超时（>{self.timeout_seconds}s）", ok=False)
except TypeError as e:
    return ToolResult(text=f"工具 {name} 参数不合法：{e}", ok=False)
except Exception as e:
    return ToolResult(text=f"工具 {name} 执行失败：{e}", ok=False)
```

`calc` 用 ast 白名单而不是 `eval()`：Agent 的输入来自模型，等于用户可控字符串。

### 记忆：超预算才压缩，压缩失败不丢历史

```python
if not self.needs_compression():
    return False        # 节流：每轮都摘一次，成本反而更高
...
except Exception:
    compressed = older_text[: self.settings.memory_token_budget]
    self.last_compression_degraded = True   # 降级但留痕
```

## 验证方式

```bash
agentkit ask "工具执行失败会不会中断整轮问答？"
agentkit chat     # 多轮，观察"记忆：N 轮 · 摘要压缩 M 次"
pytest tests/test_runtime_tools_memory.py -q
```

## 结果

离线（stub）跑 `/工具执行失败会不会中断整轮问答？`：

```
步 1 action       调用工具 search_knowledge_base
步 1 observation  观察：工具 search_knowledge_base 返回 4 个片段
                  [1] 来源：05-agent-orchestration.md §工具调用与容错
步 2 answer       工具失败不会中断问答。参数写错、工具内部异常、执行超时，
                  这三种情况都会把失败原因作为观察结果回灌给模型，
                  让它在下一步自我纠正。
                  超时是必须的。智能体卡死的大多数原因不是模型在想，
                  而是某个工具没有返回。超时后返回一个明确的失败观察，
                  保证请求一定会结束。 [1]

后端 stub · 调用 2 次 (agent×2) · token 537 · 成本 ¥0.000000 · 缓存命中 0
```

引用 `[1]` 落在 `05-agent-orchestration.md` 的 `683-821` 字符区间，
正好是 §工具调用与容错 这一节。纯计算类问题（`帮我算 (128+64)*3`）
走 `calculator` 工具，不产生引用，也不白检索一次。

## 踩了什么坑

1. **用量统计绑错实例**：`AgentRuntime.__init__` 里注册 `self._stats.add`，
   而 `ask()` 每次都新建 `_stats` —— 于是所有调用都记进了第一次问答的对象里，
   表现为"跑了 2 次调用，输出却写着 0 次"。改注册 `self._record` 解决。

2. **计算器被反复调用直到步数用尽**：观察文本里没带工具名，
   离线桩无从判断"这个工具已经调用过"，于是每轮都重新调一次。
   现在 `render_observation(tool_name, blocks)` 一定带工具名。

3. **算术正则没有捕获组**：`_ARITHMETIC` 缺分组导致 `group(1)` 抛 `IndexError`；
   顺便补了全角括号与全角运算符（`（12＋8）×2`）的归一化。

4. **答案解析把引用标记粘到下一句**：`"".join(parts)` 会把
   `"...结束。 [1]"` 和下一段的 `"网关处理一次..."` 粘成 `"[1]网关处理一次..."`，
   读起来像引用编号出现在句首。改成换行拼接。

## 下一步

- 工具集只有 3 个，且都是无副作用的查询类。真正体现 Agent 价值的是
  **有副作用、需要幂等与确认**的工具（写单、发消息），那才是权限与审计的战场。
- 收尾调用目前只看"步数"，可以再加一条"连续两轮观察无新增信息"的早停条件。
