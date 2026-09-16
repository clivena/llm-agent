# agentkit 实现文档

按步骤记录了这个检索增强智能体从切块到三后端的实现过程。
每篇固定结构：**目标 / 改动 / 关键代码 / 验证方式 / 结果 / 踩了什么坑 / 下一步**。

| 步骤 | 内容 | 文档 |
|---|---|---|
| 16 | 章节感知切块与索引落盘 | [step-16](./step-16-chunking-and-index.md) |
| 17 | 语义 + IDF 词法融合重排 | [step-17](./step-17-hybrid-rerank.md) |
| 18 | ReAct 运行时、工具容错与引用验证 | [step-18](./step-18-react-runtime.md) |
| 19 | 三后端模型客户端与观测口径 | [step-19](./step-19-three-backends.md) |
| 20 | 真机验证记录（pytest / ingest / ask / eval / bench） | [step-20](./step-20-verification.md) |

编号从 16 开始：01~06 属于配套的 `llm-gateway`，07~10 属于 `llm-eval`，
11~15 是网关与评测的迭代与修复记录。三个仓库原是同一条工程链路上的不同阶段：
**接入层（网关）→ 质量层（评测）→ 应用层（本仓库）**。

如果只看一篇，推荐 [step-20 验证记录](./step-20-verification.md)：
它把 `pytest`、`ingest`、`ask`、`eval` 的原始输出和解读都留了下来，
包括"为什么切块一改，召回侧 MRR 从 0.76 跳到 0.88"、
以及"为什么没有 API Key 就不给 LLM 重排的效果数字"。
