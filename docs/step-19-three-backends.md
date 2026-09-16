# step-19 · 三后端模型客户端与观测口径

## 目标

让同一套 Agent 代码能在三种环境下运行，且**换后端不换代码**：

- `stub`：全离线，断网可跑，作为效果基线；
- `direct`：直连任意 OpenAI 兼容端点（厂商 API / vLLM / Ollama）；
- `gateway`：走同族的自研网关 `llm-gateway`，复用它的配额、语义缓存、成本账本、traceId。

同时把"一次问答到底花了多少钱"算清楚 —— Agent 的调用次数是**乘数放大**的，
只统计最后一次调用会把成本低估到没有参考价值。

## 改动

### 本仓库

| 文件 | 职责 |
|---|---|
| `agentkit/llm.py` | `StubClient` / `OpenAiCompatibleClient` / `build_client` |
| `agentkit/transport.py` | 极简 HTTP 传输（urllib，零第三方依赖） |
| `agentkit/observability.py` | `RunReport` + Markdown 对比表 |
| `agentkit/cli.py` | `bench` 子命令：同一组问题跨后端对比 |

### 配套的 `llm-gateway`（补齐 OpenAI 兼容性）

这一步是前提：**网关之前只兼容请求侧，响应侧是扁平的**，
客户端必须写两套解析，"可切换后端"就只剩口号。

| 文件 | 改动 |
|---|---|
| `api/OpenAiProtocol.java`（新增） | 集中构造响应形状：`chatCompletionBody` / `chatCompletionChunk` / `embeddingsBody` / `modelEntry` / `statusOf` / `errorPayload` |
| `api/dto/EmbeddingsRequest.java`（新增） | `input` 兼容字符串与数组两种写法 |
| `api/OpenAiCompatController.java`（新增） | `GET /v1/models`、`POST /v1/embeddings` |
| `api/ChatController.java`（重写） | 非流式改 `choices[].message.content` + snake_case usage + `total_tokens`；流式改 OpenAI chunk 信封 + `[DONE]` 终止帧；新增 `X-Cache-Hit` 响应头 |
| `observability/GatewayMetrics.java` | 新增 `llmgw.embedding.requests` 计数 |
| `test/.../OpenAiProtocolTest.java`（新增 11 个用例） | 守住响应形状、状态码映射、embeddings 顺序 |

网关测试从 10 个增加到 **17 个，`mvn test` BUILD SUCCESS**。

## 关键代码

### 差异只有 base_url 与鉴权头

```python
@staticmethod
def for_backend(backend):
    if backend == "gateway":
        return Settings(backend="gateway", base_url="http://127.0.0.1:8080/v1",
                        api_key="agentkit", api_key_header="X-Api-Key",
                        api_key_prefix="")     # 网关认裸 key，不认 Bearer
    if backend == "direct":
        return Settings(backend="direct", base_url="https://api.openai.com/v1",
                        api_key_header="Authorization", api_key_prefix="Bearer ")
    return Settings()                          # stub
```

### 缓存命中放响应头，不放 body

```python
cache_hit=result.headers.get(CACHE_HIT_HEADER, "false").lower() == "true",
trace_id=result.headers.get(TRACE_ID_HEADER),
```

理由：body 要保持**纯 OpenAI 形状**，加自定义字段会破坏"任何 OpenAI 客户端都能直连"这个前提。
`X-Cache-Hit` / `X-Trace-Id` 是网关的额外信息，放头里不污染协议。

### 传输层显式禁用环境代理

```python
self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
```

本机普遍设着 `HTTP_PROXY=http://127.0.0.1:16540`，urllib 默认会把它套到**所有**请求上，
包括 `http://127.0.0.1:8080` 这样的本地地址。于是"连本地网关"变成"走代理再打回本机"，
表现为 `CONNECT tunnel failed`，而排查方向很容易跑偏到网关配置上。

### 观测：一次问答的全部调用

```python
def add(self, completion):
    self.calls += 1
    self.prompt_tokens += ...
    self.by_purpose[completion.purpose] = self.by_purpose.get(completion.purpose, 0) + 1
```

`by_purpose` 让"钱花在哪一步"可见：agent / rerank / summarize / embed / eval。

p95 用**最近秩法**而不是 `sorted[int(n*0.95)]` ——
后者在 n 较小时会直接返回最大值，把小样本的 p95 报成大数。
这个口径与 `llm-eval` 保持一致，两个项目的指标才可比。

## 验证方式

```bash
agentkit bench --backends stub --k 6
agentkit bench --backends stub,gateway --k 6     # 需要先起网关
pytest tests/test_llm_backends.py -q
cd ../llm-gateway && mvn test
```

## 结果

```
| 指标 | 后端 | 问题数 | 模型调用次数 | 次/问 | token/问 | 成本/问 | 缓存命中 | 命中率 | p95 延迟(ms) |
|---|---|---|---|---|---|---|---|---|---|
| stub（离线） | stub | 6 | 12 | 2.0 | 736.8 | 0.000000 | 0 | 0.00% | 0.0 |
```

`stub` 下一次问答是 **2 次调用**（1 次检索规划 + 1 次作答），
这个 2.0 是"最省"的下界：真实后端上重排、记忆压缩会各自再加调用。

网关侧 17 个用例通过，客户端侧响应解析、缓存头、traceId、embeddings 顺序各有测试守住。

## 踩了什么坑

1. **网关响应是扁平的**：`ChatResponse` 直接平铺字段、流式输出裸文本。
   客户端一开始写了两套解析分支，`--backend` 切换时行为不一致。
   改法不是在客户端兜，而是把网关的**响应侧**补齐 —— 兼容性该在被适配的一方修。

2. **`/v1/embeddings` 根本不存在**：网关只有内部 `ProviderAdapter.embed()` 供语义缓存用，
   没有对外端点。所以 `direct` 能跑、`gateway` 一入库就 404。

3. **环境代理劫持回环地址**（上面已说）。

4. **`bench` 里某个后端不可用会中断整轮对比**：现在把每个后端的初始化与执行都包在
   try 里，失败只打印并跳过，不让一个后端的配置问题吃掉整张对比表。

## 下一步

- 网关侧的 `X-Cache-Hit` 目前只有命中/未命中两态，可以再补一个"相似度分值"，
  让 Agent 能对"边界命中"做二次确认。
- `bench` 只有一个后端时表格没有对比价值，可以加一个"与上一次运行的差异"列。
