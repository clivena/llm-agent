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
| `agentkit/llm.py` | `StubClient` / `OpenAiCompatibleClient` / `build_client` / `build_embedder` |
| `agentkit/transport.py` | 极简 HTTP 传输（urllib，零第三方依赖） |
| `agentkit/observability.py` | `RunReport` + Markdown 对比表 |
| `agentkit/cli.py` | `bench` 子命令：同一组问题跨后端对比 |

### embedder 与 backend 解耦（后续补做，见"已知边界"第 1 条）

拆掉 `backend == "stub" ? hashing : client` 这个错误耦合：

| 文件 | 改动 |
|---|---|
| `agentkit/config.py` | 新增 `embedder` 配置项（`auto`/`hashing`/`client`，环境变量 `AGENTKIT_EMBEDDER`）；订正文档里"网关补齐了 `/v1/embeddings`"这句失实描述 |
| `agentkit/llm.py` | `resolve_embedder_choice()` + `build_embedder()` 改按 `embedder` 配置选，不再看 backend |
| `agentkit/store.py` | 索引身份落盘 `out/index.meta.json`（`embedder`/`kind`/`dim`/`chunks`）；`cosine` 与 `search` 加**维度硬校验**，去掉原来 `min(len(a),len(b))` 的静默截断 |
| `agentkit/embedder.py` | 各 embedder 统一带 `kind` 与 `dim`，供索引对账 |
| `agentkit/bootstrap.py` | `verify_embedder_match()`：加载索引时对账，不一致**报错**并给出两条出路；`ingest` 写入索引身份 |
| `agentkit/cli.py` | `ingest`/`eval`/`bench`/`serve`/`ask`/`chat` 新增 `--embedder` |

### 配套的 `llm-gateway`（补齐 OpenAI 兼容性）

这一步是前提：**网关之前只兼容请求侧，响应侧是扁平的**
（`{id, model, content, usage:{promptTokens, completionTokens}, cacheHit}`），
客户端必须写两套解析，"可切换后端"就只剩口号。

更麻烦的是它**半兼容**：错误响应已经是 OpenAI 形状（`{"error":{...}}`），
成功响应却不是 —— 客户端会以为握手成功了，直到解析 `choices` 才炸。

> ⚠️ **修订说明（2026-10-15）**：下面这份"前提"最初只是**设计意图**，网关代码当时**一行都没改**。
> 一次真实联调里 `agentkit bench --backends gateway` 报 `响应里没有 choices`，
> 而网关侧其实已经完整跑通（真实上游调用、真实 usage、账本也落了账），
> 才暴露出"文档写了、代码没做"这个问题（当时网关仓库 `git status` 是干净的，是硬证据）。
> 本节表格已按**实际落地情况**重写；未落地的部分单独列在"已知边界"里，不再写成已完成。

| 文件 | 实际改动 |
|---|---|
| `api/OpenAiProtocol.java`（新增） | 集中构造响应形状：`chatCompletionBody` / `chatCompletionChunk` / `modelsBody` / `modelEntry` / `statusOf` / `errorPayload` / `DONE` |
| `api/OpenAiCompatController.java`（新增） | **只有** `GET /v1/models`；`POST /v1/embeddings` **未实现**（见"已知边界"） |
| `api/ChatController.java`（重写） | 非流式改 `choices[].message.content` + snake_case usage + `total_tokens`；流式改 OpenAI chunk 信封 + `[DONE]` 终止帧；新增 `X-Cache-Hit` 响应头；异常映射与错误体收归 `OpenAiProtocol` |
| `scripts/run-gateway-smoke.py` | 冒烟脚本同步改读新形状。**这一步不能漏**：它原来读 `content` / `promptTokens`，改完不会报错、只会静默拿到 `None` |
| `test/.../OpenAiProtocolTest.java`（新增 14 个用例） | 断言**序列化后的 JSON**（契约是线上字节，不是 Map）：choices 结构、usage 键名、`cacheHit` 不得进 body、状态码映射、`/v1/models` 形状 |

网关测试从 10 个增加到 **20 个，`mvn test` BUILD SUCCESS**。
原 `ChatControllerStatusTest` 的 4 个状态码用例已被 `OpenAiProtocolTest` 完整覆盖，合并删除。

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

### embedder 与 backend 是两件事，别绑

```python
# ✗ 旧实现：把"谁向量化"绑到了"谁生成"，跨向量空间比余弦还不报错
return HashingEmbedder() if settings.backend == "stub" else ClientEmbedder(client, ...)

# ✓ 现在：由独立配置项决定，auto 的含义是"沿用索引自己记录的那个 embedder"
def resolve_embedder_choice(settings) -> str:
    if settings.embedder in ("hashing", "client"):
        return settings.embedder
    kind = (read_index_meta(settings.index_path) or {}).get("kind")
    return kind if kind in ("hashing", "client") else "hashing"
```

对账失败时报错，并且**必须给出出路**，否则用户只知道"不行"、不知道"怎么办"：

```
索引是用 `hashing`（本地哈希向量）建的，当前 embedder 是 `client:text-embedding-3-small`
（远程 embedding）：两者不在同一向量空间，检索出来的分数没有意义。
  继续用这个索引：设 AGENTKIT_EMBEDDER=hashing
  换 embedder    ：agentkit ingest --reset 重建整个索引
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
# 索引身份：先建库，会打印 kind / 维度 / 元数据路径
agentkit ingest                       # 输出："索引身份：kind=hashing · 维度=512 · 元数据=out\index.meta.json"

agentkit bench --backends stub --k 6
# gateway 后端必须显式给模型名：默认的 gpt-4o-mini 网关没配适配器，会 502。
# （网关侧要先起：见 ../llm-gateway/docs/step-12-live-integration.md）
AGENTKIT_CHAT_MODEL=deepseek-flash agentkit bench --backends gateway --k 6

# embedder 解耦的自查：索引是 hashing 建的，强制 client 应当报错（而不是悄悄跨空间比）
AGENTKIT_EMBEDDER=client agentkit ask "配额怎么保证不超发"   # 期望：RuntimeError + 两条出路

pytest -q                             # 143 个用例
cd ../llm-gateway && mvn test         # 20 个用例
```

网关响应形状可以不用客户端、直接看一眼：

```bash
curl -s --noproxy "*" -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "X-Api-Key: probe" \
  -d '{"model":"deepseek-flash","messages":[{"role":"user","content":"你好"}]}' \
  | python -c "import sys,json;d=json.load(sys.stdin);print(list(d.keys()));print(d['usage'])"
# 期望: ['id','object','created','model','choices','usage']
#       {'prompt_tokens': …, 'completion_tokens': …, 'total_tokens': …}
curl -s --noproxy "*" http://127.0.0.1:8080/v1/models
```

## 结果

补齐网关响应形状（`choices[].message.content` + snake_case usage）**并**把 embedder 从 backend 解耦之后，
三个后端都能真正跑完 —— 也就是说，`gateway` 这一行**包含真实的检索**：

| 指标 | 后端 | 问题数 | 模型调用次数 | 次/问 | token/问 | 成本/问 | 缓存命中率 | p95 延迟(ms) |
|---|---|---|---|---|---|---|---|---|
| stub（离线） | stub | 6 | 12 | 2.00 | 800.8 | 0.000000 | 0.00% | 0.0 |
| direct（直连 DeepSeek） | direct | 6 | 13 | 2.17 | 1201.0 | 0.000825 | 0.00% | 2,078 |
| gateway（经自研网关） | gateway | 6 | 13 | 2.17 | **2779.3** | 0.001806 | 0.00% | 3,158 |

`stub` 下一次问答是 **2 次调用**（1 次检索规划 + 1 次作答），
这个 2.0 是"最省"的下界：真实后端上重排、记忆压缩会各自再加调用。

**为什么 gateway 的 token/问 是 direct 的 2.3 倍**（2779 vs 1201）：
这两个数字不是同一个东西了。`direct` 那一次是在 embedder 还没解耦时跑的 ——
检索工具 404 失败、没有任何片段进入提示词，所以 token 少是因为**它压根没检索**。
gateway 这一行注入 5 个片段之后才是"真正做过 RAG"的量级。
两个数字放在一起看恰好是一个很好的对照：**少了检索，成本能便宜一半，但那不是优化，是功能没跑。**

> `direct` 那一行是历史数据，保留原值以便对照；要拿到可比的 direct 数字，
> 需要给 `AGENTKIT_BASE_URL` / `AGENTKIT_API_KEY` 并单独跑（bench 的 `--base-url`
> 会同时作用于所有后端，所以 `direct` 与 `gateway` 不能在同一条命令里混跑）。

网关侧 20 个用例通过（含 14 个协议形状用例）；客户端侧 143 个用例通过
（含协议解析、缓存头、traceId、embedder 解耦与索引对账）。

## 踩了什么坑

1. **网关响应是扁平的**：`ChatResponse` 直接平铺字段、流式输出裸文本。
   客户端一开始写了两套解析分支，`--backend` 切换时行为不一致。
   改法不是在客户端兜，而是把网关的**响应侧**补齐 —— 兼容性该在被适配的一方修。

2. **"文档写了、代码没做"是最难发现的一类问题**。
   本节表格早先写的是"网关已新增 `OpenAiProtocol` / `OpenAiCompatController`、
   测试 10→17、`mvn test` SUCCESS"，但网关仓库 `git status` 干净、`find` 找不到那几个文件、
   测试实际是 10 个。**文档描述的是一份从未落地的改造。**
   教训：文件的"存在性"必须用命令核（`find` / `git status` / 实际跑 `mvn test`），
   不能凭文档或记忆；写完文档要顺手验一遍它声称的东西是否真的在。

3. **改协议形状必须全局搜"旧字段名的读取点"**。
   网关改完之后，仓库自带的 `scripts/run-gateway-smoke.py` 还在读 `data["content"]`
   和 `usage["promptTokens"]` —— 这种代码**不会报错**，只会静默拿到 `None`，
   然后把日志打成一片空白。比直接崩掉更难排查，因为看上去"跑通了"。

4. **`/v1/embeddings` 根本不存在**：网关只有内部 `ProviderAdapter.embed()` 供语义缓存用，
   没有对外端点。所以 `direct` 能跑、`gateway` 一入库就 404。

5. **环境代理劫持回环地址**（上面已说）。

6. **`bench` 里某个后端不可用会中断整轮对比**：现在把每个后端的初始化与执行都包在
   try 里，失败只打印并跳过，不让一个后端的配置问题吃掉整张对比表。

## 已知边界（这一版没做、也没打算假装做了）

1. ✅ **【已修】`build_embedder` 曾把"谁向量化"和"谁生成"绑在一起 —— 这是本步最深的坑。**
   修之前的实现是 `backend == "stub" ? HashingEmbedder : ClientEmbedder`。
   这两件事本没有任何关系，绑错的后果是**静默出错**：

   - 仓库里索引是用 `HashingEmbedder` 建的，向量是 **512 维**；
   - 远程 embedding 是**另一个维度、另一个向量空间**；
   - `cosine` 当时写的是 `n = min(len(a), len(b))`，于是跨维度比较会**悄悄截断**、
     返回一个"看起来有分"的数 —— 比直接失败难发现得多；
   - 而在没有 embeddings 端点的环境里（本项目就是），`gateway` / `direct` 后端的
     `search_knowledge_base` **直接 404 失败**，而 `registry.call` 把它吞成 `ok=False`，
     模型照着"资料不足"作答 —— **整轮 bench 跑得完、指标全是假的**（实测引用数为 0）。

   **修法（已落地）**：
   - embedder 由独立配置项 `Settings.embedder` 决定（`auto | hashing | client`），与 backend 解耦；
   - `auto` 的含义是**沿用索引自己记录的那个 embedder**；
   - `ingest` 把索引身份（`embedder` / `kind` / `dim` / `chunks`）写进 `out/index.meta.json`；
   - `build_app` 加载时对账：身份不一致就**报错**，并给出两条出路
     （`AGENTKIT_EMBEDDER=<原种类>` 或 `agentkit ingest --reset` 整库重建）；
   - `VectorStore.search` 与 `cosine` 加了**维度硬校验**，杜绝跨空间硬算（老索引没元数据时的兜底）；
   - 新增 `--embedder` 选项（`ingest` / `eval` / `bench` / `serve` / `ask` / `chat`）。

   修完实测：`gateway` 后端的 `search_knowledge_base` 从 `ok=False` 变成 `ok=True`、
   **引用数 2**、答案正确引到 `02-gateway-and-quota.md`；
   token/问 从 1148.7 涨到 **2779.3**（这才是真正把 5 个片段注入提示词之后的量）。
   客户端测试 143 个全绿，其中新增了"解耦三条语义"与"对账必须报错"的守门用例。

   **仍然存在的边界**：`client` 这条路上游必须真有 embeddings 端点。
   本项目用的 DeepSeek 没有，所以 `client` 目前**只能验证"会正确地报错"**，
   不能验证"远程向量检索效果更好"。那是数据/厂商层面的缺口，不是代码缺口。

2. **`POST /v1/embeddings` 未实现。** 上游 DeepSeek 没有 embeddings 端点（实测 HTTP 404），
   转发出去只会是一个必然失败的端点。语义缓存因此仍在配置层关着。
   要真正支持，得先决定 embedding 从哪来（换厂商 / 网关本地提供哈希向量），
   那是产品决策而不是协议适配。

3. **`gateway` 后端的默认 `chat_model` 是 `gpt-4o-mini`**，而网关只配了
   `deepseek-flash` / `deepseek-v4-pro`，于是默认配置下第一句话就 502：
   `no adapter for model gpt-4o-mini`。跑之前必须
   `AGENTKIT_CHAT_MODEL=deepseek-flash agentkit bench --backends gateway`。
   客户端默认值不可能预知网关配了哪些模型，所以更合理的做法是让客户端发 `auto`
   由网关侧路由 —— 但 `RouterService` 目前对 `auto` 也会挑到没有适配器的模型，
   这条要连同路由策略一起修。

4. **流式分片里的 `model` 只能是"请求的"模型名。** 级联升级后实际服务的是更强模型，
   但 `ChatPipeline.executeStream` 目前只吐文本、不吐实际模型名，流式路径拿不到真值。
   要修得先改 `executeStream` 的返回契约（改吐 `(model, chunk)` 结构），属另一个改动。
   账本里记的是**真实**服务模型，不受此影响。
5. **流式路径不发 `X-Cache-Hit`。** 缓存的判定发生在流订阅之后，
   而响应头在首次写出时就冻结了，所以流式拿不到命中信息（非流式有）。
6. **`/admin/*` 系列（配额、账本、单价）仍不是 OpenAI 协议的一部分**，
   它们是网关自有运维接口，不打算兼容任何外部协议。

## 下一步

- 网关侧的 `X-Cache-Hit` 目前只有命中/未命中两态，可以再补一个"相似度分值"，
  让 Agent 能对"边界命中"做二次确认。
- `bench` 只有一个后端时表格没有对比价值，可以加一个"与上一次运行的差异"列。
