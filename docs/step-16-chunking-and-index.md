# step-16 · 章节感知切块与索引落盘

## 目标

把 markdown 语料切成"每个块只讲一个话题、且能精确定位回原文"的片段。
切块是 RAG 里最容易被忽视、又最先决定上限的一步 —— 后面的重排再强，
也救不回"块里混了三个话题"。

## 改动

新增三个模块：

| 文件 | 职责 |
|---|---|
| `agentkit/chunker.py` | 归一化 → 章节切分 → 章节内滑窗 → 句末吸附 |
| `agentkit/embedder.py` | 离线哈希向量 / 远程 embedding（带缓存） |
| `agentkit/store.py` | JSONL 落盘 + 内存暴力余弦；按 `chunk.id` upsert |

## 关键代码

### 章节是硬边界

```python
for raw_start, raw_end in _split_sections(text):
    seg_start = _trim_left(text, max(raw_start, content_start), content_end)
    seg_end = _trim_right(text, seg_start, min(raw_end, content_end))
    start = seg_start
    while start < seg_end:
        end = min(start + size, seg_end)
        if end < seg_end:
            end = _snap(end, start, size, breaks)   # 吸附到句末，不切在句中
        ...
        spans.append((start, end, _section_at(headings, start)))
        nxt = _advance(text, start, end, overlap, breaks, seg_end)
        if nxt <= start:      # 保险：起点必须严格前进
            break
        start = nxt
```

`_split_sections` 会把"只有标题没有正文"的段（文档开头的 H1）并进相邻章节，
避免产生 7 个字的垃圾块。

### 块起点必须落在句末之后

`_advance` 三级回退，越往后越宽松：

1. 块尾两倍重叠长度内的句末断点 —— 既保住重叠，又不至于让新块几乎没推进；
2. 块内任意句末断点 —— 长句段落里窗口内可能一个断点都没有；
3. 直接取 `end - overlap` —— 整块都没有句末标点（表格式文本）才会走到这里。

### 硬折行必须先还原

```python
def normalize_text(text: str) -> str:
    for para in re.compile(r"\n\s*\n").split(text.strip()):
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        joined = lines[0]
        for line in lines[1:]:
            joined += line if (_is_cjk(joined[-1]) or _is_cjk(line[0])) else " " + line
        out.append(joined)
```

`Document.text` 保存的是**归一化后**的文本，这样块的区间、`Citation.verify` 的校验、
页面上的高亮三者才指向同一份内容。

## 验证方式

```bash
python scripts/dump_chunks.py 05-agent-orchestration.md
python scripts/dump_chunks.py 08-deployment.md
pytest tests/test_chunker.py -q
```

`dump_chunks.py` 会打印每个块的 `[start-end]`、所属章节、前文 12 字与块首 24 字 ——
块是不是从半句开始，一眼就能看出来。

## 结果

`05-agent-orchestration.md` 切成 6 块，每块恰好一个章节，块首都是标题行：

```
#0 [   0- 261] §智能体编排     head=# 智能体编排\n\n## 检索链路\n\n文档先切块再
#1 [ 263- 525] §重排序        head=## 重排序\n\n向量召回按整体语义接近排序，而实
#2 [ 527- 681] §引用溯源      head=## 引用溯源\n\n答案正文里的每个引用编号都必须
#3 [ 683- 821] §工具调用与容错  head=## 工具调用与容错\n\n工具失败不会中断问答。参
#4 [ 823- 965] §记忆管理      head=## 记忆管理\n\n多轮对话的历史不能无限制地塞进
#5 [ 967-1066] §离线可运行     head=## 离线可运行\n\n整条链路支持完全离线运行：本
```

15 篇文档 / 1066~2200 字不等，共切出 66 块。

## 踩了什么坑

1. **块横跨章节** → 向量把两三个话题平均掉，召回区分度下降。
   这是本项目收益最大的一处改动，用 `scripts/ablation_chunking.py` 做了 2×2 消融验证：

   | A 归一化 | B 章节边界 | 块数 | k=4 MRR | k=1 recall |
   |---|---|---|---|---|
   | 是 | 是 | 66 | **0.8849** | **0.8571** |
   | 是 | 否 | 48 | 0.7460 | 0.5714 |
   | 否 | 是 | 66 | **0.8849** | **0.8571** |
   | 否 | 否 | 47 | 0.7619 | 0.6190 |

   章节边界贡献了几乎全部收益（k=4 MRR +14pt，k=1 recall +29pt）。

2. **硬折行被当成段落边界** → 切出以半句开头的块，
   答案里出现"而不是继续使用未重排的结果冒充已完成"这种没头没尾的开头，
   看起来非常像模型在胡说，排查时却会去怀疑模型。

   但要说清楚：**在这个语料上它对检索指标测不出差异**（上表第一、三行完全相同）——
   因为所有章节都短于 320 字的窗口，一个章节刚好一块，折行不影响块边界。
   它真正起作用的地方是"章节长于窗口"时的块起点，以及引用片段的显示文本。
   把这条如实写下来，比假装它也有收益更有用。

3. **章节尾部空白没裁掉** → `end >= seg_end` 永远不成立，
   循环每次只前进 1~2 个字符，跑出 **178 个碎片块**。
   现在有 `test_chunk_text_terminates_on_fragmented_input` 守着。

## 下一步

切块粒度固定（320 字 / 64 重叠）。真正该做的是**按内容自适应**：
标题层级深的文档切细，口语化长文切粗。这需要先有一批评测数据支撑，
否则只是把参数量从 2 个变成 3 个。
