"""装配：把各层拼成一个可用的 App。

单独抽出来的理由是**依赖方向**：CLI、评测、基准测试都要组装同一套东西，
如果各自 new 一遍，很容易出现"评测用的是启发式重排、线上用的是 LLM 重排"
这类不一致 —— 指标就没意义了。装配只有一处，参数从 Settings 和命令行来。

另外这里固定了一件事：无论用哪种重排，都常备一个启发式兜底，
这样 LLM 重排失败时是"降级"而不是"整条链路失败"。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentkit.chunker import chunk_document
from agentkit.config import Settings
from agentkit.embedder import Embedder
from agentkit.llm import LlmClient, build_client, build_embedder
from agentkit.memory import ConversationMemory
from agentkit.rerank import HeuristicReranker, build_reranker
from agentkit.retrieve import Retriever
from agentkit.runtime import AgentRuntime
from agentkit.store import JsonlVectorStore
from agentkit.tools import build_registry

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass
class App:
    settings: Settings
    client: LlmClient
    embedder: Embedder
    store: JsonlVectorStore
    retriever: Retriever
    runtime: AgentRuntime

    def close(self) -> None:
        self.runtime.close()


def build_app(settings: Settings, rerank: str | None = None,
              verify_embedder: bool = True) -> App:
    resolved = settings.model_copy(update={"rerank": rerank}) if rerank else settings

    client = build_client(resolved)
    embedder = build_embedder(resolved, client)
    store = JsonlVectorStore.load(resolved.index_path)
    # ingest 是唯一允许"换 embedder"的操作，所以它要跳过这道校验 ——
    # 否则索引和 embedder 一旦不一致，连重建索引的路都被自己堵死了。
    if verify_embedder:
        verify_embedder_match(store, embedder)
    reranker = build_reranker(resolved.rerank, client)
    retriever = Retriever(store, embedder, resolved, reranker=reranker,
                          fallback=HeuristicReranker())
    registry = build_registry(retriever, resolved.timeout_seconds)
    memory = ConversationMemory(resolved, client)
    runtime = AgentRuntime(resolved, client, registry, memory)
    return App(settings=resolved, client=client, embedder=embedder, store=store,
               retriever=retriever, runtime=runtime)


def verify_embedder_match(store: JsonlVectorStore, embedder: Embedder) -> None:
    """索引身份与当前 embedder 对账：**知道对不上就报错，不知道就不拦。**

    为什么必须是硬错误而不是警告：跨向量空间比余弦**不会抛出任何异常**，
    它只会返回一个没有意义但"看起来有分"的数，然后检索结果整体错乱、
    下游所有指标一起失真。这类错误一旦放过，排查成本极高。

    老索引没有元数据（本项目加这个字段之前的产物）时无从判断，
    不拦 —— 此时由 `VectorStore.search` 的维度硬校验兜底。
    """
    recorded = store.embedder_kind
    if not recorded or recorded == embedder.kind:
        return
    raise RuntimeError(
        f"索引是用 `{store.embedder_name}`（{embedder_kind_label(recorded)}）建的，"
        f"当前 embedder 是 `{embedder.name}`（{embedder_kind_label(embedder.kind)}）："
        "两者不在同一向量空间，检索出来的分数没有意义。\n"
        f"  继续用这个索引：设 AGENTKIT_EMBEDDER={recorded}\n"
        "  换 embedder    ：agentkit ingest --reset 重建整个索引"
    )


def embedder_kind_label(kind: str) -> str:
    return {"hashing": "本地哈希向量", "client": "远程 embedding"}.get(kind, kind)


def iter_source_files(paths: list[str | Path]) -> list[Path]:
    """展开目录，只收文本类文件。排序保证 ingest 可复现（同样的输入得到同样的索引）。"""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES)
        elif path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return sorted({p.resolve() for p in files})


def ingest(app: App, paths: list[str | Path], reset: bool = False,
           source_root: str | Path | None = None) -> dict:
    """把文档切片、向量化并写入索引。返回一份可打印的摘要。"""
    files = iter_source_files(paths)
    if not files:
        return {"files": 0, "chunks": 0, "sources": []}

    if reset:
        app.store.clear()

    root = Path(source_root).resolve() if source_root else _common_root(files)
    total_chunks = 0
    per_file: list[dict] = []

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        source = _relative(path, root)
        document = chunk_document(source, text, app.settings.chunk_size,
                                  app.settings.chunk_overlap)
        # 重新 ingest 同一文件时先删旧块：否则改了文档会出现新旧两个版本同时被召回
        app.store.drop_sources([source])
        if not document.chunks:
            per_file.append({"source": source, "chunks": 0})
            continue
        vectors = app.embedder.embed([c.text for c in document.chunks])
        app.store.add(document.chunks, vectors)
        total_chunks += len(document.chunks)
        per_file.append({"source": source, "chunks": len(document.chunks)})

    # 落盘前把"我是谁建的"写进 store，save() 会连同 meta 一起写出去。
    # 这一步是索引能安全复用的前提：下次加载时才知道该用哪个 embedder 来查询。
    app.store.embedder_name = app.embedder.name
    app.store.embedder_kind = app.embedder.kind
    app.store.save()
    return {
        "files": len(files),
        "chunks": total_chunks,
        "total_in_index": len(app.store),
        "embedder": app.embedder.name,
        "embedder_kind": app.embedder.kind,
        "embedder_dim": app.store.vector_dim(),
        "index_path": str(app.settings.index_path),
        "meta_path": str(app.store.meta_path),
        "per_file": per_file,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _common_root(files: list[Path]) -> Path:
    if not files:
        return Path.cwd()
    common = Path(files[0]).parent
    for path in files[1:]:
        while common not in path.parents and common != path.parent:
            if common.parent == common:
                break
            common = common.parent
    return common
