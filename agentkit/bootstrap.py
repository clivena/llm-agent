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


def build_app(settings: Settings, rerank: str | None = None) -> App:
    resolved = settings.model_copy(update={"rerank": rerank}) if rerank else settings

    client = build_client(resolved)
    embedder = build_embedder(resolved, client)
    store = JsonlVectorStore.load(resolved.index_path)
    reranker = build_reranker(resolved.rerank, client)
    retriever = Retriever(store, embedder, resolved, reranker=reranker,
                          fallback=HeuristicReranker())
    registry = build_registry(retriever, resolved.timeout_seconds)
    memory = ConversationMemory(resolved, client)
    runtime = AgentRuntime(resolved, client, registry, memory)
    return App(settings=resolved, client=client, embedder=embedder, store=store,
               retriever=retriever, runtime=runtime)


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

    app.store.save()
    return {
        "files": len(files),
        "chunks": total_chunks,
        "total_in_index": len(app.store),
        "embedder": app.embedder.name,
        "index_path": str(app.settings.index_path),
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
