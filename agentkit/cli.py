"""命令行入口。

设计取向：每个子命令都对应链路里一个可独立验证的环节，
而不是只暴露一个"问答"黑盒 ——
  ingest  能不能正确入库、切了多少块
  ask     一次问答的答案、引用、推理轨迹与开销
  eval    检索质量（重排前后对比）
  bench   不同后端下的调用次数/成本/缓存命中对比
  serve   浏览器里现场演示
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentkit.bootstrap import build_app, ingest
from agentkit.config import Settings
from agentkit.evaluate import evaluate_retrieval, load_cases, render_report
from agentkit.observability import (
    BACKEND_LABELS,
    build_report,
    render_comparison,
)

app = typer.Typer(add_completion=False, help="agentkit — 可溯源的检索增强智能体")
console = Console()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = ROOT / "data" / "docs"
DEFAULT_CASES = ROOT / "data" / "eval" / "queries.json"


def _settings(backend: str, index: str | None, base_url: str | None = None,
              api_key: str | None = None, chat_model: str | None = None) -> Settings:
    settings = Settings.from_env(backend)  # type: ignore[arg-type]
    updates: dict[str, object] = {}
    if index:
        updates["index_path"] = index
    if base_url:
        updates["base_url"] = base_url
    if api_key is not None:
        updates["api_key"] = api_key
    if chat_model:
        updates["chat_model"] = chat_model
    return settings.model_copy(update=updates) if updates else settings


def _print_answer(answer) -> None:
    console.print(Panel(answer.text or "（空回答）", title="回答", border_style="blue"))

    if answer.citations:
        table = Table(title="引用（可反查原文）", show_lines=False)
        table.add_column("#", width=4, justify="right")
        table.add_column("来源")
        table.add_column("字符区间", width=12)
        table.add_column("原文片段", overflow="fold")
        for citation in answer.citations:
            table.add_row(f"[{citation.index}]", citation.source,
                          f"{citation.start}-{citation.end}", citation.snippet)
        console.print(table)
    else:
        console.print("[dim]本次回答未使用检索资料（纯计算类问题会走这条路）。[/dim]")

    if answer.unresolved_citations:
        console.print(f"[red]警告：正文引用了不存在的资料编号 {answer.unresolved_citations}[/red]")

    if answer.steps:
        table = Table(title="推理轨迹", show_lines=False)
        table.add_column("步", width=4, justify="right")
        table.add_column("类型", width=12)
        table.add_column("内容", overflow="fold")
        for step in answer.steps:
            table.add_row(str(step.step_index), step.kind, step.text[:220])
        console.print(table)

    usage = answer.usage
    breakdown = "，".join(f"{k}×{v}" for k, v in sorted(usage.by_purpose.items())) or "无"
    console.print(
        f"后端 [bold]{answer.backend}[/bold] · 调用 [bold]{usage.calls}[/bold] 次 "
        f"({breakdown}) · token {usage.total_tokens()} · 成本 ¥{usage.cost:.6f} · "
        f"缓存命中 {usage.cache_hits}"
    )
    if usage.trace_ids:
        console.print("traceId：" + ", ".join(usage.trace_ids))


@app.command("ingest")
def ingest_cmd(
    paths: list[str] = typer.Argument(None, help="文档或目录，默认 data/docs"),
    backend: str = typer.Option("stub", help="stub | direct | gateway"),
    index: str = typer.Option(None, help="索引文件路径"),
    reset: bool = typer.Option(False, "--reset", help="清空已有索引后重建"),
    base_url: str = typer.Option(None, help="覆盖 base_url"),
    api_key: str = typer.Option(None, help="覆盖 api_key"),
) -> None:
    """把文档切片、向量化并写入索引。"""
    settings = _settings(backend, index, base_url, api_key)
    instance = build_app(settings)
    try:
        summary = ingest(instance, paths or [str(DEFAULT_DOCS)], reset=reset)
    finally:
        instance.close()

    if summary["files"] == 0:
        console.print("[yellow]没有找到可入库的文本文件（.md/.txt）。[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title=f"入库结果（embedder={summary['embedder']}）")
    table.add_column("来源")
    table.add_column("切块数", justify="right")
    for row in summary["per_file"]:
        table.add_row(row["source"], str(row["chunks"]))
    console.print(table)
    console.print(f"文件 {summary['files']} 个 · 本次写入 {summary['chunks']} 块 · "
                  f"索引共 {summary['total_in_index']} 块 · 落盘 {summary['index_path']}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="要问的问题"),
    backend: str = typer.Option("stub", help="stub | direct | gateway"),
    rerank: str = typer.Option(None, help="heuristic | llm | off"),
    index: str = typer.Option(None, help="索引文件路径"),
    base_url: str = typer.Option(None, help="覆盖 base_url"),
    api_key: str = typer.Option(None, help="覆盖 api_key"),
    show_trace: bool = typer.Option(True, "--trace/--no-trace", help="是否打印推理轨迹"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出（便于脚本消费）"),
) -> None:
    """问一个问题，返回带引用的答案、推理轨迹与本次开销。"""
    settings = _settings(backend, index, base_url, api_key)
    instance = build_app(settings, rerank=rerank)
    try:
        if len(instance.store) == 0:
            console.print("[red]索引为空。先运行：agentkit ingest[/red]")
            raise typer.Exit(code=1)
        answer = instance.runtime.ask(question)
    finally:
        instance.close()

    if as_json:
        console.print_json(json.dumps({
            "answer": answer.text,
            "backend": answer.backend,
            "citations": [c.model_dump() for c in answer.citations],
            "unresolved_citations": answer.unresolved_citations,
            "usage": answer.usage.model_dump(exclude={"completions"}),
            "steps": [s.model_dump() for s in answer.steps] if show_trace else [],
        }, ensure_ascii=False))
        return
    if not show_trace:
        console.print(Panel(answer.text, title="回答", border_style="blue"))
        return
    _print_answer(answer)


@app.command()
def chat(
    backend: str = typer.Option("stub", help="stub | direct | gateway"),
    rerank: str = typer.Option(None, help="heuristic | llm | off"),
    index: str = typer.Option(None, help="索引文件路径"),
    base_url: str = typer.Option(None, help="覆盖 base_url"),
) -> None:
    """多轮对话（带滚动摘要记忆）。输入 exit 退出。"""
    settings = _settings(backend, index, base_url)
    instance = build_app(settings, rerank=rerank)
    console.print("[dim]多轮模式：历史超预算会自动压缩成摘要。输入 exit 退出。[/dim]")
    try:
        while True:
            question = console.input("[bold blue]你>[/bold blue] ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                break
            if not question:
                continue
            answer = instance.runtime.ask(question)
            console.print(Panel(answer.text, title="助手", border_style="green"))
            memory = instance.runtime.memory
            console.print(f"[dim]记忆：{len(memory.turns)} 轮 · 摘要压缩 {memory.compressions} 次 · "
                          f"估算 {memory.estimated_tokens()} token[/dim]")
    finally:
        instance.close()
    console.print("再见。")


@app.command("eval")
def eval_cmd(
    backend: str = typer.Option("stub", help="stub | direct | gateway"),
    k: int = typer.Option(4, help="取前 k 条计算指标"),
    rerank: str = typer.Option(None, help="heuristic | llm | off"),
    cases: str = typer.Option(None, help="评测集路径"),
    index: str = typer.Option(None, help="索引文件路径"),
) -> None:
    """检索评测：同一次检索结果，按召回位次与重排位次分别算指标。"""
    settings = _settings(backend, index)
    instance = build_app(settings, rerank=rerank)
    try:
        if len(instance.store) == 0:
            console.print("[red]索引为空。先运行：agentkit ingest[/red]")
            raise typer.Exit(code=1)
        loaded = load_cases(cases or DEFAULT_CASES)
        outcome = evaluate_retrieval(instance.retriever, loaded, k=k)
    finally:
        instance.close()

    console.print(render_report(outcome))
    console.print(f"\n重排使 recall@{k} 变化 {outcome.delta_recall():+.4f}，"
                  f"MRR 变化 {outcome.delta_mrr():+.4f}")


@app.command()
def bench(
    backends: str = typer.Option("stub", help="逗号分隔，如 stub,direct,gateway"),
    questions: str = typer.Option(None, help="问题集路径，默认取评测集里的 query"),
    rerank: str = typer.Option("heuristic", help="heuristic | llm | off"),
    index: str = typer.Option(None, help="索引文件路径"),
    base_url: str = typer.Option(None, help="覆盖 base_url（direct 模式常用）"),
    api_key: str = typer.Option(None, help="覆盖 api_key"),
    k: int = typer.Option(6, help="问题集截断到前 k 条，控制成本"),
) -> None:
    """跨后端基准：同一组问题跑不同后端，对比调用次数 / token / 成本 / 缓存命中。"""
    loaded = load_cases(questions or DEFAULT_CASES)[:k]
    reports = []
    for name in [b.strip() for b in backends.split(",") if b.strip()]:
        settings = _settings(name, index, base_url, api_key)
        try:
            instance = build_app(settings, rerank=rerank)
        except Exception as e:  # noqa: BLE001 — 某个后端不可用不应中断整轮对比
            console.print(f"[red]{name} 初始化失败：{e}[/red]")
            continue
        try:
            usages = []
            unresolved = 0
            for case in loaded:
                answer = instance.runtime.ask(case.query)
                usages.append(answer.usage)
                unresolved += len(answer.unresolved_citations)
            reports.append(build_report(BACKEND_LABELS.get(name, name), name, usages, unresolved))
            console.print(f"[dim]{name}：{len(loaded)} 个问题完成[/dim]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]{name} 执行失败：{e}[/red]")
        finally:
            instance.close()

    if not reports:
        console.print("[red]没有任何后端跑通。[/red]")
        raise typer.Exit(code=1)
    console.print()
    console.print(render_comparison(reports))


@app.command()
def serve(
    backend: str = typer.Option("stub", help="stub | direct | gateway"),
    rerank: str = typer.Option(None, help="heuristic | llm | off"),
    index: str = typer.Option(None, help="索引文件路径"),
    host: str = typer.Option("127.0.0.1", help="监听地址"),
    port: int = typer.Option(8765, help="监听端口"),
) -> None:
    """启动本地演示控制台（浏览器里直接提问）。"""
    settings = _settings(backend, index)
    instance = build_app(settings, rerank=rerank)
    if len(instance.store) == 0:
        console.print("[red]索引为空。先运行：agentkit ingest[/red]")
        raise typer.Exit(code=1)
    try:
        from agentkit.serve import serve as run_console

        run_console(instance.runtime, host=host, port=port)
    finally:
        instance.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
