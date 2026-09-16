"""本地演示控制台。

刻意保持极简：标准库 http.server + 单页 HTML，没有前端框架、没有构建步骤、
没有鉴权。它存在的意义只有一个 —— 让人在浏览器里直接提问，而不是听描述。

这也是本项目里唯一一处"为了演示而写"的代码，所以边界写清楚：
  只监听 127.0.0.1；只读不写；不做会话持久化（每次请求独立）；
  故意不暴露工具配置接口，避免变成需要维护的后台。
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agentkit.runtime import AgentRuntime

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agentkit 演示控制台</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 32px 20px; background: #f7f7f5; color: #2c2c2a;
         font: 14px/1.65 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
  main { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 18px; font-weight: 500; margin: 0 0 4px; }
  p.sub { color: #5f5e5a; margin: 0 0 20px; font-size: 13px; }
  form { display: flex; gap: 8px; margin-bottom: 20px; }
  input { flex: 1; padding: 10px 12px; border: 1px solid #d3d1c7; border-radius: 10px;
          font-size: 14px; background: #fff; color: inherit; }
  button { padding: 10px 18px; border: 0; border-radius: 10px; background: #185fa5;
           color: #fff; font-size: 14px; cursor: pointer; }
  button:disabled { opacity: .55; cursor: default; }
  .card { background: #fff; border: 1px solid #e4e2da; border-radius: 12px;
          padding: 16px 18px; margin-bottom: 14px; }
  .card h2 { font-size: 13px; font-weight: 500; color: #5f5e5a; margin: 0 0 10px; }
  .answer { font-size: 15px; white-space: pre-wrap; }
  .cite { display: block; padding: 8px 10px; margin-top: 8px; background: #f1efe8;
          border-left: 3px solid #639922; border-radius: 6px; font-size: 12.5px; color: #444441; }
  .cite b { color: #2c2c2a; }
  .step { font-size: 12.5px; color: #444441; padding: 3px 0; }
  .step span { display: inline-block; min-width: 52px; color: #888780; }
  .stats { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: #e6f1fb; color: #185fa5; border-radius: 999px;
          padding: 4px 11px; font-size: 12.5px; }
  .err { color: #a32d2d; }
</style>
</head>
<body>
<main>
  <h1>agentkit 演示控制台</h1>
  <p class="sub">检索增强智能体 · 答案带可反查引用 · 模型调用统一经模型层</p>
  <form id="f">
    <input id="q" placeholder="问点什么，比如：配额是怎么保证不超发的？" autocomplete="off">
    <button id="b">提问</button>
  </form>
  <div id="out"></div>
</main>
<script>
const f = document.getElementById('f'), q = document.getElementById('q'),
      b = document.getElementById('b'), out = document.getElementById('out');
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
f.addEventListener('submit', async e => {
  e.preventDefault();
  if (!q.value.trim()) return;
  b.disabled = true; out.innerHTML = '<div class="card">检索中…</div>';
  try {
    const r = await fetch('/api/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q.value})
    });
    const d = await r.json();
    if (d.error) { out.innerHTML = '<div class="card err">' + esc(d.error) + '</div>'; return; }
    let html = '<div class="card"><h2>回答</h2><div class="answer">' + esc(d.answer) + '</div>';
    (d.citations || []).forEach(c => {
      html += '<span class="cite"><b>[' + c.index + ']</b> ' + esc(c.source) +
              '<br>' + esc(c.snippet) + '</span>';
    });
    html += '</div>';
    html += '<div class="card"><h2>推理轨迹</h2>';
    (d.steps || []).forEach(s => {
      html += '<div class="step"><span>' + esc(s.kind) + '</span>' + esc(s.text) + '</div>';
    });
    html += '</div>';
    html += '<div class="card"><h2>本次开销</h2><div class="stats">' +
            '<span class="chip">调用 ' + d.usage.calls + ' 次</span>' +
            '<span class="chip">token ' + d.usage.total_tokens + '</span>' +
            '<span class="chip">成本 ¥' + d.usage.cost.toFixed(6) + '</span>' +
            '<span class="chip">缓存命中 ' + d.usage.cache_hits + '</span>' +
            '<span class="chip">后端 ' + esc(d.backend) + '</span>' +
            '</div></div>';
    out.innerHTML = html;
  } catch (err) {
    out.innerHTML = '<div class="card err">请求失败：' + esc(err) + '</div>';
  } finally { b.disabled = false; }
});
</script>
</body>
</html>
"""


def _payload(answer) -> dict[str, Any]:
    usage = answer.usage
    return {
        "answer": answer.text,
        "backend": answer.backend,
        "citations": [
            {"index": c.index, "source": c.source, "snippet": c.snippet,
             "start": c.start, "end": c.end}
            for c in answer.citations
        ],
        "steps": [
            {"kind": s.kind, "text": s.text,
             "tool": s.tool.name if s.tool else None}
            for s in answer.steps
        ],
        "unresolved_citations": answer.unresolved_citations,
        "usage": {
            "calls": usage.calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens(),
            "cost": round(usage.cost, 6),
            "cache_hits": usage.cache_hits,
            "by_purpose": usage.by_purpose,
        },
    }


def serve(runtime: AgentRuntime, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _write(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler 的接口约定
            if self.path in ("/", "/index.html"):
                self._write(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            else:
                self._write(404, "application/json", b'{"error":"not found"}')

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/ask":
                self._write(404, "application/json", b'{"error":"not found"}')
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                question = str(payload.get("question") or "").strip()
                if not question:
                    raise ValueError("question 不能为空")
                body = json.dumps(_payload(runtime.ask(question)), ensure_ascii=False)
                self._write(200, "application/json; charset=utf-8", body.encode("utf-8"))
            except Exception as e:  # noqa: BLE001 — 演示服务不能因为一次异常就退出
                body = json.dumps({"error": str(e)}, ensure_ascii=False)
                self._write(200, "application/json; charset=utf-8", body.encode("utf-8"))

        def log_message(self, *args: Any) -> None:
            pass  # 静音访问日志，避免把控制台刷满

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"演示控制台已启动： http://{host}:{port}   (Ctrl+C 退出)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
