"""极简 HTTP 传输层（只用标准库 urllib，零第三方依赖）。

两个必须踩过的坑，都写在这里而不是散落在调用方：

① 环境代理。本机普遍设着 HTTP_PROXY=http://127.0.0.1:16540，
   urllib 默认会把这个代理套到**所有**请求上，包括 http://127.0.0.1:8080
   这样的本地地址。于是"连本地网关"变成"走代理再打回本机"，
   表现为连接被拒或 CONNECT tunnel failed，而排查方向很容易跑偏到网关配置上。
   这里显式构造空 ProxyHandler，永远直连。

② 错误必须解析 body。OpenAI 兼容协议把错误信息放在响应体里
   （{"error":{"message","type"}}），只抛"HTTP 502"会丢掉全部可诊断信息，
   而 429 和 502 的处置方式完全不同（退避 vs 不重试）。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class TransportError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body

    def is_rate_limited(self) -> bool:
        return self.status == 429


@dataclass
class HttpResult:
    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    latency_ms: int = 0


class JsonTransport:
    """JSON over HTTP，带计时与统一错误处理。"""

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout
        # 空 ProxyHandler：显式声明"不走任何环境代理"
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> HttpResult:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        return self._send(request)

    def get(self, url: str, headers: dict[str, str] | None = None) -> HttpResult:
        request = urllib.request.Request(url, method="GET")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        return self._send(request)

    def _send(self, request: urllib.request.Request) -> HttpResult:
        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
                latency = int((time.perf_counter() - started) * 1000)
                return HttpResult(
                    status=resp.status,
                    body=json.loads(raw) if raw.strip() else {},
                    headers={k.lower(): v for k, v in resp.getheaders()},
                    latency_ms=latency,
                )
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            latency = int((time.perf_counter() - started) * 1000)
            detail = _error_message(raw)
            raise TransportError(
                f"HTTP {e.code} from {request.full_url}: {detail}",
                status=e.code,
                body=raw,
            ) from None
        except urllib.error.URLError as e:
            raise TransportError(f"无法连接 {request.full_url}: {e.reason}") from None


def _error_message(raw: str) -> str:
    """从 OpenAI 形状的响应体里取出可读信息，取不到就退回原始前 200 字。"""
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw[:200]
    error = parsed.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    if error:
        return str(error)
    return raw[:200]


def join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")
