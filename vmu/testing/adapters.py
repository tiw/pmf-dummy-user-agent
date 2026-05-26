"""
Agent 适配器

将被测 agent 统一包装为标准化接口，
支持函数、类方法、HTTP API 等多种形式。
"""

import json
from typing import Any, Callable, Dict, Optional, Protocol


class AgentAdapter:
    """
    Agent 适配器：将被测 agent 统一为 `send(message) -> response` 接口。

    支持多种形式：
    - 普通函数/Callable
    - 有 .respond() 或 .chat() 方法的对象
    - HTTP API endpoint
    - 异步函数（在 sync 场景下会自动 run_until_complete）
    """

    def __init__(self, sender: Callable[[str], str], name: str = "agent"):
        self.sender = sender
        self.name = name

    def send(self, message: str) -> str:
        """发送消息给被测 agent，返回其回复"""
        return self.sender(message)

    # ─── 工厂方法 ───

    @classmethod
    def from_callable(cls, fn: Callable[[str], str], name: str = "agent") -> "AgentAdapter":
        """从普通函数创建适配器"""
        return cls(sender=fn, name=name)

    @classmethod
    def from_object(
        cls,
        obj: Any,
        method_name: str = "respond",
        name: Optional[str] = None,
    ) -> "AgentAdapter":
        """
        从对象创建适配器。

        会自动探测方法名：respond / chat / run / invoke / call
        """
        if not name:
            name = getattr(obj, "name", getattr(obj, "__class__", type(obj)).__name__)

        # 自动探测方法
        candidates = [method_name, "respond", "chat", "run", "invoke", "call", "execute"]
        method = None
        for cand in candidates:
            if hasattr(obj, cand) and callable(getattr(obj, cand)):
                method = getattr(obj, cand)
                break

        if method is None:
            raise ValueError(
                f"对象 {type(obj).__name__} 没有找到可用的方法，"
                f"请确保它有 {candidates} 中的一个"
            )

        def sender(message: str) -> str:
            result = method(message)
            # 如果返回的是 dict，尝试提取回复内容
            if isinstance(result, dict):
                for key in ("response", "content", "reply", "text", "message", "output"):
                    if key in result:
                        return str(result[key])
                return str(result)
            return str(result)

        return cls(sender=sender, name=name)

    @classmethod
    def http(
        cls,
        url: str,
        method: str = "POST",
        message_field: str = "message",
        response_field: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        name: str = "http_agent",
    ) -> "AgentAdapter":
        """
        从 HTTP API 创建适配器。

        Args:
            url: API endpoint
            method: HTTP 方法（默认 POST）
            message_field: 请求体中消息字段名
            response_field: 响应中回复字段名（None 则返回整个响应体）
            headers: 额外请求头
            timeout: 超时时间
        """
        import urllib.request
        import urllib.error

        _headers = {"Content-Type": "application/json"}
        if headers:
            _headers.update(headers)

        def sender(message: str) -> str:
            payload = json.dumps({message_field: message}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers=_headers,
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                    # 尝试解析 JSON
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        return body

                    if response_field:
                        return str(data.get(response_field, body))
                    # 自动探测常见字段
                    for key in ("response", "content", "reply", "text", "message", "output", "result"):
                        if key in data:
                            return str(data[key])
                    return body
            except urllib.error.HTTPError as e:
                return f"[HTTP Error {e.code}: {e.reason}]"
            except Exception as e:
                return f"[Error: {str(e)}]"

        return cls(sender=sender, name=name)

    @classmethod
    def async_http(
        cls,
        url: str,
        method: str = "POST",
        message_field: str = "message",
        response_field: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        name: str = "async_http_agent",
    ) -> "AgentAdapter":
        """
        从异步 HTTP API 创建适配器（需要 httpx 或 aiohttp）。
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("使用 async_http 需要安装 httpx: pip install httpx")

        _headers = {"Content-Type": "application/json"}
        if headers:
            _headers.update(headers)

        async def sender(message: str) -> str:
            payload = {message_field: message}
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method.upper() == "GET":
                    resp = await client.get(url, params=payload, headers=_headers)
                else:
                    resp = await client.request(method.upper(), url, json=payload, headers=_headers)
                resp.raise_for_status()
                data = resp.json()
                if response_field:
                    return str(data.get(response_field, resp.text))
                for key in ("response", "content", "reply", "text", "message", "output", "result"):
                    if key in data:
                        return str(data[key])
                return resp.text

        # 包装为 sync 接口
        import asyncio

        def sync_sender(message: str) -> str:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 在已有事件循环中（如 Jupyter）
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(sender(message))
                return loop.run_until_complete(sender(message))
            except RuntimeError:
                return asyncio.run(sender(message))

        return cls(sender=sync_sender, name=name)
