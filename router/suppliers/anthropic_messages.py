# -*- coding: utf-8 -*-
"""Anthropic Messages 协议适配层（Claude Code 接入）。

把入站的 Anthropic ``/v1/messages`` 请求转换成 Router 内部统一的 Chat
Completions 格式，从而复用既有的转发 / 重试 / 图片策略 / 推理档位管线；
响应侧再把 Chat（或经 ``ResponsesToChat`` 中转的 Responses）结果包装回
Anthropic 协议。

Anthropic 协议要点：

- ``system`` 是顶层字段（字符串或 content 块数组）；
- ``content`` 可以是字符串，也可以是块数组
  （text / image / tool_use / tool_result / thinking）；
- ``tools`` 形如 ``{"name", "description", "input_schema"}``；
- 流式响应使用 ``message_start`` / ``content_block_*`` / ``message_delta`` /
  ``message_stop`` 事件，且不以 ``data: [DONE]`` 结尾。

本模块只做纯转换，不依赖 FastAPI / httpx，便于单元测试。
"""
import json
import uuid

from .shared import content_to_text as _content_to_text

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def _num(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _iter_blocks(content) -> list:
    """把 content（字符串或块数组）统一成块列表。"""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _image_block_to_chat_part(block: dict) -> dict | None:
    """Anthropic image 块 -> OpenAI Chat 的 image_url 部分。"""
    source = block.get("source") or {}
    stype = source.get("type")
    if stype == "base64":
        media = source.get("media_type") or "image/png"
        data = source.get("data") or ""
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    if stype == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _system_to_text(system) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for block in _iter_blocks(system):
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "\n\n".join(p for p in parts if p)


def anthropic_to_chat(body: dict) -> dict:
    """Anthropic Messages 请求体 -> Chat Completions 请求体。"""
    if not isinstance(body, dict):
        body = {}
    out: dict = {
        "model": body.get("model") or "",
        # Anthropic 缺省 stream=false，与 Chat Completions 的缺省一致，
        # 但这里显式写出，便于 relay 的非流式分支正确判定。
        "stream": bool(body.get("stream", False)),
    }
    if body.get("max_tokens") is not None:
        out["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p", "seed", "user"):
        if body.get(key) is not None:
            out[key] = body[key]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]

    # 扩展思考映射为 OpenAI 系的 reasoning_effort 档位；后续
    # _apply_configured_reasoning_effort 会识别“请求已显式携带档位”。
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = _num(thinking.get("budget_tokens"))
        out["reasoning_effort"] = "max" if budget >= 24000 else "high"

    messages = []
    system_text = _system_to_text(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        blocks = _iter_blocks(m.get("content"))
        if role == "assistant":
            text_parts = []
            tool_calls = []
            for block in blocks:
                btype = block.get("type")
                if btype == "text" and block.get("text"):
                    text_parts.append(block["text"])
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": json.dumps(
                                block.get("input") if block.get("input") is not None else {},
                                ensure_ascii=False,
                            ),
                        },
                    })
                # thinking / redacted_thinking 属于 ephemeral 内容，历史回放时丢弃。
            if text_parts or tool_calls:
                msg = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                messages.append(msg)
            continue

        # user（以及其他未知角色一律按 user 处理）
        tool_messages = []
        plain = []
        for block in blocks:
            btype = block.get("type")
            if btype == "tool_result":
                text = _content_to_text(block.get("content"))
                if not isinstance(text, str):
                    text = json.dumps(text, ensure_ascii=False)
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "",
                    "content": text,
                })
            elif btype == "image":
                part = _image_block_to_chat_part(block)
                if part:
                    plain.append(part)
            elif btype == "text" and block.get("text"):
                plain.append({"type": "text", "text": block["text"]})
            elif btype == "tool_use":
                # 少数客户端把 tool_use 放进 user 消息，转为占位文本保持可读。
                plain.append({"type": "text", "text": f"[tool_use {block.get('name') or ''}]"})
            # thinking 块丢弃
        messages.extend(tool_messages)
        if plain:
            has_image = any(p.get("type") == "image_url" for p in plain)
            messages.append({
                "role": "user",
                "content": plain if has_image else "\n".join(p.get("text") or "" for p in plain),
            })

    out["messages"] = messages

    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") not in (None, "", "custom", "function"):
            # web_search 等托管工具上游不支持，直接丢弃（与 Responses 清洗策略一致）。
            continue
        name = t.get("name") or ""
        if not name:
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    if tools:
        out["tools"] = tools

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        ttype = tool_choice.get("type")
        if ttype == "auto":
            out["tool_choice"] = "auto"
        elif ttype == "any":
            out["tool_choice"] = "required"
        elif ttype == "tool" and tool_choice.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": tool_choice["name"]}}
        if tool_choice.get("disable_parallel_tool_use"):
            out["parallel_tool_calls"] = False
    return out


def anthropic_usage(usage) -> dict:
    """把 OpenAI / Anthropic 形状的 usage 统一成 Anthropic 形状。"""
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")
    result = {"input_tokens": _num(input_tokens), "output_tokens": _num(output_tokens)}
    cached = None
    for key in ("cache_read_input_tokens", "cached_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and value > 0:
            cached = int(value)
            break
    if cached is None:
        detail = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        if isinstance(detail, dict):
            value = detail.get("cached_tokens")
            if isinstance(value, (int, float)) and value > 0:
                cached = int(value)
    if cached:
        result["cache_read_input_tokens"] = cached
    return result


def chat_json_to_anthropic(chat: dict, model: str = "") -> dict:
    """非流式 Chat Completions 响应 -> Anthropic Messages 响应。"""
    chat = chat if isinstance(chat, dict) else {}
    choice = (chat.get("choices") or [{}])[0]
    if not isinstance(choice, dict):
        choice = {}
    msg = choice.get("message") or {}
    if not isinstance(msg, dict):
        msg = {}

    blocks = []
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    text = msg.get("content")
    if isinstance(text, list):
        text = _content_to_text(text)
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {"_raw": fn.get("arguments") or ""}
        if not isinstance(args, dict):
            args = {"_raw": args}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:16]),
            "name": fn.get("name") or "",
            "input": args,
        })
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": chat.get("model") or model or "",
        "content": blocks,
        "stop_reason": STOP_REASON_MAP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": anthropic_usage(chat.get("usage")),
    }


def anthropic_error_json(message, err_type: str | None = None, status: int | None = None) -> dict:
    """把任意错误包装成 Anthropic 的 error 响应体。"""
    by_status = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        503: "overloaded_error",
    }
    if not err_type:
        err_type = by_status.get(status or 0, "api_error")
    if isinstance(message, str):
        # 上游错误体通常是 JSON 文本，尝试解包出 message 字段。
        try:
            parsed = json.loads(message)
            if isinstance(parsed, dict):
                message = parsed
        except Exception:
            pass
    if isinstance(message, dict):
        err = message.get("error") if isinstance(message.get("error"), dict) else None
        message = (err or {}).get("message") or json.dumps(message, ensure_ascii=False)
    return {"type": "error", "error": {"type": err_type, "message": str(message)[:2000]}}


class ChatToAnthropic:
    """把内部 Chat Completions 事件流转换为 Anthropic Messages SSE 事件。

    输入既可以直接来自 Chat 上游（``iter_upstream_events`` 的输出），也可以
    来自 Responses 上游经 ``ResponsesToChat`` 中转后的 chat 块——两者都是
    OpenAI Chat chunk 形状的 dict。用法：逐块 ``feed()``，结束后 ``final_events()``。
    """

    def __init__(self, model: str = ""):
        self.model = model or ""
        self._started = False
        self._next_index = 0
        self._open_kind = None  # None | "text" | "thinking"
        self._open_index = None
        # Chat 的 tool_call 增量可能交错到达，而 Anthropic 的事件流不允许
        # 交错回写已关闭的块，所以按工具缓冲参数，在 finish_reason 或流结束时
        # 按首次出现顺序一次性输出完整的 tool_use 块。
        self._tool_buffers: list = []      # 首次出现顺序
        self._tool_buffer_by_key: dict = {}
        self._tools_flushed = False
        self.usage = None  # 保留 OpenAI 形状，供统计复用
        self.stop_reason = None
        self.status = "completed"

    # ---- 块管理 ----
    def _start_message(self, events: list) -> None:
        if self._started:
            return
        self._started = True
        events.append({
            "type": "message_start",
            "message": {
                "id": "msg_" + uuid.uuid4().hex[:24],
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    def _ensure_block(self, events: list, kind: str) -> int:
        """确保当前打开的块是 kind（text/thinking），返回块 index。"""
        if self._open_kind == kind:
            return self._open_index
        self._close_open_block(events)
        index = self._next_index
        self._next_index += 1
        if kind == "text":
            events.append({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
        else:  # thinking
            events.append({"type": "content_block_start", "index": index, "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
        self._open_kind = kind
        self._open_index = index
        return index

    def _close_open_block(self, events: list) -> None:
        if self._open_kind is None:
            return
        events.append({"type": "content_block_stop", "index": self._open_index})
        self._open_kind = None
        self._open_index = None

    def _flush_tools(self, events: list) -> None:
        """把缓冲的工具调用按首次出现顺序输出为完整的 tool_use 块。"""
        if self._tools_flushed:
            return
        self._tools_flushed = True
        self._close_open_block(events)
        for buf in self._tool_buffers:
            index = self._next_index
            self._next_index += 1
            events.append({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "tool_use", "id": buf["id"], "name": buf["name"], "input": {}},
            })
            args = "".join(buf["parts"])
            if args:
                events.append({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": args},
                })
            events.append({"type": "content_block_stop", "index": index})

    # ---- 主转换 ----
    def feed(self, chunk) -> list:
        """消费一个 chat chunk dict，返回 0..n 个 Anthropic 事件。"""
        events: list = []
        if not isinstance(chunk, dict):
            return events
        if chunk.get("error"):
            self.status = "failed"
            err = chunk["error"]
            if isinstance(err, dict):
                message = err.get("message") or json.dumps(err, ensure_ascii=False)
                etype = err.get("type") or "api_error"
            else:
                message = str(err)
                etype = "api_error"
            events.append({"type": "error", "error": {"type": etype, "message": str(message)[:500]}})
            return events

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = usage

        choices = chunk.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}
        finish = choice.get("finish_reason")
        if finish:
            self.stop_reason = STOP_REASON_MAP.get(finish) or self.stop_reason or "end_turn"
            if finish == "length" and self.status == "completed":
                self.status = "incomplete"

        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            self._start_message(events)
            index = self._ensure_block(events, "thinking")
            events.append({"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": reasoning}})

        content = delta.get("content")
        text_parts = []
        if isinstance(content, str) and content:
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                    text_parts.append(part["text"])
                elif isinstance(part, str) and part:
                    text_parts.append(part)
        for text in text_parts:
            self._start_message(events)
            index = self._ensure_block(events, "text")
            events.append({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}})

        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            self._start_message(events)
            key = tc.get("index")
            if key is None:
                key = tc.get("id")
            fn = tc.get("function") or {}
            buf = self._tool_buffer_by_key.get(key)
            if buf is None:
                buf = {
                    "id": tc.get("id") or ("toolu_" + uuid.uuid4().hex[:16]),
                    "name": fn.get("name") or "",
                    "parts": [],
                }
                self._tool_buffer_by_key[key] = buf
                self._tool_buffers.append(buf)
            else:
                if tc.get("id") and not buf["id"]:
                    buf["id"] = tc["id"]
                if fn.get("name") and not buf["name"]:
                    buf["name"] = fn["name"]
            args = fn.get("arguments")
            if args:
                buf["parts"].append(args)
        if finish:
            # finish 通常携带在最后一个 chunk 上：先处理完本块增量再输出工具块。
            self._flush_tools(events)
        return events

    def final_events(self) -> list:
        """流结束时的收尾事件：关闭块 + 输出工具块 + message_delta + message_stop。"""
        events: list = []
        self._start_message(events)
        self._flush_tools(events)
        self._close_open_block(events)
        usage = self.usage if isinstance(self.usage, dict) else {}
        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = usage.get("output_tokens")
        events.append({
            "type": "message_delta",
            "delta": {"stop_reason": self.stop_reason or "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": max(0, _num(output_tokens))},
        })
        events.append({"type": "message_stop"})
        return events


async def stream_chat_to_anthropic(source, model: str = "", conv: "ChatToAnthropic | None" = None):
    """把 chat chunk dict 异步流转换成 Anthropic SSE 事件流。

    ``source`` 是产出 chat chunk dict 的异步迭代器（Chat 上游的
    ``iter_upstream_events``——调用方需先过滤 ``[DONE]`` 哨兵，或 Responses
    上游经 ``ResponsesToChat.convert`` 中转后的输出）。传入 ``conv`` 可让
    调用方持有同一个转换器实例以读取 usage / status。
    """
    if conv is None:
        conv = ChatToAnthropic(model)
    async for chunk in source:
        for evt in conv.feed(chunk):
            yield evt
    for evt in conv.final_events():
        yield evt


def estimate_input_tokens(body: dict) -> int:
    """粗略估算 /v1/messages/count_tokens 的输入 token 数（约 4 字符/token）。"""
    total_chars = 0

    def count(value):
        nonlocal total_chars
        if isinstance(value, str):
            total_chars += len(value)
        elif isinstance(value, dict):
            for v in value.values():
                count(v)
        elif isinstance(value, list):
            for v in value:
                count(v)

    if isinstance(body, dict):
        count(body.get("system"))
        count(body.get("messages"))
        count(body.get("tools"))
    return max(1, total_chars // 4)
