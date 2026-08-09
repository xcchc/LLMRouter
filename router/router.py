# -*- coding: utf-8 -*-
"""
本地模型路由网关 + 控制台（LLM Router）

- 按"模型名"把 Codex 的请求转发到不同上游（不同 API、不同协议）
- responses / chat completions 双向协议转换
- 聚合各上游模型列表；请求统计（token、次数、耗时、费用估算）
- 控制台界面：http://127.0.0.1:8765/

用法：编辑 config.json 后双击 start.bat 启动。
"""
import asyncio
import json
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import stats
import vision

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent          # exe 所在目录（config/stats 放这里）
    _DATA_DIR = Path(getattr(sys, "_MEIPASS", BASE_DIR))      # 打包资源目录
else:
    BASE_DIR = Path(__file__).resolve().parent
    _DATA_DIR = BASE_DIR
CONFIG_PATH = BASE_DIR / "config.json"
DASHBOARD_PATH = _DATA_DIR / "dashboard.html"
MODEL_CACHE_TTL = 600  # 上游模型列表缓存秒数
APP_VERSION = "2026.08.05"

app = FastAPI(title="LLM Router", docs_url=None, redoc_url=None)

DONE = object()  # SSE [DONE] 哨兵


def _upstream_models_url(base_url: str) -> str:
    """拼接上游 /models 地址；非 ASCII 字符（如中文）做百分号编码，避免请求崩溃。"""
    import urllib.parse
    return urllib.parse.quote(str(base_url).rstrip("/") + "/models", safe=":/?&=#%+@")


def _ensure_config() -> None:
    if not CONFIG_PATH.exists():
        default_cfg = {
            "port": 8765,
            "open_browser": False,
            "suppliers": [],
            "model_map": {},
            "default_supplier": "",
            "model_blacklist": {},
        }
        try:
            save_config(default_cfg)
        except Exception:
            pass


def load_config() -> dict:
    """每次请求都重新读取配置，改完 config.json 不用重启（端口除外）。"""
    _ensure_config()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalized_config(cfg: dict) -> dict:
    try:
        port = int(cfg.get("port") or 8765)
    except Exception:
        port = 8765
    return {
        "port": port,
        "open_browser": bool(cfg.get("open_browser", True)),
        "suppliers": cfg.get("suppliers") or [],
        "model_map": cfg.get("model_map") or {},
        "default_supplier": cfg.get("default_supplier") or "",
        "model_blacklist": cfg.get("model_blacklist") or {},
        "model_prices": cfg.get("model_prices") or {},
    }


def save_config(cfg: dict) -> None:
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG_PATH)


# ------------------ 模型列表聚合 ------------------

_model_cache: dict = {}  # supplier name -> {"ids": [..], "at": ts}


def _blacklisted_ids(cfg: dict, supplier_name: str) -> set:
    """该供应商被用户隐藏的模型集合（model_blacklist）。"""
    bl = (cfg.get("model_blacklist") or {}).get(supplier_name) or []
    return set(bl)


async def supplier_model_ids(supplier: dict, cfg: dict | None = None) -> list:
    """获取供应商的模型列表：优先手动配置的 models，否则请求上游 /v1/models（带缓存）。
    用户隐藏的模型（model_blacklist）会被过滤掉。"""
    cfg = cfg or load_config()
    name = str(supplier.get("name", ""))
    black = _blacklisted_ids(cfg, name)
    if supplier.get("models"):
        return [m for m in supplier["models"] if m not in black]
    cached = _model_cache.get(name)
    if cached and time.time() - cached["at"] < MODEL_CACHE_TTL:
        return [m for m in cached["ids"] if m not in black]
    ids = []
    try:
        base = str(supplier.get("base_url", "")).rstrip("/")
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(_upstream_models_url(base), headers={"Authorization": f"Bearer {supplier.get('api_key', '')}"})
        if r.status_code == 200:
            data = r.json().get("data", [])
            ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
    except Exception:
        pass
    if cached and not ids:
        ids = cached["ids"]  # 获取失败时沿用旧缓存
    _model_cache[name] = {"ids": ids, "at": time.time()}
    return [m for m in ids if m not in black]


async def merged_model_list(cfg: dict) -> list:
    result = []
    seen = set()
    for s in cfg.get("suppliers", []):
        name = s.get("name", "")
        for mid in await supplier_model_ids(s, cfg):
            if mid not in seen:
                seen.add(mid)
                result.append({"id": mid, "object": "model", "owned_by": name})
    for alias, sup_name in (cfg.get("model_map") or {}).items():
        if alias not in seen:
            if alias in _blacklisted_ids(cfg, sup_name):
                continue  # 用户已隐藏该模型
            seen.add(alias)
            result.append({"id": alias, "object": "model", "owned_by": "model_map"})
    return result


async def find_supplier(cfg: dict, model: str) -> dict | None:
    """按模型名找供应商：model_map 显式指定 > 上游模型列表命中 > default_supplier。"""
    if model in cfg.get("model_map", {}):
        target = cfg["model_map"][model]
        for s in cfg.get("suppliers", []):
            if s.get("name") == target:
                return s
    for s in cfg.get("suppliers", []):
        if model in await supplier_model_ids(s, cfg):
            return s
    default = cfg.get("default_supplier")
    if default:
        for s in cfg.get("suppliers", []):
            if s.get("name") == default:
                return s
    return None


# ------------------ 请求体转换 ------------------

def _extract_reasoning_text(reasoning) -> str:
    """从 Responses 的 reasoning 字段里提取思考文本：
    支持字符串 / 字典 / 数组（如 [{"type":"reasoning","summary":[{"type":"summary_text","text":"..."}]}]）。"""
    parts = []
    def add(v):
        if isinstance(v, str) and v:
            parts.append(v)
        elif isinstance(v, dict):
            s = v.get("summary") or v.get("content") or v.get("text") or ""
            if isinstance(s, list):
                for x in s:
                    add(x)
            else:
                add(s)
        elif isinstance(v, list):
            for x in v:
                add(x)
    add(reasoning)
    return "\n".join(parts)


def _args_to_string(args) -> str:
    """arguments 字段转成 chat 的 JSON 字符串；本就是字符串时不重复转义。"""
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                t = c.get("type")
                if t in ("input_text", "output_text", "text"):
                    parts.append(str(c.get("text", "")))
                elif t == "input_image":
                    parts.append("[图片]")
                elif t == "input_file":
                    parts.append(f"[文件: {c.get('filename') or c.get('file_id') or '附件'}]")
        return "\n\n".join(p for p in parts if p)
    return str(content)


def _responses_content_to_chat(content):
    """Responses message content -> Chat content；图片保留为 image_url。"""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        return str(content)
    text_parts = []
    chat_parts = []
    has_media = False
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            chat_parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t in ("input_text", "output_text", "text"):
            text = str(part.get("text", ""))
            if text:
                text_parts.append(text)
                chat_parts.append({"type": "text", "text": text})
        elif t == "input_image":
            image_url = part.get("image_url") or part.get("url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if image_url:
                detail = part.get("detail") or "auto"
                if detail == "original":
                    detail = "high"
                chat_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": detail},
                })
                has_media = True
        elif t == "input_file":
            label = part.get("filename") or part.get("file_id") or "附件"
            text = f"[文件: {label}]"
            text_parts.append(text)
            chat_parts.append({"type": "text", "text": text})
    if has_media:
        return chat_parts
    return "\n\n".join(p for p in text_parts if p)


def _chat_content_to_responses(content, assistant: bool = False) -> list:
    part_type = "output_text" if assistant else "input_text"
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": part_type, "text": content}] if content else []
    if not isinstance(content, list):
        return [{"type": part_type, "text": str(content)}]
    parts = []
    for part in content:
        if isinstance(part, str):
            if part:
                parts.append({"type": part_type, "text": part})
            continue
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t in ("text", "input_text", "output_text"):
            text = str(part.get("text", ""))
            if text:
                parts.append({"type": part_type, "text": text})
        elif not assistant and t in ("image_url", "input_image"):
            image = part.get("image_url") or part.get("url")
            detail = part.get("detail") or "auto"
            if isinstance(image, dict):
                detail = image.get("detail") or detail
                image = image.get("url")
            if image:
                parts.append({"type": "input_image", "image_url": image, "detail": detail})
    return parts


def _sanitize_schema(value):
    if isinstance(value, dict):
        return {k: _sanitize_schema(v) for k, v in value.items() if k != "encrypted"}
    if isinstance(value, list):
        return [_sanitize_schema(v) for v in value]
    return value


def _normalize_tool_parameters_schema(value) -> dict:
    """Make function parameters acceptable to strict Chat tool validators."""
    schema = _sanitize_schema(value)
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    root_type = schema.get("type")
    if root_type in (None, ""):
        schema["type"] = "object"
    elif isinstance(root_type, list) and "object" in root_type:
        schema["type"] = "object"
    elif root_type != "object":
        return {"type": "object", "properties": {}}
    return schema


def _chat_tool_name(name: str, namespace: str | None = None) -> str:
    raw = f"{namespace}__{name}" if namespace else str(name or "tool")
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", raw) or "tool"
    if len(safe) > 64:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:55]}_{digest}"
    return safe


def _build_tool_bridge(body: dict) -> dict:
    bridge = {"chat_tools": [], "by_chat": {}, "by_response": {}, "unsupported": []}
    used = set()

    def unique_name(name: str, namespace: str | None) -> str:
        candidate = _chat_tool_name(name, namespace)
        if candidate not in used:
            used.add(candidate)
            return candidate
        raw = f"{namespace or ''}:{name}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate[:55]}_{digest}"
        used.add(candidate)
        return candidate

    def add_tool(tool: dict, namespace: str | None = None, namespace_description: str = "") -> None:
        if not isinstance(tool, dict):
            return
        kind = tool.get("type")
        if kind == "namespace":
            ns = str(tool.get("name") or namespace or "namespace")
            ns_desc = str(tool.get("description") or namespace_description or "").strip()
            for nested in tool.get("tools") or []:
                add_tool(nested, ns, ns_desc)
            return
        if kind == "tool_search":
            original_name = "tool_search"
        elif kind in ("function", "custom"):
            original_name = str(tool.get("name") or "")
            if not original_name:
                return
        else:
            if kind:
                bridge["unsupported"].append(str(kind))
            return
        chat_name = unique_name(original_name, namespace)
        description = str(tool.get("description") or "").strip()
        if namespace_description:
            description = f"{namespace_description}\n\n{description}" if description else namespace_description
        spec = {
            "kind": kind,
            "name": original_name,
            "namespace": namespace,
            "chat_name": chat_name,
            "execution": tool.get("execution"),
        }
        bridge["by_chat"][chat_name] = spec
        bridge["by_response"][(kind, namespace or "", original_name)] = chat_name
        if kind == "custom":
            bridge_note = "Pass the complete freeform tool input in the required `input` string."
            format_note = ""
            custom_format = tool.get("format")
            if isinstance(custom_format, dict) and (
                "syntax" in custom_format or "definition" in custom_format
            ):
                format_lines = [
                    "Responses custom-tool format constraint "
                    "(Chat Completions has no native custom grammar field):"
                ]
                if "syntax" in custom_format:
                    syntax = custom_format.get("syntax")
                    if not isinstance(syntax, str):
                        syntax = json.dumps(syntax, ensure_ascii=False)
                    format_lines.append(f"format.syntax: {syntax}")
                if "definition" in custom_format:
                    definition = custom_format.get("definition")
                    if not isinstance(definition, str):
                        definition = json.dumps(definition, ensure_ascii=False)
                    format_lines.append(f"format.definition:\n{definition}")
                format_note = "\n".join(format_lines)
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The complete raw input for this freeform tool.",
                    },
                },
                "required": ["input"],
                "additionalProperties": False,
            }
            notes = [part for part in (description, bridge_note, format_note) if part]
            description = "\n\n".join(notes)
        elif kind == "tool_search":
            parameters = _normalize_tool_parameters_schema(tool.get("parameters") or {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Describe the tool capability that should be loaded.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum number of matching tools to return.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            })
        else:
            parameters = _normalize_tool_parameters_schema(
                tool.get("parameters") or {"type": "object", "properties": {}}
            )
        chat_function = {
            "name": chat_name,
            "description": description,
            "parameters": parameters,
        }
        if kind == "function" and "strict" in tool:
            chat_function["strict"] = tool["strict"]
        bridge["chat_tools"].append({
            "type": "function",
            "function": chat_function,
        })

    for tool in body.get("tools") or []:
        add_tool(tool)
    for item in body.get("input") or []:
        if not isinstance(item, dict) or item.get("type") not in ("tool_search_output", "additional_tools"):
            continue
        candidates = item.get("tools") or item.get("output") or []
        if isinstance(candidates, dict):
            candidates = candidates.get("tools") or [candidates]
        if isinstance(candidates, list):
            for tool in candidates:
                add_tool(tool)
    bridge["unsupported"] = sorted(set(bridge["unsupported"]))
    return bridge


def _bridge_chat_name(bridge: dict, kind: str, name: str, namespace=None) -> str:
    return (bridge.get("by_response") or {}).get(
        (kind, namespace or "", name),
        _chat_tool_name(name, namespace),
    )


def _tool_choice_to_chat(tc, bridge=None):
    if isinstance(tc, dict) and tc.get("type") in ("function", "custom"):
        name = tc.get("name", "")
        if name:
            kind = tc.get("type")
            chat_name = _bridge_chat_name(bridge or {}, kind, name, tc.get("namespace"))
            return {"type": "function", "function": {"name": chat_name}}
    return tc


def _tool_choice_to_responses(tc):
    if isinstance(tc, dict) and tc.get("type") == "function":
        fn = tc.get("function") or {}
        return {"type": "function", "name": fn.get("name", tc.get("name", ""))}
    return tc


def _tool_output_to_text(output) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        text = _content_to_text(output)
        if text:
            return text
    return json.dumps(output, ensure_ascii=False)


def _tool_output_to_chat_messages(item: dict) -> list:
    output = item.get("output", "")
    tool_call_id = item.get("call_id", "")
    if not isinstance(output, list):
        return [{"role": "tool", "tool_call_id": tool_call_id, "content": _tool_output_to_text(output)}]
    text_parts = []
    media_parts = []
    for part in output:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t in ("input_text", "output_text", "text"):
            text = str(part.get("text", ""))
            if text:
                text_parts.append(text)
        elif t == "input_image":
            converted = _responses_content_to_chat([part])
            if isinstance(converted, list):
                media_parts.extend(converted)
        elif t == "input_file":
            text_parts.append(f"[文件: {part.get('filename') or part.get('file_id') or '附件'}]")
    content = "\n\n".join(text_parts) or ("Tool returned image content." if media_parts else "")
    messages = [{"role": "tool", "tool_call_id": tool_call_id, "content": content}]
    if media_parts:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Image returned by the preceding tool call."}, *media_parts],
        })
    return messages


def _responses_text_format_to_chat(text_config):
    if not isinstance(text_config, dict):
        return None
    fmt = text_config.get("format")
    if not isinstance(fmt, dict):
        return None
    kind = fmt.get("type")
    if kind == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.get("name") or "response",
                "description": fmt.get("description"),
                "schema": _sanitize_schema(fmt.get("schema") or {}),
                "strict": bool(fmt.get("strict", False)),
            },
        }
    if kind in ("json_object", "text"):
        return {"type": kind}
    return None


def _verbosity_instruction(text_config) -> str:
    if not isinstance(text_config, dict):
        return ""
    verbosity = str(text_config.get("verbosity") or "").lower()
    instructions = {
        "low": "Keep the final answer concise and include only essential details.",
        "medium": "Use a moderate level of detail in the final answer.",
        "high": "Provide a detailed final answer with the relevant reasoning and caveats.",
    }
    return instructions.get(verbosity, "")


def _chat_response_format_to_responses(response_format):
    if not isinstance(response_format, dict):
        return None
    kind = response_format.get("type")
    if kind == "json_schema":
        spec = response_format.get("json_schema") or {}
        return {
            "type": "json_schema",
            "name": spec.get("name") or "response",
            "description": spec.get("description"),
            "schema": spec.get("schema") or {},
            "strict": bool(spec.get("strict", False)),
        }
    if kind in ("json_object", "text"):
        return {"type": kind}
    return None


def _custom_tool_arguments(tool_input) -> str:
    if not isinstance(tool_input, str):
        tool_input = json.dumps(tool_input, ensure_ascii=False)
    return json.dumps({"input": tool_input}, ensure_ascii=False)


def _custom_input_from_chat_arguments(arguments: str) -> str:
    if not isinstance(arguments, str):
        arguments = _args_to_string(arguments)
    try:
        parsed = json.loads(arguments or "{}")
    except Exception:
        return arguments
    if isinstance(parsed, dict) and "input" in parsed:
        value = parsed["input"]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if isinstance(parsed, str):
        return parsed
    return arguments


def _custom_tool_names(body: dict) -> set:
    bridge = _build_tool_bridge(body)
    return {name for name, spec in bridge["by_chat"].items() if spec.get("kind") == "custom"}


def responses_to_chat(body: dict, tool_bridge=None) -> dict:
    """Responses 请求体 -> Chat Completions 请求体"""
    tool_bridge = tool_bridge or _build_tool_bridge(body)
    messages = []
    assistant_message = None
    pending_tool_media = []

    def ensure_assistant_message() -> dict:
        nonlocal assistant_message
        if assistant_message is None:
            assistant_message = {"role": "assistant", "content": None}
        return assistant_message

    def append_assistant_text(text: str) -> None:
        if not text:
            return
        msg = ensure_assistant_message()
        current = msg.get("content")
        msg["content"] = f"{current}\n\n{text}" if current else text

    def append_reasoning(reasoning) -> None:
        text = _extract_reasoning_text(reasoning)
        if not text:
            return
        msg = ensure_assistant_message()
        current = msg.get("reasoning_content")
        if not current:
            msg["reasoning_content"] = text
        elif text != current and text not in current:
            msg["reasoning_content"] = f"{current}\n{text}"

    def flush_assistant_message() -> None:
        nonlocal assistant_message
        if assistant_message is not None:
            # Chat 上游不接受只有 reasoning_content 的 assistant 历史消息。
            # 正文或工具调用存在时，reasoning_content 仍与该 assistant 一并保留。
            if assistant_message.get("content") or assistant_message.get("tool_calls"):
                messages.append(assistant_message)
            assistant_message = None

    def flush_pending_tool_media() -> None:
        nonlocal pending_tool_media
        if pending_tool_media:
            messages.extend(pending_tool_media)
            pending_tool_media = []

    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    if tool_bridge.get("unsupported"):
        unsupported = ", ".join(tool_bridge["unsupported"])
        messages.append({
            "role": "system",
            "content": (
                "[Local router capability notice] These Responses server-hosted tools are unavailable on this "
                f"Chat-compatible upstream: {unsupported}. Do not claim to have used them."
            ),
        })
    response_format = _responses_text_format_to_chat(body.get("text"))
    verbosity_instruction = _verbosity_instruction(body.get("text"))
    if verbosity_instruction:
        messages.append({"role": "system", "content": verbosity_instruction})
    if response_format and response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema") or {}
        messages.append({
            "role": "system",
            "content": "Return only JSON matching this schema: " + json.dumps(schema, ensure_ascii=False),
        })
    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, str):
                flush_assistant_message()
                flush_pending_tool_media()
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t not in ("function_call_output", "custom_tool_call_output", "tool_search_output"):
                flush_pending_tool_media()
            if t == "reasoning":
                # Responses 把思考作为独立输出项；Chat 要求它挂回同一轮 assistant 消息。
                append_reasoning(item)
            elif t == "message":
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                if role == "assistant":
                    append_reasoning(item.get("reasoning"))
                    append_assistant_text(_content_to_text(item.get("content")))
                else:
                    flush_assistant_message()
                    messages.append({"role": role, "content": _responses_content_to_chat(item.get("content"))})
            elif t == "agent_message":
                flush_assistant_message()
                content = item.get("content") or []
                text = _content_to_text(content)
                encrypted = any(
                    isinstance(part, dict) and part.get("type") == "encrypted_content"
                    for part in content
                )
                prefix = f"Message from agent {item.get('author', 'unknown')}:"
                if encrypted:
                    text = f"{text}\n[Encrypted agent payload is unavailable to this non-OpenAI backend.]".strip()
                messages.append({"role": "user", "content": f"{prefix}\n{text}".strip()})
            elif t in ("function_call", "custom_tool_call", "tool_search_call"):
                msg = ensure_assistant_message()
                if t == "custom_tool_call":
                    arguments = _custom_tool_arguments(item.get("input", ""))
                    kind = "custom"
                elif t == "tool_search_call":
                    arguments = _args_to_string(item.get("arguments") or item.get("action") or {})
                    kind = "tool_search"
                else:
                    arguments = _args_to_string(item.get("arguments", {}))
                    kind = "function"
                original_name = item.get("name") or ("tool_search" if t == "tool_search_call" else "")
                chat_name = _bridge_chat_name(tool_bridge, kind, original_name, item.get("namespace"))
                msg.setdefault("tool_calls", []).append({
                        "id": item.get("call_id", "call_" + uuid.uuid4().hex[:8]),
                        "type": "function",
                        "function": {
                            "name": chat_name,
                            "arguments": arguments,
                        },
                })
            elif t in ("function_call_output", "custom_tool_call_output"):
                flush_assistant_message()
                converted_messages = _tool_output_to_chat_messages(item)
                for message in converted_messages:
                    if message.get("role") == "tool":
                        messages.append(message)
                    else:
                        pending_tool_media.append(message)
            elif t == "tool_search_output":
                flush_assistant_message()
                loaded = []
                candidates = item.get("tools") or item.get("output") or []
                if isinstance(candidates, dict):
                    candidates = candidates.get("tools") or [candidates]
                for tool in (candidates if isinstance(candidates, list) else []):
                    if not isinstance(tool, dict):
                        continue
                    if tool.get("type") == "namespace":
                        ns = tool.get("name") or "namespace"
                        loaded.extend(f"{ns}__{x.get('name')}" for x in tool.get("tools") or [] if isinstance(x, dict))
                    elif tool.get("name"):
                        loaded.append(str(tool["name"]))
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": "Loaded tools: " + (", ".join(loaded) if loaded else "none"),
                })
        flush_assistant_message()
        flush_pending_tool_media()
    chat = {
        "model": body.get("model"),
        "messages": messages,
        "stream": body.get("stream", True),
    }
    for k in ("temperature", "top_p", "max_tokens", "seed", "stop", "presence_penalty", "frequency_penalty", "logit_bias", "user", "metadata"):
        if body.get(k) is not None:
            chat[k] = body[k]
    if body.get("max_output_tokens"):
        chat["max_completion_tokens"] = body["max_output_tokens"]
    if isinstance(body.get("reasoning"), dict) and body["reasoning"].get("effort"):
        chat["reasoning_effort"] = body["reasoning"]["effort"]
    if tool_bridge.get("chat_tools"):
        chat["tools"] = tool_bridge["chat_tools"]
    if body.get("tool_choice") is not None:
        chat["tool_choice"] = _tool_choice_to_chat(body["tool_choice"], tool_bridge)
    if body.get("parallel_tool_calls") is not None:
        chat["parallel_tool_calls"] = body["parallel_tool_calls"]
    if response_format:
        chat["response_format"] = response_format
    return chat


def chat_to_responses(body: dict) -> dict:
    """Chat Completions 请求体 -> Responses 请求体"""
    items = []
    instruction_parts = []
    for m in body.get("messages", []):
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        if role in ("system", "developer"):
            text = _content_to_text(m.get("content"))
            if text:
                instruction_parts.append(text)
            continue
        if role == "tool":
            out = m.get("content", "")
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False)
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id", ""), "output": out})
            continue
        if role == "assistant":
            reasoning_text = _extract_reasoning_text(m.get("reasoning_content"))
            if reasoning_text:
                items.append({
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": reasoning_text}],
                })
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if args is None:
                    args = "{}"
                args = _args_to_string(args)
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", "call_" + uuid.uuid4().hex[:8]),
                    "name": fn.get("name", ""),
                    "arguments": args,
                })
            content = _chat_content_to_responses(m.get("content"), assistant=True)
            if content:
                items.append({"type": "message", "role": "assistant", "content": content})
            continue
        content = _chat_content_to_responses(m.get("content"), assistant=False)
        if content:
            items.append({"type": "message", "role": role, "content": content})
    resp = {
        "model": body.get("model"),
        "input": items,
        "stream": body.get("stream", True),
    }
    if instruction_parts:
        resp["instructions"] = "\n\n".join(instruction_parts)
    for k in ("temperature", "top_p", "seed", "stop", "presence_penalty", "frequency_penalty", "logit_bias", "user", "metadata"):
        if body.get(k) is not None:
            resp[k] = body[k]
    if body.get("max_completion_tokens"):
        resp["max_output_tokens"] = body["max_completion_tokens"]
    elif body.get("max_tokens"):
        resp["max_output_tokens"] = body["max_tokens"]
    effort = body.get("reasoning_effort")
    if not effort and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    if not effort and isinstance(body.get("reasoning"), str):
        effort = body["reasoning"]
    if effort:
        resp["reasoning"] = {"effort": effort}
    if body.get("tools"):
        resp["tools"] = []
        for t in body["tools"]:
            if isinstance(t, dict) and t.get("type") == "function":
                fn = t.get("function", {})
                resp["tools"].append({
                    "type": "function",
                    "name": fn.get("name", t.get("name", "")),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
    if body.get("tool_choice") is not None:
        resp["tool_choice"] = _tool_choice_to_responses(body["tool_choice"])
    if body.get("parallel_tool_calls") is not None:
        resp["parallel_tool_calls"] = body["parallel_tool_calls"]
    text_format = _chat_response_format_to_responses(body.get("response_format"))
    if text_format:
        resp["text"] = {"format": text_format}
    return resp

# ------------------ SSE 解析 ------------------

def sse_bytes(obj) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n"


def _parse_sse_text(text: str):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                yield DONE
                return
            try:
                yield json.loads(payload)
            except Exception:
                pass


async def iter_upstream_events(resp: httpx.Response):
    """把上游响应解析成一个个 JSON 事件；[DONE] 用哨兵表示。支持 SSE 或整段 JSON。"""
    ctype = resp.headers.get("content-type", "")
    if "text/event-stream" not in ctype:
        text = (await resp.aread()).decode("utf-8", "replace")
        try:
            yield json.loads(text)
        except Exception:
            for evt in _parse_sse_text(text):
                yield evt
        return
    buffer = b""
    async for chunk in resp.aiter_bytes():
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                yield DONE
                return
            try:
                yield json.loads(payload)
            except Exception:
                pass
    if buffer.strip():
        line = buffer.strip()
        if line.startswith(b"data:"):
            payload = line[5:].strip()
            if payload != b"[DONE]":
                try:
                    yield json.loads(payload)
                except Exception:
                    pass


class UsageSniffer:
    """边转发边嗅探流里的 usage 字段（responses 的 response.completed 或 chat 的末块）。"""

    def __init__(self):
        self.usage = None
        self.status = "ok"
        self._buf = b""

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload in (b"[DONE]", b""):
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("type") == "response.failed":
                self.status = "failed"
            u = obj.get("usage")
            if u is None and isinstance(obj.get("response"), dict):
                u = obj["response"].get("usage")
            if isinstance(u, dict) and (u.get("total_tokens") is not None or u.get("input_tokens") is not None or u.get("prompt_tokens") is not None):
                self.usage = u


# ------------------ Chat -> Responses 转换 ------------------

class ChatToResponses:
    """把上游 Chat Completions 响应（SSE 或 JSON）转换为 Responses SSE 事件。"""

    def __init__(self, model: str, custom_tool_names=None, tool_bridge=None):
        self.model = model or ""
        self.tool_bridge = tool_bridge or {"by_chat": {}}
        self.custom_tool_names = set(custom_tool_names or [])
        self.response_id = "resp_" + uuid.uuid4().hex
        self.msg_id = "msg_" + uuid.uuid4().hex
        self.rs_id = "rs_" + uuid.uuid4().hex
        self.created_at = int(time.time())
        self.output = []
        self.msg_started = False
        self.msg_phase = None
        self.rs_started = False
        self.msg_text = []
        self.rs_text = []
        self.tool_calls = {}
        self.has_tool_calls = False
        self.usage = None
        self.finish_reason = None
        self.upstream_error = None

    def created_event(self) -> dict:
        return {
            "type": "response.created",
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "in_progress",
                "model": self.model,
                "output": [],
            },
        }

    def _set_upstream_error(self, error, default_message="Upstream request failed") -> None:
        if isinstance(error, str):
            error = {"message": error}
        elif not isinstance(error, dict):
            error = {}
        error_type = error.get("type")
        code = error.get("code")
        if not code and error_type not in (None, "error"):
            code = error_type
        normalized = {
            "code": str(code or "upstream_error"),
            "message": str(error.get("message") or default_message),
        }
        if error.get("param") is not None:
            normalized["param"] = error["param"]
        self.upstream_error = normalized

    @classmethod
    def _normalize_text_chunk(cls, value, preferred_keys=None) -> str:
        """Normalize provider-specific structured text chunks into stable strings."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8", errors="replace")
        if isinstance(value, (list, tuple)):
            return "".join(cls._normalize_text_chunk(item, preferred_keys) for item in value)
        if isinstance(value, dict):
            keys = preferred_keys or (
                "text",
                "content",
                "summary",
                "reasoning_content",
                "reasoning",
                "value",
                "delta",
            )
            for key in keys:
                if key not in value:
                    continue
                text = cls._normalize_text_chunk(value[key], preferred_keys)
                if text:
                    return text
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (bool, int, float)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)

    def _msg_index(self) -> int:
        return len(self.output) - 1

    def _start_message(self, phase: str) -> list:
        if self.msg_started or not self.msg_text:
            return []
        self.msg_started = True
        self.msg_phase = phase
        item = {
            "id": self.msg_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
            "phase": phase,
        }
        self.output.append(item)
        output_index = self._msg_index()
        events = [
            {"type": "response.output_item.added", "output_index": output_index, "item": item},
            {
                "type": "response.content_part.added",
                "item_id": self.msg_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ]
        for text in self.msg_text:
            events.append({
                "type": "response.output_text.delta",
                "item_id": self.msg_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            })
        return events

    def _start_tool_call(self, rec: dict) -> list:
        if rec.get("started"):
            return []
        rec["started"] = True
        rec["id"] = rec.get("id") or ("call_" + uuid.uuid4().hex[:8])
        chat_name = rec.get("chat_name") or rec.get("name") or ""
        spec = (self.tool_bridge.get("by_chat") or {}).get(chat_name) or {}
        rec["kind"] = spec.get("kind") or ("custom" if chat_name in self.custom_tool_names else "function")
        rec["name"] = spec.get("name") or chat_name
        rec["namespace"] = spec.get("namespace")
        rec["execution"] = spec.get("execution")
        if rec["kind"] == "custom":
            prefix, item_type, value_key, empty_value = "ctc_", "custom_tool_call", "input", ""
        elif rec["kind"] == "tool_search":
            prefix, item_type, value_key, empty_value = "tsc_", "tool_search_call", "arguments", {}
        else:
            prefix, item_type, value_key, empty_value = "fc_", "function_call", "arguments", ""
        item = {
            "id": prefix + uuid.uuid4().hex[:16],
            "type": item_type,
            "status": "in_progress",
            "call_id": rec["id"],
        }
        if rec["kind"] != "tool_search":
            item["name"] = rec.get("name", "")
        if rec.get("namespace"):
            item["namespace"] = rec["namespace"]
        if rec["kind"] == "tool_search":
            item["execution"] = rec.get("execution") or "client"
        item[value_key] = empty_value
        rec["item_id"] = item["id"]
        self.output.append(item)
        return [{"type": "response.output_item.added", "output_index": self._msg_index(), "item": item}]

    def on_chunk(self, chunk: dict) -> list:
        events = []
        if chunk.get("error") is not None:
            self._set_upstream_error(chunk.get("error"))
            return events
        if chunk.get("type") == "error":
            self._set_upstream_error(chunk)
            return events
        if chunk.get("type") == "response.failed":
            response = chunk.get("response") or {}
            self._set_upstream_error(response.get("error"), "Upstream response failed")
            return events
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            return events
        c = choices[0]
        if c.get("finish_reason"):
            self.finish_reason = c["finish_reason"]
        payload = c.get("delta") or c.get("message") or {}
        if not isinstance(payload, dict):
            payload = {"content": payload}
        reasoning_keys = (
            "summary",
            "text",
            "content",
            "reasoning_content",
            "reasoning",
            "value",
            "delta",
        )
        reasoning = self._normalize_text_chunk(payload.get("reasoning_content"), reasoning_keys)
        if not reasoning:
            reasoning = self._normalize_text_chunk(payload.get("reasoning"), reasoning_keys)
        if reasoning:
            if not self.rs_started:
                self.rs_started = True
                item = {"id": self.rs_id, "type": "reasoning", "status": "in_progress", "summary": [], "content": []}
                self.output.append(item)
                events.append({"type": "response.output_item.added", "output_index": self._msg_index(), "item": item})
            self.rs_text.append(reasoning)
            events.append({
                "type": "response.reasoning_summary_text.delta",
                "item_id": self.rs_id,
                "output_index": self._msg_index(),
                "summary_index": 0,
                "delta": reasoning,
            })
        content = self._normalize_text_chunk(payload.get("content"))
        if content:
            self.msg_text.append(content)
            if self.msg_started:
                events.append({
                    "type": "response.output_text.delta",
                    "item_id": self.msg_id,
                    "output_index": self._item_index(self.msg_id),
                    "content_index": 0,
                    "delta": content,
                })
        tool_chunks = payload.get("tool_calls") or []
        if tool_chunks:
            self.has_tool_calls = True
            events.extend(self._start_message("commentary"))
        for tc in tool_chunks:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", len(self.tool_calls))
            rec = self.tool_calls.setdefault(idx, {
                "id": "",
                "name": "",
                "chat_name": "",
                "args": [],
                "started": False,
                "item_id": "",
                "kind": None,
            })
            fn = tc.get("function") or {}
            if tc.get("id") and not rec.get("id"):
                rec["id"] = tc["id"]
            if fn.get("name"):
                rec["chat_name"] = fn["name"]
            previous_args = "".join(rec["args"])
            just_started = False
            if not rec["started"] and rec.get("chat_name"):
                events.extend(self._start_tool_call(rec))
                just_started = True
            if just_started and rec.get("kind") == "function" and previous_args:
                events.append({
                    "type": "response.function_call_arguments.delta",
                    "item_id": rec["item_id"],
                    "output_index": self._item_index(rec["item_id"]),
                    "delta": previous_args,
                })
            args_delta = fn.get("arguments") or ""
            if args_delta:
                rec["args"].append(args_delta)
                if rec.get("started") and rec.get("kind") == "function":
                    events.append({
                    "type": "response.function_call_arguments.delta",
                    "item_id": rec["item_id"],
                    "output_index": self._item_index(rec["item_id"]),
                    "delta": args_delta,
                })
        if self.finish_reason and not self.msg_started and self.msg_text:
            phase = "commentary" if self.has_tool_calls or self.finish_reason == "tool_calls" else "final_answer"
            events.extend(self._start_message(phase))
        return events

    def _item_index(self, item_id: str) -> int:
        for i, it in enumerate(self.output):
            if it.get("id") == item_id:
                return i
        return 0

    def final_events(self) -> list:
        events = []
        if self.msg_text and not self.msg_started:
            phase = "commentary" if self.has_tool_calls or self.finish_reason == "tool_calls" else "final_answer"
            events.extend(self._start_message(phase))
        if self.msg_started:
            text = "".join(self.msg_text)
            events.append({
                "type": "response.output_text.done",
                "item_id": self.msg_id,
                "output_index": self._item_index(self.msg_id),
                "content_index": 0,
                "text": text,
            })
            events.append({
                "type": "response.content_part.done",
                "item_id": self.msg_id,
                "output_index": self._item_index(self.msg_id),
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            })
            events.append({
                "type": "response.output_item.done",
                "output_index": self._item_index(self.msg_id),
                "item": {
                    "id": self.msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                    "phase": self.msg_phase,
                },
            })
            for it in self.output:
                if it.get("id") == self.msg_id:
                    it["status"] = "completed"
                    it["content"] = [{"type": "output_text", "text": text, "annotations": []}]
                    it["phase"] = self.msg_phase
        if self.rs_started:
            text = "".join(self.rs_text)
            events.append({
                "type": "response.reasoning_summary_text.done",
                "item_id": self.rs_id,
                "output_index": self._item_index(self.rs_id),
                "summary_index": 0,
                "text": text,
            })
            events.append({
                "type": "response.output_item.done",
                "output_index": self._item_index(self.rs_id),
                "item": {
                    "id": self.rs_id,
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": text}],
                    "content": [],
                },
            })
            for it in self.output:
                if it.get("id") == self.rs_id:
                    it["status"] = "completed"
                    it["summary"] = [{"type": "summary_text", "text": text}]
                    it["content"] = []
        for idx in sorted(self.tool_calls):
            rec = self.tool_calls[idx]
            if not rec.get("started"):
                events.extend(self._start_tool_call(rec))
            if not rec.get("started"):
                continue
            args = "".join(rec["args"])
            item_id = rec.get("item_id", "")
            output_index = self._item_index(item_id)
            if rec.get("kind") == "custom":
                tool_input = _custom_input_from_chat_arguments(args)
                if tool_input:
                    events.append({
                        "type": "response.custom_tool_call_input.delta",
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": tool_input,
                    })
                events.append({
                    "type": "response.custom_tool_call_input.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "input": tool_input,
                })
                done_item = {
                    "id": item_id,
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": rec["id"],
                    "name": rec["name"],
                    "input": tool_input,
                }
            elif rec.get("kind") == "tool_search":
                try:
                    search_args = json.loads(args or "{}")
                except Exception:
                    search_args = {"query": args}
                done_item = {
                    "id": item_id,
                    "type": "tool_search_call",
                    "status": "completed",
                    "call_id": rec["id"],
                    "arguments": search_args,
                    "execution": rec.get("execution") or "client",
                }
            else:
                events.append({
                    "type": "response.function_call_arguments.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "arguments": args,
                })
                done_item = {
                    "id": item_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": rec["id"],
                    "name": rec["name"],
                    "arguments": args,
                }
            if rec.get("namespace") and rec.get("kind") in ("function", "custom"):
                done_item["namespace"] = rec["namespace"]
            events.append({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": done_item,
            })
            for it in self.output:
                if it.get("id") == item_id:
                    it.update(done_item)
        status = "completed"
        terminal_type = "response.completed"
        incomplete = None
        error = None
        if self.upstream_error:
            status = "failed"
            terminal_type = "response.failed"
            error = self.upstream_error
        elif self.finish_reason == "length":
            status = "incomplete"
            terminal_type = "response.incomplete"
            incomplete = {"reason": "max_output_tokens"}
        elif self.finish_reason == "content_filter":
            status = "incomplete"
            terminal_type = "response.incomplete"
            incomplete = {"reason": "content_filter"}
        elif not self.finish_reason:
            status = "failed"
            terminal_type = "response.failed"
            error = {
                "code": "upstream_stream_terminated",
                "message": "Upstream stream ended before finish_reason",
            }
        elif self.finish_reason not in ("stop", "tool_calls", "function_call"):
            status = "failed"
            terminal_type = "response.failed"
            error = {
                "code": "unexpected_finish_reason",
                "message": f"Unexpected upstream finish_reason: {self.finish_reason}",
            }
        response = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": self.output,
            "usage": self._map_usage(self.usage),
        }
        if incomplete:
            response["incomplete_details"] = incomplete
        if error:
            response["error"] = error
        events.append({"type": terminal_type, "response": response})
        return events

    @staticmethod
    def _map_usage(u) -> dict | None:
        if not u:
            return None
        return {
            "input_tokens": u.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)},
            "output_tokens": u.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)},
            "total_tokens": u.get("total_tokens", 0),
        }

    async def convert(self, resp: httpx.Response):
        yield self.created_event()
        try:
            async for evt in iter_upstream_events(resp):
                if evt is DONE:
                    break
                if isinstance(evt, dict):
                    for e in self.on_chunk(evt):
                        yield e
                    if self.upstream_error:
                        break
        except Exception as exc:
            self._set_upstream_error(
                {"code": "upstream_stream_error", "message": str(exc)},
                "Upstream stream failed",
            )
        for e in self.final_events():
            yield e


# ------------------ Responses -> Chat 转换 ------------------

class ResponsesToChat:
    """把上游 Responses 响应（SSE 或 JSON）转换为 Chat Completions SSE 块。"""

    def __init__(self, model: str):
        self.model = model or ""
        self.fc_index = {}
        self.tool_records = []
        self.tool_by_item_id = {}
        self.tool_by_call_id = {}
        self.has_tool_calls = False
        self.role_sent = False
        self.usage = None
        self.status = "completed"

    @staticmethod
    def _chunk(delta: dict, finish_reason=None) -> dict:
        return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    def _ensure_role(self, events: list) -> None:
        if not self.role_sent:
            self.role_sent = True
            events.append(self._chunk({"role": "assistant", "content": ""}))

    def _find_tool(self, item_id="", call_id=""):
        if call_id and call_id in self.tool_by_call_id:
            return self.tool_by_call_id[call_id]
        if item_id and item_id in self.tool_by_item_id:
            return self.tool_by_item_id[item_id]
        return None

    def _register_tool(self, item: dict, events: list):
        item_type = item.get("type")
        if item_type not in ("function_call", "custom_tool_call", "tool_search_call"):
            return None
        item_id = str(item.get("id") or "")
        call_id = str(item.get("call_id") or item_id or ("call_" + uuid.uuid4().hex[:8]))
        rec = self._find_tool(item_id, call_id)
        if rec is not None:
            if item_id:
                self.tool_by_item_id[item_id] = rec
                self.fc_index[item_id] = rec["index"]
            if call_id:
                self.tool_by_call_id[call_id] = rec
                self.fc_index[call_id] = rec["index"]
            return rec

        name = "tool_search" if item_type == "tool_search_call" else item.get("name", "")
        rec = {
            "index": len(self.tool_records),
            "item_id": item_id,
            "call_id": call_id,
            "type": item_type,
            "name": _chat_tool_name(name, item.get("namespace")),
            "argument_parts": [],
            "custom_input_parts": [],
            "arguments_sent": False,
        }
        self.tool_records.append(rec)
        if item_id:
            self.tool_by_item_id[item_id] = rec
            self.fc_index[item_id] = rec["index"]
        self.tool_by_call_id[call_id] = rec
        self.fc_index[call_id] = rec["index"]
        self.has_tool_calls = True
        self._ensure_role(events)
        events.append(self._chunk({
            "tool_calls": [{
                "index": rec["index"],
                "id": call_id,
                "type": "function",
                "function": {"name": rec["name"], "arguments": ""},
            }]
        }))
        return rec

    def _emit_arguments(self, rec: dict, arguments: str, events: list) -> None:
        if rec.get("arguments_sent"):
            return
        events.append(self._chunk({
            "tool_calls": [{
                "index": rec["index"],
                "function": {"arguments": arguments},
            }]
        }))
        rec["arguments_sent"] = True
        rec["argument_parts"].append(arguments)

    def _emit_item_arguments(self, rec: dict, item: dict, events: list, final=False) -> None:
        if rec.get("arguments_sent"):
            return
        item_type = rec.get("type")
        if item_type == "custom_tool_call":
            tool_input = item.get("input")
            if tool_input is None:
                tool_input = "".join(rec["custom_input_parts"])
            if tool_input or final:
                self._emit_arguments(rec, _custom_tool_arguments(tool_input), events)
        elif item_type == "tool_search_call":
            arguments = item.get("arguments")
            if arguments is not None and (arguments or final):
                self._emit_arguments(rec, _args_to_string(arguments), events)
        else:
            arguments = item.get("arguments")
            if arguments is not None and (arguments or final):
                self._emit_arguments(rec, _args_to_string(arguments), events)

    def _handle_tool_item(self, item: dict, events: list, final=False) -> None:
        rec = self._register_tool(item, events)
        if rec is not None:
            self._emit_item_arguments(rec, item, events, final=final)

    def on_event(self, evt: dict) -> list:
        events = []
        t = evt.get("type")
        if t == "response.output_item.added":
            item = evt.get("item") or {}
            if item.get("type") == "message":
                self._ensure_role(events)
            else:
                self._handle_tool_item(item, events)
        elif t == "response.output_item.done":
            self._handle_tool_item(evt.get("item") or {}, events, final=True)
        elif t in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            d = evt.get("delta", "")
            if d:
                self._ensure_role(events)
                events.append(self._chunk({"reasoning_content": d, "reasoning": d}))
        elif t == "response.output_text.delta":
            d = evt.get("delta", "")
            if d:
                self._ensure_role(events)
                events.append(self._chunk({"content": d}))
        elif t == "response.function_call_arguments.delta":
            d = evt.get("delta", "")
            if d:
                item_id = evt.get("item_id", "")
                rec = self._find_tool(item_id=item_id)
                idx = rec["index"] if rec is not None else self.fc_index.get(item_id, 0)
                self.has_tool_calls = True
                events.append(self._chunk({"tool_calls": [{"index": idx, "function": {"arguments": d}}]}))
                if rec is not None:
                    rec["arguments_sent"] = True
                    rec["argument_parts"].append(d)
        elif t == "response.function_call_arguments.done":
            rec = self._find_tool(item_id=evt.get("item_id", ""))
            if rec is not None and not rec.get("arguments_sent"):
                self._emit_arguments(rec, _args_to_string(evt.get("arguments", "")), events)
        elif t == "response.custom_tool_call_input.delta":
            rec = self._find_tool(item_id=evt.get("item_id", ""))
            if rec is not None:
                rec["custom_input_parts"].append(str(evt.get("delta", "")))
        elif t == "response.custom_tool_call_input.done":
            rec = self._find_tool(item_id=evt.get("item_id", ""))
            if rec is not None and not rec.get("arguments_sent"):
                tool_input = evt.get("input")
                if tool_input is None:
                    tool_input = "".join(rec["custom_input_parts"])
                self._emit_arguments(rec, _custom_tool_arguments(tool_input), events)
        elif t == "response.completed":
            resp_obj = evt.get("response") or {}
            self.status = resp_obj.get("status", "completed")
            self.usage = resp_obj.get("usage")
            for item in resp_obj.get("output") or []:
                if isinstance(item, dict):
                    self._handle_tool_item(item, events, final=True)
        elif t == "response.failed":
            self.status = "failed"
        return events

    def _synthesize(self, response: dict) -> list:
        """非流式上游 JSON -> 合成事件序列"""
        events = []
        for item in response.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                events.append({
                    "type": "response.output_item.added",
                    "item": {"id": item.get("id", "msg_x"), "type": "message", "role": "assistant"},
                })
                text = ""
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                        text += part.get("text", "")
                if text:
                    events.append({"type": "response.output_text.delta", "item_id": item.get("id", "msg_x"), "delta": text})
            elif item.get("type") in ("function_call", "custom_tool_call", "tool_search_call"):
                item_type = item.get("type")
                prefix = {
                    "function_call": "fc",
                    "custom_tool_call": "ctc",
                    "tool_search_call": "tsc",
                }[item_type]
                item_id = item.get("id") or f"{prefix}_{len(events)}"
                added_item = {
                    key: item[key]
                    for key in ("type", "call_id", "name", "namespace", "execution")
                    if key in item
                }
                added_item["id"] = item_id
                events.append({
                    "type": "response.output_item.added",
                    "item": added_item,
                })
                if item_type == "function_call":
                    args = _args_to_string(item.get("arguments", ""))
                    if args:
                        events.append({"type": "response.function_call_arguments.delta", "item_id": item_id, "delta": args})
                elif item_type == "custom_tool_call":
                    events.append({
                        "type": "response.custom_tool_call_input.done",
                        "item_id": item_id,
                        "input": item.get("input", ""),
                    })
                else:
                    done_item = dict(item)
                    done_item["id"] = item_id
                    events.append({"type": "response.output_item.done", "item": done_item})
        events.append({
            "type": "response.completed",
            "response": {"status": response.get("status", "completed"), "usage": response.get("usage")},
        })
        return events

    def final_chunk(self) -> dict:
        fr = "stop"
        if self.has_tool_calls:
            fr = "tool_calls"
        elif self.status in ("incomplete", "failed"):
            fr = "length"
        chunk = self._chunk({}, finish_reason=fr)
        if self.usage:
            chunk["usage"] = self._map_usage(self.usage)
        return chunk

    @staticmethod
    def _map_usage(u) -> dict:
        return {
            "prompt_tokens": u.get("input_tokens", 0),
            "prompt_tokens_details": {"cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0)},
            "completion_tokens": u.get("output_tokens", 0),
            "completion_tokens_details": {"reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)},
            "total_tokens": u.get("total_tokens", 0),
        }

    async def convert(self, resp: httpx.Response):
        async for evt in iter_upstream_events(resp):
            if evt is DONE:
                break
            if isinstance(evt, dict):
                if evt.get("object") == "response" and not evt.get("type"):
                    evts = []
                    for synthesized in self._synthesize(evt):
                        evts.extend(self.on_event(synthesized))
                else:
                    evts = self.on_event(evt)
                for c in evts:
                    yield c
        yield self.final_chunk()

# ------------------ 转发主逻辑 ------------------

def upstream_headers(supplier: dict, request: Request) -> dict:
    headers = {
        "Authorization": f"Bearer {supplier.get('api_key', '')}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    beta = request.headers.get("openai-beta")
    if beta:
        headers["OpenAI-Beta"] = beta
    return headers


def _is_unsupported_parameter_error(error_text: str, parameter: str) -> bool:
    text = " ".join(str(error_text or "").lower().split())
    if parameter.lower() not in text:
        return False
    markers = (
        "unsupported",
        "not supported",
        "does not support",
        "doesn't support",
        "unknown parameter",
        "unrecognized parameter",
        "unexpected parameter",
        "not a valid parameter",
        "does not accept",
        "not permitted",
        "not allowed",
        "unavailable",
        "extra inputs are not permitted",
        "extra_forbidden",
    )
    return any(marker in text for marker in markers)


def _chat_images_to_text(body: dict) -> dict | None:
    """Replace Chat image parts with explicit text placeholders without mutating body."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None

    changed = False
    converted_messages = []
    placeholder = "[Image unavailable: this upstream accepts text-only messages.]"
    for message in messages:
        if not isinstance(message, dict):
            converted_messages.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            converted_messages.append(message)
            continue
        converted_content = []
        message_changed = False
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                converted_content.append({"type": "text", "text": placeholder})
                changed = True
                message_changed = True
            else:
                converted_content.append(part)
        if message_changed:
            converted_message = dict(message)
            converted_message["content"] = converted_content
            converted_messages.append(converted_message)
        else:
            converted_messages.append(message)

    if not changed:
        return None
    converted_body = dict(body)
    converted_body["messages"] = converted_messages
    return converted_body


def _forced_tool_instruction(tool_choice) -> str:
    name = ""
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "")
        if not name:
            name = str(tool_choice.get("name") or "")
    if name:
        return (
            f'Compatibility requirement: You must call the "{name}" tool before '
            "producing a final answer. Do not answer without making that tool call."
        )
    return (
        "Compatibility requirement: You must call one of the available tools before "
        "producing a final answer. Do not answer without making a tool call."
    )


def _next_chat_compat_fallback(
    body: dict,
    status_code: int,
    error_text: str,
    applied_fallbacks=(),
):
    """Return a compatible request copy and fallback name, or (None, None)."""
    if status_code != 400 or not isinstance(body, dict):
        return None, None

    applied = set(applied_fallbacks or ())
    normalized_error = " ".join(str(error_text or "").lower().split())
    tool_choice = body.get("tool_choice")
    forced_tool_choice = tool_choice not in (None, "auto", "none")
    image_variant_rejected = (
        "unknown variant" in normalized_error
        and "image_url" in normalized_error
        and "expected" in normalized_error
        and "text" in normalized_error
    )
    if "image_url" not in applied and image_variant_rejected:
        next_body = _chat_images_to_text(body)
        if next_body is not None:
            return next_body, "image_url"

    if (
        "tool_choice" not in applied
        and forced_tool_choice
        and "thinking mode does not support this tool_choice" in normalized_error
    ):
        next_body = dict(body)
        next_body["tool_choice"] = "auto"
        messages = body.get("messages")
        if not isinstance(messages, list):
            messages = []
        next_body["messages"] = [
            {"role": "system", "content": _forced_tool_instruction(tool_choice)},
            *messages,
        ]
        return next_body, "tool_choice"

    if (
        "response_format" not in applied
        and "response_format" in body
        and _is_unsupported_parameter_error(error_text, "response_format")
    ):
        next_body = dict(body)
        next_body.pop("response_format", None)
        return next_body, "response_format"

    if (
        "max_completion_tokens" not in applied
        and "max_completion_tokens" in body
        and _is_unsupported_parameter_error(error_text, "max_completion_tokens")
    ):
        next_body = dict(body)
        value = next_body.pop("max_completion_tokens")
        next_body["max_tokens"] = value
        return next_body, "max_completion_tokens"

    for parameter in ("parallel_tool_calls", "reasoning_effort"):
        if (
            parameter not in applied
            and parameter in body
            and _is_unsupported_parameter_error(error_text, parameter)
        ):
            next_body = dict(body)
            next_body.pop(parameter, None)
            return next_body, parameter

    return None, None


def _chat_json_to_responses_json(payload: dict, model: str, tool_bridge=None) -> dict:
    """Convert one complete Chat Completions response into a Responses object."""
    if not isinstance(payload, dict):
        raise ValueError("Upstream Chat response is not a JSON object")
    converter = ChatToResponses(model, tool_bridge=tool_bridge)
    converter.on_chunk(payload)
    terminal = converter.final_events()[-1]
    response = terminal.get("response") if isinstance(terminal, dict) else None
    if not isinstance(response, dict):
        raise ValueError("Unable to convert upstream Chat response")
    return response


def _responses_json_to_chat_json(payload: dict, model: str) -> dict:
    """Convert one complete Responses object into a Chat Completions response."""
    if not isinstance(payload, dict):
        raise ValueError("Upstream Responses response is not a JSON object")
    if payload.get("type") in ("response.completed", "response.incomplete", "response.failed"):
        payload = payload.get("response") or {}
    if not isinstance(payload, dict):
        raise ValueError("Upstream Responses response is missing its response object")

    text_parts = []
    reasoning_parts = []
    refusal_parts = []
    tool_calls = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message" and item.get("role", "assistant") == "assistant":
            for part in item.get("content") or []:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                    text_parts.append(str(part.get("text", "")))
                elif isinstance(part, dict) and part.get("type") == "refusal":
                    refusal_parts.append(str(part.get("refusal") or part.get("text") or ""))
            message_reasoning = _extract_reasoning_text(item.get("reasoning"))
            if message_reasoning:
                reasoning_parts.append(message_reasoning)
        elif item_type == "reasoning":
            reasoning = _extract_reasoning_text(item)
            if reasoning:
                reasoning_parts.append(reasoning)
        elif item_type in ("function_call", "custom_tool_call", "tool_search_call"):
            if item_type == "custom_tool_call":
                name = item.get("name", "")
                arguments = _custom_tool_arguments(item.get("input", ""))
            elif item_type == "tool_search_call":
                name = "tool_search"
                arguments = _args_to_string(item.get("arguments") or item.get("action") or {})
            else:
                name = item.get("name", "")
                arguments = _args_to_string(item.get("arguments", {}))
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                "type": "function",
                "function": {
                    "name": _chat_tool_name(name, item.get("namespace")),
                    "arguments": arguments,
                },
            })

    message = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else (None if tool_calls else ""),
    }
    if reasoning_parts:
        reasoning = "\n".join(reasoning_parts)
        message["reasoning_content"] = reasoning
        message["reasoning"] = reasoning
    if refusal_parts:
        message["refusal"] = "\n".join(refusal_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    status = payload.get("status", "completed")
    incomplete_reason = (payload.get("incomplete_details") or {}).get("reason")
    if status in ("incomplete", "failed"):
        finish_reason = "content_filter" if incomplete_reason == "content_filter" else "length"
    elif tool_calls:
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"

    result = {
        "id": payload.get("id") or ("chatcmpl_" + uuid.uuid4().hex),
        "object": "chat.completion",
        "created": payload.get("created_at", int(time.time())),
        "model": payload.get("model") or model,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason,
        }],
    }
    usage = payload.get("usage")
    if isinstance(usage, dict):
        result["usage"] = ResponsesToChat._map_usage(usage)
    return result


def _update_record_usage(rec: dict, usage) -> None:
    if not isinstance(usage, dict):
        return
    rec["input_tokens"] = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    rec["cached_tokens"] = (
        (usage.get("input_tokens_details") or {}).get("cached_tokens")
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or usage.get("cached_tokens")
        or 0
    )
    rec["output_tokens"] = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    rec["reasoning_tokens"] = (
        (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        or 0
    )
    rec["total_tokens"] = usage.get("total_tokens") or (rec["input_tokens"] + rec["output_tokens"])


async def relay(request: Request, incoming_wire: str):
    cfg = load_config()
    try:
        body = await request.json()
    except Exception:
        body = {}
    model = (body or {}).get("model", "")
    supplier = await find_supplier(cfg, model)
    if not supplier:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"模型 {model} 没有可用的上游：请在控制台中添加供应商或配置 model_map。"}},
        )
    upstream_wire = str(supplier.get("wire_api", "responses")).lower()
    if upstream_wire not in ("responses", "chat"):
        upstream_wire = "responses"
    path = "responses" if upstream_wire == "responses" else "chat/completions"
    base = str(supplier.get("base_url", "")).rstrip("/")
    url = f"{base}/{path}"
    tool_bridge = _build_tool_bridge(body) if incoming_wire == "responses" and upstream_wire == "chat" else None

    if incoming_wire == upstream_wire:
        out_body = body
    elif incoming_wire == "responses":
        out_body = responses_to_chat(body, tool_bridge=tool_bridge)
    else:
        out_body = chat_to_responses(body)

    try:
        await vision.apply_image_policy(out_body, supplier, cfg, model)
    except Exception:
        # 图片策略异常时退化为纯文本占位，避免上游因 image_url 直接失败。
        vision.strip_images(out_body)

    rec = {
        "ts": time.time(),
        "model": model,
        "supplier": supplier.get("name", ""),
        "upstream_wire": upstream_wire,
        "incoming_wire": incoming_wire,
        "http_status": None,
        "status": "ok",
        "input_tokens": None,
        "cached_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "duration_ms": None,
        "reasoning_effort": (
            body.get("reasoning_effort")
            or ((body.get("reasoning") or {}).get("effort") if isinstance(body.get("reasoning"), dict) else None)
        ),
        "error": None,
        "retry_count": 0,
    }
    start_mono = time.monotonic()

    client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    resp = None
    last_exc = None
    response_error_text = None
    transport_retries = 0
    applied_compat_fallbacks = frozenset()
    request_body = out_body
    while True:
        try:
            req = client.build_request("POST", url, json=request_body, headers=upstream_headers(supplier, request))
            resp = await client.send(req, stream=True)
            last_exc = None
        except Exception as exc:
            last_exc = exc
            resp = None
            if transport_retries < 2:
                await asyncio.sleep(0.5 * (2 ** transport_retries))
                transport_retries += 1
                rec["retry_count"] += 1
                continue
            break
        if resp.status_code >= 500 and transport_retries < 2:
            await resp.aread()
            await resp.aclose()
            resp = None
            await asyncio.sleep(0.5 * (2 ** transport_retries))
            transport_retries += 1
            rec["retry_count"] += 1
            continue

        response_error_text = None
        if resp.status_code == 400 and upstream_wire == "chat":
            response_error_text = (await resp.aread()).decode("utf-8", "replace")
            next_body, fallback = _next_chat_compat_fallback(
                request_body,
                resp.status_code,
                response_error_text,
                applied_compat_fallbacks,
            )
            if fallback:
                await resp.aclose()
                resp = None
                request_body = next_body
                applied_compat_fallbacks = applied_compat_fallbacks | {fallback}
                rec["retry_count"] += 1
                continue
        break

    if resp is None:
        await client.aclose()
        exc = last_exc or RuntimeError("上游连续返回临时错误")
        rec["status"] = "error"
        rec["http_status"] = 0
        rec["error"] = str(exc)[:300]
        rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
        stats.record(rec)
        return JSONResponse(status_code=502, content={"error": {"message": f"连接上游 {base} 失败：{exc}"}})

    rec["http_status"] = resp.status_code
    if resp.status_code >= 400:
        text = response_error_text
        if text is None:
            text = (await resp.aread()).decode("utf-8", "replace")
        await resp.aclose()
        await client.aclose()
        rec["status"] = "error"
        rec["error"] = text[:300]
        rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
        stats.record(rec)
        return Response(content=text, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))

    if body.get("stream") is False:
        try:
            raw_body = await resp.aread()
            upstream_json = None
            try:
                upstream_json = resp.json()
            except Exception:
                if incoming_wire != upstream_wire:
                    raise ValueError("Upstream returned invalid JSON for a non-stream request")

            if incoming_wire == upstream_wire:
                content_type = resp.headers.get("content-type") or "application/json"
                result = Response(
                    content=raw_body,
                    status_code=resp.status_code,
                    headers={"content-type": content_type},
                )
                result_json = upstream_json
            elif incoming_wire == "responses":
                result_json = _chat_json_to_responses_json(upstream_json, model, tool_bridge=tool_bridge)
                result = JSONResponse(content=result_json, status_code=resp.status_code)
            else:
                result_json = _responses_json_to_chat_json(upstream_json, model)
                result = JSONResponse(content=result_json, status_code=resp.status_code)

            if isinstance(result_json, dict):
                _update_record_usage(rec, result_json.get("usage"))
                if rec["status"] == "ok" and result_json.get("status") == "failed":
                    rec["status"] = "failed"
            rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
            stats.record(rec)
            return result
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = str(exc)[:300]
            rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
            stats.record(rec)
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Failed to convert non-stream upstream response: {exc}"}},
            )
        finally:
            await resp.aclose()
            await client.aclose()

    sniffer = UsageSniffer()

    async def gen():
        try:
            if incoming_wire == upstream_wire:
                async for chunk in resp.aiter_bytes():
                    sniffer.feed(chunk)
                    yield chunk
            elif incoming_wire == "responses":
                conv = ChatToResponses(model, tool_bridge=tool_bridge)
                async for evt in conv.convert(resp):
                    chunk = sse_bytes(evt)
                    sniffer.feed(chunk)
                    yield chunk
            else:
                conv = ResponsesToChat(model)
                async for evt in conv.convert(resp):
                    chunk = sse_bytes(evt)
                    sniffer.feed(chunk)
                    yield chunk
                yield b"data: [DONE]\n\n"
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = str(exc)[:300]
            raise
        finally:
            u = sniffer.usage
            _update_record_usage(rec, u)
            if rec["status"] == "ok" and sniffer.status == "failed":
                rec["status"] = "failed"
            rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
            stats.record(rec)
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(gen(), status_code=resp.status_code, media_type="text/event-stream")


# ------------------ 控制台 API ------------------

@app.get("/")
async def index():
    return FileResponse(DASHBOARD_PATH)


@app.get("/api/health")
async def api_health():
    return {"ok": True, "time": time.time()}


@app.get("/api/config")
async def api_get_config():
    return normalized_config(load_config())


def validate_supplier(s: dict) -> str | None:
    if not isinstance(s, dict):
        return "供应商必须是对象"
    if not s.get("name"):
        return "供应商缺少名称 name"
    if not s.get("base_url"):
        return f"供应商 {s.get('name')} 缺少接口地址 base_url"
    if s.get("wire_api") not in (None, "", "responses", "chat"):
        return f"供应商 {s.get('name')} 的 wire_api 只能是 responses 或 chat"
    if s.get("vlm_prompt_mode") not in (None, "", "main-model", "template"):
        return f"供应商 {s.get('name')} 的 vlm_prompt_mode 只能是 main-model 或 template"
    image_handling = s.get("image_handling")
    if image_handling is not None:
        if isinstance(image_handling, str):
            if image_handling.strip().lower() not in ("send-as-is", "strip", "vlm"):
                return f"供应商 {s.get('name')} 的 image_handling 只能是 send-as-is、strip 或 vlm"
        elif isinstance(image_handling, dict):
            for model, mode in image_handling.items():
                if not isinstance(mode, str) or mode.strip().lower() not in ("send-as-is", "strip", "vlm"):
                    return f"供应商 {s.get('name')} 的模型 {model} 图片处理模式无效"
        else:
            return f"供应商 {s.get('name')} 的 image_handling 必须是字符串或对象"
    return None


@app.put("/api/config")
async def api_put_config(request: Request):
    try:
        cfg = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})
    if not isinstance(cfg.get("suppliers"), list):
        return JSONResponse(status_code=400, content={"error": "suppliers 必须是数组"})
    for s in cfg.get("suppliers", []):
        err = validate_supplier(s)
        if err:
            return JSONResponse(status_code=400, content={"error": err})
    if not isinstance(cfg.get("model_map", {}), dict):
        return JSONResponse(status_code=400, content={"error": "model_map 必须是对象"})
    try:
        save_config(cfg)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"保存失败：{exc}"})
    return {"ok": True}


@app.post("/api/suppliers")
async def api_add_supplier(request: Request):
    cfg = load_config()
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})
    err = validate_supplier(data)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    suppliers = cfg.setdefault("suppliers", [])
    if any(s.get("name") == data.get("name") for s in suppliers):
        return JSONResponse(status_code=400, content={"error": f"供应商 {data.get('name')} 已存在"})
    suppliers.append(data)
    save_config(cfg)
    return {"ok": True}


@app.put("/api/suppliers/{name}")
async def api_update_supplier(name: str, request: Request):
    cfg = load_config()
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})
    err = validate_supplier(data)
    if err:
        return JSONResponse(status_code=400, content={"error": err})
    suppliers = cfg.setdefault("suppliers", [])
    for i, s in enumerate(suppliers):
        if s.get("name") == name:
            new_name = str(data.get("name", "")).strip()
            if new_name != name and any(x.get("name") == new_name for x in suppliers):
                return JSONResponse(status_code=400, content={"error": f"供应商 {new_name} 已存在"})
            suppliers[i] = data
            if new_name != name:
                # 改名后级联更新：路由映射 / 默认供应商 / 模型缓存 / 隐藏列表
                mm = cfg.get("model_map", {})
                for k, v in list(mm.items()):
                    if v == name:
                        mm[k] = new_name
                if cfg.get("default_supplier") == name:
                    cfg["default_supplier"] = new_name
                if name in _model_cache:
                    _model_cache[new_name] = _model_cache.pop(name)
                bl = cfg.get("model_blacklist") or {}
                if name in bl:
                    bl[new_name] = bl.pop(name)
            save_config(cfg)
            return {"ok": True}
    return JSONResponse(status_code=404, content={"error": f"供应商 {name} 不存在"})


@app.delete("/api/suppliers/{name}")
async def api_delete_supplier(name: str):
    cfg = load_config()
    suppliers = cfg.setdefault("suppliers", [])
    new_list = [s for s in suppliers if s.get("name") != name]
    if len(new_list) == len(suppliers):
        return JSONResponse(status_code=404, content={"error": f"供应商 {name} 不存在"})
    cfg["suppliers"] = new_list
    mm = cfg.get("model_map", {})
    cfg["model_map"] = {k: v for k, v in mm.items() if v != name}
    if cfg.get("default_supplier") == name:
        cfg["default_supplier"] = ""
    save_config(cfg)
    return {"ok": True}


@app.get("/api/model_map")
async def api_get_model_map():
    cfg = load_config()
    return {"model_map": cfg.get("model_map", {}), "default_supplier": cfg.get("default_supplier", "")}


@app.put("/api/model_map")
async def api_put_model_map(request: Request):
    cfg = load_config()
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})
    names = {s.get("name") for s in cfg.get("suppliers", [])}
    mm = data.get("model_map", {})
    if not isinstance(mm, dict):
        return JSONResponse(status_code=400, content={"error": "model_map 必须是对象"})
    for model, sup in mm.items():
        if sup not in names:
            return JSONResponse(status_code=400, content={"error": f"供应商 {sup} 不存在"})
    default = data.get("default_supplier", cfg.get("default_supplier", ""))
    if default and default not in names:
        return JSONResponse(status_code=400, content={"error": f"默认供应商 {default} 不存在"})
    cfg["model_map"] = mm
    cfg["default_supplier"] = default or ""
    save_config(cfg)
    return {"ok": True}


@app.delete("/api/model_map/{model}")
async def api_delete_model_map(model: str):
    cfg = load_config()
    mm = cfg.get("model_map", {})
    if model not in mm:
        return JSONResponse(status_code=404, content={"error": f"模型 {model} 不在 model_map 中"})
    del mm[model]
    save_config(cfg)
    return {"ok": True}


@app.post("/api/models")
async def api_add_model(request: Request):
    """添加模型：mode=map 加入路由映射；mode=manual 加入供应商手动列表"""
    cfg = load_config()
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})
    model = str(data.get("model", "")).strip()
    supplier_name = data.get("supplier", "")
    mode = data.get("mode", "map")
    if not model:
        return JSONResponse(status_code=400, content={"error": "模型名不能为空"})
    suppliers = cfg.get("suppliers", [])
    target = next((s for s in suppliers if s.get("name") == supplier_name), None)
    if not target:
        return JSONResponse(status_code=400, content={"error": f"供应商 {supplier_name} 不存在"})
    if mode == "manual":
        models = target.setdefault("models", [])
        if model not in models:
            models.append(model)
    else:
        cfg.setdefault("model_map", {})[model] = supplier_name
    save_config(cfg)
    return {"ok": True}


@app.post("/api/test-supplier")
async def api_test_supplier(request: Request):
    """测试供应商连通性：请求它的 /v1/models"""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})
    base = str(data.get("base_url", "")).strip().rstrip("/")
    key = str(data.get("api_key", "")).strip()
    if not base:
        return JSONResponse(status_code=400, content={"error": "缺少 base_url"})
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(_upstream_models_url(base), headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 200:
            data_list = r.json().get("data", [])
            ids = [m.get("id") for m in data_list if isinstance(m, dict) and m.get("id")]
            return {"ok": True, "models": ids, "count": len(ids)}
        return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


@app.get("/api/merged-models")
async def api_merged_models():
    cfg = load_config()
    data = await merged_model_list(cfg)
    manual_map = {s.get("name", ""): set(s.get("models") or []) for s in cfg.get("suppliers", [])}
    models = []
    for m in data:
        owner = m["owned_by"]
        if owner == "model_map":
            source = "map"
        elif m["id"] in manual_map.get(owner, set()):
            source = "manual"
        else:
            source = "upstream"
        models.append({"id": m["id"], "supplier": owner, "source": source})
    bl = cfg.get("model_blacklist") or {}
    hidden = [{"id": mid, "supplier": name} for name, mids in bl.items() for mid in (mids or [])]
    return {"models": models, "hidden": hidden}


@app.post("/api/fetch-models")
async def api_fetch_all_models():
    """强制从所有上游重新拉取模型列表（只对未配置手动列表的供应商生效）。"""
    cfg = load_config()
    refreshed, failed = [], []
    async with httpx.AsyncClient(timeout=15) as client:
        for s in cfg.get("suppliers", []):
            if s.get("models"):
                continue
            name = str(s.get("name", ""))
            try:
                base = str(s.get("base_url", "")).rstrip("/")
                r = await client.get(_upstream_models_url(base), headers={"Authorization": f"Bearer {s.get('api_key', '')}"})
                if r.status_code == 200:
                    ids = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")]
                    _model_cache[name] = {"ids": ids, "at": time.time()}
                    refreshed.append(name)
                else:
                    failed.append(name)
            except Exception:
                failed.append(name)
    return {"ok": True, "refreshed": refreshed, "failed": failed}


@app.post("/api/suppliers/{name}/fetch-models")
async def api_fetch_supplier_models(name: str):
    """强制从该供应商上游拉取模型列表，返回可见模型（过滤已隐藏）。"""
    cfg = load_config()
    s = next((x for x in cfg.get("suppliers", []) if x.get("name") == name), None)
    if not s:
        return JSONResponse(status_code=404, content={"error": f"供应商 {name} 不存在"})
    try:
        base = str(s.get("base_url", "")).rstrip("/")
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(_upstream_models_url(base), headers={"Authorization": f"Bearer {s.get('api_key', '')}"})
        if r.status_code != 200:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"})
        ids = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")]
    except Exception as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)[:300]})
    _model_cache[name] = {"ids": ids, "at": time.time()}
    black = _blacklisted_ids(cfg, name)
    visible = [m for m in ids if m not in black]
    return {"ok": True, "models": visible, "count": len(visible), "hidden": len(ids) - len(visible)}


@app.delete("/api/suppliers/{name}/models/{model}")
async def api_delete_supplier_model(name: str, model: str):
    """隐藏一个模型：从手动列表移除，并加入该供应商的隐藏名单（不会重新出现）。"""
    cfg = load_config()
    s = next((x for x in cfg.get("suppliers", []) if x.get("name") == name), None)
    if not s:
        return JSONResponse(status_code=404, content={"error": f"供应商 {name} 不存在"})
    if "models" in s and model in s["models"]:
        s["models"].remove(model)
    bl = cfg.setdefault("model_blacklist", {}).setdefault(name, [])
    if model not in bl:
        bl.append(model)
    save_config(cfg)
    return {"ok": True}


@app.post("/api/suppliers/{name}/unhide-models")
async def api_unhide_supplier_models(name: str):
    """恢复该供应商所有被隐藏的模型。"""
    cfg = load_config()
    bl = cfg.get("model_blacklist") or {}
    if name in bl:
        del bl[name]
        save_config(cfg)
    return {"ok": True}


@app.post("/api/unhide-models")
async def api_unhide_all_models():
    """恢复所有供应商被隐藏的模型。"""
    cfg = load_config()
    if cfg.get("model_blacklist"):
        cfg["model_blacklist"] = {}
        save_config(cfg)
    return {"ok": True}


@app.get("/api/stats/summary")
async def api_stats_summary(model: str = "", hours: int = 24, start: float = None, end: float = None):
    try:
        hours = int(hours)
    except Exception:
        hours = 24
    if hours < 0:
        hours = 0
    start_f = float(start) if start is not None else None
    end_f = float(end) if end is not None else None
    return stats.summary(load_config(), model=str(model).strip(), hours=hours, start=start_f, end=end_f)


@app.get("/api/stats/raw")
async def api_stats_raw(
    limit: int = 50,
    offset: int = 0,
    hours: int = 0,
    start: float = None,
    end: float = None,
    model: str = "",
):
    """最近请求（分页）：支持模型与时间范围筛选。"""
    try:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        hours = max(0, int(hours))
        start = float(start) if start is not None else None
        end = float(end) if end is not None else None
    except Exception:
        limit, offset, hours = 50, 0, 0
    total, records = stats.recent_filtered(
        limit,
        offset,
        hours,
        start=start,
        end=end,
        model=str(model).strip(),
    )
    return {"records": records, "total": total}


@app.put("/api/model-price")
async def api_set_model_price(request: Request):
    """设置某个模型的价格（可覆盖供应商默认价）；字段为 null 表示不覆盖。"""
    cfg = load_config()
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})
    model = str(data.get("model", "")).strip()
    if not model:
        return JSONResponse(status_code=400, content={"error": "缺少模型名"})
    entry = {}
    for key in ("price_input_per_1m", "price_output_per_1m", "price_input_cached_per_1m"):
        v = data.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            return JSONResponse(status_code=400, content={"error": f"{key} 必须是数字"})
        entry[key] = v
    mp = cfg.setdefault("model_prices", {})
    if not entry:
        mp.pop(model, None)
    else:
        mp[model] = entry
    save_config(cfg)
    return {"ok": True}


@app.delete("/api/model-price/{model}")
async def api_delete_model_price(model: str):
    cfg = load_config()
    (cfg.setdefault("model_prices", {})).pop(model, None)
    save_config(cfg)
    return {"ok": True}


@app.post("/api/stats/clear")
async def api_stats_clear():
    stats.clear()
    return {"ok": True}


# ------------------ 更新与重启 ------------------

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exe_file_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": _file_sha256(path),
    }


def _update_file_status(base_dir: Path) -> dict:
    return {
        "ok": True,
        "frozen": bool(getattr(sys, "frozen", False)),
        "version": APP_VERSION,
        "current_exe": _exe_file_info(base_dir / "LLMRouter.exe"),
        "new_exe": _exe_file_info(base_dir / "LLMRouter.new.exe"),
    }


def _build_update_script(
    base_dir: Path,
    *,
    wait_for_process: bool = True,
    launch_new: bool = True,
    force_fail: bool = False,
) -> str:
    """Build the PowerShell updater; wait/launch flags exist for unit tests."""
    dir_literal = str(base_dir.resolve()).replace("'", "''")
    wait_block = (
        "$deadline = (Get-Date).AddSeconds(90)\n"
        "while (Get-Process -Name 'LLMRouter' -ErrorAction SilentlyContinue) {\n"
        "    if ((Get-Date) -ge $deadline) {\n"
        "        Get-Process -Name 'LLMRouter' -ErrorAction SilentlyContinue | Stop-Process -Force\n"
        "        break\n"
        "    }\n"
        "    Start-Sleep -Milliseconds 500\n"
        "}\n"
        "Start-Sleep -Milliseconds 800\n"
    ) if wait_for_process else ""
    force_fail_block = (
        "    if ($env:LLMROUTER_FORCE_UPDATE_FAIL -eq '1') {\n"
        "        throw 'forced update failure for tests'\n"
        "    }\n"
    ) if force_fail else ""
    launch_call = "Start-Router" if launch_new else "# launch disabled for test"
    return f"""$ErrorActionPreference = 'Stop'
$dir = '{dir_literal}'
Set-Location -LiteralPath $dir

{wait_block}
$old = Join-Path $dir 'LLMRouter.exe'
$prev = Join-Path $dir 'LLMRouter.previous.exe'
$new = Join-Path $dir 'LLMRouter.new.exe'
Remove-Item -LiteralPath $prev -Force -ErrorAction SilentlyContinue

function Start-Router {{
    # A restarted PyInstaller app must not inherit the old process's _MEI runtime.
    Get-ChildItem Env: | Where-Object {{ $_.Name -like '_PYI_*' }} | Remove-Item -ErrorAction SilentlyContinue
    $env:PYINSTALLER_RESET_ENVIRONMENT = '1'
    Start-Process -FilePath $old -WorkingDirectory $dir
}}

try {{
    if (-not (Test-Path -LiteralPath $new)) {{
        throw 'LLMRouter.new.exe not found'
    }}
    $moved = $false
    for ($i = 0; $i -lt 20; $i++) {{
        try {{
            Move-Item -LiteralPath $old -Destination $prev -Force
            $moved = $true
            break
        }} catch {{
            Start-Sleep -Milliseconds 500
        }}
    }}
    if (-not $moved) {{
        throw 'old executable is still locked'
    }}
{force_fail_block}    Copy-Item -LiteralPath $new -Destination $old -Force
    $hashNew = (Get-FileHash -LiteralPath $new -Algorithm SHA256).Hash
    $hashOld = (Get-FileHash -LiteralPath $old -Algorithm SHA256).Hash
    if ($hashNew -ne $hashOld) {{
        throw 'SHA256 mismatch after copy'
    }}
    Remove-Item -LiteralPath $new -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $prev -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $dir 'crash.log') -Force -ErrorAction SilentlyContinue
    {launch_call}
}} catch {{
    Remove-Item -LiteralPath $old -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $prev) {{
        Move-Item -LiteralPath $prev -Destination $old -Force
    }}
    Set-Content -LiteralPath (Join-Path $dir 'update-error.log') -Value $_.Exception.Message -Encoding UTF8
    if (Test-Path -LiteralPath $old) {{
        {launch_call}
    }}
}}
Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue
"""


@app.get("/api/update/status")
async def api_update_status():
    return _update_file_status(BASE_DIR)


@app.post("/api/update/apply")
async def api_update_apply():
    if not getattr(sys, "frozen", False):
        return JSONResponse(status_code=400, content={"error": "源码模式无法自动更新 EXE，请使用 apply-update.bat"})
    new_exe = BASE_DIR / "LLMRouter.new.exe"
    if not new_exe.exists():
        return JSONResponse(status_code=400, content={"error": "未找到 LLMRouter.new.exe"})

    script = _build_update_script(BASE_DIR)
    fd, script_path = tempfile.mkstemp(prefix="llmrouter-update-", suffix=".ps1")
    with os.fdopen(fd, "w", encoding="utf-8-sig") as f:
        f.write(script)

    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            script_path,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        close_fds=True,
    )
    threading.Timer(1.5, lambda: os._exit(0)).start()
    return {"ok": True, "restarting": True}


# ------------------ 网关运行状态与手动重载 ------------------

_current_server = None
_current_thread = None
_current_port = None


def set_server(server, thread, port=None):
    global _current_server, _current_thread, _current_port
    _current_server = server
    _current_thread = thread
    if port is not None:
        _current_port = port


def get_server():
    return _current_server, _current_thread


def get_current_port():
    return _current_port


def _reload_runtime_state() -> dict:
    """重载缓存与统计；端口变化只报告，不创建或重启进程。"""
    cfg = load_config()
    try:
        new_port = int(cfg.get("port") or 8765)
    except Exception:
        new_port = 8765
    _model_cache.clear()
    stats.load()
    return {
        "ok": True,
        "port": new_port,
        "restart_required": new_port != get_current_port(),
        "restarting": False,
    }


@app.post("/api/reload")
async def api_reload():
    return _reload_runtime_state()


@app.post("/api/restart")
async def api_restart_legacy():
    """兼容旧控制台调用，但绝不自动重启或拉起新进程。"""
    return _reload_runtime_state()


# ------------------ 模型/转发入口 ------------------

@app.get("/v1/models")
async def get_models():
    cfg = load_config()
    data = await merged_model_list(cfg)
    return {"object": "list", "data": data}


@app.post("/v1/responses")
async def post_responses(request: Request):
    return await relay(request, "responses")


@app.post("/v1/chat/completions")
async def post_chat(request: Request):
    return await relay(request, "chat")


if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    stats.load()
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    try:
        port = int(cfg.get("port") or 8765)
    except Exception:
        port = 8765
    if cfg.get("open_browser", True):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    print(f"LLM Router 控制台：http://127.0.0.1:{port}/")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
