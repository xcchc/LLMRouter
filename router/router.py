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
import copy
import json
import hashlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import stats
import vision

from suppliers.shared import (
    OPENAI_ENCRYPTED_PLACEHOLDER as _OPENAI_ENCRYPTED_PLACEHOLDER,
    args_to_string as _args_to_string,
    bridge_chat_name as _bridge_chat_name,
    build_tool_bridge as _build_tool_bridge,
    chat_tool_name as _chat_tool_name,
    content_to_text as _content_to_text,
    custom_tool_arguments as _custom_tool_arguments,
    normalize_tool_parameters_schema as _normalize_tool_parameters_schema,
    sanitize_schema as _sanitize_schema,
    unwrap_encrypted_content_parts as _unwrap_encrypted_content_parts,
)
from suppliers.minimax import (
    is_minimax_responses_supplier as _is_minimax_responses_supplier,
    MINIMAX_SENSITIVE_IMAGE_PLACEHOLDER,
    sanitize_responses as _sanitize_responses_for_minimax,
    sensitive_image_error as _minimax_sensitive_image_error,
    strip_sensitive_image_candidate as _strip_sensitive_image_candidate,
)
from suppliers.anthropic_messages import (
    anthropic_error_json as _anthropic_error_json,
    anthropic_to_chat as _anthropic_to_chat,
    chat_json_to_anthropic as _chat_json_to_anthropic_json,
    estimate_input_tokens as _anthropic_estimate_input_tokens,
    ChatToAnthropic as _ChatToAnthropic,
    stream_chat_to_anthropic as _stream_chat_to_anthropic,
)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import paths as _paths

BASE_DIR = _paths.BASE_DIR                 # 程序所在目录（Windows 版 config/stats 放这里）
DATA_DIR = _paths.DATA_DIR                 # 用户数据目录（macOS 在 ~/Library/Application Support）
CONFIG_PATH = _paths.CONFIG_PATH
DASHBOARD_PATH = _paths.DASHBOARD_PATH
MODEL_CACHE_TTL = 600  # 上游模型列表缓存秒数
APP_VERSION = _paths.APP_VERSION

ERROR_LOG_PATH = _paths.ERROR_LOG_PATH
_error_logger = logging.getLogger("llm_router")
_error_logger.setLevel(logging.ERROR)
if not _error_logger.handlers:
    _error_handler = RotatingFileHandler(
        ERROR_LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    _error_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    _error_logger.addHandler(_error_handler)

app = FastAPI(title="LLM Router", docs_url=None, redoc_url=None)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """记录 FastAPI 未处理异常，避免只看到 500 却没有原因。"""
    try:
        _error_logger.exception(
            "Unhandled error on %s %s (content_length=%s)",
            request.method,
            request.url.path,
            request.headers.get("content-length"),
            exc_info=exc,
        )
    except Exception:
        pass
    return Response(
        content="Internal Server Error",
        status_code=500,
        media_type="text/plain",
    )

DONE = object()  # SSE [DONE] 哨兵
SSE_HEARTBEAT = object()
SSE_HEARTBEAT_INTERVAL_SECONDS = 15.0

_TITLE_INSTRUCTION_PREFIXES = (
    "You are a helpful assistant. You will be presented with a user prompt",
    "You are a helpful assistant. You will be presented with the most recent messages",
)
_last_session_model: str | None = None


def _is_title_generation_body(body: dict) -> bool:
    """Codex 新任务标题生成使用独立的 helper 模型，可按 prompt 特征识别。

    Older Codex versions put the helper prompt in ``instructions``.  Codex
    Desktop 0.153+ instead includes it as the final user message in the
    Responses ``input`` array, so inspect both shapes.
    """
    if not isinstance(body, dict):
        return False

    def texts(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    yield item
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        yield text
                    content = item.get("content")
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                text = part.get("text")
                                if isinstance(text, str):
                                    yield text

    for value in (body.get("instructions"), body.get("input"), body.get("messages")):
        for text in texts(value):
            if text.startswith(_TITLE_INSTRUCTION_PREFIXES):
                return True
    return False

def _upstream_models_url(base_url: str) -> str:
    """拼接上游 /models 地址；非 ASCII 字符（如中文）做百分号编码，避免请求崩溃。"""
    import urllib.parse
    url = _join_api_path(base_url, "models")
    return urllib.parse.quote(url, safe=":/?&=#%+@")


def _join_api_path(base_url: str, path: str) -> str:
    """拼接 OpenAI 兼容路径，兼容 base_url 已带 /v1 或只写域名的情况。"""
    base = str(base_url or "").rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/{path}"
    return f"{base}/v1/{path}"


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
    """每次请求都重新读取配置，改完 config.json 不用重启（端口除外）。

    用 utf-8-sig 读取：记事本「另存为 UTF-8」会写入 BOM，普通 utf-8 解码会让
    json.loads 直接抛错，表现为双击后进程立刻退出。
    """
    _ensure_config()
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
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
        "model_max_output_tokens": cfg.get("model_max_output_tokens") or {},
        "agent_profiles": cfg.get("agent_profiles") or {},
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
    return sorted(result, key=lambda item: model_sort_key(item["id"]))


# ------------------ 模型名：<供应商>/<模型名> ------------------

MODEL_ID_SEPARATOR = "/"


def qualified_model_id(supplier_name: str, model: str) -> str:
    """Agent 侧看到的模型名：``<供应商>/<模型名>``。

    模型名本身可以带 ``/``（例如 xkiro 渠道的 ``z-ai/glm-5.3``），这里只在
    前面追加供应商，不动模型名内部的斜杠，所以公开名是
    ``xkiro/z-ai/glm-5.3``；解析时按第一个 ``/`` 切分即可还原。
    """
    name = str(supplier_name or "").strip()
    model = str(model or "").strip()
    if not name or not model:
        return model
    return f"{name}{MODEL_ID_SEPARATOR}{model}"


def _supplier_by_name(cfg: dict, name: str) -> dict | None:
    target = str(name or "").strip()
    if not target:
        return None
    for supplier in cfg.get("suppliers", []) or []:
        if str(supplier.get("name") or "").strip() == target:
            return supplier
    return None


def split_qualified_model_id(cfg: dict, model: str) -> tuple[str, str] | None:
    """``<供应商>/<模型名>`` -> (供应商名, 上游模型名)；不是限定名时返回 None。

    只在第一个 ``/`` 处切分：``xkiro/z-ai/glm-5.3`` ->
    ``("xkiro", "z-ai/glm-5.3")``。第一段必须是已配置的供应商名，否则视为
    上游本来就带 ``/`` 的普通模型名。
    """
    text = str(model or "").strip()
    head, sep, rest = text.partition(MODEL_ID_SEPARATOR)
    head = head.strip()
    rest = rest.strip()
    if not sep or not head or not rest:
        return None
    if _supplier_by_name(cfg, head) is None:
        return None
    return head, rest


def model_sort_key(text: str) -> list:
    """模型名排序键：忽略大小写，数字段按数值比较（``gpt-5.9`` 排在 ``gpt-5.10`` 前）。

    与面板里的 ``localeCompare(..., {numeric:true, sensitivity:'base'})`` 保持一致，
    这样控制台和 Agent 看到的模型列表顺序相同。
    """
    parts = re.split(r"(\d+)", str(text or "").casefold())
    return [(0, int(part)) if part.isdigit() else (1, part) for part in parts]


def _supplier_known_models(supplier: dict, cfg: dict) -> set:
    """供应商已登记或已缓存的模型名（只读缓存，不触发网络请求）。"""
    name = str(supplier.get("name") or "").strip()
    known = {str(m).strip() for m in (supplier.get("models") or []) if str(m).strip()}
    cached = _model_cache.get(name) or {}
    known.update(str(m).strip() for m in (cached.get("ids") or []) if str(m).strip())
    return known


def configured_model_list(cfg: dict) -> list:
    """Return only models explicitly added in the model-routing UI.

    The dashboard still uses :func:`merged_model_list` to inspect every model
    discovered from upstreams.  The public ``/v1/models`` catalog is kept
    intentionally narrower so clients such as Codex Switch only see models
    that the user explicitly routed or added to a supplier's manual list.

    ``model_max_output_tokens`` publishes the real output ceiling of a model
    in the listing.  Clients that read ``max_output_tokens`` (DeepSeek Harness
    among them) otherwise fall back to their own generic default and ask the
    upstream for far less than the model can produce.

    Every entry is published as ``<供应商>/<模型名>`` so the same model name on
    two suppliers stays distinguishable in the client's model picker.
    """
    result = []
    seen = set()
    limits = cfg.get("model_max_output_tokens") or {}

    def _entry(public_id: str, owner: str) -> dict:
        item = {"id": public_id, "object": "model", "owned_by": owner}
        limit = limits.get(public_id)
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            item["max_output_tokens"] = limit
        return item

    # model_map 的键就是公开模型名 ``<供应商>/<模型名>``。
    for public_id, supplier_name in (cfg.get("model_map") or {}).items():
        public_id = str(public_id).strip()
        supplier_name = str(supplier_name).strip()
        if not public_id or public_id in seen:
            continue
        prefix = f"{supplier_name}{MODEL_ID_SEPARATOR}"
        upstream = public_id[len(prefix):] if public_id.startswith(prefix) else public_id
        if upstream in _blacklisted_ids(cfg, supplier_name):
            continue
        seen.add(public_id)
        result.append(_entry(public_id, supplier_name))

    # A non-empty supplier.models list is the other explicit way the UI adds
    # a model.  Do not consult the upstream cache here: undisclosed upstream
    # models must stay out of the client-facing catalog.
    for supplier in cfg.get("suppliers", []):
        supplier_name = str(supplier.get("name", ""))
        hidden = _blacklisted_ids(cfg, supplier_name)
        manual_models = supplier.get("models") or []
        if not isinstance(manual_models, (list, tuple)):
            continue
        for model in manual_models:
            model = str(model).strip()
            public_id = qualified_model_id(supplier_name, model)
            if not model or public_id in seen or model in hidden:
                continue
            seen.add(public_id)
            result.append(_entry(public_id, supplier_name))

    # Agent 侧（/v1/models、Agent 接入页）看到的顺序固定按模型名升序。
    return sorted(result, key=lambda item: model_sort_key(item["id"]))


async def find_supplier(cfg: dict, model: str) -> dict | None:
    """按公开模型名 ``<供应商>/<模型名>`` 找供应商；不是限定名就返回 None。"""
    split = split_qualified_model_id(cfg, model)
    if split is None:
        return None
    supplier = _supplier_by_name(cfg, split[0])
    return copy.deepcopy(supplier) if supplier is not None else None


async def resolve_model_target(cfg: dict, model: str) -> tuple[dict | None, str]:
    """客户端模型名 -> (供应商配置, 真正转发给上游的模型名)。

    客户端只使用 ``<供应商>/<模型名>``，这里在第一个 ``/`` 处切分，剩下的
    部分原样转发；上游永远收不到供应商前缀。
    """
    requested = str(model or "").strip()
    if not requested:
        return None, requested
    supplier = await find_supplier(cfg, requested)
    if supplier is None:
        return None, requested
    split = split_qualified_model_id(cfg, requested)
    upstream = split[1] if split is not None else requested
    return supplier, upstream


def migrate_model_ids(cfg: dict) -> bool:
    """把旧配置里的裸模型名迁移成公开模型名 ``<供应商>/<模型名>``。

    只在启动时调用一次；返回配置是否发生变化。
    """
    changed = False
    model_map = cfg.get("model_map")
    renamed = {}
    if isinstance(model_map, dict):
        migrated = {}
        for raw, supplier_name in model_map.items():
            raw = str(raw).strip()
            supplier_name = str(supplier_name or "").strip()
            if not raw or not supplier_name:
                continue
            supplier = _supplier_by_name(cfg, supplier_name)
            known = _supplier_known_models(supplier, cfg) if supplier else set()
            prefix = f"{supplier_name}{MODEL_ID_SEPARATOR}"
            # 已经带供应商前缀、且不是"模型名本身就以供应商名开头"的，保持原样。
            if raw.startswith(prefix) and raw not in known:
                public_id = raw
            else:
                public_id = qualified_model_id(supplier_name, raw)
            if public_id != raw:
                renamed[raw] = public_id
                changed = True
            head, sep, rest = public_id.partition(MODEL_ID_SEPARATOR)
            owner = head if sep and rest and _supplier_by_name(cfg, head) is not None else supplier_name
            migrated[public_id] = owner
        if migrated != model_map:
            cfg["model_map"] = migrated
            changed = True

    def _migrate_keys(field: str) -> None:
        nonlocal changed
        table = cfg.get(field)
        if not isinstance(table, dict):
            return
        migrated = {}
        for raw, value in table.items():
            raw = str(raw).strip()
            if not raw:
                continue
            migrated[renamed.get(raw) or _public_id_for(cfg, raw)] = value
        if migrated != table:
            cfg[field] = migrated
            changed = True

    _migrate_keys("model_prices")
    _migrate_keys("model_max_output_tokens")

    for profile in (cfg.get("agent_profiles") or {}).values():
        if not isinstance(profile, dict):
            continue
        default_model = str(profile.get("default_model") or "").strip()
        if not default_model:
            continue
        public_id = renamed.get(default_model) or _public_id_for(cfg, default_model)
        if public_id != default_model:
            profile["default_model"] = public_id
            changed = True

    return changed


def _public_id_for(cfg: dict, model: str) -> str:
    """迁移用：把裸模型名补成公开名；已经是公开名（或查不到供应商）时原样返回。"""
    text = str(model or "").strip()
    if not text:
        return text
    if split_qualified_model_id(cfg, text) is not None:
        return text
    for supplier in cfg.get("suppliers", []) or []:
        name = str(supplier.get("name") or "").strip()
        if name and text in _supplier_known_models(supplier, cfg):
            return qualified_model_id(name, text)
    return text


def migrate_config_on_startup() -> bool:
    """启动时把 config.json 里的模型名统一成公开名；返回是否写回。"""
    try:
        cfg = load_config()
    except Exception:
        return False
    if not isinstance(cfg, dict) or not migrate_model_ids(cfg):
        return False
    try:
        save_config(cfg)
    except Exception:
        return False
    return True


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


def _is_openai_native_supplier(supplier: dict) -> bool:
    base = str(supplier.get("base_url") or "").lower()
    return "openai.com" in base or "chatgpt.com" in base


def _should_sanitize_openai_fields(supplier: dict) -> bool:
    configured = supplier.get("openai_sanitize")
    if isinstance(configured, bool):
        return configured
    if configured is not None:
        return bool(configured)
    return str(supplier.get("wire_api", "responses")).lower() == "responses" and not _is_openai_native_supplier(supplier)


_HOSTED_TOOL_TYPES = {"web_search", "web_search_preview", "web_search_preview_2025_03_11", "file_search", "computer_use_preview", "code_interpreter"}
_HOSTED_ITEM_TYPES = {"web_search_call", "web_search_result", "web_search_call_result", "file_search_call", "computer_call", "code_interpreter_call"}
_HOSTED_TOOL_PLACEHOLDER = "[Hosted tool call/result removed by the router because this upstream does not support it.]"


def _strip_hosted_tools_in_place(body: dict) -> None:
    """Remove hosted-tool definitions, tool_choice refs, and history items from *body*.

    Third-party Responses backends reject ``web_search`` and similar hosted
    tools with "cannot satisfy this hosted … request", so they must never
    reach the upstream — even when ``openai_sanitize`` is turned off by the
    user for other fields.
    """
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        kept = [t for t in tools if not (
            (isinstance(t, dict) and str(t.get("type", "")).lower() in _HOSTED_TOOL_TYPES)
            or (isinstance(t, str) and t.strip().lower() in _HOSTED_TOOL_TYPES)
        )]
        if len(kept) != len(tools):
            body["tools"] = kept or None
    tc = body.get("tool_choice")
    if isinstance(tc, dict) and str(tc.get("type", "")).lower() in _HOSTED_TOOL_TYPES:
        body.pop("tool_choice", None)
    elif isinstance(tc, str) and tc.strip().lower() in _HOSTED_TOOL_TYPES:
        body.pop("tool_choice", None)
    items = body.get("input")
    if isinstance(items, list) and items:
        filtered = []
        for item in items:
            if isinstance(item, dict) and str(item.get("type", "")).lower() in _HOSTED_ITEM_TYPES:
                filtered.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": _HOSTED_TOOL_PLACEHOLDER}]})
            else:
                filtered.append(item)
        if len(filtered) != len(items):
            body["input"] = filtered


def _sanitize_responses_for_non_openai(body: dict) -> dict:
    """Replace OpenAI-only agent/encrypted fields before forwarding to a third-party Responses backend."""
    if not isinstance(body, dict):
        return body

    sanitized = copy.deepcopy(body)
    changed = False

    # Strip hosted tools (web_search, file_search, etc.) that third-party
    # Responses backends unconditionally reject.
    _pre_tool_count = len(sanitized.get("tools") or [])
    _strip_hosted_tools_in_place(sanitized)
    if len(sanitized.get("tools") or []) != _pre_tool_count:
        changed = True

    input_items = sanitized.get("input")
    if isinstance(input_items, list):
        filtered_input = []
        for item in input_items:
            if not isinstance(item, dict):
                filtered_input.append(item)
                continue
            if str(item.get("type", "")).lower() in _HOSTED_ITEM_TYPES:
                changed = True
                filtered_input.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": _HOSTED_TOOL_PLACEHOLDER}]})
                continue
            for key in ("author", "recipient"):
                if key in item:
                    del item[key]
                    changed = True
            content = item.get("content")
            unwrapped = _unwrap_encrypted_content_parts(content)
            if item.get("type") == "agent_message":
                if not isinstance(unwrapped, list) or not unwrapped:
                    unwrapped = [{"type": "input_text", "text": _content_to_text(content) or "Agent message content is unavailable."}]
                item["type"] = "message"
                item["role"] = "user"
                item["content"] = unwrapped
                changed = True
            elif isinstance(unwrapped, list) and unwrapped != content:
                item["content"] = unwrapped
                changed = True
            filtered_input.append(item)
        if changed:
            sanitized["input"] = filtered_input

    def strip_encrypted(value, in_content=False):
        nonlocal changed
        if isinstance(value, list):
            out = []
            for entry in value:
                if isinstance(entry, dict) and entry.get("type") == "encrypted_content":
                    changed = True
                    if in_content:
                        out.append({"type": "input_text", "text": _OPENAI_ENCRYPTED_PLACEHOLDER})
                    continue
                out.append(strip_encrypted(entry, in_content))
            return out
        if isinstance(value, dict):
            out = {}
            for key, entry in value.items():
                if key == "encrypted_content":
                    changed = True
                    continue
                out[key] = strip_encrypted(entry, in_content or key == "content")
            return out
        return value

    sanitized = strip_encrypted(sanitized)
    return sanitized if changed else body


def _strip_prompt_cache_key(body: dict) -> dict:
    """Drop prompt_cache_key before forwarding so third-party gateways do not reject Codex requests."""
    if not isinstance(body, dict) or "prompt_cache_key" not in body:
        return body
    sanitized = copy.deepcopy(body)
    sanitized.pop("prompt_cache_key", None)
    return sanitized


def _should_forward_prompt_cache_key(supplier: dict) -> bool:
    if not isinstance(supplier, dict):
        return False
    return supplier.get("forward_prompt_cache_key") is True


def _ensure_chat_stream_usage(body: dict) -> dict:
    """Request the standalone usage frame for streamed Chat upstreams."""
    if not isinstance(body, dict) or body.get("stream") is not True:
        return body
    stream_options = body.get("stream_options")
    if isinstance(stream_options, dict) and "include_usage" in stream_options:
        return body
    out = copy.deepcopy(body)
    merged_options = dict(stream_options) if isinstance(stream_options, dict) else {}
    merged_options["include_usage"] = True
    out["stream_options"] = merged_options
    return out


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
    # xKiro (and OpenAI-compatible Chat providers) emit token usage for a
    # streamed request only when this opt-in is present.  The usage frame is a
    # separate chunk with an empty choices array, so preserve an explicit
    # caller setting but enable it by default for the router's accounting.
    stream_options = body.get("stream_options")
    if isinstance(stream_options, dict):
        chat["stream_options"] = copy.deepcopy(stream_options)
        if body.get("stream") is True and "include_usage" not in chat["stream_options"]:
            chat["stream_options"]["include_usage"] = True
    elif body.get("stream") is True:
        chat["stream_options"] = {"include_usage": True}
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
            # 正文必须排在 function_call 之前。Responses 历史里一条 assistant 的输出项
            # 顺序是 reasoning -> message -> function_call；把正文塞在 function_call
            # 与 function_call_output 之间，严格校验的上游（pixel 的 /v1/responses）
            # 会判定 "tool calls and tool results do not match"（code 11148）。
            content = _chat_content_to_responses(m.get("content"), assistant=True)
            if content:
                items.append({"type": "message", "role": "assistant", "content": content})
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

# ------------------ 悬空工具调用补齐 ------------------
#
# 一轮任务被 max-tokens / 上游断流打断时，客户端历史里会留下"有 tool_call、
# 没有对应 tool_result"的残缺结构。上游对此一律拒绝：
#   chat   : "assistant message before pending tool results" (400)
#   responses: "tool calls and tool results do not match" (code 11148)
# 一旦出现，这个会话后续每次请求都会 400，只能新开——用户看到的就是"突然断了，
# 继续也没用"。这里在转发前给缺失的结果补一条占位说明，把历史修成上游能接受
# 的结构，让会话能继续下去。

DANGLING_TOOL_PLACEHOLDER = (
    "[local router] This tool call was interrupted before it produced a result. "
    "Treat it as having returned nothing and continue with the task."
)

_RESPONSES_CALL_OUTPUT_TYPES = {
    "function_call": "function_call_output",
    "custom_tool_call": "custom_tool_call_output",
    "tool_search_call": "tool_search_output",
}
_RESPONSES_OUTPUT_TYPES = frozenset(_RESPONSES_CALL_OUTPUT_TYPES.values())


def _should_repair_tool_calls(supplier: dict) -> bool:
    """是否补齐悬空工具调用（供应商可用 ``repair_tool_calls: false`` 关掉）。"""
    configured = supplier.get("repair_tool_calls")
    return bool(configured) if isinstance(configured, bool) else True


def _repair_responses_tool_calls(body: dict) -> int:
    """Responses 请求体：补悬空调用，删掉没有对应调用的孤儿结果。

    上游对这两种错配一律拒绝（pixel 返回 ``tool_call_mismatch`` / 11148），
    而且一次错配就让整段会话作废，所以转发前先修成上游能接受的结构。
    """
    items = body.get("input")
    if not isinstance(items, list) or not items:
        return 0

    calls = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in _RESPONSES_CALL_OUTPUT_TYPES:
            call_id = item.get("call_id") or item.get("id")
            if isinstance(call_id, str) and call_id:
                calls.setdefault(call_id, item_type)

    # 先删"有结果、没调用"的孤儿项：上游同样视为错配。
    kept = []
    orphans = 0
    for item in items:
        if isinstance(item, dict) and item.get("type") in _RESPONSES_OUTPUT_TYPES:
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id and call_id not in calls:
                orphans += 1
                continue
        kept.append(item)
    if orphans:
        items = body["input"] = kept

    if not calls:
        return orphans

    answered = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") in _RESPONSES_OUTPUT_TYPES:
            call_id = item.get("call_id")
            if isinstance(call_id, str) and call_id:
                answered.add(call_id)

    pending = {cid: kind for cid, kind in calls.items() if cid not in answered}
    if not pending:
        return orphans

    repaired = []
    filled = 0
    for item in items:
        repaired.append(item)
        if not isinstance(item, dict):
            continue
        call_id = item.get("call_id") or item.get("id")
        kind = pending.get(call_id) if isinstance(call_id, str) else None
        if kind:
            repaired.append({
                "type": _RESPONSES_CALL_OUTPUT_TYPES[kind],
                "call_id": call_id,
                "output": DANGLING_TOOL_PLACEHOLDER,
            })
            pending.pop(call_id, None)
            filled += 1
    if filled:
        body["input"] = repaired
    return filled + orphans


def _repair_chat_tool_calls(body: dict) -> int:
    """Chat 请求体：补悬空调用的占位结果，并删掉无对应调用的孤儿结果。"""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return 0

    known = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and isinstance(call.get("id"), str) and call.get("id"):
                known.add(call["id"])

    dropped = 0
    filtered = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            # 结果找不到对应调用（或根本没有 id）时上游一律 400，直接删掉。
            if not (isinstance(call_id, str) and call_id and call_id in known):
                dropped += 1
                continue
        filtered.append(message)
    if dropped:
        messages = body["messages"] = filtered

    repaired = []
    filled = 0
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        repaired.append(message)
        index += 1
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue

        call_ids = [
            call.get("id")
            for call in tool_calls
            if isinstance(call, dict) and isinstance(call.get("id"), str) and call.get("id")
        ]
        if not call_ids:
            continue

        answered = set()
        while index < total:
            follower = messages[index]
            if not (isinstance(follower, dict) and follower.get("role") == "tool"):
                break
            repaired.append(follower)
            answered.add(follower.get("tool_call_id"))
            index += 1

        for call_id in call_ids:
            if call_id in answered:
                continue
            repaired.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": DANGLING_TOOL_PLACEHOLDER,
            })
            filled += 1

    if filled:
        body["messages"] = repaired
    return filled + dropped


def repair_dangling_tool_calls(body: dict, wire: str) -> int:
    """按上游线协议补齐悬空工具调用，返回补了几条。"""
    if not isinstance(body, dict):
        return 0
    if wire == "chat":
        return _repair_chat_tool_calls(body)
    return _repair_responses_tool_calls(body)


# ------------------ SSE 解析 ------------------

def sse_bytes(obj) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n"


def _tidy_upstream_error_text(text) -> str:
    """把上游层层转义的错误文本拆成一条人能看懂的消息。

    上游错误经常被包成 ``{"error":{"message":"{\\"error\\":{...}}"}}``，原样
    转发出去客户端只能看到一坨转义 JSON，看不到真正的失败原因。
    """
    text = str(text or "").strip()
    for _ in range(4):
        try:
            obj = json.loads(text)
        except Exception:
            return text
        if not isinstance(obj, dict):
            return text
        inner = obj.get("error") if isinstance(obj.get("error"), dict) else obj
        message = inner.get("message") or inner.get("msg")
        if not message:
            return text
        text = str(message).strip()
        code = inner.get("code")
        if code and str(code) not in text:
            text = f"{text}（upstream code {code}）"
    return text


def upstream_stream_failure_event(incoming_wire: str, model: str, message: str) -> bytes:
    """Build a terminal SSE frame after an upstream stream breaks."""
    message = _tidy_upstream_error_text(message)[:500]
    if incoming_wire == "responses":
        return sse_bytes({
            "type": "response.failed",
            "response": {
                "id": "resp_" + uuid.uuid4().hex,
                "object": "response",
                "created_at": int(time.time()),
                "status": "failed",
                "model": model,
                "output": [],
                "error": {
                    "code": "upstream_stream_error",
                    "message": message,
                },
            },
        })
    if incoming_wire == "anthropic":
        return sse_bytes({
            "type": "error",
            "error": {"type": "api_error", "message": message},
        })
    return sse_bytes({
        "error": {
            "code": "upstream_stream_error",
            "message": message,
            "type": "upstream_stream_error",
        }
    })


async def _with_sse_heartbeats(source, interval: float | None = None):
    """Yield source items and emit an SSE comment while the upstream is silent.

    Some chat upstreams stay quiet for minutes during long reasoning.  Local
    proxies such as Codex Switch treat that as an idle stream and close the
    connection before ``response.completed``.  SSE comments are ignored by
    clients but reset their idle timer.
    """
    interval = SSE_HEARTBEAT_INTERVAL_SECONDS if interval is None else interval
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def pump():
        try:
            async for item in source:
                await queue.put(item)
            await queue.put(None)
        except Exception as exc:
            await queue.put(exc)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                yield SSE_HEARTBEAT
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not pump_task.done():
            pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


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
        self.saw_terminal = False
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
            terminal_type = obj.get("type")
            if isinstance(obj.get("error"), dict) and terminal_type not in ("response.completed", "response.incomplete"):
                # 中转层自己合成的错误帧：本轮实际失败了，统计里要看得见。
                self.status = "error"
            if terminal_type in ("response.completed", "response.failed", "response.incomplete"):
                self.saw_terminal = True
            if terminal_type == "response.failed":
                self.status = "failed"
            elif terminal_type == "response.incomplete":
                self.status = "incomplete"
            elif terminal_type == "response.completed" and isinstance(obj.get("response"), dict):
                self.status = obj["response"].get("status", "completed")
            u = obj.get("usage")
            if u is None and isinstance(obj.get("response"), dict):
                u = obj["response"].get("usage")
            if isinstance(u, dict) and (u.get("total_tokens") is not None or u.get("input_tokens") is not None or u.get("prompt_tokens") is not None):
                self.usage = u


# ------------------ 畸形协议标记清洗 ------------------
#
# 有些上游模型（实测 pixel/deepseek-v4.1-flash）在把工具调用吐成文本时，
# 会把训练期的特殊标记和控制串一起写进正文，例如：
#   ="{" command":"ok">Let me just call the tool properly.</thinking>
#   ｜DSML｜ calls>
#   <｜begin▁of▁sentence｜>var">I made another malformed call.
# 这些片段既不是回答也不是工具调用，却会污染 assistant 正文：客户端把它当
# 正常回复渲染，收起工作步骤后仍然挂在界面上（Dsh 的 keepReplies=all），
# 看起来就像"思考链没有折叠"。所以在转发前就地剥掉。

# 特殊标记：<｜begin▁of▁sentence｜> 这类 token 用全角竖线包裹，必须单独匹配。
_SPECIAL_TOKEN_RE = re.compile(r"<[｜|][^<>\n]{0,40}?[｜|]>?")
# 明确的协议标记按名字匹配，避免误伤普通文本。
_PROTOCOL_TOKEN_RE = re.compile(
    r"[<｜|]{1,2}\s*"
    r"(?:begin|end)[▁_]{1,3}of[▁_]{1,3}(?:sentence|text)"
    r"\s*[｜|]?>{0,2}"
    r"|[<｜|]{1,2}\s*DSML\s*[｜|]?\s*(?:calls?|invokes?|invoke|tool[▁_]?calls?)?\s*>{0,2}",
    re.IGNORECASE,
)
# 常见的"半截"残留：泄露的收尾标签与调用片段。
_PROTOCOL_FRAGMENT_RES = (
    re.compile(r"</?thinking>", re.IGNORECASE),
    re.compile(r"</?(?:tool_call|function_call|antml:invoke|antml:parameter)[^>]{0,40}>", re.IGNORECASE),
    # 以 = 或 " 开头、紧跟 JSON 键、然后被 > 截断的调用残骸。
    re.compile(r'^[\s`]*=\s*"\{.*?>\s*'),
    re.compile(r'^\s*"\s*command\s*"\s*:\s*">\s*'),
    # 标记剥完后剩下的空壳引号/箭头碎片。
    re.compile(r'^[\s"]*(?:command|arguments|calls?)\s*"\s*:\s*">\s*', re.IGNORECASE),
)
# 标记被剥掉之后紧跟的孤儿碎片，例如 "<｜begin▁of▁sentence｜>var\">" 里的 var">。
_ORPHAN_STUB_RE = re.compile(r'^\s*[A-Za-z_]{1,12}"\s*>\s*')
# 被流切开的孤儿碎片（可能只到 var" 或 var 为止）。
_ORPHAN_STUB_PARTIAL_RE = re.compile(r'^[A-Za-z_]{1,12}"?>?$')
# 只在"回复开头"才用的规则：模型偶尔把 @@ 之类当控制前缀吐出来。
_HEAD_ONLY_RES = (re.compile(r"^@@+\s*"),)
# 流首暂存：只有这些"确凿的调用残骸开头"才值得等后续 chunk，避免正常
# 回复（以 = 或引号开头）被卡住。
_HEAD_HINT_RE = re.compile(
    r'^[\s`]*=\s*"\{'
    r'|^\s*"\s*(?:command|arguments|calls?)\s*"'
    r'|^\s*(?:command|arguments|calls?)\s*"?\s*>',
    re.IGNORECASE,
)
# 流式判断：缓冲区尾部是否停在一个还没写完的标记上。
_MARKER_OPEN_RE = re.compile(r"<(?![^<>\n]*>)|｜(?![^<>\n]*>)")


def _strip_protocol_markup(text: str, at_start: bool = True) -> str:
    """剥掉正文里的特殊标记和泄露的调用残骸，保留正常内容。

    只在"这一段本身就是标记"时才整段丢弃；混合正文只删除标记本身，
    避免把真正的回答一起吃掉。``at_start`` 表示这仍是本段回答的开头，
    此时才应用锚定在开头的残骸规则。
    """
    if not text:
        return text
    cleaned = _PROTOCOL_TOKEN_RE.sub("", text)
    cleaned = _SPECIAL_TOKEN_RE.sub("", cleaned)
    cleaned = re.sub(r"</?thinking>", "", cleaned, flags=re.IGNORECASE)
    if not at_start:
        return cleaned
    # 开头的残骸可能被裁成多段，反复剥到稳定为止。
    while True:
        before = cleaned
        for pattern in _PROTOCOL_FRAGMENT_RES:
            cleaned = pattern.sub("", cleaned)
        cleaned = _ORPHAN_STUB_RE.sub("", cleaned)
        for pattern in _HEAD_ONLY_RES:
            cleaned = pattern.sub("", cleaned)
        if cleaned == before:
            return cleaned


class _MarkupScrubber:
    """流式清洗：跨 chunk 被切断的标记要拼起来再判断。

    只暂存两类内容：尾部"没写完的标记"，以及回答最开头疑似调用残骸的
    前缀。其余内容立刻放行，正常文本不会因此多等。
    """

    _MAX_HOLD = 96
    # 只有离结尾这么近的标记起点才可能是被切断的，避免误存整段正文。
    _HOLD_WINDOW = 33

    def __init__(self):
        self._pending = ""
        self._emitted = ""
        # 上一块结尾刚剥掉标记时，紧跟的孤儿碎片可能被切到下一块。
        self._after_marker = False

    @property
    def _at_start(self) -> bool:
        return not self._emitted.strip()

    def feed(self, text: str) -> str:
        if not text:
            return ""
        buf = self._pending + text
        self._pending = ""
        # 回答开头：残骸可能还没写完，等它出现 > 或换行再判定。
        if self._at_start and self._head_unsettled(buf):
            self._pending = buf[: self._MAX_HOLD]
            return ""
        # 刚剥过标记：孤儿碎片可能还没写完，留着等下一块。
        if self._after_marker and _ORPHAN_STUB_PARTIAL_RE.match(buf) and len(buf) < 16:
            self._pending = buf
            return ""
        hold = self._marker_tail_start(buf)
        if hold is not None:
            self._pending = buf[hold:]
            buf = buf[:hold]
        at_start = self._at_start
        out = _strip_protocol_markup(buf, at_start=at_start)
        if out != buf:
            # 这一段剥掉过标记，后面紧跟的碎片也要跟着剥。
            out = _ORPHAN_STUB_RE.sub("", out)
        self._after_marker = out != buf and not out.strip()
        self._emitted += out
        return out

    @classmethod
    def _head_unsettled(cls, buf: str) -> bool:
        """回答开头看起来像残骸、但还没看到终止符时，先攒着。"""
        if len(buf) > cls._MAX_HOLD:
            return False
        head = buf.lstrip()
        if not head:
            return True
        if not (_HEAD_HINT_RE.match(head) or head[0] in "<｜|"):
            return False
        if head[0] in "<｜|":
            # 特殊标记自带 -> 包裹，没闭合就还得等。
            return _MARKER_OPEN_RE.search(buf) is not None
        return ">" not in buf and "\n" not in buf

    @classmethod
    def _marker_tail_start(cls, buf: str):
        """返回应当暂存的尾部起点；``None`` 表示全部可以放行。"""
        window_start = max(0, len(buf) - cls._HOLD_WINDOW)
        match = _MARKER_OPEN_RE.search(buf, window_start)
        if match is None:
            return None
        start = match.start()
        # 暂存长度受 _MAX_HOLD 约束，超长的说明不是标记。
        if len(buf) - start > cls._MAX_HOLD:
            return None
        return start

    def flush(self) -> str:
        tail, self._pending = self._pending, ""
        out = _strip_protocol_markup(tail, at_start=self._at_start)
        self._emitted += out
        return out


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
        # 正文里的畸形协议标记就地剥掉，别让它变成客户端可见的"回复"。
        self._text_scrubber = _MarkupScrubber()

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
            content = self._text_scrubber.feed(content)
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
        # 流结束时把清洗器里挂着的尾巴放出来（可能是半个标记，剥完可能是空串）。
        tail = self._text_scrubber.flush()
        if tail:
            self.msg_text.append(tail)
            if self.msg_started:
                # 正文已经开始下发：尾巴也要按 delta 补发，否则只出现在
                # output_text.done 里会被只认 delta 的客户端丢掉。
                events.append({
                    "type": "response.output_text.delta",
                    "item_id": self.msg_id,
                    "output_index": self._item_index(self.msg_id),
                    "content_index": 0,
                    "delta": tail,
                })
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
            "input_tokens_details": {"cached_tokens": _usage_cached_tokens(u)},
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

def _is_complete_json(text) -> bool:
    """工具参数是否已经是一段完整的 JSON。

    上游把流截断时，``function_call`` 的增量会停在一句半中间；这种参数绝不能
    转发给客户端，否则客户端会解析失败并把整轮任务打断。
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not text:
        return False
    try:
        json.loads(text)
    except Exception:
        return False
    return True


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
        self._reasoning_parts = {}
        self._reasoning_all = ""
        # 上游是否给出了终止事件（completed/incomplete/failed）。没有它就是被截断
        # 的流：usage 不会下发，工具参数也可能停在半句上。
        self.terminal_seen = False
        # 已经流给客户端的正文，用来识别"上游把 reasoning 复读成正文"。
        self._text_all = ""
        self.reasoning_echo = False
        # 正文里的畸形协议标记就地剥掉（与 ChatToResponses 同一套规则）。
        self._text_scrubber = _MarkupScrubber()

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
            # 已按增量收到但还没下发；下发前必须确认是完整 JSON。
            "arguments_streamed": False,
            # 终止事件里带的完整参数（上游有时只在收尾处给全量）。
            "final_arguments": None,
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

    def _collected_arguments(self, rec: dict):
        """返回 ``(可下发的参数, 是否被截断)``。

        ``final_arguments`` 优先（终止事件给的是全量），其次才用流式累计值；
        两者都拼不成 JSON 时判定为截断，交给收尾逻辑换成一个明确的错误帧。
        """
        final_args = rec.get("final_arguments")
        streamed = "".join(rec.get("argument_parts") or [])
        if isinstance(final_args, str) and not final_args.strip() and not streamed:
            # 无参工具：上游给的本来就是空串，不算截断。
            return None, False
        if final_args is not None and _is_complete_json(final_args):
            return final_args, False
        if streamed and _is_complete_json(streamed):
            return streamed, False
        if streamed or final_args is not None:
            return None, True
        return None, False

    def _emit_collected_arguments(self, rec: dict, events: list) -> None:
        if rec.get("arguments_sent"):
            return
        if rec.get("type") != "function_call":
            return
        arguments, broken = self._collected_arguments(rec)
        if arguments is not None and not broken:
            self._emit_arguments(rec, arguments, events)

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
            if isinstance(item, dict) and item.get("arguments") is not None:
                rec["final_arguments"] = _args_to_string(item.get("arguments"))
            if not final:
                # 参数先攒着：确认是完整 JSON 之后再一次性下发，半截的不落地。
                return
            self._emit_collected_arguments(rec, events)

    def _handle_tool_item(self, item: dict, events: list, final=False) -> None:
        rec = self._register_tool(item, events)
        if rec is not None:
            self._emit_item_arguments(rec, item, events, final=final)

    def _emit_reasoning_summary(self, summary, events: list, item_id="") -> None:
        """Emit only reasoning text not already sent by summary deltas."""
        text = _extract_reasoning_text(summary)
        if not text:
            return
        key = str(item_id or "__default__")
        emitted = self._reasoning_parts.get(key, "")
        if emitted:
            if text == emitted:
                return
            if emitted.startswith(text):
                # The summary is a prefix of what was already streamed.
                return
            if text.startswith(emitted):
                text = text[len(emitted):]
            if not text:
                return
        elif self._reasoning_already_streamed(text):
            # Unknown item id: the gateway re-issued finished reasoning under a
            # fresh id in the terminal envelope. It was already streamed.
            return
        self._ensure_role(events)
        events.append(self._chunk({"reasoning_content": text, "reasoning": text}))
        self._reasoning_parts[key] = emitted + text
        self._reasoning_all += text

    def _reasoning_already_streamed(self, text: str) -> bool:
        """True when `text` was already streamed as reasoning.

        A gateway may hand the finished reasoning item back in the terminal
        response envelope under a different item id than the one used while
        streaming, so per-item bookkeeping cannot spot the repeat on its own.
        The repeat is normally the whole summary, which a suffix match catches.
        A containment match is only trusted for long text so that a short
        repeated phrase is never swallowed.
        """
        if not text:
            return True
        if self._reasoning_all.endswith(text):
            return True
        return len(text) >= 32 and text in self._reasoning_all

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
            item = evt.get("item") or {}
            if item.get("type") == "reasoning":
                self._emit_reasoning_summary(
                    item.get("summary") or item.get("content") or item.get("text"),
                    events,
                    item.get("id"),
                )
            else:
                self._handle_tool_item(item, events, final=True)
        elif t in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            d = evt.get("delta", "")
            if d:
                self._ensure_role(events)
                events.append(self._chunk({"reasoning_content": d, "reasoning": d}))
                key = str(evt.get("item_id") or "__default__")
                self._reasoning_parts[key] = self._reasoning_parts.get(key, "") + d
                self._reasoning_all += d
        elif t == "response.output_text.delta":
            d = evt.get("delta", "")
            if d and self._is_reasoning_echo(d):
                # 上游把 reasoning_content 灌进 content 兜底（正文为空时），
                # 整段思考会被客户端当成回答显示。丢掉它并标记，收尾时报错。
                self.reasoning_echo = True
                return events
            if d:
                d = self._text_scrubber.feed(d)
            if d:
                self._ensure_role(events)
                events.append(self._chunk({"content": d}))
                self._text_all += d
        elif t == "response.function_call_arguments.delta":
            d = evt.get("delta", "")
            if d:
                item_id = evt.get("item_id", "")
                rec = self._find_tool(item_id=item_id)
                idx = rec["index"] if rec is not None else self.fc_index.get(item_id, 0)
                self.has_tool_calls = True
                if rec is not None:
                    # 先攒着，等 done/终止事件确认参数完整后再一次性下发。
                    rec["argument_parts"].append(d)
                    rec["arguments_streamed"] = True
                else:
                    events.append(self._chunk({"tool_calls": [{"index": idx, "function": {"arguments": d}}]}))
        elif t == "response.function_call_arguments.done":
            rec = self._find_tool(item_id=evt.get("item_id", ""))
            if rec is not None and not rec.get("arguments_sent"):
                if evt.get("arguments") is not None:
                    rec["final_arguments"] = _args_to_string(evt.get("arguments"))
                self._emit_collected_arguments(rec, events)
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
        elif t in ("response.completed", "response.incomplete"):
            resp_obj = evt.get("response") or {}
            self.terminal_seen = True
            self.status = resp_obj.get("status", "incomplete" if t == "response.incomplete" else "completed")
            self.usage = resp_obj.get("usage")
            for item in resp_obj.get("output") or []:
                if isinstance(item, dict):
                    if item.get("type") == "reasoning":
                        self._emit_reasoning_summary(
                            item.get("summary") or item.get("content") or item.get("text"),
                            events,
                            item.get("id"),
                        )
                    else:
                        self._handle_tool_item(item, events, final=True)
            # 终止事件里没带到的调用（例如只流了增量）在这里补发完整参数。
            for rec in self.tool_records:
                if not rec.get("arguments_sent"):
                    self._emit_collected_arguments(rec, events)
        elif t == "response.failed":
            self.terminal_seen = True
            self.status = "failed"
        return events

    def _is_reasoning_echo(self, delta: str) -> bool:
        """这一块正文是否就是已经流出去的思考原样复读。

        像素上游在模型只产出 thinking、正文为空时，会把 ``reasoning_content``
        塞进 ``content`` 兜底：一个完整的 ``output_text.delta`` 落在结束前一瞬，
        内容与已流出的思考逐字相同。这种正文会被客户端当成最终回答显示。
        """
        if self._text_all:
            return False
        reasoning = self._reasoning_all.strip()
        if not reasoning:
            return False
        return delta.strip() == reasoning

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
    def _error_chunk(message: str) -> dict:
        """Chat Completions 错误帧：客户端据此报错，而不是收到半截工具调用。"""
        return {
            "error": {
                "code": "upstream_stream_error",
                "type": "upstream_stream_error",
                "message": message,
            }
        }

    def finish_events(self) -> list:
        """流结束时的收尾事件。

        上游没有给出终止事件、且工具参数停在半句时，宁可发一条明确的错误帧，
        也不要把半截 JSON 当成成功结果交给客户端执行。
        """
        broken = [
            rec for rec in self.tool_records
            if not rec.get("arguments_sent") and self._collected_arguments(rec)[1]
        ]
        if broken:
            self.status = "failed"
            return [self._error_chunk(
                "上游在工具调用参数写完前中断了流（没有 response.completed），"
                "本次工具调用已丢弃，请直接重试。"
            )]
        if self.reasoning_echo and not self._text_all:
            self.status = "failed"
            return [self._error_chunk(
                "上游只返回了思考内容（reasoning），没有正文；请重试。"
            )]
        events = []
        for rec in self.tool_records:
            if not rec.get("arguments_sent"):
                self._emit_collected_arguments(rec, events)
        events.append(self.final_chunk())
        return events

    @staticmethod
    def _map_usage(u) -> dict:
        return {
            "prompt_tokens": u.get("input_tokens", 0),
            "prompt_tokens_details": {"cached_tokens": _usage_cached_tokens(u)},
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
        # 流结束时把清洗器里挂着的尾巴补发出去。
        tail = self._text_scrubber.flush()
        if tail:
            tail_events = []
            self._ensure_role(tail_events)
            tail_events.append(self._chunk({"content": tail}))
            for c in tail_events:
                yield c
        for c in self.finish_events():
            yield c

# ------------------ 转发主逻辑 ------------------

def upstream_headers(supplier: dict, request: Request, body: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {supplier.get('api_key', '')}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if supplier.get("forward_client_identity"):
        incoming_ua = request.headers.get("user-agent")
        if incoming_ua:
            headers["User-Agent"] = incoming_ua
        for key, value in request.headers.items():
            lower = key.lower()
            # This header identifies the local Codex Switch proxy. It is not
            # the client's identity and must not be exposed to an upstream.
            if lower == "x-openai-actor-authorization":
                continue
            if (
                lower.startswith("openai")
                or lower.startswith("codex")
                or lower.startswith("x-codex")
                or lower.startswith("x-openai")
                or lower.startswith("x-oai")
                or lower.startswith("x-client-request-id")
                or lower.startswith("x-responsesapi")
                or lower.startswith("x-reasoning-included")
                or lower.startswith("x-models-etag")
            ):
                headers[key] = value
    beta = request.headers.get("openai-beta")
    if beta:
        headers["OpenAI-Beta"] = beta
    _apply_opencode_session_header(headers, supplier, request, body)
    return headers


def _is_opencode_supplier(supplier: dict) -> bool:
    """Identify the OpenCode Go endpoint without changing other suppliers."""
    if not isinstance(supplier, dict):
        return False
    base = str(supplier.get("base_url") or "").strip()
    if not base:
        return False
    from urllib.parse import urlparse

    parsed = urlparse(base if "://" in base else f"//{base}")
    host = (parsed.netloc or "").lower()
    return host == "opencode.ai" or host.endswith(".opencode.ai")


def _opencode_session_id(request: Request, body: dict | None) -> str | None:
    """Find a stable per-conversation ID to use as ``x-opencode-session``.

    Codex sends its native ``session-id`` and ``thread-id`` headers.  Some
    auxiliary requests rely on body metadata instead.  Prefer an already
    canonical OpenCode session header, then native IDs, then stable Responses
    metadata so related turns stay on the same upstream route.
    """
    candidate_keys = (
        "x-opencode-session",
        "session-id",
        "thread-id",
        "x-client-request-id",
    )
    for key in candidate_keys:
        value = request.headers.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if not isinstance(body, dict):
        return None

    value = body.get("prompt_cache_key")
    if isinstance(value, str) and value.strip():
        return value.strip()

    metadata = body.get("client_metadata")
    if isinstance(metadata, dict):
        for key in ("session-id", "session_id", "thread-id", "thread_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _apply_opencode_session_header(
    headers: dict,
    supplier: dict,
    request: Request,
    body: dict | None = None,
) -> None:
    """Map a stable Codex conversation ID to OpenCode's required header."""
    if not _is_opencode_supplier(supplier):
        return
    session_id = _opencode_session_id(request, body)
    if session_id:
        headers["x-opencode-session"] = session_id


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


def _raw_chat_usage(payload: dict):
    """Return the original Chat usage object for provider-specific accounting."""
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


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
        "content": _strip_protocol_markup("".join(text_parts)) if text_parts else (None if tool_calls else ""),
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


def _usage_number(value):
    """Return a non-negative integer token count, or None when absent."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, number)


def _first_usage_number(*values):
    for value in values:
        number = _usage_number(value)
        if number is not None:
            return number
    return 0


def _usage_detail_number(usage: dict, detail_key: str, *field_names):
    detail = usage.get(detail_key)
    if not isinstance(detail, dict):
        return None
    for field_name in field_names:
        number = _usage_number(detail.get(field_name))
        if number is not None:
            return number
    return None


def _usage_alias_number(usage: dict, *field_names):
    for field_name in field_names:
        number = _usage_number(usage.get(field_name))
        if number is not None:
            return number
    return None


def _reasoning_effort_value(body: dict):
    """Extract the requested reasoning level from either wire format."""
    if not isinstance(body, dict):
        return None
    effort = body.get("reasoning_effort")
    if effort not in (None, ""):
        return effort
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort not in (None, ""):
            return effort
    elif isinstance(reasoning, str) and reasoning:
        return reasoning
    return None


REASONING_EFFORT_LEVELS = frozenset((
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
))


def _apply_configured_reasoning_effort(
    body: dict,
    upstream_wire: str,
    source_body: dict | None = None,
    profile_effort: str | None = None,
) -> dict:
    """仅按 Agent 预设档位补推理强度，且只在客户端未显式携带时生效。

    推理强度跟随 Agent：客户端请求里的档位永远优先，Agent 预设只做兜底；
    Router 不再按模型（model_reasoning_efforts）固定注入档位。
    显式值（包括显式空值）永远保留。
    """
    if not isinstance(body, dict):
        return body
    if isinstance(source_body, dict) and (
        "reasoning" in source_body or "reasoning_effort" in source_body
    ):
        return body
    if _reasoning_effort_value(body) is not None:
        return body

    effort = profile_effort
    if not effort:
        return body

    out = copy.deepcopy(body)
    if upstream_wire == "chat":
        out["reasoning_effort"] = effort
    else:
        out["reasoning"] = {"effort": effort}
    return out


def _is_xkiro_supplier(supplier: dict) -> bool:
    """Identify xKiro without changing behavior for other suppliers."""
    if not isinstance(supplier, dict):
        return False
    if str(supplier.get("name") or "").strip().casefold() == "xkiro":
        return True
    base_url = str(supplier.get("base_url") or "").strip()
    if not base_url:
        return False
    try:
        from urllib.parse import urlparse

        parsed = urlparse(base_url if "://" in base_url else f"//{base_url}")
        host = (parsed.hostname or "").casefold()
    except ValueError:
        return False
    return host == "api.xkiro.com"


def _normalize_xkiro_chat_reasoning(body: dict, source_body: dict | None = None) -> dict:
    """Convert Responses-style reasoning fields to xKiro Chat's parameter."""
    if not isinstance(body, dict):
        return body

    source_effort = _reasoning_effort_value(source_body)
    effort = source_effort if source_effort not in (None, "") else _reasoning_effort_value(body)
    has_reasoning = "reasoning" in body
    has_empty_effort = "reasoning_effort" in body and body.get("reasoning_effort") in (None, "")
    if effort in (None, "") and not has_reasoning and not has_empty_effort:
        return body

    out = copy.deepcopy(body)
    out.pop("reasoning", None)
    if effort in (None, ""):
        out.pop("reasoning_effort", None)
    else:
        out["reasoning_effort"] = effort
    return out


def _usage_cached_tokens(usage: dict) -> int:
    """Read cached-input usage across common OpenAI-compatible schemas."""
    detail_fields = (
        "cached_tokens",
        "cache_read_input_tokens",
        "cache_read_tokens",
        "cache_hit_tokens",
    )
    # Prefer an explicit non-zero cache count from either details object. Some
    # gateways include an empty ``cached_tokens: 0`` details object while also
    # returning a useful cache-read alias at the top level.
    zero_seen = False
    for detail_key in ("input_tokens_details", "prompt_tokens_details"):
        detail = usage.get(detail_key)
        if not isinstance(detail, dict):
            continue
        detail_values = []
        for field_name in detail_fields:
            value = _usage_number(detail.get(field_name))
            if value is not None:
                detail_values.append(value)
        if any(value > 0 for value in detail_values):
            return next(value for value in detail_values if value > 0)
        if detail_values:
            zero_seen = True
    value = _usage_alias_number(usage, *detail_fields)
    if value is not None:
        if value > 0 or not zero_seen:
            return value
    return 0


def _update_record_usage(rec: dict, usage) -> None:
    if not isinstance(usage, dict):
        return
    rec["input_tokens"] = _first_usage_number(
        usage.get("input_tokens"), usage.get("prompt_tokens")
    )
    rec["cached_tokens"] = _usage_cached_tokens(usage)
    rec["output_tokens"] = _first_usage_number(
        usage.get("output_tokens"), usage.get("completion_tokens")
    )
    rec["reasoning_tokens"] = _first_usage_number(
        _usage_detail_number(usage, "output_tokens_details", "reasoning_tokens"),
        _usage_detail_number(usage, "completion_tokens_details", "reasoning_tokens"),
        usage.get("reasoning_tokens"),
    )
    total = _usage_number(usage.get("total_tokens"))
    rec["total_tokens"] = total if total is not None else (
        rec["input_tokens"] + rec["output_tokens"]
    )


def _exc_message(exc: BaseException) -> str:
    """Human-readable message for stats records and failure events.

    Some transport exceptions (for example ``httpx.WriteError`` raised while
    uploading a large body) carry an empty ``str()``; falling back to the
    exception class name keeps the recorded error meaningful.
    """
    message = str(exc).strip()
    return message or type(exc).__name__


# ------------------ Agent 接入（客户端预设） ------------------

AGENT_PRESETS = {
    "codex": {"label": "Codex 桌面版", "wire": "responses", "endpoint": "/v1/responses"},
    "dsh": {"label": "DeepSeek Harness", "wire": "chat", "endpoint": "/v1/chat/completions"},
    "claude-code": {"label": "Claude Code", "wire": "anthropic", "endpoint": "/v1/messages"},
}


def detect_agent(request: Request, incoming_wire: str) -> str:
    """按入站协议和 User-Agent 识别客户端 Agent，用于统计归因与预设覆盖。

    - /v1/messages 只有 Anthropic 协议客户端会调（Claude Code）；
    - /v1/responses 只有 Responses 协议客户端会调（Codex 一侧，含 Remote AI 链路）；
    - /v1/chat/completions 是通用入口，只能按 UA 尽力识别。
    """
    if incoming_wire == "anthropic":
        return "claude-code"
    if incoming_wire == "responses":
        return "codex"
    try:
        ua = (request.headers.get("user-agent") or "").lower()
    except Exception:
        ua = ""
    if "codex" in ua:
        return "codex"
    if "claude" in ua:
        return "claude-code"
    if "deepseek" in ua or "dsh" in ua or "pi-ai" in ua or "undici" in ua or ua.startswith("node"):
        return "dsh"
    return ""


def _agent_profile(cfg: dict, agent: str) -> dict:
    profiles = cfg.get("agent_profiles") if isinstance(cfg, dict) else None
    if not isinstance(profiles, dict):
        return {}
    profile = profiles.get(agent)
    return profile if isinstance(profile, dict) else {}


def _agent_default_model(cfg: dict, agent: str) -> str:
    """Agent 预设的默认模型（仅作客户端未指定模型时的兜底）。"""
    profile = _agent_profile(cfg, agent)
    default = str(profile.get("default_model") or "").strip()
    if default:
        return default
    legacy = profile.get("model_map")
    if isinstance(legacy, dict):
        return str(legacy.get("*") or "").strip()
    return ""


def _apply_agent_profile(cfg: dict, agent: str, body: dict) -> None:
    """Agent 预设默认模型：仅在客户端没有指定模型时填入。

    客户端选择的模型永远原样使用，不再强制改写——保证控制台"最近请求"
    与统计里显示的就是实际干活的模型。
    """
    if agent not in AGENT_PRESETS or not isinstance(body, dict):
        return
    current = body.get("model")
    if isinstance(current, str) and current.strip():
        return
    default = _agent_default_model(cfg, agent)
    if default:
        body["model"] = default


def _agent_reasoning_effort(cfg: dict, agent: str) -> str | None:
    """Agent 预设的推理档位（仅在请求未显式携带档位时生效）。"""
    effort = _agent_profile(cfg, agent).get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        effort = effort.strip().lower()
        return effort if effort in REASONING_EFFORT_LEVELS else None
    return None


async def _iter_chat_chunks_for_anthropic(resp: httpx.Response):
    """Chat 上游事件流 -> 内部 chat chunk dict（过滤 [DONE] 哨兵）。"""
    async for evt in iter_upstream_events(resp):
        if evt is DONE:
            break
        if isinstance(evt, dict):
            yield evt


# --- 上游"早失败"探测 ------------------------------------------------------
# 上游可能先回 200，再在事件流里塞一个 error 帧（例如中转网关的
# "Upstream service temporarily unavailable"），或者干脆在吐出任何内容前断流
# （IncompleteRead）。这类失败发生时客户端还没收到一个字节，整条请求可以安全
# 重发；不重发的话，客户端只会看到一次莫名其妙的中断。
#
# 探测只缓存流开头的一小段字节：心跳照常转发，一旦出现内容帧立刻放行，所以
# 正常请求不会因此多等一个字节。只有"确认失败"才重发。
EARLY_STREAM_MAX_RETRIES = 2
EARLY_STREAM_RETRY_BACKOFF = 0.5
EARLY_STREAM_PROBE_SECONDS = 180.0
EARLY_STREAM_PROBE_BYTES = 256 * 1024

# responses 事件里这些只是开场白，不代表模型已经开始产出内容。
_RESPONSES_PREAMBLE_EVENTS = frozenset({
    "response.created",
    "response.in_progress",
    "response.queued",
    "response.output_item.added",
    "response.content_part.added",
})


def _early_stream_retry_budget(supplier: dict) -> int:
    """每个供应商可以重发的次数（默认 2，即最多尝试 3 次）。"""
    raw = supplier.get("early_stream_retries", EARLY_STREAM_MAX_RETRIES)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return EARLY_STREAM_MAX_RETRIES
    return max(0, min(value, 5))


def _complete_sse_payloads(buf: bytes):
    """取出缓冲区里已经收全的 data: 负载；尾部不完整的行留到下一次。"""
    for line in buf.split(b"\n")[:-1]:
        line = line.strip()
        if not line or line.startswith(b":"):
            continue
        if line.startswith(b"data:"):
            yield line[5:].strip()


def _stream_event_has_content(obj: dict, wire: str) -> bool:
    """这一帧是否代表上游已经开始产出内容（拿到就可以停止探测）。"""
    if wire == "responses":
        evt = obj.get("type")
        return (
            isinstance(evt, str)
            and evt.startswith("response.")
            and evt not in _RESPONSES_PREAMBLE_EVENTS
        )
    if obj.get("usage"):
        # 带用量的帧说明这次请求已经跑完了，即使没有正文也不算"早失败"。
        return True
    choices = obj.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                return True
            delta = choice.get("delta")
            if isinstance(delta, dict) and any(
                delta.get(key)
                for key in (
                    "content",
                    "reasoning_content",
                    "reasoning",
                    "tool_calls",
                    "function_call",
                )
            ):
                return True
    return False


def _event_error_text(obj: dict) -> str:
    error = obj.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    if isinstance(error, dict):
        for key in ("message", "msg", "detail", "code", "type"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    response = obj.get("response")
    if isinstance(response, dict):
        return _event_error_text(response)
    return ""


def _stream_head_verdict(buf: bytes, wire: str) -> str:
    """判定流开头：content 已产出内容 / error 上游报错 / pending 继续等。"""
    for payload in _complete_sse_payloads(buf):
        if payload == b"[DONE]":
            # 正常收尾（哪怕内容为空）不算失败，避免对合法空响应反复重发。
            return "content"
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("error") or obj.get("type") in ("error", "response.failed"):
            return "error"
        if _stream_event_has_content(obj, wire):
            return "content"
    return "pending"


def _stream_head_error_text(buf: bytes) -> str:
    for payload in _complete_sse_payloads(buf):
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict):
            text = _event_error_text(obj)
            if text:
                return text
    return ""


class _StreamHeadProbe:
    """缓存上游流开头，判断上游是否"还没产出内容就失败了"。"""

    def __init__(self, wire: str):
        self.wire = wire
        self.pending = bytearray()
        self.committed = False
        self.error = None

    def feed(self, chunk: bytes) -> bool:
        """喂入原始字节；返回 True 表示探测结束。"""
        self.pending += chunk
        verdict = _stream_head_verdict(bytes(self.pending), self.wire)
        if verdict == "error":
            self.error = _stream_head_error_text(bytes(self.pending)) or "上游返回了错误事件"
            return True
        if verdict == "content" or len(self.pending) >= EARLY_STREAM_PROBE_BYTES:
            self.committed = True
            return True
        return False

    def commit(self) -> None:
        """放弃探测（上游只是慢），按正常流转发。"""
        self.committed = True


class _PrefixedResponse:
    """把探测阶段缓存的字节接回流里，同时保证底层只有一条迭代器。"""

    def __init__(self, response, prefix: bytes, stream):
        self._response = response
        self._prefix = prefix
        self._stream = stream

    async def aiter_bytes(self):
        if self._prefix:
            prefix, self._prefix = self._prefix, b""
            yield prefix
        while True:
            try:
                item = await self._stream.__anext__()
            except StopAsyncIteration:
                return
            yield b": keep-alive\n\n" if item is SSE_HEARTBEAT else item

    def __getattr__(self, name):
        return getattr(self._response, name)


class _FirstChunkTimer:
    """包装上游响应，记录第一个数据块到达的耗时（首字）。"""

    def __init__(self, response, started: float):
        self._response = response
        self._started = started
        self.first_ms = None

    async def aiter_bytes(self):
        async for chunk in self._response.aiter_bytes():
            if self.first_ms is None:
                self.first_ms = int((time.monotonic() - self._started) * 1000)
            yield chunk

    def __getattr__(self, name):
        return getattr(self._response, name)


async def relay(request: Request, incoming_wire: str):
    global _last_session_model
    cfg = load_config()
    try:
        body = await request.json()
    except Exception:
        body = {}
    agent = detect_agent(request, incoming_wire)
    if incoming_wire == "anthropic":
        # Anthropic Messages 入站先转成内部统一的 Chat 请求，
        # 之后复用整条 Chat 管线（选路 / 重试 / 图片 / 推理档位）。
        body = _anthropic_to_chat(body)
    if agent:
        _apply_agent_profile(cfg, agent, body)
    model = (body or {}).get("model", "")
    is_title_generation = _is_title_generation_body(body)
    if not is_title_generation and isinstance(model, str) and model:
        _last_session_model = model
    elif is_title_generation and _last_session_model and model != _last_session_model:
        model = _last_session_model
        if isinstance(body, dict):
            # The substituted model must also reach the upstream payload:
            # otherwise the helper model name is forwarded while the supplier
            # (and stats) use the session model, and upstreams that do not
            # host the helper model reply with a not-found error.
            body["model"] = model
    supplier, upstream_model = await resolve_model_target(cfg, model)
    if not supplier:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": (
                        f"模型 {model} 没有可用的上游：模型名必须是 <供应商>/<模型名>，"
                        f"例如 xkiro/z-ai/glm-5.3。请在控制台的模型路由里添加该模型。"
                    )
                }
            },
        )
    if isinstance(body, dict) and upstream_model and body.get("model") != upstream_model:
        # 客户端看到的是 <供应商>/<模型名>，上游只认模型名本身。
        body["model"] = upstream_model
    upstream_wire = str(supplier.get("wire_api", "responses")).lower()
    if upstream_wire not in ("responses", "chat"):
        upstream_wire = "responses"
    path = "responses" if upstream_wire == "responses" else "chat/completions"
    base = str(supplier.get("base_url", "")).rstrip("/")
    url = _join_api_path(base, path)
    tool_bridge = _build_tool_bridge(body) if incoming_wire == "responses" and upstream_wire == "chat" else None
    is_minimax_responses = (
        incoming_wire == "responses"
        and upstream_wire == "responses"
        and _is_minimax_responses_supplier(supplier)
    )

    if incoming_wire == upstream_wire:
        out_body = body
        if is_minimax_responses:
            out_body = _sanitize_responses_for_minimax(out_body)
        elif incoming_wire == "responses" and _should_sanitize_openai_fields(supplier):
            out_body = _sanitize_responses_for_non_openai(out_body)
        elif incoming_wire == "responses":
            # openai_sanitize is disabled, but hosted tools must still be
            # stripped: third-party backends reject them unconditionally.
            _strip_hosted_tools_in_place(out_body)
    elif incoming_wire == "responses":
        out_body = responses_to_chat(body, tool_bridge=tool_bridge)
    else:
        out_body = chat_to_responses(body)

    if incoming_wire in ("chat", "anthropic"):
        out_body = _apply_configured_reasoning_effort(
            out_body,
            upstream_wire,
            source_body=body,
            profile_effort=_agent_reasoning_effort(cfg, agent) if agent else None,
        )

    if upstream_wire == "chat":
        out_body = _ensure_chat_stream_usage(out_body)
        if _is_xkiro_supplier(supplier):
            out_body = _normalize_xkiro_chat_reasoning(out_body, source_body=body)
    if not _should_forward_prompt_cache_key(supplier):
        out_body = _strip_prompt_cache_key(out_body)

    # 被 max-tokens / 断流打断的轮次会留下悬空工具调用，上游一律 400 拒绝。
    # 在发给上游之前补占位结果，避免整段会话作废。
    repaired_tool_calls = 0
    if _should_repair_tool_calls(supplier):
        try:
            repaired_tool_calls = repair_dangling_tool_calls(out_body, upstream_wire)
        except Exception:
            repaired_tool_calls = 0

    try:
        await vision.apply_image_policy(out_body, supplier, cfg, upstream_model)
    except Exception:
        # 图片策略异常时退化为纯文本占位，避免上游因 image_url 直接失败。
        vision.strip_images(out_body)

    rec = {
        "ts": time.time(),
        "model": qualified_model_id(supplier.get("name", ""), upstream_model),
        "supplier": supplier.get("name", ""),
        "upstream_wire": upstream_wire,
        "incoming_wire": incoming_wire,
        "agent": agent or None,
        "http_status": None,
        "status": "ok",
        "input_tokens": None,
        "cached_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "duration_ms": None,
        "first_token_ms": None,
        # Keep the recorded level aligned with either the incoming request
        # or the converted payload sent to the upstream provider.
        "reasoning_effort": _reasoning_effort_value(body) or _reasoning_effort_value(out_body),
        "error": None,
        "retry_count": 0,
        "repaired_tool_calls": repaired_tool_calls,
    }
    start_mono = time.monotonic()

    client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    resp = None
    last_exc = None
    response_error_text = None
    transport_retries = 0
    applied_compat_fallbacks = frozenset()
    minimax_image_retried = False
    request_body = out_body
    while True:
        try:
            req = client.build_request(
                "POST",
                url,
                json=request_body,
                headers=upstream_headers(supplier, request, body),
            )
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
        if (resp.status_code == 429 or resp.status_code >= 500) and transport_retries < 2:
            await resp.aread()
            await resp.aclose()
            resp = None
            await asyncio.sleep(0.5 * (2 ** transport_retries))
            transport_retries += 1
            rec["retry_count"] += 1
            continue

        response_error_text = None
        if (
            is_minimax_responses
            and not minimax_image_retried
            and resp.status_code in (400, 422)
        ):
            response_error_text = (await resp.aread()).decode("utf-8", "replace")
            if _minimax_sensitive_image_error(response_error_text):
                next_body = _strip_sensitive_image_candidate(request_body)
                if next_body is not None:
                    await resp.aclose()
                    resp = None
                    request_body = next_body
                    minimax_image_retried = True
                    rec["retry_count"] += 1
                    continue
        if resp.status_code == 400 and upstream_wire == "chat":
            if response_error_text is None:
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
        rec["error"] = _exc_message(exc)[:300]
        rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
        stats.record(rec)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"连接上游 {base} 失败：{_exc_message(exc)}"}},
        )

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
        if incoming_wire == "anthropic":
            # Claude Code 只认 Anthropic 的 error 结构。
            return JSONResponse(
                status_code=resp.status_code,
                content=_anthropic_error_json(text, status=resp.status_code),
            )
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

            raw_usage_for_record = (
                _raw_chat_usage(upstream_json)
                if incoming_wire == "responses" and upstream_wire == "chat"
                else None
            )
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
            elif incoming_wire == "anthropic":
                chat_json = (
                    upstream_json
                    if upstream_wire == "chat"
                    else _responses_json_to_chat_json(upstream_json, model)
                )
                result_json = _chat_json_to_anthropic_json(chat_json, model)
                result = JSONResponse(content=result_json, status_code=resp.status_code)
            else:
                result_json = _responses_json_to_chat_json(upstream_json, model)
                result = JSONResponse(content=result_json, status_code=resp.status_code)

            if isinstance(result_json, dict):
                _update_record_usage(rec, raw_usage_for_record or result_json.get("usage"))
                if rec["status"] == "ok" and result_json.get("status") in ("failed", "incomplete"):
                    rec["status"] = result_json["status"]
            rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
            # 非流式响应一次性返回，首字时间就是总耗时。
            rec["first_token_ms"] = rec["duration_ms"]
            stats.record(rec)
            return result
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = _exc_message(exc)[:300]
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
    anth_conv = None
    # 只有 SSE 才需要探测；上游直接回整段 JSON 时保持原来的透传行为。
    probe_stream = "text/event-stream" in (resp.headers.get("content-type") or "")
    early_retry_budget = _early_stream_retry_budget(supplier)

    async def gen():
        nonlocal anth_conv, resp
        guarded = None
        early_error = None
        attempt = 0
        iterator = None
        timed = _FirstChunkTimer(resp, start_mono)
        try:
            while True:
                attempt += 1
                # 首字耗时按"当前这一次尝试"算，重发后不会把上一次的等待算进来。
                timed = _FirstChunkTimer(
                    resp, start_mono if attempt == 1 else time.monotonic()
                )
                # 探测阶段独占这一条底层迭代器：心跳照常转发给客户端，数据字节
                # 先攒着，确认上游真的开始产出内容后再放行。
                iterator = _with_sse_heartbeats(timed.aiter_bytes()).__aiter__()
                probe = _StreamHeadProbe(upstream_wire)
                if not probe_stream:
                    probe.commit()
                deadline = time.monotonic() + EARLY_STREAM_PROBE_SECONDS
                while not probe.committed:
                    try:
                        item = await iterator.__anext__()
                    except StopAsyncIteration:
                        early_error = "上游在返回任何内容前结束了流"
                        break
                    except Exception as exc:
                        early_error = _exc_message(exc)
                        break
                    if item is SSE_HEARTBEAT:
                        yield b": keep-alive\n\n"
                        continue
                    if probe.feed(item):
                        break
                    if time.monotonic() >= deadline:
                        # 上游只是慢（长推理），不是失败：放行，别让客户端干等。
                        probe.commit()
                if probe.committed:
                    guarded = _PrefixedResponse(resp, bytes(probe.pending), iterator)
                    break
                early_error = probe.error or early_error
                await iterator.aclose()
                if resp is not None:
                    await resp.aclose()
                    resp = None
                if attempt > early_retry_budget:
                    break
                # 内容一个字节都还没到客户端，重发整条请求是安全的，
                # 代价只是多算一次输入 token。
                rec["retry_count"] += 1
                await asyncio.sleep(EARLY_STREAM_RETRY_BACKOFF * attempt)
                try:
                    req = client.build_request(
                        "POST",
                        url,
                        json=request_body,
                        headers=upstream_headers(supplier, request, body),
                    )
                    resp = await client.send(req, stream=True)
                except Exception as exc:
                    early_error = _exc_message(exc)
                    resp = None
                    break
                if resp.status_code >= 400:
                    early_error = (await resp.aread()).decode("utf-8", "replace")[:300]
                    await resp.aclose()
                    resp = None
                    break
            if guarded is None:
                raise RuntimeError(early_error or "上游在返回任何内容前失败了")
            if incoming_wire == upstream_wire:
                # 心跳在探测阶段就已经接进这条流里了，这里只管原样转发。
                async for chunk in guarded.aiter_bytes():
                    sniffer.feed(chunk)
                    yield chunk
            elif incoming_wire == "responses":
                conv = ChatToResponses(model, tool_bridge=tool_bridge)
                async for evt in _with_sse_heartbeats(conv.convert(guarded)):
                    if evt is SSE_HEARTBEAT:
                        chunk = b": keep-alive\n\n"
                    else:
                        chunk = sse_bytes(evt)
                    sniffer.feed(chunk)
                    yield chunk
            elif incoming_wire == "anthropic":
                # Anthropic SSE 不以 [DONE] 结尾，message_stop 就是终止事件。
                anth_conv = _ChatToAnthropic(model)
                if upstream_wire == "chat":
                    chat_source = _iter_chat_chunks_for_anthropic(guarded)
                else:
                    chat_source = ResponsesToChat(model).convert(guarded)
                async for evt in _with_sse_heartbeats(_stream_chat_to_anthropic(chat_source, model, anth_conv)):
                    if evt is SSE_HEARTBEAT:
                        chunk = b": keep-alive\n\n"
                    else:
                        chunk = sse_bytes(evt)
                    sniffer.feed(chunk)
                    yield chunk
            else:
                conv = ResponsesToChat(model)
                async for evt in _with_sse_heartbeats(conv.convert(guarded)):
                    if evt is SSE_HEARTBEAT:
                        chunk = b": keep-alive\n\n"
                    else:
                        chunk = sse_bytes(evt)
                    sniffer.feed(chunk)
                    yield chunk
                yield b"data: [DONE]\n\n"
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = _exc_message(exc)[:300]
            # A broken upstream chunked body must not leave Codex with an
            # abrupt transport error when a protocol-level terminal event can
            # still be delivered. Do not transparently retry after output.
            chunk = upstream_stream_failure_event(
                incoming_wire, model, _exc_message(exc)
            )
            sniffer.feed(chunk)
            yield chunk
            if incoming_wire == "chat":
                yield b"data: [DONE]\n\n"
        else:
            if incoming_wire == "responses" and not sniffer.saw_terminal:
                rec["status"] = "failed"
                rec["error"] = "Upstream Responses stream ended without response.completed"
                chunk = upstream_stream_failure_event(
                    incoming_wire,
                    model,
                    rec["error"],
                )
                sniffer.feed(chunk)
                yield chunk
        finally:
            if iterator is not None:
                # 客户端中途断开时也要把心跳泵一起收掉，避免留下悬挂任务。
                try:
                    await iterator.aclose()
                except Exception:
                    pass
            u = anth_conv.usage if anth_conv is not None else sniffer.usage
            _update_record_usage(rec, u)
            stream_status = anth_conv.status if anth_conv is not None else sniffer.status
            if rec["status"] == "ok" and stream_status in ("failed", "incomplete", "error"):
                rec["status"] = stream_status
            rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
            rec["first_token_ms"] = timed.first_ms
            stats.record(rec)
            if resp is not None:
                await resp.aclose()
            await client.aclose()

    return StreamingResponse(gen(), status_code=resp.status_code, media_type="text/event-stream")


def _image_request_headers(supplier: dict, request: Request, content_type: str) -> dict:
    headers = {
        "Authorization": f"Bearer {supplier.get('api_key', '')}",
        "Accept": "application/json",
    }
    if content_type:
        headers["Content-Type"] = content_type
    beta = request.headers.get("openai-beta")
    if beta:
        headers["OpenAI-Beta"] = beta
    return headers


_MULTIPART_MODEL_RE = re.compile(rb'name\s*=\s*"model"\r?\n\r?\n([^\r\n]+)')


def _multipart_model(raw: bytes) -> str:
    match = _MULTIPART_MODEL_RE.search(raw)
    if not match:
        return ""
    return match.group(1).decode("utf-8", "replace").strip()


def _replace_multipart_model(raw: bytes, old: str, new: str) -> bytes:
    """把 multipart 表单里 ``model`` 字段的值换成上游真实模型名。"""
    match = _MULTIPART_MODEL_RE.search(raw)
    if not match:
        return raw
    current = match.group(1).decode("utf-8", "replace").strip()
    if current != old:
        return raw
    start, end = match.span(1)
    return raw[:start] + new.encode("utf-8") + raw[end:]


async def _relay_image(request: Request, path: str) -> Response:
    """把 /v1/images/* 请求按 model 转发到已配置的图片供应商。"""
    cfg = load_config()
    content_type = request.headers.get("content-type", "")
    raw_body = None
    body = None
    if "multipart/form-data" in content_type.lower():
        raw_body = await request.body()
        model = _multipart_model(raw_body)
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        model = (body or {}).get("model", "")

    supplier, upstream_model = await resolve_model_target(cfg, model)
    if not supplier:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": (
                        f"模型 {model} 没有可用的上游：模型名必须是 <供应商>/<模型名>。"
                    )
                }
            },
        )
    if upstream_model and upstream_model != model:
        # 限定名只用于客户端选路，转发前换成上游真实模型名。
        if raw_body is not None:
            raw_body = _replace_multipart_model(raw_body, model, upstream_model)
        elif isinstance(body, dict):
            body["model"] = upstream_model

    base = str(supplier.get("base_url", "")).rstrip("/")
    url = _join_api_path(base, path)
    headers = _image_request_headers(supplier, request, content_type)
    rec = {
        "ts": time.time(),
        "model": qualified_model_id(supplier.get("name", ""), upstream_model),
        "supplier": supplier.get("name", ""),
        "upstream_wire": "image",
        "incoming_wire": "image",
        "http_status": None,
        "status": "ok",
        "input_tokens": None,
        "cached_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "duration_ms": None,
        "first_token_ms": None,
        "reasoning_effort": None,
        "error": None,
        "retry_count": 0,
    }
    start_mono = time.monotonic()
    client = None
    resp = None
    response_body = b""
    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        if raw_body is not None:
            req = client.build_request("POST", url, content=raw_body, headers=headers)
        else:
            req = client.build_request("POST", url, json=body, headers=headers)
        resp = await client.send(req)
        response_body = await resp.aread()
    except Exception as exc:
        rec["status"] = "error"
        rec["http_status"] = 0
        rec["error"] = str(exc)[:300]
        rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
        stats.record(rec)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"连接上游 {base} 失败：{exc}"}},
        )
    finally:
        if resp is not None:
            await resp.aclose()
        if client is not None:
            await client.aclose()

    rec["http_status"] = resp.status_code
    rec["duration_ms"] = int((time.monotonic() - start_mono) * 1000)
    if resp.status_code >= 400:
        rec["status"] = "error"
        rec["error"] = response_body[:300].decode("utf-8", "replace")
    stats.record(rec)
    return Response(
        content=response_body,
        status_code=resp.status_code,
        headers={"content-type": resp.headers.get("content-type") or "application/json"},
    )


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
    if MODEL_ID_SEPARATOR in str(s.get("name")):
        # 公开模型名是 <供应商>/<模型名>，供应商名里的斜杠会让切分产生歧义。
        return f"供应商名称不能包含 {MODEL_ID_SEPARATOR}"
    if not s.get("base_url"):
        return f"供应商 {s.get('name')} 缺少接口地址 base_url"
    if s.get("wire_api") not in (None, "", "responses", "chat"):
        return f"供应商 {s.get('name')} 的 wire_api 只能是 responses 或 chat"
    if s.get("vlm_prompt_mode") not in (None, "", "main-model", "template"):
        return f"供应商 {s.get('name')} 的 vlm_prompt_mode 只能是 main-model 或 template"
    if s.get("openai_sanitize") is not None and not isinstance(s.get("openai_sanitize"), bool):
        return f"供应商 {s.get('name')} 的 openai_sanitize 必须是布尔值"
    responses_compat = s.get("responses_compat")
    if responses_compat is not None and (
        not isinstance(responses_compat, str)
        or responses_compat.strip().lower() not in ("", "minimax")
    ):
        return f"供应商 {s.get('name')} 的 responses_compat 只能是空值或 minimax"
    forward_cache_key = s.get("forward_prompt_cache_key")
    if forward_cache_key is not None and not isinstance(forward_cache_key, bool):
        return f"供应商 {s.get('name')} 的 forward_prompt_cache_key 必须是布尔值"
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
                # 改名后级联更新：路由映射 / 模型价格 / 输出上限 / 默认供应商 /
                # 模型缓存 / 隐藏列表 / Agent 默认模型
                prefix = f"{name}{MODEL_ID_SEPARATOR}"
                renamed = {}
                for k, v in (cfg.get("model_map") or {}).items():
                    key = str(k)
                    new_key = new_name + key[len(name):] if key.startswith(prefix) else key
                    renamed[new_key] = new_name if v == name else v
                cfg["model_map"] = renamed
                for field in ("model_prices", "model_max_output_tokens"):
                    table = cfg.get(field)
                    if not isinstance(table, dict):
                        continue
                    cfg[field] = {
                        (new_name + str(k)[len(name):] if str(k).startswith(prefix) else k): v
                        for k, v in table.items()
                    }
                if cfg.get("default_supplier") == name:
                    cfg["default_supplier"] = new_name
                for profile in (cfg.get("agent_profiles") or {}).values():
                    if not isinstance(profile, dict):
                        continue
                    default_model = str(profile.get("default_model") or "")
                    if default_model.startswith(prefix):
                        # Agent 默认模型用的是 <供应商>/<模型名>，前缀要跟着换。
                        profile["default_model"] = new_name + default_model[len(name):]
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
    return {
        "model_map": cfg.get("model_map", {}),
        "default_supplier": cfg.get("default_supplier", ""),
    }


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
        if split_qualified_model_id(cfg, str(model)) is None:
            return JSONResponse(
                status_code=400,
                content={"error": f"模型名必须是 <供应商>/<模型名>：{model}"},
            )
    default = data.get("default_supplier", cfg.get("default_supplier", ""))
    if default and default not in names:
        return JSONResponse(status_code=400, content={"error": f"默认供应商 {default} 不存在"})
    cfg["model_map"] = mm
    cfg.pop("model_overrides", None)  # 按模型的兼容覆盖已移除，供应商级设置生效
    cfg["default_supplier"] = default or ""
    save_config(cfg)
    return {"ok": True}


async def _delete_model_map_entry(model: str):
    """Delete a model-map entry by its complete model ID."""
    model = model.strip()
    cfg = load_config()
    mm = cfg.get("model_map", {})
    if model not in mm:
        return JSONResponse(status_code=404, content={"error": f"模型 {model} 不在 model_map 中"})
    del mm[model]
    cfg.pop("model_overrides", None)  # 按模型的兼容覆盖已移除
    save_config(cfg)
    return {"ok": True}


@app.delete("/api/model_map")
async def api_delete_model_map(model: str):
    """Delete a model-map entry supplied as a query parameter.

    Model IDs commonly contain slashes (for example ``openai/gpt-5.6-sol``).
    The query form keeps the complete ID intact for the dashboard.
    """
    return await _delete_model_map_entry(model)


@app.delete("/api/model_map/{model:path}")
async def api_delete_model_map_path(model: str):
    """Backward-compatible path form for model IDs, including slashes."""
    return await _delete_model_map_entry(model)


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
        cfg.setdefault("model_map", {})[qualified_model_id(supplier_name, model)] = supplier_name
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
            try:
                data_list = r.json().get("data", [])
                ids = [m.get("id") for m in data_list if isinstance(m, dict) and m.get("id")]
                return {"ok": True, "models": ids, "count": len(ids)}
            except Exception:
                preview = (r.text or repr(r.content[:200]))[:200]
                return {"ok": False, "error": f"HTTP 200 但响应不是 JSON（Content-Type: {r.headers.get('content-type', '?')}）：{preview}"}
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


# ------------------ Agent 接入 API ------------------

@app.get("/api/agents")
async def api_agents():
    """Agent 接入预设：状态、覆盖配置和可选模型列表。"""
    cfg = load_config()
    since = time.time() - 24 * 3600
    recent = stats.recent(2000)
    agents = []
    for agent_id, preset in AGENT_PRESETS.items():
        profile = _agent_profile(cfg, agent_id)
        rows = [r for r in recent if r.get("agent") == agent_id]
        day_rows = [r for r in rows if (r.get("ts") or 0) >= since]
        last = rows[-1] if rows else None
        default_model = _agent_default_model(cfg, agent_id)
        agents.append({
            "id": agent_id,
            "label": preset["label"],
            "wire": preset["wire"],
            "endpoint": preset["endpoint"],
            # 控制台与 Agent 看到的是 <供应商>/<模型名>。
            "default_model": default_model,
            "reasoning_effort": profile.get("reasoning_effort") or "",
            "requests_24h": len(day_rows),
            "errors_24h": len([r for r in day_rows if r.get("status") != "ok"]),
            "last_ts": (last or {}).get("ts"),
            "last_model": (last or {}).get("model"),
            "last_supplier": (last or {}).get("supplier"),
        })
    # 模型组 = 模型路由界面显式添加的模型（路由映射 + 手动模型），不含上游目录杂项。
    models = [m["id"] for m in configured_model_list(cfg)]
    try:
        port = get_current_port() or int(cfg.get("port") or 8765)
    except Exception:
        port = int(cfg.get("port") or 8765)
    return {"agents": agents, "models": models, "base_url": f"http://127.0.0.1:{port}/v1"}


@app.put("/api/agents/{agent_id}")
async def api_put_agent(agent_id: str, request: Request):
    """保存单个 Agent 的预设覆盖（默认模型 / 推理档位）。"""
    if agent_id not in AGENT_PRESETS:
        return JSONResponse(status_code=404, content={"error": f"未知 Agent：{agent_id}"})
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON"})
    cfg = load_config()
    profile = cfg.setdefault("agent_profiles", {}).setdefault(agent_id, {})
    if not isinstance(profile, dict):
        profile = {}
        cfg["agent_profiles"][agent_id] = profile
    if "default_model" in data or "model_map" in data:
        default = str(data.get("default_model") or "").strip()
        if not default:
            legacy = data.get("model_map") or {}
            if isinstance(legacy, dict):
                default = str(legacy.get("*") or "").strip()
        profile["default_model"] = default
        profile.pop("model_map", None)  # 旧的强制改写映射已废弃
    if "reasoning_effort" in data:
        effort = str(data.get("reasoning_effort") or "").strip().lower()
        if effort and effort not in REASONING_EFFORT_LEVELS:
            return JSONResponse(status_code=400, content={"error": f"无效的推理档位：{effort}"})
        profile["reasoning_effort"] = effort
    save_config(cfg)
    return {"ok": True}


@app.post("/api/agents/{agent_id}/test")
async def api_test_agent(agent_id: str, request: Request):
    """端到端测试：从 Router 本机按该 Agent 的协议发一条最小请求。"""
    if agent_id not in AGENT_PRESETS:
        return JSONResponse(status_code=404, content={"error": f"未知 Agent：{agent_id}"})
    try:
        data = await request.json()
    except Exception:
        data = {}
    model = str((data or {}).get("model") or "").strip()
    if not model:
        cfg = load_config()
        model = _agent_default_model(cfg, agent_id)
        if not model and cfg.get("model_map"):
            model = next(iter(cfg["model_map"]))
        if not model:
            models = configured_model_list(cfg)
            model = models[0]["id"] if models else ""
    if not model:
        return {"ok": False, "error": "没有可用模型：请先在模型路由里添加"}
    endpoint = AGENT_PRESETS[agent_id]["endpoint"]
    if endpoint == "/v1/responses":
        payload = {"model": model, "stream": False, "input": "ping", "max_output_tokens": 16}
    elif endpoint == "/v1/messages":
        payload = {"model": model, "stream": False, "max_tokens": 16, "messages": [{"role": "user", "content": "ping"}]}
    else:
        payload = {"model": model, "stream": False, "max_tokens": 16, "messages": [{"role": "user", "content": "ping"}]}
    try:
        port = get_current_port() or int(load_config().get("port") or 8765)
    except Exception:
        port = 8765
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"http://127.0.0.1:{port}{endpoint}",
                json=payload,
                headers={"Authorization": "Bearer agent-test"},
            )
        ms = int((time.monotonic() - started) * 1000)
        preview = ""
        try:
            body = r.json()
            if isinstance(body, dict):
                if endpoint == "/v1/messages":
                    preview = " ".join(
                        (b.get("text") or "") for b in (body.get("content") or []) if isinstance(b, dict)
                    )[:120]
                else:
                    choices = body.get("choices") or body.get("output") or []
                    if choices and isinstance(choices[0], dict):
                        preview = str((choices[0].get("message") or {}).get("content") or "")[:120]
        except Exception:
            preview = (r.text or "")[:120]
        return {
            "ok": r.status_code < 400,
            "http_status": r.status_code,
            "duration_ms": ms,
            "model": model,
            "preview": preview,
            "error": None if r.status_code < 400 else (r.text or "")[:300],
        }
    except Exception as exc:
        return {"ok": False, "error": _exc_message(exc)[:300], "model": model}


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
    status: str = "",
):
    """最近请求（分页）：支持模型、状态与时间范围筛选。"""
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
        status=str(status).strip(),
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


async def _delete_model_price_entry(model: str):
    cfg = load_config()
    (cfg.setdefault("model_prices", {})).pop(model, None)
    save_config(cfg)
    return {"ok": True}


@app.delete("/api/model-price")
async def api_delete_model_price(model: str):
    """Clear a model-specific price supplied as a query parameter."""
    return await _delete_model_price_entry(model)


@app.delete("/api/model-price/{model:path}")
async def api_delete_model_price_path(model: str):
    """Backward-compatible path form for model IDs, including slashes."""
    return await _delete_model_price_entry(model)


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
        "platform": "windows" if _paths.IS_WINDOWS else ("macos" if _paths.IS_MACOS else sys.platform),
        "self_update": _paths.SELF_UPDATE_SUPPORTED,
        "exe_name": _paths.EXE_NAME,
        "new_exe_name": _paths.NEW_EXE_NAME,
        "current_exe": _exe_file_info(base_dir / _paths.EXE_NAME),
        "new_exe": _exe_file_info(base_dir / _paths.NEW_EXE_NAME),
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
    proc_name = _paths.APP_NAME.replace("'", "''")
    wait_block = (
        "$deadline = (Get-Date).AddSeconds(90)\n"
        f"while (Get-Process -Name '{proc_name}' -ErrorAction SilentlyContinue) {{\n"
        "    if ((Get-Date) -ge $deadline) {\n"
        f"        Get-Process -Name '{proc_name}' -ErrorAction SilentlyContinue | Stop-Process -Force\n"
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
    exe_name = _paths.EXE_NAME
    prev_name = _paths.PREV_EXE_NAME
    new_name = _paths.NEW_EXE_NAME
    return f"""$ErrorActionPreference = 'Stop'
$dir = '{dir_literal}'
Set-Location -LiteralPath $dir

{wait_block}
$old = Join-Path $dir '{exe_name}'
$prev = Join-Path $dir '{prev_name}'
$new = Join-Path $dir '{new_name}'
Remove-Item -LiteralPath $prev -Force -ErrorAction SilentlyContinue

function Start-Router {{
    # A restarted PyInstaller app must not inherit the old process's _MEI runtime.
    Get-ChildItem Env: | Where-Object {{ $_.Name -like '_PYI_*' }} | Remove-Item -ErrorAction SilentlyContinue
    $env:PYINSTALLER_RESET_ENVIRONMENT = '1'
    Start-Process -FilePath $old -WorkingDirectory $dir
}}

function Get-Sha256([string] $path) {{
    # Use .NET instead of Get-FileHash for compatibility with minimal Windows PowerShell 5.1 images.
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {{
        $bytes = [System.IO.File]::ReadAllBytes($path)
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '')
    }} finally {{
        $sha.Dispose()
    }}
}}

try {{
    if (-not (Test-Path -LiteralPath $new)) {{
        throw '{new_name} not found'
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
    $hashNew = Get-Sha256 $new
    $hashOld = Get-Sha256 $old
    if ($hashNew -ne $hashOld) {{
        throw 'SHA256 mismatch after copy'
    }}
    Remove-Item -LiteralPath $new -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $prev -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $dir '{_paths.CRASH_LOG_PATH.name}') -Force -ErrorAction SilentlyContinue
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
    if not _paths.SELF_UPDATE_SUPPORTED:
        return JSONResponse(
            status_code=400,
            content={
                "error": "macOS 版不支持在线替换程序：请下载新的 LLMRouter-macos-*.zip，"
                "解压后把 LLMRouter.app 拖到 /Applications 覆盖旧版本",
            },
        )
    if not getattr(sys, "frozen", False):
        return JSONResponse(status_code=400, content={"error": "源码模式无法自动更新 EXE，请使用 apply-update.bat"})
    new_exe = BASE_DIR / _paths.NEW_EXE_NAME
    if not new_exe.exists():
        return JSONResponse(status_code=400, content={"error": f"未找到 {_paths.NEW_EXE_NAME}"})

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
    data = configured_model_list(cfg)
    return {"object": "list", "data": data}


@app.post("/v1/responses")
async def post_responses(request: Request):
    return await relay(request, "responses")


@app.post("/v1/chat/completions")
async def post_chat(request: Request):
    return await relay(request, "chat")


@app.post("/v1/messages")
async def post_messages(request: Request):
    """Anthropic Messages 协议入口（Claude Code 等客户端）。"""
    return await relay(request, "anthropic")


@app.post("/v1/messages/count_tokens")
async def post_messages_count_tokens(request: Request):
    """Anthropic count_tokens：本地粗估，不请求上游。"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return {"input_tokens": _anthropic_estimate_input_tokens(body)}


@app.post("/v1/images/generations")
async def post_images_generations(request: Request):
    return await _relay_image(request, "images/generations")


@app.post("/v1/images/edits")
async def post_images_edits(request: Request):
    return await _relay_image(request, "images/edits")


if __name__ == "__main__":
    import threading
    import webbrowser

    import uvicorn

    stats.load()
    migrate_config_on_startup()
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
