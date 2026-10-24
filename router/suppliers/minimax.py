# -*- coding: utf-8 -*-
"""MiniMax M3 Responses-subset compatibility adaptor.

MiniMax's Responses endpoint implements a smaller schema than OpenAI's:
- No ``custom`` / ``tool_search`` tool types (only ``function``).
- No ``agent_message`` input items (converted to plain user messages).
- ``encrypted_content`` items are unwrapped into ``input_text``.
- Images rejected by content-safety (1026) are stripped one at a time so a
  subsequent retry can succeed without re-uploading all images.

All heavy lifting (tool bridging, content flattening) is delegated to
:mod:`suppliers.shared`.
"""
import copy
import json
import re
import uuid

from suppliers.shared import (
    OPENAI_ENCRYPTED_PLACEHOLDER,
    args_to_string,
    bridge_chat_name,
    build_tool_bridge,
    content_to_text,
    custom_tool_arguments,
    unwrap_encrypted_content_parts,
)


MINIMAX_SENSITIVE_IMAGE_PLACEHOLDER = "[图片已被 MiniMax 内容安全拒绝]"


def is_minimax_responses_supplier(supplier: dict) -> bool:
    if not isinstance(supplier, dict):
        return False
    return str(supplier.get("responses_compat") or "").strip().lower() == "minimax"


def _normalize_image(part: dict) -> dict:
    image = part.get("image_url") or part.get("url")
    detail = part.get("detail") or "auto"
    if isinstance(image, dict):
        detail = image.get("detail") or detail
        image = image.get("url") or image.get("image_url") or ""
    return {"type": "input_image", "image_url": image or "", "detail": detail}


def _normalize_content(content):
    """Normalize Responses content parts for MiniMax's smaller InputItem schema."""
    if isinstance(content, str):
        return content if content else None
    if not isinstance(content, list):
        text = str(content)
        return text if text.strip() else None
    parts = []
    for part in content:
        if isinstance(part, str):
            if part:
                parts.append({"type": "input_text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "encrypted_content":
            payload = part.get("encrypted_content")
            text = payload if isinstance(payload, str) and payload else OPENAI_ENCRYPTED_PLACEHOLDER
            parts.append({"type": "input_text", "text": text})
        elif part_type in ("input_image", "image_url"):
            parts.append(_normalize_image(part))
        elif part_type == "input_file":
            label = part.get("filename") or part.get("file_id") or "附件"
            parts.append({"type": "input_text", "text": f"[文件: {label}]"})
        elif part_type in ("input_text", "output_text", "text"):
            text = part.get("text")
            if text is not None:
                parts.append({"type": "input_text", "text": str(text)})
        else:
            text = content_to_text([part]) or json.dumps(part, ensure_ascii=False)
            if text:
                parts.append({"type": "input_text", "text": text})
    return parts if parts else None


def _output_is_content_parts(output) -> bool:
    if not isinstance(output, list):
        return False
    content_types = {
        "input_text",
        "output_text",
        "text",
        "input_image",
        "image_url",
        "input_file",
        "refusal",
    }
    for part in output:
        if isinstance(part, str):
            continue
        if not isinstance(part, dict) or part.get("type") not in content_types:
            return False
    return True


def _output_value(output):
    if isinstance(output, str):
        return output
    if _output_is_content_parts(output):
        return _normalize_content(output) or []
    if output is None:
        return ""
    return json.dumps(output, ensure_ascii=False)


def _tool_search_text(item: dict) -> str:
    output = item.get("output")
    if _output_is_content_parts(output):
        return _output_value(output)
    loaded = []
    candidates = item.get("tools") or output or []
    if isinstance(candidates, dict):
        candidates = candidates.get("tools") or [candidates]
    if isinstance(candidates, list):
        for tool in candidates:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "namespace":
                namespace = tool.get("name") or "namespace"
                loaded.extend(
                    f"{namespace}__{x.get('name')}"
                    for x in tool.get("tools") or []
                    if isinstance(x, dict) and x.get("name")
                )
            elif tool.get("name"):
                loaded.append(str(tool["name"]))
    if loaded:
        return "Loaded tools: " + ", ".join(loaded)
    payload = output if output is not None else item.get("tools")
    return _output_value(payload)


def _unknown_item_text(item: dict) -> str:
    for key in ("content", "output", "input", "arguments"):
        payload = item.get(key)
        if payload is None:
            continue
        if isinstance(payload, str):
            if payload.strip():
                return payload
            continue
        if isinstance(payload, list):
            text = content_to_text(payload)
            if text:
                return text
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=False)
        if str(payload).strip():
            return str(payload)
    return ""


def _user_message(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _message_item(item: dict) -> dict | None:
    role = item.get("role") or "user"
    if role == "developer":
        role = "system"
    if role not in ("user", "system", "assistant"):
        role = "user"
    content = _normalize_content(unwrap_encrypted_content_parts(item.get("content")))
    if not content:
        return None
    return {"type": "message", "role": role, "content": content}


def _agent_item(item: dict) -> dict:
    content = _normalize_content(unwrap_encrypted_content_parts(item.get("content")))
    if not content:
        return _user_message("Agent message content is unavailable.")
    return {"type": "message", "role": "user", "content": content}


def _call_id(item: dict) -> str:
    return item.get("call_id") or item.get("id") or ("call_" + uuid.uuid4().hex[:16])


def _call_item(item: dict, bridge: dict) -> dict:
    item_type = item.get("type")
    call_id = _call_id(item)
    if item_type == "function_call":
        return {
            "type": "function_call",
            "call_id": call_id,
            "name": str(item.get("name") or ""),
            "arguments": args_to_string(item.get("arguments", {})),
        }
    if item_type == "tool_search_call":
        name = str(item.get("name") or "tool_search")
        arguments = item.get("arguments") if "arguments" in item else item.get("action") or {}
        if not isinstance(arguments, str):
            arguments = args_to_string(arguments or {})
        return {
            "type": "function_call",
            "call_id": call_id,
            "name": bridge_chat_name(bridge, "tool_search", name, item.get("namespace")),
            "arguments": arguments,
        }
    name = str(item.get("name") or "custom_tool")
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": bridge_chat_name(bridge, "custom", name, item.get("namespace")),
        "arguments": custom_tool_arguments(item.get("input", item.get("arguments", {}))),
    }


def _tools_from_bridge(bridge: dict) -> list:
    tools = []
    for tool in bridge.get("chat_tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        name = fn.get("name") or tool.get("name") or ""
        if not name:
            continue
        item = {"type": "function", "name": str(name)}
        if fn.get("description") is not None:
            item["description"] = str(fn["description"])
        if isinstance(fn.get("parameters"), dict):
            item["parameters"] = fn["parameters"]
        tools.append(item)
    return tools


def _tool_choice(tool_choice, bridge: dict):
    if not isinstance(tool_choice, dict):
        return tool_choice
    kind = tool_choice.get("type")
    if kind == "namespace":
        nested = tool_choice.get("tools") or []
        if nested and isinstance(nested[0], dict):
            inner = dict(nested[0])
            inner.setdefault("namespace", tool_choice.get("name"))
            return _tool_choice(inner, bridge)
        return tool_choice
    if kind not in ("function", "custom", "tool_search"):
        return tool_choice
    name = tool_choice.get("name")
    if not name:
        fn = tool_choice.get("function") or {}
        name = fn.get("name") or ("tool_search" if kind == "tool_search" else "")
    fn_name = bridge_chat_name(bridge, kind, str(name), tool_choice.get("namespace"))
    return {"type": "function", "name": fn_name}


def sanitize_responses(body: dict) -> dict:
    """Adapt Codex Responses bodies to MiniMax's supported Responses subset."""
    if not isinstance(body, dict):
        return body

    sanitized = copy.deepcopy(body)
    bridge = build_tool_bridge(sanitized)
    changed = False

    if sanitized.get("tools") is not None or bridge.get("chat_tools"):
        sanitized["tools"] = _tools_from_bridge(bridge)
        changed = True

    if sanitized.get("tool_choice") is not None:
        sanitized["tool_choice"] = _tool_choice(sanitized["tool_choice"], bridge)
        changed = True

    input_value = sanitized.get("input")
    if isinstance(input_value, str):
        return sanitized
    if not isinstance(input_value, list):
        sanitized["input"] = []
        changed = True
        return sanitized

    known_call_ids = set()
    for item in input_value:
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("function_call", "tool_search_call", "custom_tool_call"):
            call_id = item.get("call_id") or item.get("id") or ""
            if call_id:
                known_call_ids.add(call_id)

    new_input = []
    for item in input_value:
        if isinstance(item, str):
            if item.strip():
                new_input.append(_user_message(item))
                changed = True
            continue
        if not isinstance(item, dict):
            changed = True
            continue

        item_type = item.get("type")
        if item_type in ("function_call", "tool_search_call", "custom_tool_call"):
            new_input.append(_call_item(item, bridge))
            changed = True
        elif item_type in ("message", "reasoning"):
            if item_type == "message":
                message = _message_item(item)
                if message is not None:
                    new_input.append(message)
                    changed = True
            else:
                reasoning = dict(item)
                reasoning.pop("author", None)
                reasoning.pop("recipient", None)
                new_input.append(reasoning)
                changed = True
        elif item_type == "agent_message":
            new_input.append(_agent_item(item))
            changed = True
        elif item_type == "function_call_output":
            call_id = item.get("call_id") or item.get("id") or ""
            if call_id:
                new_input.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _output_value(item.get("output")),
                })
            else:
                text = _unknown_item_text(item)
                if text:
                    new_input.append(_user_message(text))
            changed = True
        elif item_type == "tool_search_output":
            call_id = item.get("call_id") or item.get("id") or ""
            if call_id in known_call_ids:
                new_input.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _tool_search_text(item),
                })
            else:
                text = _tool_search_text(item)
                if text:
                    new_input.append(_user_message(text))
            changed = True
        elif item_type == "custom_tool_call_output":
            call_id = item.get("call_id") or item.get("id") or ""
            if call_id in known_call_ids:
                new_input.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _output_value(item.get("output")),
                })
            else:
                text = _unknown_item_text(item)
                if text:
                    new_input.append(_user_message(text))
            changed = True
        else:
            text = _unknown_item_text(item)
            if text:
                new_input.append(_user_message(text))
            changed = True

    sanitized["input"] = new_input
    sanitized.pop("prompt_cache_key", None)
    return sanitized if changed else sanitized


def sensitive_image_error(error_text: str) -> bool:
    if not error_text:
        return False
    text = error_text.lower()
    return (
        "new_sensitive" in text
        or "image is sensitive" in text
        or "(1026)" in text
        or re.search(r"messages\[\d+\].*\bcontent\[\d+\]", text) is not None
    )


def _image_candidates(body: dict) -> list:
    candidates = []

    def visit(value, path):
        if isinstance(value, dict):
            if value.get("type") in ("input_image", "image_url"):
                candidates.append((path, value))
            for key, child in value.items():
                visit(child, path + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (index,))

    visit(body, ())
    return candidates


def _replace_at_path(value, path, replacement):
    if not path:
        return replacement
    key = path[0]
    if isinstance(value, list):
        value[key] = _replace_at_path(value[key], path[1:], replacement)
    elif isinstance(value, dict):
        value[key] = _replace_at_path(value[key], path[1:], replacement)
    return value


def strip_sensitive_image_candidate(body: dict) -> dict | None:
    candidates = _image_candidates(body)
    if not candidates:
        return None
    sanitized = copy.deepcopy(body)
    path, _part = candidates[-1]
    _replace_at_path(
        sanitized,
        path,
        {"type": "input_text", "text": MINIMAX_SENSITIVE_IMAGE_PLACEHOLDER},
    )
    return sanitized
