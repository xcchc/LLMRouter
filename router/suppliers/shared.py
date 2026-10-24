# -*- coding: utf-8 -*-
"""Shared protocol-conversion utilities used by the main router and supplier adaptors.

These functions bridge OpenAI Responses / Chat Completions concepts (tool names,
content parts, encrypted payloads, JSON Schema sanitisation) and are not tied
to any single upstream provider.
"""
import copy
import hashlib
import json
import re


def args_to_string(args) -> str:
    """arguments 字段转成 chat 的 JSON 字符串；本就是字符串时不重复转义。"""
    if isinstance(args, str):
        return args
    return json.dumps(args, ensure_ascii=False)


def content_to_text(content) -> str:
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


OPENAI_ENCRYPTED_PLACEHOLDER = (
    "[Encrypted agent payload was removed by the router because this upstream "
    "cannot decrypt it.]"
)


def unwrap_encrypted_content_parts(content):
    """Convert encrypted_content parts into input_text for third-party Responses backends."""
    if not isinstance(content, list):
        return content
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "encrypted_content":
            payload = part.get("encrypted_content")
            if isinstance(payload, str) and payload:
                out.append({"type": "input_text", "text": payload})
            else:
                out.append({"type": "input_text", "text": OPENAI_ENCRYPTED_PLACEHOLDER})
        else:
            out.append(part)
    return out


def sanitize_schema(value):
    if isinstance(value, dict):
        return {k: sanitize_schema(v) for k, v in value.items() if k != "encrypted"}
    if isinstance(value, list):
        return [sanitize_schema(v) for v in value]
    return value


def normalize_tool_parameters_schema(value) -> dict:
    """Make function parameters acceptable to strict Chat tool validators."""
    schema = sanitize_schema(value)
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


def chat_tool_name(name: str, namespace: str | None = None) -> str:
    raw = f"{namespace}__{name}" if namespace else str(name or "tool")
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", raw) or "tool"
    if len(safe) > 64:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:55]}_{digest}"
    return safe


def build_tool_bridge(body: dict) -> dict:
    bridge = {"chat_tools": [], "by_chat": {}, "by_response": {}, "unsupported": []}
    used = set()
    seen = set()

    def unique_name(name: str, namespace: str | None) -> str:
        candidate = chat_tool_name(name, namespace)
        if candidate not in used:
            used.add(candidate)
            return candidate
        raw = f"{namespace or ''}:{name}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate[:55]}_{digest}"
        counter = 2
        while candidate in used:
            candidate = f"{candidate[:55]}_{digest}_{counter}"
            counter += 1
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
        tool_key = (kind, namespace or "", original_name)
        if tool_key in seen:
            return
        seen.add(tool_key)
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
            parameters = normalize_tool_parameters_schema(tool.get("parameters") or {
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
            parameters = normalize_tool_parameters_schema(
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


def bridge_chat_name(bridge: dict, kind: str, name: str, namespace=None) -> str:
    return (bridge.get("by_response") or {}).get(
        (kind, namespace or "", name),
        chat_tool_name(name, namespace),
    )


def custom_tool_arguments(tool_input) -> str:
    if not isinstance(tool_input, str):
        tool_input = json.dumps(tool_input, ensure_ascii=False)
    return json.dumps({"input": tool_input}, ensure_ascii=False)
