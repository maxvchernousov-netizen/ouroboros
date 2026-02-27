"""
Ouroboros — LLM client (Claude CLI backend).

Uses Claude Code CLI (`claude -p`) instead of OpenRouter API.
All LLM calls go through the ClaudeCliClient class.
Requires an active Claude Max subscription.

Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "anthropic/claude-sonnet-4.6"

# Model name mapping: OpenRouter-style -> Claude CLI --model flag
_MODEL_MAP = {
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4-5",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-opus-4.6": "claude-opus-4-6",
    "anthropic/claude-opus-4": "claude-opus-4",
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5",
}

# Effort mapping: Ouroboros levels -> Claude CLI --effort values
_EFFORT_MAP = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}

# JSON schema for structured tool-use output
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["text_response", "tool_calls"],
            "description": "Use 'tool_calls' to invoke tools, 'text_response' for final answer"
        },
        "content": {
            "type": "string",
            "description": "Your thinking/status notes (for tool_calls) or final answer (for text_response)"
        },
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Tool function name"
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Tool arguments as key-value pairs"
                    }
                },
                "required": ["name", "arguments"]
            },
            "description": "Tools to call (only when action='tool_calls')"
        }
    },
    "required": ["action", "content"]
}

# Short system instruction for response format (appended via --append-system-prompt)
_FORMAT_INSTRUCTION = (
    "Respond ONLY with valid JSON matching the provided schema. "
    "When you need to use tools, set action='tool_calls' and provide tool_calls array. "
    "When giving a final answer with no more tools needed, set action='text_response'. "
    "You may call multiple tools at once in a single tool_calls response."
)


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _resolve_claude_binary() -> str:
    """Find the claude CLI binary. Checks PATH and well-known locations."""
    # Check PATH first
    found = shutil.which("claude")
    if found:
        return found

    # Well-known locations on macOS
    well_known = [
        os.path.expanduser("~/Library/Application Support/Claude/claude-code"),
        "/usr/local/bin/claude",
    ]

    # Search versioned directories under claude-code/
    claude_code_base = os.path.expanduser("~/Library/Application Support/Claude/claude-code")
    if os.path.isdir(claude_code_base):
        try:
            versions = sorted(os.listdir(claude_code_base), reverse=True)
            for ver in versions:
                candidate = os.path.join(claude_code_base, ver, "claude")
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
        except OSError:
            pass

    for path in well_known:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "Claude CLI binary not found. Ensure Claude Code is installed. "
        "Install via: npm install -g @anthropic-ai/claude-code"
    )


def _map_model(model: str) -> str:
    """Convert OpenRouter-style model name to Claude CLI --model flag value."""
    if model in _MODEL_MAP:
        return _MODEL_MAP[model]
    # If already in CLI format (e.g., "claude-sonnet-4-6"), return as-is
    if model.startswith("claude-"):
        return model
    # Unknown model — try stripping provider prefix
    if "/" in model:
        bare = model.split("/", 1)[1]
        # Convert dots to hyphens (sonnet-4.6 -> sonnet-4-6)
        bare = bare.replace(".", "-")
        if not bare.startswith("claude-"):
            bare = "claude-" + bare
        return bare
    log.warning("Unknown model %s, falling back to claude-sonnet-4-6", model)
    return "claude-sonnet-4-6"


def _flatten_messages(messages: List[Dict[str, Any]]) -> str:
    """Serialize a messages array into a single text prompt for claude -p stdin."""
    parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            # System content can be multipart (list of text blocks with cache_control)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                parts.append("[SYSTEM]\n" + "\n\n".join(text_parts))
            else:
                parts.append("[SYSTEM]\n" + str(content))

        elif role == "user":
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "image_url":
                            text_parts.append("[Image attached]")
                parts.append("[USER]\n" + "\n".join(text_parts))
            else:
                parts.append("[USER]\n" + str(content))

        elif role == "assistant":
            text = str(content or "")
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                calls_desc = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "unknown")
                    fn_args = fn.get("arguments", "{}")
                    # Truncate long arguments for context
                    if len(fn_args) > 500:
                        fn_args = fn_args[:500] + "..."
                    calls_desc.append(f"  - {fn_name}({fn_args})")
                text += "\n[Tool calls]:\n" + "\n".join(calls_desc)
            parts.append("[ASSISTANT]\n" + text)

        elif role == "tool":
            tool_id = msg.get("tool_call_id", "")
            parts.append(f"[TOOL RESULT ({tool_id})]\n{content}")

    return "\n\n".join(parts)


def _embed_tool_schemas(prompt: str, tools: List[Dict[str, Any]]) -> str:
    """Embed tool definitions into the prompt text."""
    if not tools:
        return prompt

    tool_section = "\n\n## Available Tools\n\n"
    tool_section += (
        "You have access to the following tools. To use a tool, respond with "
        "action='tool_calls' and provide the tool_calls array with name and arguments.\n\n"
    )

    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        tool_section += f"### {name}\n{desc}\n"
        if props:
            tool_section += "Parameters:\n"
            for pname, pschema in props.items():
                req_marker = " (required)" if pname in required else ""
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                enum_vals = pschema.get("enum")
                type_info = ptype
                if enum_vals:
                    type_info += f" [{', '.join(str(v) for v in enum_vals)}]"
                tool_section += f"  - {pname}: {type_info}{req_marker} — {pdesc}\n"
        tool_section += "\n"

    return prompt + tool_section


class ClaudeCliClient:
    """Claude Code CLI wrapper. All LLM calls go through this class.

    Uses `claude -p` with --output-format json and --json-schema
    for structured output. Requires Claude Max subscription.
    """

    AVAILABLE_MODELS = [
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-opus-4",
        "anthropic/claude-haiku-4.5",
    ]

    def __init__(self, api_key: Optional[str] = None, base_url: str = ""):
        # api_key and base_url accepted for backward compatibility but ignored
        self._claude_bin: Optional[str] = None

    def _get_binary(self) -> str:
        if self._claude_bin is None:
            self._claude_bin = _resolve_claude_binary()
        return self._claude_bin

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call via claude -p. Returns: (response_message_dict, usage_dict)."""
        claude_bin = self._get_binary()
        cli_model = _map_model(model)
        effort = _EFFORT_MAP.get(normalize_reasoning_effort(reasoning_effort), "medium")

        # Build the prompt
        prompt_text = _flatten_messages(messages)
        if tools:
            prompt_text = _embed_tool_schemas(prompt_text, tools)

        # Build command
        max_turns = "1"
        cmd = [
            claude_bin, "-p",
            "--output-format", "json",
            "--model", cli_model,
            "--no-session-persistence",
        ]

        if tools:
            # Use JSON schema for structured tool-use output
            # --json-schema needs 2 turns (model output + schema validation)
            max_turns = "2"
            cmd.extend(["--json-schema", json.dumps(RESPONSE_SCHEMA)])
            cmd.extend(["--append-system-prompt", _FORMAT_INSTRUCTION])
            # Disable built-in Claude Code tools — we use our own
            cmd.extend(["--tools", ""])

        cmd.extend(["--max-turns", max_turns])

        # Run the CLI, feeding prompt via stdin
        # Remove CLAUDECODE env var to prevent "nested session" error
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        try:
            result = subprocess.run(
                cmd,
                input=prompt_text,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI timed out after 300s")
        except FileNotFoundError:
            raise RuntimeError(f"Claude CLI binary not found at {claude_bin}")

        if result.returncode != 0:
            stderr = (result.stderr or "")[:500]
            raise RuntimeError(f"Claude CLI error (exit {result.returncode}): {stderr}")

        # Parse response
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed output
            stdout = result.stdout.strip()
            # Find the last JSON object in the output
            last_brace = stdout.rfind("}")
            if last_brace >= 0:
                first_brace = stdout.rfind("{", 0, last_brace)
                if first_brace >= 0:
                    try:
                        payload = json.loads(stdout[first_brace:last_brace + 1])
                    except json.JSONDecodeError:
                        raise RuntimeError(f"Failed to parse Claude CLI JSON output: {stdout[:500]}")
                else:
                    raise RuntimeError(f"Failed to parse Claude CLI output: {stdout[:500]}")
            else:
                raise RuntimeError(f"Failed to parse Claude CLI output: {stdout[:500]}")

        # Debug: log payload keys for troubleshooting
        log.info("CLI payload keys: %s, subtype=%s, has_result=%s, has_structured=%s",
                 list(payload.keys()),
                 payload.get("subtype"),
                 bool(payload.get("result")),
                 bool(payload.get("structured_output")))

        # Extract usage from CLI response (real token counts available)
        cli_usage = payload.get("usage", {})
        prompt_tokens = int(cli_usage.get("input_tokens", 0)) + int(cli_usage.get("cache_read_input_tokens", 0))
        completion_tokens = int(cli_usage.get("output_tokens", 0))
        if prompt_tokens == 0:
            prompt_tokens = _estimate_tokens(prompt_text)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_tokens": int(cli_usage.get("cache_read_input_tokens", 0)),
            "cache_write_tokens": int(cli_usage.get("cache_creation_input_tokens", 0)),
            "cost": 0.0,  # Free with Max subscription
        }

        # If we used JSON schema (tools mode), parse structured_output
        if tools:
            structured = payload.get("structured_output")
            if structured and isinstance(structured, dict):
                return self._parse_structured_dict(structured, usage)
            # Fallback: try result field
            result_text = payload.get("result", "")
            if result_text:
                return self._parse_structured_response(result_text, usage)
            return {"content": "", "tool_calls": None}, usage

        # No tools — plain text response from result field
        result_text = payload.get("result", "")
        return {"content": result_text, "tool_calls": None}, usage

    def _parse_structured_dict(
        self, structured: Dict[str, Any], usage: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Parse structured_output dict (from --json-schema) into loop.py format."""
        action = structured.get("action", "text_response")
        content = structured.get("content", "")

        if action == "tool_calls":
            raw_calls = structured.get("tool_calls") or []
            if not raw_calls:
                return {"content": content, "tool_calls": None}, usage

            tool_calls = []
            for tc in raw_calls:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(
                            tc.get("arguments", {}),
                            ensure_ascii=False,
                        ),
                    },
                })
            return {"content": content, "tool_calls": tool_calls}, usage

        return {"content": content, "tool_calls": None}, usage

    def _parse_structured_response(
        self, result_text: str, usage: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Parse JSON string result into the format loop.py expects (fallback)."""
        try:
            structured = json.loads(result_text)
        except json.JSONDecodeError:
            log.warning("Failed to parse structured response, treating as text: %s", result_text[:200])
            return {"content": result_text, "tool_calls": None}, usage

        return self._parse_structured_dict(structured, usage)

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """Send a vision query. Uses Anthropic SDK if ANTHROPIC_API_KEY is set."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            return self._vision_query_anthropic(prompt, images, model, max_tokens, api_key)

        # Fallback: describe images textually and use chat()
        image_desc = f"[{len(images)} image(s) attached but cannot be processed without ANTHROPIC_API_KEY]"
        messages = [{"role": "user", "content": f"{prompt}\n\n{image_desc}"}]
        msg, usage = self.chat(messages=messages, model=model, reasoning_effort=reasoning_effort, max_tokens=max_tokens)
        return msg.get("content") or "", usage

    def _vision_query_anthropic(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        api_key: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Vision query using Anthropic SDK directly."""
        try:
            import anthropic
        except ImportError:
            log.warning("anthropic package not installed, vision degraded")
            return "(Vision unavailable: pip install anthropic)", {}

        client = anthropic.Anthropic(api_key=api_key)
        content = [{"type": "text", "text": prompt}]
        for img in images:
            if "base64" in img:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.get("mime", "image/png"),
                        "data": img["base64"],
                    },
                })
            elif "url" in img:
                content.append({
                    "type": "image",
                    "source": {"type": "url", "url": img["url"]},
                })

        try:
            response = client.messages.create(
                model=_map_model(model),
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            text = response.content[0].text if response.content else ""
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "cost": 0.0,
            }
            return text, usage
        except Exception as e:
            log.warning("Anthropic vision query failed: %s", e)
            return f"(Vision query failed: {e})", {}

    def default_model(self) -> str:
        """Return the default model from env."""
        model = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        if not model.startswith("anthropic/"):
            log.warning("Non-Claude model %s not supported by CLI, using sonnet", model)
            return "anthropic/claude-sonnet-4.6"
        return model

    def available_models(self) -> List[str]:
        """Return list of available Claude models."""
        main = self.default_model()
        models = [main]
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        for m in (code, light):
            if m and m.startswith("anthropic/") and m not in models:
                models.append(m)
        # Add all known models not already in list
        for m in self.AVAILABLE_MODELS:
            if m not in models:
                models.append(m)
        return models


# Backward-compatible alias — all existing imports continue to work
LLMClient = ClaudeCliClient
