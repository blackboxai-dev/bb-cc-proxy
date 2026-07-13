"""Local OpenAI- and Anthropic-compatible HTTP surface.

Runs on the user's trusted machine (localhost). Each incoming chat request has
its ``messages`` array extracted and sent, sealed, to the confidential worker
via ``/message`` (or ``/message_stream``). The worker returns encrypted plain
text; this layer wraps it back into the OpenAI (``chat.completion``) or
Anthropic (``messages``) shape the caller expected. Anthropic requests are
translated to a plain messages array on this trusted side, sent the same way,
and translated back.
"""

import json
import uuid

import requests
from flask import Flask, Response, jsonify, request, stream_with_context

from .session import ConfidentialSession, UpstreamStreamError


# ------------------ Anthropic <-> OpenAI translation ------------------


def _text_from_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text" and "text" in block:
            parts.append(block["text"])
    return "\n".join(p for p in parts if p)


def anthropic_to_openai(a: dict) -> dict:
    """Translate an Anthropic /v1/messages body into an OpenAI chat body."""
    msgs = []
    if a.get("system"):
        msgs.append({"role": "system", "content": _text_from_content(a["system"])})

    for m in a.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue

        text_parts, tool_calls, tool_results = [], [], []
        for b in content or []:
            t = b.get("type")
            if t == "text":
                text_parts.append(b.get("text", ""))
            elif t == "tool_use":  # assistant asked to call a tool
                tool_calls.append({
                    "id": b.get("id"),
                    "type": "function",
                    "function": {"name": b.get("name"),
                                 "arguments": json.dumps(b.get("input", {}))},
                })
            elif t == "tool_result":  # user returns a tool's output
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id"),
                    "content": _text_from_content(b.get("content")),
                })

        if role == "assistant" and tool_calls:
            msgs.append({"role": "assistant",
                         "content": "\n".join(text_parts) or None,
                         "tool_calls": tool_calls})
        elif tool_results:
            msgs.extend(tool_results)
            if text_parts:
                msgs.append({"role": role, "content": "\n".join(text_parts)})
        else:
            msgs.append({"role": role, "content": "\n".join(text_parts)})

    out = {"model": "anthropic-proxied", "messages": msgs}
    if a.get("max_tokens"):
        out["max_tokens"] = a["max_tokens"]
    if a.get("temperature") is not None:
        out["temperature"] = a["temperature"]
    if a.get("tools"):
        out["tools"] = [{"type": "function",
                         "function": {"name": t["name"],
                                      "description": t.get("description", ""),
                                      "parameters": t.get("input_schema", {})}}
                        for t in a["tools"]]
    tc = a.get("tool_choice")
    if tc:
        kind = tc.get("type")
        if kind == "auto":
            out["tool_choice"] = "auto"
        elif kind == "any":
            out["tool_choice"] = "required"
        elif kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": tc.get("name")}}
    return out


_STOP_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def openai_to_anthropic_message(resp: dict, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    blocks = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            inp = {}
        blocks.append({"type": "tool_use", "id": tc.get("id"),
                       "name": fn.get("name"), "input": inp})
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id", f"msg_{uuid.uuid4().hex}"),
        "type": "message", "role": "assistant", "model": model,
        "content": blocks or [{"type": "text", "text": ""}],
        "stop_reason": _STOP_MAP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


def anthropic_sse_from_message(msg: dict):
    """Pseudo-stream a complete Anthropic message as SSE events."""
    def ev(kind, data):
        return f"event: {kind}\ndata: {json.dumps(data)}\n\n"

    yield ev("message_start", {"type": "message_start",
             "message": {**msg, "content": [], "stop_reason": None}})
    for i, block in enumerate(msg["content"]):
        if block["type"] == "text":
            yield ev("content_block_start", {"type": "content_block_start", "index": i,
                     "content_block": {"type": "text", "text": ""}})
            yield ev("content_block_delta", {"type": "content_block_delta", "index": i,
                     "delta": {"type": "text_delta", "text": block["text"]}})
        else:  # tool_use
            yield ev("content_block_start", {"type": "content_block_start", "index": i,
                     "content_block": {"type": "tool_use", "id": block["id"],
                                       "name": block["name"], "input": {}}})
            yield ev("content_block_delta", {"type": "content_block_delta", "index": i,
                     "delta": {"type": "input_json_delta",
                               "partial_json": json.dumps(block["input"])}})
        yield ev("content_block_stop", {"type": "content_block_stop", "index": i})
    yield ev("message_delta", {"type": "message_delta",
             "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
             "usage": {"output_tokens": msg["usage"]["output_tokens"]}})
    yield ev("message_stop", {"type": "message_stop"})


# ------------------ Responses API input normalization ------------------


def _normalize_responses_input(body: dict) -> dict:
    """Make a Responses API ``input`` array digestible by vLLM.

    Clients like Codex echo prior turns back as structured message items, e.g.
    ``{"type":"message","role":"assistant","content":[{"type":"output_text",
    "text":"..."}]}``. vLLM's Responses schema rejects these: as a
    ``ResponseOutputMessageParam`` they must carry ``id``/``status`` and each
    ``output_text`` part needs ``annotations`` — none of which Codex sends. This
    yields a 400 that (mid-stream) surfaces to the client as a 502 retry storm.

    Fix: flatten any message item whose ``content`` is a list of text parts into
    the forgiving ``{"role", "content": "<text>"}`` shape (EasyInputMessageParam),
    which vLLM accepts. Non-message items (function_call, function_call_output,
    reasoning, …) are passed through untouched.
    """
    if not isinstance(body, dict):
        return body
    inp = body.get("input")
    if not isinstance(inp, list):
        return body

    roles = {"user", "assistant", "system", "developer"}
    changed = False
    out = []
    for item in inp:
        # Drop reasoning items. Clients like Codex echo back the model's
        # reasoning turns (``{"type":"reasoning","content":[{"type":
        # "reasoning_text",...}],"encrypted_content":...}``). Non-reasoning
        # models served by vLLM reject this shape (400), and the encrypted
        # content is unusable to them anyway.
        if isinstance(item, dict) and item.get("type") == "reasoning":
            changed = True
            continue
        if (isinstance(item, dict) and item.get("role") in roles
                and isinstance(item.get("content"), list)):
            parts = []
            for p in item["content"]:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    # input_text / output_text / text all carry a "text" field
                    t = p.get("text")
                    if t:
                        parts.append(t)
            out.append({"role": item["role"], "content": "".join(parts)})
            changed = True
        else:
            out.append(item)

    if not changed:
        return body
    body = dict(body)
    body["input"] = out
    return body


# ------------------ app factory ------------------


def _extract_client_bearer() -> str:
    """Pull a bearer token from the incoming request (Authorization header,
    then x-api-key). Returns "" when none is present."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def create_app(
    session: ConfidentialSession,
    model_name: str,
    *,
    local_api_key: str = None,
    passthrough_api_key: bool = False,
) -> Flask:
    """Build the Flask app.

    ``local_api_key`` — if set, incoming requests must present this exact
        bearer token. Used to gate local access when you *don't* want
        pass-through. Ignored when ``passthrough_api_key`` is True.
    ``passthrough_api_key`` — when True, the bearer token on each incoming
        request is forwarded upstream verbatim (BYO-key). ``session.api_key``
        remains as a fallback default if the client sent no token.
    """
    app = Flask(__name__)

    def _authorized() -> bool:
        # Pass-through mode delegates auth to the upstream enclave — any key
        # (or the default) will be sent, and upstream will 401 bad ones.
        if passthrough_api_key:
            return True
        if not local_api_key:
            return True
        return _extract_client_bearer() == local_api_key

    @app.before_request
    def _guard():
        if request.method != "POST":
            return None
        if not _authorized():
            return jsonify({"error": {"message": "invalid local api key", "type": "auth"}}), 401
        if passthrough_api_key:
            token = _extract_client_bearer()
            if not session.has_upstream_credentials(token):
                return jsonify({"error": {
                    "message": "no upstream credentials: send Authorization: Bearer <sk-...> "
                               "or configure a default via --upstream-api-key/$BLACKBOX_API_KEY",
                    "type": "auth",
                }}), 401

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "connected": session.connected,
                        "enc_endpoint": session.enc_endpoint, "model": model_name})

    @app.get("/v1/models")
    def models():
        # The supervisor has no model-discovery route — advertise the single
        # configured model.
        return jsonify({"object": "list",
                        "data": [{"id": model_name, "object": "model", "owned_by": "confidential"}]})

    def _client_override() -> str:
        """Per-request upstream bearer, active only in pass-through mode."""
        if not passthrough_api_key:
            return None
        token = _extract_client_bearer()
        return token or None

    @app.post("/v1/chat/completions")
    def chat_completions():
        body = request.get_json(force=True)
        if body.get("stream"):
            gen = session.stream_openai(body, api_key_override=_client_override())
            # Peek the first frame so an upstream error (e.g. tool-choice 400) or a
            # dropped connection becomes a clean HTTP status, not a mid-stream crash.
            try:
                first = next(gen)
            except StopIteration:
                first = None
            except UpstreamStreamError as e:
                rbody = e.body if isinstance(e.body, dict) else {"error": {"message": str(e.body)}}
                return jsonify(rbody), e.status
            except requests.RequestException as e:
                return jsonify({"error": {"message": f"upstream connection failed: {e}"}}), 502

            def stream():
                if first is not None:
                    yield f"data: {first}\n\n"
                try:
                    for chunk in gen:
                        yield f"data: {chunk}\n\n"
                except (UpstreamStreamError, requests.RequestException) as e:
                    err = getattr(e, "body", None) or {"error": {"message": str(e)}}
                    yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(stream()), mimetype="text/event-stream")
        status, rbody = session.send_openai(body, api_key_override=_client_override())
        return jsonify(rbody), status

    @app.post("/v1/responses")
    def responses():
        body = _normalize_responses_input(request.get_json(force=True))
        if body.get("stream"):
            gen = session.stream_openai(body, path="/v1/responses",
                                        api_key_override=_client_override())
            try:
                first = next(gen)
            except StopIteration:
                first = None
            except UpstreamStreamError as e:
                rbody = e.body if isinstance(e.body, dict) else {"error": {"message": str(e.body)}}
                return jsonify(rbody), e.status
            except requests.RequestException as e:
                return jsonify({"error": {"message": f"upstream connection failed: {e}"}}), 502

            def stream():
                if first is not None:
                    yield f"data: {first}\n\n"
                try:
                    for chunk in gen:
                        yield f"data: {chunk}\n\n"
                except (UpstreamStreamError, requests.RequestException) as e:
                    err = getattr(e, "body", None) or {"error": {"message": str(e)}}
                    yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(stream()), mimetype="text/event-stream")
        status, rbody = session.send_openai(body, path="/v1/responses",
                                            api_key_override=_client_override())
        return jsonify(rbody), status

    @app.post("/v1/messages")
    def anthropic_messages():
        body = request.get_json(force=True)
        oai = anthropic_to_openai(body)
        oai["stream"] = False  # translate the full reply, then (pseudo-)stream if asked
        status, rbody = session.send_openai(oai, api_key_override=_client_override())
        if status != 200:
            return jsonify({"type": "error",
                            "error": {"type": "api_error", "message": str(rbody)}}), status
        msg = openai_to_anthropic_message(rbody, model_name)
        if not body.get("stream"):
            return jsonify(msg)
        return Response(stream_with_context(anthropic_sse_from_message(msg)),
                        mimetype="text/event-stream")

    return app
