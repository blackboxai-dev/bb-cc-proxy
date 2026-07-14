"""Confidential session: attestation handshake + encrypted transport.

Establishes one attested ECDH session with the remote secure worker
(``enc_endpoint``) and exposes plaintext ``send``/``stream`` helpers. All wire
payloads are AES-GCM sealed; plaintext exists only here (on the user's trusted
machine) and inside the confidential VM.

Wire protocol (single, uniform):

* ``GET  {enc_endpoint}/enc/{provider}/{model}/attestation`` — returns the
  worker's public key, a nonce, and (optionally) a ``session_id`` that must be
  echoed in every subsequent request body.
* ``POST {enc_endpoint}/enc/{provider}/{model}/message`` — non-streaming
  chat: encrypted messages in, encrypted plain-text reply out.
* ``POST {enc_endpoint}/enc/{provider}/{model}/message_stream`` — streaming
  chat: encrypted messages in, newline-delimited encrypted chunks out, plus a
  plaintext ``{"eos": true}`` trailer.

Each POST includes an ``Authorization: Bearer <api_key>`` header. The key may
be a proxy-wide default (``api_key``) or a per-request override forwarded from
the local client (``api_key_override``, see BYO-key pass-through).
"""

import base64
import json
import logging
import threading
import time
import uuid

import requests

from . import crypto

logger = logging.getLogger("cc_proxy.session")


class AttestationError(RuntimeError):
    """Raised when the supervisor fails attestation verification."""


class UpstreamStreamError(RuntimeError):
    """Raised when the worker reports an upstream (vLLM) error on a stream."""

    def __init__(self, status, body):
        super().__init__(f"upstream error {status}")
        self.status = status
        self.body = body


class ConfidentialSession:
    def __init__(self, enc_endpoint: str, model: str, *,
                 insecure_skip_attest: bool = False,
                 timeout: float = 120.0, verify_tls: bool = True,
                 api_key: str = None):
        if not model:
            raise ValueError("model is required, e.g. 'google/gemma-4-31b-it'")
        self.enc_endpoint = enc_endpoint.rstrip("/")
        self.model = model
        self.insecure_skip_attest = insecure_skip_attest
        self.timeout = timeout
        self.verify_tls = verify_tls
        self.api_key = api_key

        self._lock = threading.Lock()
        self._nonce = 1000
        self._connected = False
        self._shared_key = None
        self._supervisor_pub = None
        self._local_priv = None
        self._local_pub = None
        self._session_id = None  # supplied by /attestation on supervisors that use it
        # Wire protocol mode. Newer workers accept a full ``{path, method, body}``
        # envelope (preserves tool_calls / usage / finish_reason). Some older
        # workers only accept a bare messages array. We start optimistic and
        # auto-downgrade on the first HTTP 500 from an envelope attempt.
        self._wire = "envelope"  # "envelope" | "legacy"

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Fetch + verify attestation, then derive the shared key via ECDH."""
        logger.info("requesting attestation report from %s", self._attest_url())
        att = requests.get(
            self._attest_url(),
            timeout=self.timeout,
            verify=self.verify_tls,
        ).json()

        supervisor_pub = crypto.deserialize_public_key(att["public_key"])
        nonce = base64.b64decode(att["nonce_b64"])
        session_id = att.get("session_id")  # optional per-supervisor

        if self.insecure_skip_attest:
            logger.warning(
                "INSECURE: --insecure-skip-attestation set; NOT verifying GPU "
                "attestation. Do not use against production/real workloads."
            )
        else:
            try:
                crypto.verify_attestation_full(
                    report_json=att["report_json"].encode(),
                    signature_b64=att["signature"],
                    gpu_eat_json=att["gpu_eat"],
                    public_key=supervisor_pub,
                    expected_nonce=nonce,
                )
            except (ValueError, KeyError) as e:
                raise AttestationError(f"supervisor attestation failed: {e}") from e
            logger.info("attestation verified: worker GPU is in Confidential Computing mode")

        local_priv, local_pub = crypto.generate_key_pair()
        with self._lock:
            self._supervisor_pub = supervisor_pub
            self._local_priv = local_priv
            self._local_pub = local_pub
            self._shared_key = crypto.derive_shared_key(local_priv, supervisor_pub)
            self._session_id = session_id
            self._nonce = 1000  # per-session monotonic counter, reset on (re)connect
            self._connected = True
        logger.info("ECDH shared key established%s",
                    f" (session_id={session_id})" if session_id else "")

    # -- URL helpers -------------------------------------------------------

    def _base_url(self) -> str:
        """``{enc_endpoint}/enc/{provider}/{model}`` — the prefix shared by
        ``/attestation``, ``/message`` and ``/message_stream``."""
        return f"{self.enc_endpoint}/enc/{self.model.strip('/')}"

    def _attest_url(self) -> str:
        return f"{self._base_url()}/attestation"

    @property
    def connected(self) -> bool:
        return self._connected

    def _next_nonce(self) -> int:
        with self._lock:
            n = self._nonce
            self._nonce += 1
            return n

    def _request_body(self, plaintext: str) -> dict:
        nonce = self._next_nonce()
        payload = crypto.encrypt_and_sign(
            plaintext, self._shared_key, self._local_priv, nonce
        )
        body = {
            "peer_public_key": crypto.serialize_public_key(self._local_pub),
            "payload": payload,
        }
        if self._session_id:
            body["session_id"] = self._session_id
        return body

    def _upstream_headers(self, override: str = None) -> dict:
        """Build headers for an upstream request.

        ``override`` — a per-request bearer token forwarded from the local
        client. When present it takes precedence over ``self.api_key`` (the
        default configured at proxy startup). This enables BYO-key pass-through
        so each caller can supply their own upstream credentials.
        """
        h = {"Content-Type": "application/json"}
        key = override or self.api_key
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def has_upstream_credentials(self, override: str = None) -> bool:
        """True iff we have *some* bearer token to send upstream (either a
        per-request override or a proxy-wide default)."""
        return bool(override or self.api_key)

    # -- transport ---------------------------------------------------------

    def _post_message(self, payload, *, stream: bool, api_key_override: str = None):
        """POST to /message or /message_stream.

        ``payload`` is sealed as-is (JSON-encoded). It may be a bare messages
        list (legacy) or a full ``{path, method, body}`` envelope (passthrough).

        Handles 409 (session expired) by reconnecting once and retrying.
        Returns the raw ``requests.Response``; caller is responsible for
        decrypting the body / streaming lines.
        """
        leaf = "/message_stream" if stream else "/message"
        url = f"{self._base_url()}{leaf}"

        for attempt in range(2):
            body = self._request_body(json.dumps(payload))
            resp = requests.post(
                url,
                json=body,
                headers=self._upstream_headers(api_key_override),
                timeout=self.timeout,
                stream=stream,
                verify=self.verify_tls,
            )
            if resp.status_code == 409 and attempt == 0:
                logger.warning("session expired (409); re-attesting and retrying")
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass
                self.connect()
                continue
            return resp
        return resp  # pragma: no cover (loop always returns)

    def send(self, messages: list, *, api_key_override: str = None) -> str:
        """Non-streaming: send a messages array, return the assistant text
        (raw plaintext, exactly as the worker sealed it — no OpenAI wrapping).
        """
        if not self._connected:
            raise RuntimeError("session not connected")
        resp = self._post_message(messages, stream=False, api_key_override=api_key_override)
        resp.raise_for_status()
        return crypto.decrypt_and_verify(resp.json(), self._shared_key, self._supervisor_pub)

    def stream(self, messages: list, *, api_key_override: str = None):
        """Streaming: yield decrypted text pieces as the worker produces them.

        The worker emits one JSON object per line — encrypted chunks, then a
        plaintext ``{"eos": true, ...}`` trailer that ends the stream.
        """
        if not self._connected:
            raise RuntimeError("session not connected")
        resp = self._post_message(messages, stream=True, api_key_override=api_key_override)
        resp.raise_for_status()
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if isinstance(chunk, dict) and chunk.get("eos"):
                    break
                if isinstance(chunk, dict) and "error" in chunk:
                    err = chunk["error"]
                    raise UpstreamStreamError(err.get("status", 502), err.get("body"))
                yield crypto.decrypt_and_verify(chunk, self._shared_key, self._supervisor_pub)
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass

    # -- OpenAI-shaped adapters (used by the local HTTP server) ------------

    @staticmethod
    def _extract_text(reply: str) -> str:
        """The reply plaintext may be either a raw string or a JSON-encoded
        string. Normalise to plain text for wrapping."""
        s = reply.strip()
        if s.startswith('"') and s.endswith('"'):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
        return reply

    def _wrap_legacy_reply(self, reply, body: dict) -> dict:
        """Wrap a legacy worker reply into an OpenAI ``chat.completion`` dict.

        ``reply`` may be:
          * bare assistant text (older workers, envelope-fallback text) — wrapped as ``content``
          * a JSON string encoding ``{"content": "...", "tool_calls": [...]}``
            (nemotron-49b when tools are engaged) — surfaced with proper
            ``tool_calls[]`` and ``finish_reason="tool_calls"``
        """
        model_id = body.get("model") or self.model or "encrypted-model"
        text = self._extract_text(reply) if not isinstance(reply, str) else reply

        # Try to parse a structured {content, tool_calls} reply.
        parsed = None
        if isinstance(text, str) and text.lstrip().startswith("{"):
            try:
                candidate = json.loads(text)
                if isinstance(candidate, dict) and ("tool_calls" in candidate or "content" in candidate):
                    parsed = candidate
            except (json.JSONDecodeError, ValueError):
                pass

        if parsed and isinstance(parsed.get("tool_calls"), list) and parsed["tool_calls"]:
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": parsed.get("content") or None,
                        "tool_calls": parsed["tool_calls"],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        # Fall through: bare content (either extracted from parsed dict or raw text)
        content = parsed.get("content") if parsed else text
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _send_openai_legacy(self, body: dict, *, api_key_override: str = None) -> tuple:
        """Legacy transport: send a bare messages array (no envelope, no path
        routing, no tool_calls / usage / finish_reason preservation). Used
        automatically when the worker rejects the ``{path, method, body}``
        envelope with HTTP 500.
        """
        # Send the full OpenAI body as a bare dict (no {path,method,body} envelope).
        # The nemotron-49b enclave accepts this shape and returns proper tool_calls;
        # older workers that predate tools still accept it and return bare text.
        payload = {k: v for k, v in body.items() if k != "stream"}
        resp = self._post_message(payload, stream=False, api_key_override=api_key_override)
        if resp.status_code >= 400:
            try:
                err_body = resp.json()
            except ValueError:
                err_body = {"error": {"message": resp.text[:500]}}
            return resp.status_code, err_body
        try:
            reply = crypto.decrypt_and_verify(resp.json(), self._shared_key, self._supervisor_pub)
        except (ValueError, KeyError) as e:
            return 502, {"error": {"message": f"failed to decrypt reply: {e}"}}
        return 200, self._wrap_legacy_reply(reply, body)

    def send_openai(self, body: dict, *, path: str = "/v1/chat/completions",
                    api_key_override: str = None) -> tuple:
        """Send a full OpenAI request body via the supervisor.

        ``path`` selects the upstream route (e.g. ``/v1/responses``).
        Returns ``(status_code, response_dict)``.

        Some older workers only accept a bare messages array (no
        ``{path, method, body}`` envelope). If the envelope attempt returns
        HTTP 500 we auto-downgrade the session to legacy mode and retry once;
        subsequent requests skip the envelope attempt entirely.
        """
        if not self._connected:
            raise RuntimeError("session not connected")

        # Legacy mode: skip the envelope, it's known to 500 for this worker.
        # Only chat/completions has a legacy fallback — /v1/responses etc. only
        # ever worked on new workers, so we can't downgrade those.
        if self._wire == "legacy" and path == "/v1/chat/completions":
            return self._send_openai_legacy(body, api_key_override=api_key_override)

        # Send the FULL OpenAI body (messages + tools + tool_choice + sampling …)
        # in a {path, body} envelope so the worker forwards it verbatim to vLLM
        # and tool_calls / usage / finish_reason survive end-to-end.
        envelope = {"path": path, "method": "POST", "body": body}
        resp = self._post_message(envelope, stream=False, api_key_override=api_key_override)
        if resp.status_code == 500 and path == "/v1/chat/completions":
            # Likely a legacy worker rejecting the envelope. Downgrade the
            # session and retry once with the bare messages array.
            logger.warning(
                "worker returned 500 to envelope; downgrading session to legacy "
                "wire protocol (usage token counts will be zero) — %s",
                self.model,
            )
            self._wire = "legacy"
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
            return self._send_openai_legacy(body, api_key_override=api_key_override)
        if resp.status_code >= 400:
            try:
                err_body = resp.json()
            except ValueError:
                err_body = {"error": {"message": resp.text[:500]}}
            return resp.status_code, err_body

        try:
            reply = crypto.decrypt_and_verify(resp.json(), self._shared_key, self._supervisor_pub)
        except (ValueError, KeyError) as e:
            return 502, {"error": {"message": f"failed to decrypt reply: {e}"}}

        # New worker: reply is {"status", "body": <full OpenAI completion>} — or
        # a bare completion dict. Return it as-is so tool_calls survive.
        try:
            obj = json.loads(reply)
        except (json.JSONDecodeError, TypeError):
            obj = None
        if isinstance(obj, dict):
            if isinstance(obj.get("body"), dict):
                return obj.get("status", 200), obj["body"]
            if "choices" in obj:
                return 200, obj

        # Legacy worker: reply is bare assistant text — wrap it OpenAI-style.
        return 200, self._wrap_legacy_reply(self._extract_text(reply), body)

    def stream_openai(self, body: dict, *, path: str = "/v1/chat/completions",
                      api_key_override: str = None):
        """Stream a full OpenAI request body via the supervisor.

        ``path`` selects the upstream route (e.g. ``/v1/responses``).
        Yields raw JSON strings (without the ``data: `` SSE prefix — the
        server layer adds that).

        On HTTP 500 from the envelope (legacy worker), auto-downgrades the
        session to bare-messages-array mode and retries once. The per-frame
        loop below already handles both structured (new) and bare-text (legacy)
        chunk shapes, so no other stream-side changes are needed.
        """
        if not self._connected:
            raise RuntimeError("session not connected")
        model_id = body.get("model") or self.model or "encrypted-model"
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        def _frame(delta: dict, finish=None) -> str:
            return json.dumps({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })

        def _post_envelope():
            envelope = {"path": path, "method": "POST", "body": body}
            return self._post_message(envelope, stream=True, api_key_override=api_key_override)

        def _post_legacy():
            # Same shape as _send_openai_legacy: full body dict (minus stream flag,
            # which is carried in the URL leaf). Enclaves that support tools in the
            # legacy dict will now stream tool_calls chunks; older workers still
            # receive a valid messages+... payload and return bare text as before.
            payload = {k: v for k, v in body.items() if k != "stream"}
            return self._post_message(payload, stream=True,
                                      api_key_override=api_key_override)

        # Fire the upstream request BEFORE yielding the opening role frame,
        # so that HTTP-level failures (4xx) and the very first streamed frame
        # (which may itself be a plaintext ``{"error": ...}`` from the worker)
        # can surface as a clean HTTP status instead of a mid-stream error.
        if self._wire == "legacy" and path == "/v1/chat/completions":
            resp = _post_legacy()
        else:
            resp = _post_envelope()
            if resp.status_code == 500 and path == "/v1/chat/completions":
                logger.warning(
                    "worker returned 500 to envelope on stream; downgrading session "
                    "to legacy wire protocol (usage token counts will be zero) — %s", self.model,
                )
                self._wire = "legacy"
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass
                resp = _post_legacy()
        if resp.status_code >= 400:
            try:
                err_body = resp.json()
            except ValueError:
                err_body = {"error": {"message": resp.text[:500]}}
            raise UpstreamStreamError(resp.status_code, err_body)

        line_iter = resp.iter_lines()

        def _next_nonempty():
            for raw in line_iter:
                if raw:
                    return raw
            return None

        first = _next_nonempty()
        if first is not None:
            try:
                first_frame = json.loads(first)
            except json.JSONDecodeError:
                first_frame = None
            if isinstance(first_frame, dict) and "error" in first_frame:
                err = first_frame["error"]
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass
                raise UpstreamStreamError(err.get("status", 502), err.get("body"))

        # Emit an opening role frame only for the chat completions path.
        # Other paths (e.g. /v1/responses) have their own event shapes —
        # injecting a chat.completion.chunk object would cause SDK parse errors
        # which trigger client retries and cascade into 502s.
        if path == "/v1/chat/completions":
            yield _frame({"role": "assistant"})

        def _decrypt(frame):
            """Return (piece_str_or_None, done). Raises on worker error frames."""
            if isinstance(frame, dict) and frame.get("eos"):
                return None, True
            if isinstance(frame, dict) and "error" in frame:
                err = frame["error"]
                raise UpstreamStreamError(err.get("status", 502), err.get("body"))
            piece = crypto.decrypt_and_verify(frame, self._shared_key, self._supervisor_pub)
            return piece, False

        def _frames():
            if first is not None:
                yield first_frame if first_frame is not None else json.loads(first)
            for raw in line_iter:
                if raw:
                    yield json.loads(raw)

        saw_raw_chunk = False
        try:
            for frame in _frames():
                piece, done = _decrypt(frame)
                if done:
                    break
                s = piece.strip()
                obj = None
                if s.startswith("{"):
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError:
                        obj = None
                if isinstance(obj, dict):
                    # New worker: a full structured frame — either a
                    # chat.completion.chunk (delta.tool_calls / finish_reason
                    # intact) or a Responses-API streaming event (has a "type").
                    # Pass through verbatim so nothing is lost.
                    saw_raw_chunk = True
                    yield s
                else:
                    # Legacy worker: bare assistant text — wrap as a content delta.
                    text = self._extract_text(piece)
                    if text:
                        yield _frame({"content": text})
            # The passthrough chunks already carried vLLM's finish_reason; the
            # legacy text path did not, so synthesize a terminal frame there.
            if not saw_raw_chunk:
                yield _frame({}, finish="stop")
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
