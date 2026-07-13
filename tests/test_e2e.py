"""End-to-end proxy test on a normal machine — no CC hardware required.

Spins up the mock worker (echo backend) and the real proxy in background
threads, then drives both the OpenAI and Anthropic surfaces (streaming and
non-streaming). Exercises the real ECDH handshake + AES-GCM seal/open; only
the GPU attestation is skipped (impossible without CC hardware).

The unified protocol is plain-text messages in / plain-text reply out via
``/message`` and ``/message_stream``. There is no tool-calling channel at
this layer (tool calls would require a richer wire format inside the enclave),
so the historic tool-call tests are intentionally absent.
"""

import json
import socket
import threading
import time

import pytest
import requests
from werkzeug.serving import make_server

from cc_proxy.server import create_app as create_proxy
from cc_proxy.session import ConfidentialSession
from tests.mock_worker import create_app as create_worker


class _Server:
    def __init__(self, app, port):
        self.srv = make_server("127.0.0.1", port, app, threaded=True)
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait(url):
    for _ in range(50):
        try:
            if requests.get(url, timeout=1).ok:
                return
        except requests.RequestException:
            time.sleep(0.1)
    raise RuntimeError(f"server never came up: {url}")


@pytest.fixture()
def proxy_url():
    wport, pport = _free_port(), _free_port()
    with _Server(create_worker(), wport):
        _wait(f"http://127.0.0.1:{wport}/health")
        # Point at the /enc/{model} prefix the real enc endpoint uses; the mock
        # worker accepts both prefixed and bare paths, so this exercises the
        # exact URL shape the real code generates.
        session = ConfidentialSession(
            f"http://127.0.0.1:{wport}",
            "test/model",
            insecure_skip_attest=True,
        )
        session.connect()
        with _Server(create_proxy(session, "test/model"), pport):
            _wait(f"http://127.0.0.1:{pport}/health")
            yield f"http://127.0.0.1:{pport}"


def test_openai_non_stream(proxy_url):
    r = requests.post(f"{proxy_url}/v1/chat/completions", json={
        "model": "test/model", "messages": [{"role": "user", "content": "ping"}]}).json()
    assert r["choices"][0]["message"]["content"] == "[mock] you said: ping"


def test_openai_stream(proxy_url):
    r = requests.post(f"{proxy_url}/v1/chat/completions", json={
        "model": "test/model", "stream": True,
        "messages": [{"role": "user", "content": "hello world"}]}, stream=True)
    out = ""
    sse_done = "[" + "DONE" + "]"
    for line in r.iter_lines():
        if not line:
            continue
        body = line.decode()[6:]
        if body == sse_done:
            break
        out += json.loads(body)["choices"][0]["delta"].get("content", "")
    assert out.strip() == "[mock] you said: hello world"


def test_anthropic_non_stream(proxy_url):
    r = requests.post(f"{proxy_url}/v1/messages", json={
        "model": "test/model", "max_tokens": 64, "system": "be brief",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "yo"}]}]}).json()
    assert r["content"][0]["text"] == "[mock] you said: yo"


def test_stream_upstream_error_is_clean(proxy_url):
    # A streaming upstream error should surface as a clean HTTP status, not a
    # mid-stream connection drop / 500. The mock worker triggers this when
    # "force-error" appears in the user message.
    r = requests.post(f"{proxy_url}/v1/chat/completions", json={
        "model": "test/model", "stream": True,
        "messages": [{"role": "user", "content": "force-error please"}]})
    assert r.status_code == 400
    assert "error" in r.json()


def test_anthropic_stream(proxy_url):
    r = requests.post(f"{proxy_url}/v1/messages", json={
        "model": "test/model", "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": "count up"}]}, stream=True)
    out, saw_stop = "", False
    for line in r.iter_lines():
        if not line or not line.decode().startswith("data: "):
            continue
        obj = json.loads(line.decode()[6:])
        if obj.get("type") == "content_block_delta":
            out += obj["delta"]["text"]
        if obj.get("type") == "message_stop":
            saw_stop = True
    assert out.strip() == "[mock] you said: count up"
    assert saw_stop
