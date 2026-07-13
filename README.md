# bb-cc-proxy

A local proxy that lets any OpenAI- or Anthropic-compatible CLI talk to a
**confidential (end-to-end-encrypted) model endpoint**.

A confidential model worker doesn't serve a plain OpenAI API — it performs
NVIDIA GPU attestation and AES-GCM-encrypts every request/response. `bb-cc-proxy`
runs on **your** machine: it verifies the GPU attestation, performs an ECDH key
exchange, and exposes a normal **OpenAI + Anthropic** endpoint on `localhost`.
Your CLI talks plaintext to localhost; the network hop stays encrypted into the
enclave, so plaintext only ever exists on your machine and inside the CVM.

```
CLI ──localhost──▶ bb-cc-proxy ──attested + AES-GCM──▶ worker :5056 (CVM) ──▶ vLLM
```

Requests are tunneled **verbatim**, so tool calls, JSON mode, sampling params,
logprobs, and streaming all work with no per-feature handling.

> ⚠️ This process is your attestation verifier — it decides whether to trust the
> remote enclave. Only run a build you trust: install from a pinned commit or a
> signed tag, and read [`cc_proxy/crypto.py`](cc_proxy/crypto.py) (the ~130 lines
> that verify the GPU attestation).

## Install

```bash
# from a local checkout
uv tool install .          # or: pipx install .

# from a pinned commit of this repo (recommended: a SHA, not a branch)
uv tool install "git+https://github.com/<org>/bb-cc-proxy.git@<commit-sha>"
```

## Run

```bash
bb-cc-proxy --enc-endpoint http://<worker-ip>:5056 --model <model-name>
```

- Verifies GPU attestation + runs the ECDH handshake **before** serving, and
  refuses to start if attestation fails.
- Serves `http://127.0.0.1:8080`: OpenAI (`/v1/chat/completions`, `/v1/models`)
  and Anthropic (`/v1/messages`).
- Keep it running (a dedicated terminal or a background service).

Useful flags:

| Flag | Purpose |
|------|---------|
| `--port` | local bind port (default `8080`) |
| `--local-api-key` | require a bearer token from your CLI to the proxy |
| `--no-verify-tls` | skip TLS verification to the worker (self-signed cert) |
| `--insecure-skip-attestation` | **testing only** — skip GPU attestation (non-CC worker) |

## Point a CLI at it

| CLI | Setting |
|-----|---------|
| OpenAI-compatible (opencode, Codex, Hermes, OpenClaw, …) | `base_url = http://127.0.0.1:8080/v1` |
| Claude Code | `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` (no `/v1`) |
| any | `api_key` = anything, or the value passed to `--local-api-key` |

Example — OpenClaw (`~/.openclaw/openclaw.json`):

```json5
{
  models: { providers: { vllm: {
    baseUrl: "http://127.0.0.1:8080/v1",
    apiKey: "local",
    api: "openai-completions",
    models: [{ id: "<model-name>", contextWindow: 128000, maxTokens: 8192 }],
  } } },
  agents: { defaults: { model: { primary: "vllm/<model-name>" } } },
}
```

## Testing on a normal machine (no CC hardware)

The bundled mock worker speaks the real wire protocol but skips attestation, so
you can exercise the whole path — handshake, ECDH, AES-GCM, tunnel, tool calls —
on any laptop. Run the proxy with `--insecure-skip-attestation`.

```bash
# terminal 1 — mock worker (echo backend, no model needed)
python -m tests.mock_worker --port 5056
#   real completions instead? point it at any OpenAI-compatible server:
#   python -m tests.mock_worker --upstream http://localhost:11434/v1 --model llama3.2

# terminal 2 — the proxy, skipping attestation (INSECURE, testing only)
bb-cc-proxy --enc-endpoint http://127.0.0.1:5056 --model demo --insecure-skip-attestation

# terminal 3 — hit it like any OpenAI endpoint
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"demo","messages":[{"role":"user","content":"hi"}]}'
```

Automated end-to-end tests (spin up mock worker + proxy, assert both APIs incl.
tool calls):

```bash
uv tool install --with pytest --editable .   # or: pip install -e ".[test]"
pytest
```

## How it works

- **Attestation + key exchange** ([`session.py`](cc_proxy/session.py)): `GET
  /attestation` → verify the NVIDIA EAT + signed key binding → ECDH → per-session
  AES-GCM key.
- **Transport** ([`session.py`](cc_proxy/session.py)): the `messages` array is
  sealed and POSTed to `{enc_endpoint}/enc/{provider}/{model}/message` (or
  `/message_stream`) with an `Authorization: Bearer` header; the worker returns
  encrypted plain text (streaming or single reply).
- **Surface** ([`server.py`](cc_proxy/server.py)): the decrypted text is wrapped
  into OpenAI `chat.completion` / `chat.completion.chunk` shapes for the local
  HTTP surface. Anthropic `/v1/messages` is translated to/from a plain messages
  array on this (trusted) side.

## Security notes

- `--insecure-skip-attestation` disables the confidential-compute guarantee. It
  exists only for non-CC testing and logs a loud warning. Never use it against a
  real workload.
- The proxy holds the session key and sees plaintext. Run it only on a machine
  in your own trust domain.
- Wire TLS to the worker is optional and independent of the payload encryption;
  the confidentiality comes from the attested AES-GCM layer, not TLS.
