"""Mock confidential worker for testing on a normal (non-CC) machine.

Speaks the real wire protocol of the production secure worker — the same
``/attestation`` handshake (ECDH), AES-GCM sealing, and the ``/message`` /
``/message_stream`` endpoints — but performs **no GPU attestation** (stub
``gpu_eat``). Run the proxy against it with ``--insecure-skip-attestation``.

The reply is a deterministic echo (``"[mock] you said: <last-user-message>"``);
sending ``model="force-error"`` in the request triggers a streaming error frame
so tests can verify graceful upstream-error handling.

This worker intentionally does NOT enforce the ``Authorization: Bearer`` header
or the ``session_id`` echo — those are the real supervisor's concerns; here we
just want to exercise the crypto + streaming shape.
"""

import argparse
import base64
import json
import os

from flask import Flask, Response, jsonify, request

from cc_proxy import crypto

_PRIV, _PUB = crypto.generate_key_pair()
_STUB_GPU_EAT = json.dumps({"mock": True, "note": "NON-CC MOCK WORKER — no attestation"})


def create_app(model: str = "mock-model") -> Flask:
    app = Flask("mock_worker")

    def _shared_key(data):
        peer = crypto.deserialize_public_key(data["peer_public_key"])
        return crypto.derive_shared_key(_PRIV, peer), peer

    def _decrypt_messages(data):
        key, peer = _shared_key(data)
        plaintext = crypto.decrypt_and_verify(data["payload"], key, peer)
        return key, json.loads(plaintext)

    def _last_user(messages):
        return next((m.get("content", "") for m in reversed(messages)
                     if m.get("role") == "user"), "")

    def _seal(text: str, key, nonce: int = 3000) -> dict:
        return crypto.encrypt_and_sign(text, key, _PRIV, nonce)

    @app.get("/health")
    def health():
        return jsonify({"status": "healthy", "backend": "echo"})

    # The real supervisor exposes /attestation under /enc/{provider}/{model};
    # register both so tests can point ConfidentialSession at either shape.
    @app.get("/attestation")
    @app.get("/enc/<path:model_path>/attestation")
    def attestation(model_path=None):
        return jsonify({
            "report": {}, "report_json": "{}", "signature": "",
            "public_key": crypto.serialize_public_key(_PUB),
            "gpu_eat": _STUB_GPU_EAT,
            "nonce_b64": base64.b64encode(os.urandom(32)).decode(),
        })

    def _messages_from_request():
        data = request.json
        key, messages = _decrypt_messages(data)
        return key, messages

    @app.post("/message")
    @app.post("/enc/<path:model_path>/message")
    def message(model_path=None):
        key, messages = _messages_from_request()
        # Deterministic echo reply so tests can assert exact content.
        reply = f"[mock] you said: {_last_user(messages)}"
        return jsonify(_seal(reply, key))

    @app.post("/message_stream")
    @app.post("/enc/<path:model_path>/message_stream")
    def message_stream(model_path=None):
        key, messages = _messages_from_request()

        # Look at *any* message content to trigger a simulated upstream error —
        # tests use model="force-error" in the OpenAI wrapper; that model id
        # isn't visible here because /message_stream only sees the messages
        # array, so we key on a magic marker in the user text instead.
        force_error = "force-error" in _last_user(messages)

        def gen():
            if force_error:
                yield json.dumps({"error": {"status": 400,
                                            "body": {"error": {"message": "simulated upstream error"}}}}) + "\n"
                return
            reply = f"[mock] you said: {_last_user(messages)}"
            n = 3000
            for word in reply.split(" "):
                yield json.dumps(_seal(word + " ", key, n)) + "\n"
                n += 1
            yield json.dumps({"eos": True}) + "\n"

        return Response(gen(), mimetype="text/event-stream")

    return app


def main(argv=None):
    p = argparse.ArgumentParser(description="Mock confidential worker (non-CC testing)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5056)
    p.add_argument("--model", default="mock-model")
    args = p.parse_args(argv)
    app = create_app(args.model)
    print(f"mock worker on http://{args.host}:{args.port} (echo backend)")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
