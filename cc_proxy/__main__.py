"""CLI entry point: ``bb-cc-proxy`` / ``python -m cc_proxy``.

Establishes the attested session to a confidential worker, then serves a local
OpenAI/Anthropic-compatible endpoint that coding CLIs can point at.
"""

import argparse
import logging
import os
import sys

from .server import create_app
from .session import AttestationError, ConfidentialSession


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bb-cc-proxy",
        description="Local decrypting proxy for confidential (E2E-encrypted) model endpoints.",
    )
    p.add_argument("--enc-endpoint", required=True,
                   help="URL of the confidential worker org host, "
                        "e.g. https://{organisation}.blackbox.ai")
    p.add_argument("--model", required=True,
                   help="Model path segment under /enc/, e.g. google/gemma-4-31b-it")
    p.add_argument("--host", default="127.0.0.1", help="Local bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="Local bind port (default: 8080)")
    p.add_argument("--local-api-key", default=None,
                   help="Optional bearer token CLIs must present to this proxy")
    p.add_argument("--timeout", type=float, default=120.0, help="Upstream request timeout (s)")
    p.add_argument("--no-verify-tls", action="store_true",
                   help="Skip TLS cert verification to the worker (self-signed certs)")
    p.add_argument("--insecure-skip-attestation", action="store_true",
                   help="INSECURE: skip GPU attestation verification (non-CC testing only)")
    p.add_argument("--upstream-api-key", default=os.environ.get("BLACKBOX_API_KEY"),
                   help="Bearer token sent upstream. Defaults to $BLACKBOX_API_KEY. "
                        "Optional when --passthrough-api-key is set.")
    p.add_argument("--passthrough-api-key", action="store_true",
                   help="Forward the client's Authorization: Bearer header upstream verbatim "
                        "(BYO-key). Falls back to --upstream-api-key when the client sends no token.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("cc_proxy")

    if not args.upstream_api_key and not args.passthrough_api_key:
        log.error("must provide either --upstream-api-key / $BLACKBOX_API_KEY "
                  "or --passthrough-api-key (BYO-key from the client)")
        return 2
    if args.passthrough_api_key and not args.upstream_api_key:
        log.info("--passthrough-api-key: no default upstream key configured; every request "
                 "must include Authorization: Bearer <sk-...>")

    session = ConfidentialSession(
        args.enc_endpoint,
        args.model,
        insecure_skip_attest=args.insecure_skip_attestation,
        timeout=args.timeout,
        verify_tls=not args.no_verify_tls,
        api_key=args.upstream_api_key,
    )
    try:
        session.connect()
    except AttestationError as e:
        log.error("%s", e)
        log.error("refusing to serve — the worker did not prove it is confidential hardware")
        return 2
    except Exception as e:  # noqa: BLE001 — surface any connection failure clearly
        log.error("could not establish confidential session: %s", e)
        return 1

    app = create_app(
        session,
        args.model,
        local_api_key=args.local_api_key,
        passthrough_api_key=args.passthrough_api_key,
    )
    log.info("serving OpenAI (/v1/chat/completions) + Anthropic (/v1/messages) on "
             "http://%s:%d  ->  %s%s", args.host, args.port, args.enc_endpoint,
             "  [passthrough-api-key ON]" if args.passthrough_api_key else "")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
