"""Client-side cryptography for the confidential-compute proxy.

Self-contained: vendored from the Minions secure stack so this package can be
lifted into its own public repo without depending on the server code. Only the
*client* half lives here — attestation verification, ECDH, and AES-GCM
seal/open. There is deliberately no NVIDIA attestation SDK dependency: the
client only *verifies* the Entity Attestation Token (EAT), it never produces
one.
"""

import base64
import hashlib
import json
import os

import jwt  # PyJWT
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# ------------------ Key management ------------------


def generate_key_pair():
    private_key = ec.generate_private_key(ec.SECP384R1())
    return private_key, private_key.public_key()


def serialize_public_key(public_key) -> str:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


def deserialize_public_key(pem_data: str):
    return serialization.load_pem_public_key(pem_data.encode())


def derive_shared_key(private_key, peer_public_key) -> bytes:
    shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"handshake data",
    ).derive(shared_secret)


# ------------------ Attestation verification ------------------


def verify_attestation_full(
    report_json: bytes,
    signature_b64: str,
    gpu_eat_json: str,
    public_key,
    expected_nonce: bytes,
) -> bool:
    """Verify the supervisor's signed attestation report.

    Raises ValueError on any mismatch. Checks, in order:
      1. the report's ECDSA signature under the supervisor public key,
      2. that the report binds this public key and this gpu_eat,
      3. the anti-replay nonce,
      4. that the NVIDIA GPU EAT reports a verified attestation signature.
    """
    # 1) software-level signature over the report
    sig = base64.b64decode(signature_b64)
    try:
        public_key.verify(sig, report_json, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as e:
        raise ValueError("software-level signature invalid") from e

    report = json.loads(report_json)

    pub_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    digest = base64.b64encode(hashlib.sha256(pub_key_bytes).digest()).decode()
    if report["pubkey_hash"] != digest:
        raise ValueError("pubkey hash mismatch")
    if (
        report["gpu_eat_hash"]
        != base64.b64encode(hashlib.sha256(gpu_eat_json.encode()).digest()).decode()
    ):
        raise ValueError("gpu_eat hash mismatch")
    if report["nonce"] != base64.b64encode(expected_nonce).decode():
        raise ValueError("nonce mismatch (replay?)")

    # 2) GPU evidence check.
    # Freshness is already bound above via the signed report's `nonce`. The
    # NVIDIA attestation-SDK EAT (this SDK version) carries `eat_nonce` on the
    # per-GPU token rather than the outer platform token, so we additionally
    # bind it there when present. Both nonce fields are checked tolerantly
    # (only when present) to match the reference client, which must interop
    # across SDK/token-layout variations; the signature-verified flag is
    # always required.
    eat = json.loads(gpu_eat_json)
    platform_jwt, gpu_jwt = eat["platform_token"], eat["gpu_token"]
    if not gpu_jwt:
        raise ValueError("attestation contained no GPU tokens")

    expected_hex = expected_nonce.hex()

    plat_claims = jwt.decode(platform_jwt, options={"verify_signature": False})
    plat_nonce = plat_claims.get("eat_nonce")
    if plat_nonce is not None and plat_nonce != expected_hex:
        raise ValueError("platform nonce mismatch")

    for gpu_idx, gpu_token in gpu_jwt.items():
        gpu_claims = jwt.decode(gpu_token, options={"verify_signature": False})
        if not gpu_claims.get("x-nvidia-gpu-attestation-report-signature-verified"):
            raise ValueError(f"GPU attestation signature check failed for GPU {gpu_idx}")
        gpu_nonce = gpu_claims.get("eat_nonce")
        if gpu_nonce is not None and gpu_nonce != expected_hex:
            raise ValueError(f"GPU {gpu_idx} nonce mismatch (replay?)")
        if gpu_claims.get("x-nvidia-gpu-attestation-report-nonce-match") is False:
            raise ValueError(f"GPU {gpu_idx} reports attestation nonce mismatch")

    return True


# ------------------ AEAD messaging ------------------


def encrypt_and_sign(message: str, key: bytes, signing_key, nonce: int) -> dict:
    aesgcm = AESGCM(key)
    iv = os.urandom(12)
    ciphertext = aesgcm.encrypt(iv, message.encode(), None)
    data_to_sign = nonce.to_bytes(8, "big") + iv + ciphertext
    signature = signing_key.sign(data_to_sign, ec.ECDSA(hashes.SHA256()))
    return {
        "nonce": nonce,
        "iv": base64.b64encode(iv).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "signature": base64.b64encode(signature).decode(),
    }


def decrypt_and_verify(payload: dict, key: bytes, verifying_key) -> str:
    nonce = payload["nonce"]
    iv = base64.b64decode(payload["iv"])
    ciphertext = base64.b64decode(payload["ciphertext"])
    signature = base64.b64decode(payload["signature"])
    data_to_verify = nonce.to_bytes(8, "big") + iv + ciphertext

    try:
        verifying_key.verify(signature, data_to_verify, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as e:
        raise ValueError("response signature invalid") from e

    return AESGCM(key).decrypt(iv, ciphertext, None).decode()
