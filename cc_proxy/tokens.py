"""Model-family-agnostic local tokenizer facade.

One tokenizer per proxy instance (the tokenizer is a property of the model,
not the API surface — so it covers /v1/chat/completions, /v1/responses and
/v1/messages alike). Add a new family by appending one row to
``TOKENIZER_FAMILIES``.
"""

import json
import logging
import os

logger = logging.getLogger("cc_proxy.tokens")


# family short-name -> canonical HF repo id (or "tiktoken:<encoding>")
TOKENIZER_FAMILIES = {
    "gemma":    "google/gemma-2-2b",
    "llama":    "meta-llama/Meta-Llama-3-8B",
    "nemotron": "nvidia/Nemotron-4-Mini-4B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "minimax":  "MiniMaxAI/MiniMax-Text-01",
    "qwen":     "Qwen/Qwen2.5-7B",
    "openai":   "tiktoken:o200k_base",
}

# per-message overhead when the tokenizer has no embedded chat template.
# ~matches OpenAI's num_tokens_from_messages heuristic and is close enough
# for Gemma/Llama/etc. turn markers.
_MSG_OVERHEAD = 4


class Tokenizer:
    """Thin wrapper around whatever backend we managed to load.

    Backends (in priority order): HuggingFace ``tokenizers``, ``tiktoken``,
    whitespace × 1.3 heuristic. Callers only see ``count_*`` methods.
    """

    def __init__(self, name: str, source: str, encode):
        self.name = name
        self.source = source  # family | hf | local | tiktoken | heuristic
        self._encode = encode  # str -> int (token count)

    def count_text(self, s) -> int:
        if not s:
            return 0
        if not isinstance(s, str):
            s = str(s)
        try:
            return self._encode(s)
        except Exception:  # noqa: BLE001 — never break a request over counting
            return _heuristic_count(s)

    def count_messages(self, messages, tools=None) -> int:
        total = 0
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            total += _MSG_OVERHEAD
            total += self.count_text(m.get("role"))
            content = m.get("content")
            if isinstance(content, str):
                total += self.count_text(content)
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, str):
                        total += self.count_text(b)
                    elif isinstance(b, dict):
                        total += self.count_text(b.get("text") or b.get("content") or "")
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                total += self.count_text(fn.get("name"))
                total += self.count_text(fn.get("arguments"))
        for t in tools or []:
            total += self.count_text(json.dumps(t, default=str))
        return total

    def count_openai_response(self, rbody) -> int:
        if not isinstance(rbody, dict):
            return 0
        total = 0
        for choice in rbody.get("choices") or []:
            msg = (choice or {}).get("message") or {}
            total += self.count_text(msg.get("content"))
            for tc in msg.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                total += self.count_text(fn.get("name"))
                total += self.count_text(fn.get("arguments"))
        return total

    def count_stream_chunks(self, chunks) -> int:
        """Fold OpenAI chat.completion.chunk deltas into a single count."""
        content_parts = []
        tool_slots = {}  # index -> {"name": str, "args": [str]}
        for c in chunks or []:
            if not isinstance(c, dict):
                continue
            for choice in c.get("choices") or []:
                delta = (choice or {}).get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    slot = tool_slots.setdefault(idx, {"name": "", "args": []})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"].append(fn["arguments"])
        total = self.count_text("".join(content_parts))
        for slot in tool_slots.values():
            total += self.count_text(slot["name"])
            total += self.count_text("".join(slot["args"]))
        return total


# ------------------ loader ------------------


def _heuristic_count(s: str) -> int:
    # ~1.3 tokens per whitespace-split word — crude but bounded, and free.
    return max(1, int(len(s.split()) * 1.3)) if s else 0


def _load_tokenizers_from_file(path: str):
    from tokenizers import Tokenizer as HFTok  # type: ignore
    tok = HFTok.from_file(path)
    return lambda s: len(tok.encode(s).ids)


def _load_tokenizers_from_pretrained(repo: str):
    from tokenizers import Tokenizer as HFTok  # type: ignore
    tok = HFTok.from_pretrained(repo)
    return lambda s: len(tok.encode(s).ids)


def _load_tiktoken(encoding: str):
    import tiktoken  # type: ignore
    enc = tiktoken.get_encoding(encoding)
    return lambda s: len(enc.encode(s, disallowed_special=()))


def _auto_family(model_hint: str):
    if not model_hint:
        return None
    h = model_hint.lower()
    for fam in TOKENIZER_FAMILIES:
        if fam in h:
            return fam
    return None


def load(spec: str = None, model_hint: str = None) -> Tokenizer:
    """Resolve ``spec`` into a Tokenizer. Never raises — falls back to a
    heuristic with a warning if nothing else is available.

    Accepted forms for ``spec``:
      - None                     -> auto-detect from ``model_hint``
      - "gemma"/"llama"/...      -> registry lookup
      - "org/repo"               -> HF repo id
      - "/path/to/tokenizer.*"   -> local file
      - "tiktoken:<encoding>"    -> tiktoken passthrough
    """
    # 1. explicit local path
    if spec and os.path.exists(spec):
        try:
            return Tokenizer(spec, "local", _load_tokenizers_from_file(spec))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not load tokenizer from %s: %s", spec, e)

    # 2. tiktoken passthrough
    if spec and spec.startswith("tiktoken:"):
        enc = spec.split(":", 1)[1]
        try:
            return Tokenizer(f"tiktoken:{enc}", "tiktoken", _load_tiktoken(enc))
        except Exception as e:  # noqa: BLE001
            logger.warning("could not load tiktoken encoding %s: %s", enc, e)

    # 3./5. resolve family / auto-detect
    resolved = None
    source = None
    if spec and spec in TOKENIZER_FAMILIES:
        resolved = TOKENIZER_FAMILIES[spec]
        source = "family"
    elif spec and "/" in spec:
        resolved = spec
        source = "hf"
    elif spec is None:
        fam = _auto_family(model_hint)
        if fam:
            resolved = TOKENIZER_FAMILIES[fam]
            source = "family"

    if resolved:
        if resolved.startswith("tiktoken:"):
            enc = resolved.split(":", 1)[1]
            try:
                return Tokenizer(resolved, "tiktoken", _load_tiktoken(enc))
            except Exception as e:  # noqa: BLE001
                logger.warning("could not load tiktoken encoding %s: %s", enc, e)
        else:
            try:
                return Tokenizer(resolved, source, _load_tokenizers_from_pretrained(resolved))
            except Exception as e:  # noqa: BLE001
                logger.warning("could not load HF tokenizer %s (%s); "
                               "falling back", resolved, e)

    # 6. fallbacks
    try:
        return Tokenizer("tiktoken:cl100k_base", "tiktoken",
                         _load_tiktoken("cl100k_base"))
    except Exception:  # noqa: BLE001
        logger.warning("no tokenizer backend available — using whitespace "
                       "heuristic; token counts will be approximate. Install "
                       "with: pip install \"bb-cc-proxy[tokens]\"")
        return Tokenizer("heuristic", "heuristic", _heuristic_count)
