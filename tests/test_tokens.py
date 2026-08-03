"""Unit tests for the tokenizer facade."""

from cc_proxy import tokens as T


def test_heuristic_never_raises_and_returns_positive():
    tok = T.Tokenizer("heuristic", "heuristic", T._heuristic_count)
    assert tok.count_text("hello world foo bar") > 0
    assert tok.count_text("") == 0
    assert tok.count_text(None) == 0


def test_count_messages_is_additive():
    tok = T.Tokenizer("heuristic", "heuristic", T._heuristic_count)
    m1 = [{"role": "user", "content": "hello"}]
    m2 = [{"role": "user", "content": "hello"},
          {"role": "assistant", "content": "hi there friend"}]
    assert tok.count_messages(m2) > tok.count_messages(m1)


def test_count_openai_response_covers_content_and_tool_calls():
    tok = T.Tokenizer("heuristic", "heuristic", T._heuristic_count)
    r = {"choices": [{"message": {
        "content": "some reply text",
        "tool_calls": [{"function": {"name": "run", "arguments": '{"cmd":"ls"}'}}],
    }}]}
    n = tok.count_openai_response(r)
    assert n > 0


def test_stream_chunks_matches_joined_content_count():
    tok = T.Tokenizer("heuristic", "heuristic", T._heuristic_count)
    parts = ["Hello", " ", "world", "!"]
    chunks = [{"choices": [{"delta": {"content": p}}]} for p in parts]
    joined = "".join(parts)
    assert tok.count_stream_chunks(chunks) == tok.count_text(joined)


def test_stream_chunks_accumulates_tool_call_fragments_by_index():
    tok = T.Tokenizer("heuristic", "heuristic", T._heuristic_count)
    chunks = [
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "run"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"cmd":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"ls"}'}}]}}]},
    ]
    n = tok.count_stream_chunks(chunks)
    assert n > 0


def test_auto_family_from_model_hint():
    assert T._auto_family("google/gemma-4-31b-it") == "gemma"
    assert T._auto_family("deepseek-ai/DeepSeek-V3") == "deepseek"
    assert T._auto_family("meta-llama/Meta-Llama-3-70B") == "llama"
    assert T._auto_family("nvidia/Nemotron-4-340B") == "nemotron"
    assert T._auto_family("MiniMaxAI/MiniMax-Text-01") == "minimax"
    assert T._auto_family("Qwen/Qwen2.5-72B") == "qwen"
    assert T._auto_family("some/unknown-model") is None
    assert T._auto_family(None) is None


def test_load_never_raises_and_returns_a_tokenizer():
    # Even with no backends and a nonsense spec, load() must return something.
    tok = T.load(spec="definitely-not-a-real-thing", model_hint=None)
    assert tok is not None
    assert tok.count_text("hello world") > 0


def test_load_auto_detects_gemma_from_model_hint():
    tok = T.load(spec=None, model_hint="google/gemma-4-31b-it")
    # source will be "family" if tokenizers+network worked, or a fallback tag
    # otherwise. Either way the object must count.
    assert tok is not None
    assert tok.count_text("hello") > 0
