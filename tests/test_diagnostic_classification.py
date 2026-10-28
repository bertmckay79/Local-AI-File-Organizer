"""Ensure diagnostics expose errors without copying source book passages."""
from types import SimpleNamespace
from scripts.diagnose_book_classification import inspect_response


def test_empty_answer_with_thinking_is_visible_in_diagnostics():
    response = SimpleNamespace(status_code=200, json=lambda: {
        "message": {"content": "", "thinking": "analyzing"},
        "done_reason": "length", "eval_count": 768, "prompt_eval_count": 2000,
    })
    report = inspect_response({
        "model": "qwen3.5:4b", "think": False,
        "options": {"num_predict": 768},
        "messages": [{"role": "user", "content": "private book text"}],
    }, response=response)
    assert report["response_characters"] == 0
    assert report["thinking_characters"] == 9
    assert report["done_reason"] == "length"
    assert "private book text" not in str(report)


def test_http_status_and_error_are_captured():
    response = SimpleNamespace(status_code=404, text="model not found")
    report = inspect_response({"model": "missing"}, response=response)
    assert report["http_status"] == 404
    assert report["error"] == "model not found"


def test_timeout_captured_without_prompt_contents():
    report = inspect_response({
        "messages": [{"role": "user", "content": "sensitive text"}],
    }, error=TimeoutError("Ollama timed out"))
    assert "TimeoutError" in report["error"]
    assert "sensitive text" not in str(report)
