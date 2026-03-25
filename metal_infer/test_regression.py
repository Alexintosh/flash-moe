#!/usr/bin/env python3
"""
Flash-MoE Regression Test Suite

Validates inference engine output via the OpenAI-compatible API server.
Start the server separately:  ./infer --model MODEL --openai-api 8080
Then run:                      python3 test_regression.py [--port 8080]
"""

import argparse
import json
import re
import sys
import time
from collections import Counter

try:
    import requests
    _USE_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    _USE_REQUESTS = False


# ---------------------------------------------------------------------------
# HTTP helpers (requests with urllib fallback)
# ---------------------------------------------------------------------------

def http_get(url, timeout=30):
    """GET request. Returns (status_code, body_text)."""
    if _USE_REQUESTS:
        r = requests.get(url, timeout=timeout)
        return r.status_code, r.text
    else:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")


def http_post_json(url, payload, timeout=60):
    """POST JSON request (non-streaming). Returns (status_code, body_text)."""
    body = json.dumps(payload).encode("utf-8")
    if _USE_REQUESTS:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code, r.text
    else:
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")


def http_post_stream(url, payload, timeout=60):
    """POST JSON request with streaming. Yields raw lines as strings."""
    body = json.dumps(payload).encode("utf-8")
    if _USE_REQUESTS:
        r = requests.post(url, json=payload, timeout=timeout, stream=True)
        for line in r.iter_lines(decode_unicode=True):
            if line is not None:
                yield line
    else:
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                yield raw_line.decode("utf-8").rstrip("\n\r")


# ---------------------------------------------------------------------------
# Chat / completions helpers
# ---------------------------------------------------------------------------

def chat(base_url, messages, max_tokens=50, temperature=0, stream=False, timeout=60):
    """Send a chat completion request."""
    payload = {
        "model": "flash-moe",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    url = f"{base_url}/v1/chat/completions"
    if stream:
        return http_post_stream(url, payload, timeout=timeout)
    else:
        status, body = http_post_json(url, payload, timeout=timeout)
        return status, body


def completions(base_url, prompt, max_tokens=20, temperature=0, timeout=60):
    """Send a legacy completions request."""
    payload = {
        "model": "flash-moe",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = f"{base_url}/v1/completions"
    return http_post_json(url, payload, timeout=timeout)


# ---------------------------------------------------------------------------
# Repetition detector
# ---------------------------------------------------------------------------

def has_excessive_repetition(text, phrase_len=5, max_repeats=3):
    """Check if any phrase_len-word sequence appears max_repeats or more times."""
    words = text.split()
    if len(words) < phrase_len:
        return False
    phrases = []
    for i in range(len(words) - phrase_len + 1):
        phrases.append(" ".join(words[i:i + phrase_len]))
    counts = Counter(phrases)
    for phrase, count in counts.items():
        if count >= max_repeats:
            return True
    return False


# ---------------------------------------------------------------------------
# Test runner infrastructure
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name, passed, detail=""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self):
        tag = "[PASS]" if self.passed else "[FAIL]"
        if self.detail:
            return f"{tag} {self.name}: {self.detail}"
        return f"{tag} {self.name}"


def run_tests(base_url):
    results = []

    # ------------------------------------------------------------------
    # 1. Health check
    # ------------------------------------------------------------------
    try:
        status, _ = http_get(f"{base_url}/health", timeout=10)
        if status == 200:
            results.append(TestResult("Health check /health", True))
        else:
            results.append(TestResult("Health check /health", False,
                                      f"expected 200, got {status}"))
    except Exception as e:
        results.append(TestResult("Health check /health", False, str(e)))

    try:
        status, body = http_get(f"{base_url}/v1/models", timeout=10)
        if status != 200:
            results.append(TestResult("Health check /v1/models", False,
                                      f"expected 200, got {status}"))
        elif "flash-moe" not in body.lower():
            results.append(TestResult("Health check /v1/models", False,
                                      "response does not contain 'flash-moe'"))
        else:
            results.append(TestResult("Health check /v1/models", True))
    except Exception as e:
        results.append(TestResult("Health check /v1/models", False, str(e)))

    # ------------------------------------------------------------------
    # 2. Basic generation (non-streaming)
    # ------------------------------------------------------------------
    try:
        t0 = time.monotonic()
        status, body = chat(base_url, [
            {"role": "user", "content": "Say hello"}
        ], max_tokens=20, temperature=0)
        elapsed = time.monotonic() - t0
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        errors = []
        if not content:
            errors.append("content is empty")
        if "!!!!" in content:
            errors.append(f"gibberish detected: {content!r}")
        if "You are a helpful" in content:
            errors.append("system prompt leaked into output")
        if errors:
            results.append(TestResult("Basic generation", False, "; ".join(errors)))
        else:
            results.append(TestResult("Basic generation", True,
                                      f"{len(content)} chars, {elapsed:.1f}s"))
    except Exception as e:
        results.append(TestResult("Basic generation", False, str(e)))

    # ------------------------------------------------------------------
    # 3. Longer generation
    # ------------------------------------------------------------------
    try:
        status, body = chat(base_url, [
            {"role": "user", "content": "Explain what a CPU is in 3 sentences"}
        ], max_tokens=100, temperature=0)
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        errors = []
        if len(content) <= 20:
            errors.append(f"too short ({len(content)} chars)")
        if has_excessive_repetition(content):
            errors.append(f"excessive repetition detected")
        if errors:
            results.append(TestResult("Longer generation", False, "; ".join(errors)))
        else:
            results.append(TestResult("Longer generation", True,
                                      f"{len(content)} chars"))
    except Exception as e:
        results.append(TestResult("Longer generation", False, str(e)))

    # ------------------------------------------------------------------
    # 4. JSON output test
    # ------------------------------------------------------------------
    try:
        status, body = chat(base_url, [
            {"role": "user",
             "content": ("Respond with a JSON object containing name and age fields. "
                         "Only output the JSON, nothing else.")}
        ], max_tokens=50, temperature=0)
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        errors = []
        if "{" not in content or "}" not in content:
            errors.append(f"no JSON braces in output: {content!r}")
        if '"name"' not in content:
            # Check for 2-bit corruption pattern: \name\ instead of "name"
            if "\\name\\" in content or "\\name" in content:
                errors.append(f'2-bit corruption: backslash-quoted keys in: {content!r}')
            else:
                errors.append(f'missing "name" field in output: {content!r}')
        if errors:
            results.append(TestResult("JSON output", False, "; ".join(errors)))
        else:
            results.append(TestResult("JSON output", True,
                                      f"valid JSON structure"))
    except Exception as e:
        results.append(TestResult("JSON output", False, str(e)))

    # ------------------------------------------------------------------
    # 5. Streaming test
    # ------------------------------------------------------------------
    try:
        lines = list(http_post_stream(
            f"{base_url}/v1/chat/completions",
            {
                "model": "flash-moe",
                "messages": [{"role": "user", "content": "Count from 1 to 5"}],
                "max_tokens": 50,
                "temperature": 0,
                "stream": True,
            },
            timeout=60,
        ))
        sse_events = [l for l in lines if l.startswith("data: ")]
        # Combine content from SSE events
        combined = ""
        for evt in sse_events:
            payload = evt[len("data: "):]
            if payload.strip() == "[DONE]":
                continue
            try:
                d = json.loads(payload)
                delta = d.get("choices", [{}])[0].get("delta", {})
                combined += delta.get("content", "")
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
        errors = []
        # Need at least 3 real SSE data events (excluding [DONE])
        real_events = [e for e in sse_events if "[DONE]" not in e]
        if len(real_events) < 3:
            errors.append(f"only {len(real_events)} SSE events (expected >=3)")
        if not combined.strip():
            errors.append("combined streamed content is empty")
        if errors:
            results.append(TestResult("Streaming", False, "; ".join(errors)))
        else:
            results.append(TestResult("Streaming", True,
                                      f"{len(real_events)} events, {len(combined)} chars"))
    except Exception as e:
        results.append(TestResult("Streaming", False, str(e)))

    # ------------------------------------------------------------------
    # 6. Completions endpoint (legacy)
    # ------------------------------------------------------------------
    try:
        status, body = completions(base_url,
                                   prompt="The capital of France is",
                                   max_tokens=20, temperature=0)
        data = json.loads(body)
        text = data["choices"][0]["text"]
        if not text.strip():
            results.append(TestResult("Completions endpoint", False, "text is empty"))
        else:
            results.append(TestResult("Completions endpoint", True,
                                      f"{len(text)} chars"))
    except Exception as e:
        results.append(TestResult("Completions endpoint", False, str(e)))

    # ------------------------------------------------------------------
    # 7. Performance check
    # ------------------------------------------------------------------
    try:
        t0 = time.monotonic()
        status, body = chat(base_url, [
            {"role": "user", "content": "What is 2+2?"}
        ], max_tokens=30, temperature=0, timeout=60)
        elapsed = time.monotonic() - t0
        data = json.loads(body)
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        errors = []
        if elapsed > 60:
            errors.append(f"too slow: {elapsed:.1f}s")
        tok_s = completion_tokens / elapsed if elapsed > 0 else 0
        detail = f"{elapsed:.1f}s, {completion_tokens} tokens"
        if tok_s > 0:
            detail += f", {tok_s:.1f} tok/s"
        if errors:
            results.append(TestResult("Performance check", False, "; ".join(errors)))
        else:
            results.append(TestResult("Performance check", True, detail))
    except Exception as e:
        results.append(TestResult("Performance check", False, str(e)))

    # ------------------------------------------------------------------
    # 8. Multi-turn (continuation)
    # ------------------------------------------------------------------
    try:
        # Turn 1
        status1, body1 = chat(base_url, [
            {"role": "user", "content": "Remember the number 42"}
        ], max_tokens=30, temperature=0)
        data1 = json.loads(body1)
        assistant_reply = data1["choices"][0]["message"]["content"]

        # Turn 2
        status2, body2 = chat(base_url, [
            {"role": "user", "content": "Remember the number 42"},
            {"role": "assistant", "content": assistant_reply},
            {"role": "user", "content": "What number did I mention?"},
        ], max_tokens=30, temperature=0)
        data2 = json.loads(body2)
        content2 = data2["choices"][0]["message"]["content"]
        if "42" in content2:
            results.append(TestResult("Multi-turn continuation", True))
        else:
            results.append(TestResult("Multi-turn continuation", False,
                                      f"expected '42' in response: {content2!r}"))
    except Exception as e:
        results.append(TestResult("Multi-turn continuation", False, str(e)))

    # ------------------------------------------------------------------
    # 9. Edge cases
    # ------------------------------------------------------------------
    # 9a. Empty message
    try:
        status, body = chat(base_url, [
            {"role": "user", "content": ""}
        ], max_tokens=20, temperature=0)
        # Just check it doesn't crash — any valid JSON response is fine
        data = json.loads(body)
        results.append(TestResult("Edge case: empty message", True,
                                  f"status {status}"))
    except Exception as e:
        results.append(TestResult("Edge case: empty message", False, str(e)))

    # 9b. Very long prompt
    try:
        long_prompt = ("Please summarize the following: " +
                       "The quick brown fox jumps over the lazy dog. " * 55)
        # ~500 words
        status, body = chat(base_url, [
            {"role": "user", "content": long_prompt}
        ], max_tokens=30, temperature=0, timeout=90)
        data = json.loads(body)
        _ = data["choices"][0]["message"]["content"]
        results.append(TestResult("Edge case: long prompt", True,
                                  f"status {status}"))
    except Exception as e:
        results.append(TestResult("Edge case: long prompt", False, str(e)))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flash-MoE regression tests")
    parser.add_argument("--port", type=int, default=8080,
                        help="API server port (default: 8080)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="API server host (default: localhost)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("Flash-MoE Regression Test Suite")
    print("================================")
    print(f"Server: {base_url}")
    print(f"Model:  flash-moe")
    print()

    # Quick connectivity check before running all tests
    try:
        http_get(f"{base_url}/health", timeout=5)
    except Exception:
        print(f"ERROR: Cannot connect to {base_url}")
        print("Start the server first:  ./infer --model MODEL --openai-api 8080")
        sys.exit(1)

    results = run_tests(base_url)

    print()
    for r in results:
        print(r)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    failed = total - passed
    print()
    print(f"Results: {passed}/{total} passed, {failed} failed")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
