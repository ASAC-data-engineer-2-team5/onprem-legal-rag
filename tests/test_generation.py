"""생성 단위 테스트 — 프롬프트 구성/백엔드 가용성(네트워크 불필요한 로직만)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import build_answer_prompt, is_backend_available


def test_build_answer_prompt_includes_context_and_question():
    prompt = build_answer_prompt("징계 종류는?", ["제56조 ...", "제1조 ..."])
    assert "징계 종류는?" in prompt
    assert "제56조 ..." in prompt
    assert "제1조 ..." in prompt
    # 근거 없으면 모른다고 답하도록 지시 포함
    assert "모른다" in prompt


def test_is_backend_available_non_ollama():
    cfg = {"generation": {"backend": "vllm", "endpoint": "http://localhost:11434"}}
    assert is_backend_available(cfg) is False


def test_is_backend_available_unreachable():
    # 닫힌 포트로 빠르게 실패해야 함
    cfg = {"generation": {"backend": "ollama", "endpoint": "http://localhost:9"}}
    assert is_backend_available(cfg, timeout=0.5) is False


if __name__ == "__main__":
    test_build_answer_prompt_includes_context_and_question()
    test_is_backend_available_non_ollama()
    test_is_backend_available_unreachable()
    print("모든 테스트 통과")
