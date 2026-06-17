"""생성 — 로컬 LLM(Ollama) 답변 생성.

온프레미스 제약: 외부 LLM API 금지. 로컬 Ollama 엔드포인트만 사용한다.
표준 라이브러리(urllib)만으로 Ollama HTTP API를 호출해 의존성을 최소화한다.

이 모듈은 두 곳에서 쓰인다.
  - 인덱싱: 가상 질문 생성(src.indexing.generate_hypothetical_questions)
  - 검색 후: 검색된 조 전체를 컨텍스트로 최종 답변 생성

LLM이 아직 구동되지 않았을 수 있으므로, is_backend_available로 가용성을 먼저 확인한다.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def is_backend_available(config: dict[str, Any], timeout: float = 1.5) -> bool:
    """로컬 LLM 백엔드(Ollama)가 응답 가능한지 확인한다."""
    if config["generation"]["backend"] != "ollama":
        return False
    endpoint = config["generation"]["endpoint"].rstrip("/")
    try:
        with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def generate(config: dict[str, Any], prompt: str) -> str:
    """프롬프트를 로컬 LLM에 보내 생성 결과 텍스트를 반환한다."""
    gen = config["generation"]
    if gen["backend"] != "ollama":
        raise NotImplementedError(f"지원하지 않는 backend: {gen['backend']}")

    endpoint = gen["endpoint"].rstrip("/")
    payload = {
        "model": gen["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": gen["temperature"],
            "num_predict": gen["max_tokens"],
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/api/generate", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "")


def build_answer_prompt(question: str, context_articles: list[str]) -> str:
    """검색된 조 전체를 컨텍스트로 최종 답변 프롬프트를 구성한다."""
    context = "\n\n---\n\n".join(context_articles)
    return (
        "당신은 사내 규정집에 근거해 답변하는 어시스턴트입니다. "
        "아래 [참고 조항]에 있는 내용만 근거로, 한국어로 정확하게 답하세요. "
        "참고 조항에 근거가 없으면 모른다고 답하세요.\n\n"
        f"[참고 조항]\n{context}\n\n"
        f"[질문]\n{question}\n\n[답변]\n"
    )
