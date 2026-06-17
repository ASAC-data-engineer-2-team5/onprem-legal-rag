"""단일 질문 실행 — 검색 → 조 전체 컨텍스트로 로컬 LLM 답변 생성(수동 확인용).

사용법:
    python scripts/run_query.py "전결 금액 기준이 어떻게 되나요"
    python scripts/run_query.py "..." --config config/experiments/exp1_retrieval.yaml
    python scripts/run_query.py "..." --no-generate   # 검색 결과만 보고 LLM은 생략

선행 조건: scripts/build_index.py로 인덱스가 빌드되어 있어야 한다.
답변 생성을 쓰려면 로컬 LLM(Ollama)이 구동 중이어야 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import generation, retrieval
from src.config import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 + 답변 생성")
    parser.add_argument("question", help="질문 문장")
    parser.add_argument("--config", default=None, help="실험 오버라이드 YAML 경로")
    parser.add_argument("--no-generate", action="store_true", help="LLM 답변 생성 생략")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    # 1) 검색
    results = retrieve = retrieval.retrieve(config, args.question)
    print(f"\n[질문] {args.question}\n")
    print(f"[검색된 조 {len(results)}개]")
    for r in results:
        print(f"  - {r['jo']} ({r['metadata'].get('jo_title','')})  score={r['score']:.4f}")

    if args.no_generate:
        return

    # 2) 답변 생성 (로컬 LLM)
    if not generation.is_backend_available(config):
        print("\n[안내] 로컬 LLM(Ollama)이 응답하지 않아 답변 생성을 생략합니다.")
        print("       Ollama를 구동하고 모델을 받은 뒤 다시 실행하세요.")
        return

    context = [r["text"] for r in results]
    prompt = generation.build_answer_prompt(args.question, context)
    answer = generation.generate(config, prompt)
    print("\n[답변]")
    print(answer.strip())


if __name__ == "__main__":
    main()
