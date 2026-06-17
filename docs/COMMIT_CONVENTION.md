# 커밋 메시지 컨벤션

이 프로젝트는 [Conventional Commits](https://www.conventionalcommits.org/) 스타일을 따른다.

## 형식

```
<type>: <제목>

<본문 — 변경 요점 bullet (선택)>
```

- **type**: 아래 표의 용어 중 하나(소문자).
- **제목**: 한국어, 명령형 요약. 마침표 없이.
- 본문: 무엇을/왜 바꿨는지 bullet로. 필요 시.
- 자동 서명 트레일러(Co-Authored-By 등)는 넣지 않는다.

## type 용어

| type | 의미 | 예시 |
| --- | --- | --- |
| `feat` | 새 기능 추가 | `feat: Hybrid 검색 RRF 결합 구현` |
| `fix` | 버그 수정 | `fix: BM25 한국어 토큰화 누락 수정` |
| `docs` | 문서만 변경(주석·md) | `docs: README에 Ollama 설치 절차 추가` |
| `test` | 테스트 추가·수정 | `test: 청킹 Parent-Child 케이스 추가` |
| `refactor` | 기능 변화 없는 구조 개선 | `refactor: config 로더 깊은 병합 분리` |
| `perf` | 성능 개선 | `perf: 임베딩 배치 크기 조정` |
| `style` | 포맷·공백 등 비기능 변경 | `style: import 정렬` |
| `chore` | 빌드·설정·의존성 등 잡일 | `chore: requirements에 ragas 추가` |
| `config` | 실험 설정값 변경(config/) | `config: exp1 RRF sparse 가중치 2.0` |
| `data` | 평가셋·원본 데이터 변경 | `data: 평가셋 질문 10개 추가` |
| `exp` | 실험 실행·결과 기록 | `exp: 검색방식 비교 결과표 추가` |

## scope (선택)

단계/모듈을 괄호로 덧붙일 수 있다.

```
feat(retrieval): 메타데이터 필터 적용
fix(chunking): 표 요약 위치 보정
```

진행 단계를 강조하고 싶으면 본문 첫 줄이나 제목에 단계 번호를 함께 적어도 된다
(예: `feat: 3단계 검색 — Hybrid + RRF`).

## 예시

```
feat: 2단계 인덱싱 — 벡터 색인 + BM25 통계 + parent lookup

- 검색 단위(항/조) 임베딩 → Chroma 저장
- 한국어 토큰화 후 BM25 코퍼스 통계 영속화
- parent(조) lookup 저장, 가상질문은 LLM 미구동 시 스킵
```

```
fix(generation): 생성 타임아웃 config 분리 및 thinking 비활성화
```
