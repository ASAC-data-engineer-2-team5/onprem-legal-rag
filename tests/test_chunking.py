"""청킹 단위 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import (
    annotate_tables,
    parse_document,
    records_to_chunks,
)

SAMPLE = """# 제1편 총칙

## 제1장 일반

### 제1절 통칙

**제1조 (목적)**

① [정의 및 목적]
본 규정의 목적은 가나다이다.
1. 첫째 항목
2. 둘째 항목

② [적용 대상]
모든 임직원에게 적용된다.

---

**제2조 (용어)**

① [정의]
용어를 정의한다.
"""


def test_parse_structure():
    recs = parse_document(SAMPLE)
    assert len(recs) == 2
    r0 = recs[0]
    assert r0["pyeon"] == "제1편 총칙"
    assert r0["jang"] == "제1장 일반"
    assert r0["jeol"] == "제1절 통칙"
    assert r0["jo"] == "제1조"
    assert r0["jo_no"] == 1
    assert r0["jo_title"] == "목적"
    assert len(r0["hangs"]) == 2
    assert r0["hangs"][0]["hang_no"] == 1
    assert r0["hangs"][0]["label"] == "정의 및 목적"
    # 호(1. 2.)는 항 본문 안에 포함
    assert "첫째 항목" in r0["hangs"][0]["text"]


def test_parent_child_chunks():
    recs = parse_document(SAMPLE)
    chunks = records_to_chunks(recs, mode="parent_child", table_summary=True)
    parents = [c for c in chunks if c["type"] == "parent"]
    children = [c for c in chunks if c["type"] == "child"]
    assert len(parents) == 2          # 제1조, 제2조
    assert len(children) == 3         # 제1조 2항 + 제2조 1항
    # child는 parent_id로 조에 연결
    c0 = children[0]
    assert c0["parent_id"] == "jo-1"
    assert c0["metadata"]["path"] == "제1편 총칙 > 제1장 일반 > 제1절 통칙 > 제1조"
    # parent는 조 전체(헤더 + 모든 항)를 담음
    p0 = parents[0]
    assert "제1조 (목적)" in p0["text"]
    assert "모든 임직원에게 적용된다." in p0["text"]


def test_single_mode():
    recs = parse_document(SAMPLE)
    chunks = records_to_chunks(recs, mode="single", table_summary=False)
    assert all(c["type"] == "article" for c in chunks)
    assert len(chunks) == 2


def test_table_summary():
    text = "앞 문장\n| 구분 | 금액 |\n| --- | --- |\n| 부서장 | 100만원 |\n뒤 문장"
    out = annotate_tables(text)
    assert "[표 요약: 컬럼 — 구분, 금액]" in out
    assert "부서장" in out


if __name__ == "__main__":
    test_parse_structure()
    test_parent_child_chunks()
    test_single_mode()
    test_table_summary()
    print("모든 테스트 통과")
