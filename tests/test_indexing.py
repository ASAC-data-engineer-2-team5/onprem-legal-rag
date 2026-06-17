"""인덱싱 단위 테스트 (모델 다운로드 불필요한 로직만)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing import split_search_and_parents, _clean_meta

CHUNKS = [
    {"id": "jo-1", "type": "parent", "text": "조 전체", "metadata": {"jo_no": 1, "path": "p"}},
    {
        "id": "jo-1-hang-1",
        "type": "child",
        "parent_id": "jo-1",
        "text": "항1",
        "metadata": {"jo_no": 1, "hang_no": 1, "hang_label": None},
    },
    {
        "id": "jo-1-hang-2",
        "type": "child",
        "parent_id": "jo-1",
        "text": "항2",
        "metadata": {"jo_no": 1, "hang_no": 2, "hang_label": "라벨"},
    },
]


def test_split_search_and_parents():
    search, parents = split_search_and_parents(CHUNKS)
    # child 2개가 검색 단위, parent 1개가 lookup
    assert len(search) == 2
    assert all(c["type"] == "child" for c in search)
    assert "jo-1" in parents
    assert parents["jo-1"]["text"] == "조 전체"


def test_single_mode_split():
    chunks = [
        {"id": "jo-1", "type": "article", "text": "조1", "metadata": {"jo_no": 1}},
        {"id": "jo-2", "type": "article", "text": "조2", "metadata": {"jo_no": 2}},
    ]
    search, parents = split_search_and_parents(chunks)
    assert len(search) == 2
    assert parents == {}  # article은 자기 자신이 곧 조


def test_clean_meta_none_to_empty():
    out = _clean_meta({"a": None, "b": 1, "c": "x"})
    assert out["a"] == ""
    assert out["b"] == 1
    assert out["c"] == "x"


if __name__ == "__main__":
    test_split_search_and_parents()
    test_single_mode_split()
    test_clean_meta_none_to_empty()
    print("모든 테스트 통과")
