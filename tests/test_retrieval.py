"""검색 단위 테스트 — RRF 결합/순위 환원 로직(모델 불필요)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import _to_parent_ranking, rrf_combine


def test_to_parent_ranking_dedup():
    hits = [
        ("jo-1-hang-1", {"parent_id": "jo-1"}),
        ("jo-1-hang-2", {"parent_id": "jo-1"}),  # 같은 조 → 무시(첫 등장 유지)
        ("jo-3-hang-1", {"parent_id": "jo-3"}),
    ]
    assert _to_parent_ranking(hits) == ["jo-1", "jo-3"]


def test_rrf_combine_basic():
    rankings = {
        "dense": ["jo-1", "jo-2", "jo-3"],
        "sparse": ["jo-2", "jo-1", "jo-4"],
    }
    out = rrf_combine(rankings, {"dense": 1.0, "sparse": 1.0}, rrf_k=60)
    ids = [pid for pid, _ in out]
    # jo-1(1위+2위)과 jo-2(2위+1위)가 상위, jo-3/jo-4는 하위
    assert set(ids[:2]) == {"jo-1", "jo-2"}
    assert ids[2] in ("jo-3", "jo-4")


def test_rrf_weight_effect():
    rankings = {"dense": ["jo-A"], "sparse": ["jo-B"]}
    # sparse 가중치를 높이면 jo-B가 앞서야 함
    out = rrf_combine(rankings, {"dense": 1.0, "sparse": 5.0}, rrf_k=60)
    assert out[0][0] == "jo-B"


if __name__ == "__main__":
    test_to_parent_ranking_dedup()
    test_rrf_combine_basic()
    test_rrf_weight_effect()
    print("모든 테스트 통과")
