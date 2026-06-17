"""설정 로더.

2층 구조로 설정을 관리한다.
  - config/default.yaml          : 모든 변수의 고정 기준값(baseline)
  - config/experiments/*.yaml    : 이번 실험에서 바뀌는 변수만 담은 오버라이드

load_config()가 default 위에 오버라이드를 깊은 병합(deep merge)으로 덮어쓴 최종
설정을 돌려준다. 실험 변수를 코드에 하드코딩하지 않기 위한 단일 진입점이다.
또한 재현성을 위해 seed를 고정하는 set_seed()를 제공한다.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

# 프로젝트 루트 (이 파일 기준 한 단계 위가 src/, 그 위가 루트)
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """override를 base 위에 깊은 병합한다.

    두 값이 모두 dict면 재귀적으로 병합하고, 그 외에는 override 값으로 덮어쓴다.
    base는 변형하지 않고 새 dict를 반환한다.
    """
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """YAML 파일을 dict로 읽는다. 빈 파일은 빈 dict로 처리."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(
    experiment_path: str | os.PathLike[str] | None = None,
    *,
    default_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """최종 설정을 반환한다.

    Args:
        experiment_path: 실험 오버라이드 YAML 경로. None이면 default만 사용.
        default_path: 기준값 YAML 경로(기본 config/default.yaml).

    Returns:
        default에 오버라이드를 병합한 설정 dict.
    """
    config = _load_yaml(default_path)
    if experiment_path is not None:
        override = _load_yaml(experiment_path)
        config = _deep_merge(config, override)
    return config


def set_seed(seed: int) -> None:
    """재현성을 위해 랜덤 시드를 고정한다.

    numpy/torch는 설치돼 있을 때만 시드를 건다(0단계에서는 선택적 의존성).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
