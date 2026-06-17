"""청킹 — 법조문 구조 기반 + Parent-Child.

원본 규정집(구조화 MD)을 편-장-절-조-항-호 체계로 파싱한다.
설계 원칙(CLAUDE.md):
  - 검색 단위는 '항'(child), LLM에 전달하는 단위는 '조 전체'(parent).
  - 각 청크에 편-장-절-조 경로를 메타데이터로 저장.
  - 표는 위에 요약 텍스트를 붙여 함께 인덱싱.

입력 구조 패턴:
  편  : '# 제N편 ...'
  장  : '## 제N장 ...'
  절  : '### 제N절 ...'
  조  : '**제N조 (제목)**'
  항  : '① [라벨] 본문 ...'   (원문자 ①~⑳로 시작)
  호  : '1. ...'              (항 본문 내부에 포함, 별도 청크로 쪼개지 않음)

출력: chunks.jsonl (한 줄당 청크 1개)
  공통 필드: id, type, text, metadata
  parent_child 모드 -> type in {parent, child}, child는 parent_id 보유
  single 모드       -> type = article (조 단위 단일 청크)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

# 항 마커로 쓰이는 원문자 (제1항 ~ 제20항)
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_CIRCLED_INDEX = {ch: i + 1 for i, ch in enumerate(CIRCLED)}

# 구조 마커 정규식
RE_PYEON = re.compile(r"^#\s+(제\d+편.*)$")
RE_JANG = re.compile(r"^##\s+(제\d+장.*)$")
RE_JEOL = re.compile(r"^###\s+(제\d+절.*)$")
RE_JO = re.compile(r"^\*\*\s*(제(\d+)조)\s*(?:\(([^)]*)\))?\s*\*\*\s*$")
RE_HANG = re.compile(rf"^([{CIRCLED}])\s*(.*)$")
# 항 본문 첫머리의 대괄호 라벨: '[정의 및 목적]'
RE_HANG_LABEL = re.compile(r"^\[([^\]]*)\]\s*")


# ---------------------------------------------------------------------------
# 표 처리
# ---------------------------------------------------------------------------
def _is_table_line(line: str) -> bool:
    """마크다운 표 행 여부(파이프로 시작)."""
    return line.lstrip().startswith("|")


def annotate_tables(text: str) -> str:
    """본문에서 마크다운 표 블록을 찾아 위에 요약 텍스트를 덧붙인다.

    표는 임베딩 시 의미가 흐려지기 쉬우므로, 헤더(컬럼명)를 추출해
    '[표 요약: 컬럼 — a, b, c]' 한 줄을 표 바로 위에 삽입한다.
    (외부 API 없이 결정적으로 동작하는 휴리스틱 요약.)
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if _is_table_line(lines[i]):
            # 연속된 표 블록 수집
            start = i
            while i < n and _is_table_line(lines[i]):
                i += 1
            block = lines[start:i]
            header_cells = [
                c.strip() for c in block[0].strip().strip("|").split("|") if c.strip()
            ]
            if header_cells:
                out.append(f"[표 요약: 컬럼 — {', '.join(header_cells)}]")
            out.extend(block)
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------
def _split_hang(jo_body_lines: list[str]) -> list[dict[str, Any]]:
    """조 본문 라인들을 항 단위로 분리한다.

    Returns: [{"hang_no": int, "label": str, "text": str}, ...]
    항 마커가 없는 머리말(드문 경우)은 hang_no=0의 항으로 담는다.
    """
    hangs: list[dict[str, Any]] = []
    cur_no: int | None = None
    cur_label = ""
    buf: list[str] = []

    def flush() -> None:
        if cur_no is None and not any(s.strip() for s in buf):
            return
        body = "\n".join(buf).strip()
        hangs.append(
            {"hang_no": cur_no if cur_no is not None else 0, "label": cur_label, "text": body}
        )

    for line in jo_body_lines:
        m = RE_HANG.match(line)
        if m:
            # 이전 항 마무리
            if cur_no is not None or buf:
                flush()
            cur_no = _CIRCLED_INDEX[m.group(1)]
            rest = m.group(2)
            label_m = RE_HANG_LABEL.match(rest)
            if label_m:
                cur_label = label_m.group(1).strip()
                rest = RE_HANG_LABEL.sub("", rest)
            else:
                cur_label = ""
            buf = [rest] if rest.strip() else []
        else:
            buf.append(line)
    if cur_no is not None or buf:
        flush()
    return hangs


def parse_document(md_text: str) -> list[dict[str, Any]]:
    """구조화 MD 한 편을 조 단위 레코드 리스트로 파싱한다.

    Returns: 각 조에 대해
      {pyeon, jang, jeol, jo, jo_no, jo_title, body_lines, hangs}
    """
    pyeon = jang = jeol = ""
    jo = jo_title = ""
    jo_no = 0
    body_lines: list[str] = []
    records: list[dict[str, Any]] = []

    def flush_jo() -> None:
        if not jo:
            return
        hangs = _split_hang(body_lines)
        records.append(
            {
                "pyeon": pyeon,
                "jang": jang,
                "jeol": jeol,
                "jo": jo,
                "jo_no": jo_no,
                "jo_title": jo_title,
                "body_lines": list(body_lines),
                "hangs": hangs,
            }
        )

    for raw in md_text.split("\n"):
        line = raw.rstrip("\n")
        if RE_PYEON.match(line):
            flush_jo()
            jo = ""
            body_lines = []
            pyeon = RE_PYEON.match(line).group(1).strip()
            jang = jeol = ""
            continue
        if RE_JANG.match(line):
            flush_jo()
            jo = ""
            body_lines = []
            jang = RE_JANG.match(line).group(1).strip()
            jeol = ""
            continue
        if RE_JEOL.match(line):
            flush_jo()
            jo = ""
            body_lines = []
            jeol = RE_JEOL.match(line).group(1).strip()
            continue
        m = RE_JO.match(line)
        if m:
            flush_jo()
            jo = m.group(1).strip()
            jo_no = int(m.group(2))
            jo_title = (m.group(3) or "").strip()
            body_lines = []
            continue
        if line.strip() == "---":
            continue
        if jo:
            body_lines.append(line)

    flush_jo()
    return records


# ---------------------------------------------------------------------------
# 청크 생성
# ---------------------------------------------------------------------------
def _path_str(rec: dict[str, Any]) -> str:
    parts = [rec["pyeon"], rec["jang"], rec["jeol"], rec["jo"]]
    return " > ".join(p for p in parts if p)


def _base_meta(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "pyeon": rec["pyeon"],
        "jang": rec["jang"],
        "jeol": rec["jeol"],
        "jo": rec["jo"],
        "jo_no": rec["jo_no"],
        "jo_title": rec["jo_title"],
        "path": _path_str(rec),
    }


def _jo_full_text(rec: dict[str, Any], table_summary: bool) -> str:
    """조 전체 텍스트(헤더 + 본문). parent/article의 LLM 전달용."""
    header = f"{rec['jo']} ({rec['jo_title']})" if rec["jo_title"] else rec["jo"]
    body = "\n".join(rec["body_lines"]).strip()
    text = f"{header}\n{body}".strip()
    return annotate_tables(text) if table_summary else text


def records_to_chunks(
    records: list[dict[str, Any]], mode: str, table_summary: bool
) -> list[dict[str, Any]]:
    """파싱된 조 레코드를 청크 리스트로 변환한다."""
    chunks: list[dict[str, Any]] = []
    for rec in records:
        meta = _base_meta(rec)
        parent_id = f"jo-{rec['jo_no']}"
        full_text = _jo_full_text(rec, table_summary)

        if mode == "single":
            # 조 단위 단일 청크
            chunks.append(
                {"id": parent_id, "type": "article", "text": full_text, "metadata": meta}
            )
            continue

        # parent_child: parent(조 전체) + child(항)
        chunks.append(
            {"id": parent_id, "type": "parent", "text": full_text, "metadata": meta}
        )
        for hang in rec["hangs"]:
            htext = hang["text"]
            if not htext.strip():
                continue
            if table_summary:
                htext = annotate_tables(htext)
            child_meta = dict(meta)
            child_meta["hang_no"] = hang["hang_no"]
            child_meta["hang_label"] = hang["label"]
            chunks.append(
                {
                    "id": f"{parent_id}-hang-{hang['hang_no']}",
                    "type": "child",
                    "parent_id": parent_id,
                    "text": htext,
                    "metadata": child_meta,
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# 엔트리
# ---------------------------------------------------------------------------
def _iter_source_files(raw_dir: Path) -> Iterator[Path]:
    for path in sorted(raw_dir.glob("*.md")):
        yield path
    for path in sorted(raw_dir.glob("*.txt")):
        yield path


def chunk_corpus(config: dict[str, Any]) -> list[dict[str, Any]]:
    """config에 따라 raw_dir의 모든 문서를 청킹한다."""
    mode = config["chunking"]["mode"]
    table_summary = config["chunking"]["table_summary"]
    raw_dir = Path(config["paths"]["raw_dir"])

    all_chunks: list[dict[str, Any]] = []
    for path in _iter_source_files(raw_dir):
        text = path.read_text(encoding="utf-8")
        records = parse_document(text)
        all_chunks.extend(records_to_chunks(records, mode, table_summary))
    return all_chunks


def write_chunks(chunks: list[dict[str, Any]], out_path: str | Path) -> None:
    """청크를 jsonl로 저장한다."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def run(config: dict[str, Any]) -> list[dict[str, Any]]:
    """청킹 실행: 파싱 -> 청크 생성 -> chunks.jsonl 저장."""
    chunks = chunk_corpus(config)
    write_chunks(chunks, config["paths"]["chunks_path"])
    return chunks
