"""계층 인식 청킹.

코퍼스 형식:
    # 대제목(약관명)
    <!-- product_type=...; insurer=... -->
    ## 제N관 ...           (관, 선택)
    ### 제N조(...)          (조)
    본문(정제됨)

청크 기본 단위 = 조(條). 조 본문이 길면 항(① ② / (1)(2)) 경계로 분할(의미 단위).
각 청크는 관·조 계층을 함께 들고 다닌다.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

# 항(項) 경계: ①..⑮ 또는 (1)(2) 앞에서 분리
_HANG = re.compile(r"(?=(?:[①-⑮]|\(\d{1,2}\)))")
_SENT = re.compile(r"(?<=[.다요])\s+")


@dataclass
class Chunk:
    doc_id: str
    doc_title: str       # 대제목(약관명)
    gwan: str            # 관(款), 없으면 ""
    section: str         # 소제목(조)
    section_idx: int
    chunk_id: str
    text: str
    product_type: str
    insurer: str
    source_file: str


def _split_units(text: str, max_chars: int) -> list[str]:
    """항 경계로 1차 분할 → max 넘으면 문장으로 2차 분할 → max 단위로 병합."""
    if len(text) <= max_chars:
        return [text]
    pieces = [p for p in _HANG.split(text) if p.strip()]
    expanded: list[str] = []
    for p in pieces:
        if len(p) <= max_chars:
            expanded.append(p)
        else:
            cur = ""
            for sent in _SENT.split(p):
                if cur and len(cur) + len(sent) > max_chars:
                    expanded.append(cur.strip())
                    cur = sent
                else:
                    cur = (cur + " " + sent).strip()
            if cur:
                expanded.append(cur)
    # 너무 잘게 쪼개진 조각은 max 한도 내에서 다시 합침
    merged: list[str] = []
    cur = ""
    for p in expanded:
        if cur and len(cur) + len(p) > max_chars:
            merged.append(cur.strip())
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        merged.append(cur.strip())
    return merged


def parse_doc(path: Path, doc_id: str, max_chars: int) -> list[Chunk]:
    title, product_type, insurer = "", "", ""
    gwan = ""
    jo = None
    buf: list[str] = []
    out: list[Chunk] = []
    sec_idx = 0

    def flush():
        nonlocal buf, sec_idx
        if jo is not None:
            body = "\n".join(buf).strip()
            if body:
                for j, part in enumerate(_split_units(body, max_chars)):
                    out.append(Chunk(
                        doc_id=doc_id, doc_title=title, gwan=gwan, section=jo,
                        section_idx=sec_idx, chunk_id=f"{doc_id}#s{sec_idx}.{j}",
                        text=part, product_type=product_type, insurer=insurer,
                        source_file=path.name,
                    ))
                sec_idx += 1
        buf = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("<!--") and "product_type" in line:
            m = re.search(r"product_type=([^;]+);\s*insurer=([^>]+?)\s*-->", line)
            if m:
                product_type, insurer = m.group(1).strip(), m.group(2).strip()
        elif line.startswith("## "):
            flush()
            jo = None
            gwan = line[3:].strip()
        elif line.startswith("### "):
            flush()
            jo = line[4:].strip()
        else:
            if jo is not None:
                buf.append(line)
    flush()
    return out
