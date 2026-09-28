#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAUSE-RAG / RAGON - Researcher A
DART 사업보고서 전체 Text/Table parser + table ensemble/QC

Table extraction (NO OCR)
- pdfplumber: lines / lines_strict / text / mixed
- img2table: native strict / native flexible, pdf_text_extraction=True, ocr=None
- PyMuPDF is used only for native body text and native word coordinates.
- Spatial clustering + structural quality + numeric-weighted content agreement
  + two-family consensus.

Important
- The PDFs are assumed to be born-digital / text-based DART reports.
- No Tesseract, no OCR fallback, no image-table rescue.
- pdfplumber and img2table are the TWO table engines.
- If img2table returns cell geometry but empty cell values, PyMuPDF native words
  are mapped into img2table cell bounding boxes. This is still native PDF text,
  not OCR.
- All raw table rows are preserved. The first row is NOT silently consumed as
  a DataFrame header.
- Multi-page tables remain page-level blocks for provenance, but receive a
  shared table_group_id when a conservative continuation rule matches.
- Text inside selected table boxes is excluded from text blocks.

Install
    pip install -U pymupdf pdfplumber img2table pandas

Run
    python 01_parse_business_reports.py
    python 01_parse_business_reports.py --save-debug-pdfs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import fitz
import pandas as pd
import pdfplumber

try:
    from img2table.document import PDF as Img2TablePDF
except ImportError as exc:
    raise SystemExit(
        "img2table이 없습니다.\n"
        "pip install -U img2table pymupdf pdfplumber pandas"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdf_data"
OUTPUT_DIR = BASE_DIR / "dataset" / "01_parsed"

BLOCKS_PATH = OUTPUT_DIR / "parsed_blocks.jsonl"
DOCUMENT_MANIFEST_PATH = OUTPUT_DIR / "document_manifest.csv"
DATASET_MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.json"
TABLE_CANDIDATES_PATH = OUTPUT_DIR / "table_candidates.csv"
TABLE_QC_PATH = OUTPUT_DIR / "table_qc.csv"
MULTIPAGE_GROUP_PATH = OUTPUT_DIR / "multipage_table_groups.csv"
PAGE_TEXT_QC_PATH = OUTPUT_DIR / "page_text_qc.csv"
TABLE_DIR = OUTPUT_DIR / "tables"
RAW_TEXT_DIR = OUTPUT_DIR / "raw_text"
DEBUG_DIR = OUTPUT_DIR / "debug_pdf"

TARGETS = {
    "삼성전자": {
        "stock_code": "005930",
        "aliases": ["삼성전자", "삼성전자주식회사", "samsung electronics"],
    },
    "SK하이닉스": {
        "stock_code": "000660",
        "aliases": [
            "sk하이닉스", "에스케이하이닉스", "에스케이하이닉스 주식회사",
            "sk hynix",
        ],
    },
    "현대자동차": {
        "stock_code": "005380",
        "aliases": [
            "현대자동차", "현대자동차 주식회사",
            "hyundai motor", "hyundai motor company",
        ],
    },
}
TARGET_FISCAL_YEARS = {2024, 2025}
FINANCIAL_MAJOR_SECTION = "III. 재무에 관한 사항"

IMG2TABLE_PDF_DPI = 200

# img2table validates min_confidence even when ocr=None.
# 1 is intentionally used as a near-no-filter value because this pipeline
# does NOT use OCR confidence; actual acceptance is controlled by our
# structural QC + cross-engine ensemble.
IMG2TABLE_MIN_CONFIDENCE = 1

# Front matter often contains lightly ruled / borderless corporate tables.
# Apply additional pdfplumber rescue strategies only to these early pages.
FRONT_MATTER_RESCUE_PAGES = 40

SAME_TABLE_IOU = 0.38
SAME_TABLE_COVERAGE = 0.65
MIN_ACCEPTED_SELECTION_SCORE = 0.38
NATIVE_TEXT_MIN_CHARS = 50
NATIVE_BAD_CHAR_RATIO_LIMIT = 0.03
BOTTOM_REGION_RATIO = 0.77
TOP_REGION_RATIO = 0.23
MULTIPAGE_HEADER_SIMILARITY = 0.45
MULTIPAGE_PROFILE_SIMILARITY = 0.72
QC_HIGH_THRESHOLD = 6
QC_MEDIUM_THRESHOLD = 3

MAJOR_SECTION_RULES = [
    ("I. 회사의 개요", [r"^(?:I|Ⅰ)\.\s*회사의\s*개요$"]),
    ("II. 사업의 내용", [r"^(?:II|Ⅱ)\.\s*사업의\s*내용$"]),
    ("III. 재무에 관한 사항", [r"^(?:III|Ⅲ)\.\s*재무에\s*관한\s*사항$"]),
    ("IV. 이사의 경영진단 및 분석의견", [r"^(?:IV|Ⅳ)\.\s*이사의\s*경영진단\s*및\s*분석의견$"]),
    ("V. 회계감사인의 감사의견 등", [r"^(?:V|Ⅴ)\.\s*회계감사인의\s*감사의견\s*등$"]),
    ("VI. 이사회 등 회사의 기관에 관한 사항", [r"^(?:VI|Ⅵ)\.\s*이사회\s*등\s*회사의\s*기관에\s*관한\s*사항$"]),
    ("VII. 주주에 관한 사항", [r"^(?:VII|Ⅶ)\.\s*주주에\s*관한\s*사항$"]),
    ("VIII. 임원 및 직원 등에 관한 사항", [r"^(?:VIII|Ⅷ)\.\s*임원\s*및\s*직원\s*등에\s*관한\s*사항$"]),
    ("IX. 계열회사 등에 관한 사항", [r"^(?:IX|Ⅸ)\.\s*계열회사\s*등에\s*관한\s*사항$"]),
    ("X. 대주주 등과의 거래내용", [r"^(?:X|Ⅹ)\.\s*대주주\s*등과의\s*거래내용$"]),
    ("XI. 그 밖에 투자자 보호를 위하여 필요한 사항", [r"^(?:XI|Ⅺ)\.\s*그\s*밖에\s*투자자\s*보호를\s*위하여\s*필요한\s*사항$"]),
    ("XII. 상세표", [r"^(?:XII|Ⅻ)\.\s*상세표$"]),
    ("전문가의 확인", [r"^[【\[]?\s*전문가의\s*확인\s*[】\]]?$"]),
]

PDFPLUMBER_SETTINGS = {
    "lines": {
        "vertical_strategy": "lines", "horizontal_strategy": "lines",
        "snap_tolerance": 3, "snap_x_tolerance": 2, "snap_y_tolerance": 2,
        "join_tolerance": 3, "join_x_tolerance": 3, "join_y_tolerance": 3,
        "edge_min_length": 3, "min_words_vertical": 3, "min_words_horizontal": 1,
        "intersection_tolerance": 3, "intersection_x_tolerance": 3,
        "intersection_y_tolerance": 3, "text_tolerance": 3,
        "text_x_tolerance": 3, "text_y_tolerance": 3,
    },
    "lines_strict": {
        "vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict",
        "snap_tolerance": 3, "join_tolerance": 3, "intersection_tolerance": 3,
        "edge_min_length": 3, "text_x_tolerance": 3, "text_y_tolerance": 3,
    },
    "text": {
        "vertical_strategy": "text", "horizontal_strategy": "text",
        "min_words_vertical": 3, "min_words_horizontal": 1,
        "snap_tolerance": 3, "join_tolerance": 3, "intersection_tolerance": 3,
        "text_x_tolerance": 3, "text_y_tolerance": 3,
    },
    "lines_text": {
        "vertical_strategy": "lines", "horizontal_strategy": "text",
        "min_words_horizontal": 1, "snap_tolerance": 3,
        "join_tolerance": 3, "intersection_tolerance": 3,
        "text_x_tolerance": 3, "text_y_tolerance": 3,
    },
    "text_lines": {
        "vertical_strategy": "text", "horizontal_strategy": "lines",
        "min_words_vertical": 3, "snap_tolerance": 3,
        "join_tolerance": 3, "intersection_tolerance": 3,
        "text_x_tolerance": 3, "text_y_tolerance": 3,
    },
}


# Extra strategies for the early pages of DART business reports.
# These are intentionally NOT used on all 400+ pages because text-based
# strategies can over-detect paragraph layouts as tables.
PDFPLUMBER_RESCUE_SETTINGS = {
    "front_text_loose": {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "snap_tolerance": 4,
        "join_tolerance": 4,
        "intersection_tolerance": 4,
        "text_x_tolerance": 4,
        "text_y_tolerance": 4,
    },
    "front_lines_loose": {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 5,
        "snap_x_tolerance": 5,
        "snap_y_tolerance": 5,
        "join_tolerance": 5,
        "join_x_tolerance": 5,
        "join_y_tolerance": 5,
        "edge_min_length": 1.5,
        "intersection_tolerance": 5,
        "intersection_x_tolerance": 5,
        "intersection_y_tolerance": 5,
        "text_x_tolerance": 4,
        "text_y_tolerance": 4,
    },
    "front_lines_text_loose": {
        "vertical_strategy": "lines",
        "horizontal_strategy": "text",
        "min_words_horizontal": 1,
        "snap_tolerance": 5,
        "join_tolerance": 5,
        "intersection_tolerance": 5,
        "text_x_tolerance": 4,
        "text_y_tolerance": 4,
    },
}


@dataclass
class TableCandidate:
    candidate_id: str
    page: int
    bbox_pdf: tuple[float, float, float, float]
    engine: str
    engine_family: str
    strategy: str
    rows: list[list[str]]
    cells: list[dict[str, Any]]
    source_dpi: Optional[int]
    parser_report: dict[str, Any]
    quality_score: float
    qc_score: int
    qc_priority: str
    qc_flags: list[str]
    numeric_density: float
    nonempty_ratio: float
    cluster_id: Optional[str] = None
    family_support: int = 1
    candidate_support: int = 1
    content_agreement: float = 0.0
    selection_score: float = 0.0
    selected: bool = False


@dataclass
class FinalTable:
    table_id: str
    page: int
    bbox_pdf: tuple[float, float, float, float]
    selected_candidate_id: str
    selected_engine: str
    selected_engine_family: str
    selected_strategy: str
    rows: list[list[str]]
    cells: list[dict[str, Any]]
    parser_report: dict[str, Any]
    quality_score: float
    selection_score: float
    qc_score: int
    qc_priority: str
    qc_flags: list[str]
    family_support: int
    candidate_support: int
    content_agreement: float
    table_group_id: Optional[str] = None
    continued_from_previous: bool = False
    continues_next: bool = False
    repeated_header_on_continuation: bool = False


@dataclass
class ParsedBlock:
    block_id: str
    document_id: str
    entity: str
    stock_code: str
    source_report_year: int
    filing_date: str
    filing_version: str
    version_id: str
    report_type: str
    major_section: str
    section_path: list[str]
    modality: str
    page: int
    bbox_pdf: Optional[dict[str, float]]
    source_file: str
    raw_content: str
    local_context: str
    text_extraction_method: Optional[str]
    native_text_quality: Optional[float]
    table_index: Optional[int]
    table_id: Optional[str]
    table_group_id: Optional[str]
    continued_from_previous: bool
    continues_next: bool
    repeated_header_on_continuation: bool
    table_columns: Optional[list[str]]
    table_rows: Optional[list[list[str]]]
    table_cells: Optional[list[dict[str, Any]]]
    table_html: Optional[str]
    table_markdown: Optional[str]
    table_extraction_engine: Optional[str]
    table_extraction_strategy: Optional[str]
    table_family_support: int
    table_candidate_support: int
    table_content_agreement: float
    table_selection_score: float
    is_financial_core: bool
    scope_hint: str
    statement_hint: str
    dart_gold_candidate_hint: bool
    table_qc_score: int
    table_qc_priority: str
    table_qc_flags: list[str]
    parsing_review_required: bool


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("\ufeff", "").replace("\xad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", clean_text(text)).lower()


def stable_id(*parts: Any, prefix: str = "") -> str:
    raw = "||".join(clean_text(x) for x in parts)
    return prefix + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bbox_to_dict(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "x0": round(float(bbox[0]), 3), "y0": round(float(bbox[1]), 3),
        "x1": round(float(bbox[2]), 3), "y1": round(float(bbox[3]), 3),
    }


def rect_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def intersection_area(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return 0.0 if x1 <= x0 or y1 <= y0 else (x1 - x0) * (y1 - y0)


def bbox_iou(a, b) -> float:
    inter = intersection_area(a, b)
    union = rect_area(a) + rect_area(b) - inter
    return inter / union if inter > 0 and union > 0 else 0.0


def bbox_coverage(a, b) -> float:
    inter = intersection_area(a, b)
    smaller = min(rect_area(a), rect_area(b))
    return inter / smaller if smaller > 0 else 0.0


def same_table_region(a, b) -> bool:
    return bbox_iou(a, b) >= SAME_TABLE_IOU or bbox_coverage(a, b) >= SAME_TABLE_COVERAGE


def center_inside(inner, outer) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def normalize_rows(rows: list[list[Any]]) -> list[list[str]]:
    cleaned = [[clean_text(c) for c in row] for row in rows if row is not None]
    if not cleaned:
        return []
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    while cleaned and not any(clean_text(c) for c in cleaned[-1]):
        cleaned.pop()
    return cleaned


def matrix_to_dataframe(rows: list[list[str]]) -> pd.DataFrame:
    rows = normalize_rows(rows)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[f"col_{i}" for i in range(len(rows[0]))])


def matrix_to_plain_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(clean_text(c) for c in r) for r in normalize_rows(rows))


def escape_markdown(value: str) -> str:
    return clean_text(value).replace("|", r"\|")


def looks_numeric(text: str) -> bool:
    value = clean_text(text)
    if not value:
        return False
    stripped = value.replace(",", "").replace(" ", "").replace("%", "")
    for suffix in ["원", "천원", "백만원", "억원", "조원", "명", "주", "개", "건", "대", "톤"]:
        stripped = stripped.replace(suffix, "")
    stripped = stripped.replace("(", "").replace(")", "").replace("−", "-").replace("–", "-")
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def likely_header_row(row: list[str]) -> bool:
    nonempty = [clean_text(c) for c in row if clean_text(c)]
    if not nonempty:
        return False
    period = any(re.search(r"20\d{2}|당기|전기|전전기|구분|항목|과목", c) for c in nonempty)
    mostly_text = sum(not looks_numeric(c) for c in nonempty) / len(nonempty) >= 0.5
    return period or mostly_text


def matrix_to_markdown(rows: list[list[str]]) -> str:
    rows = normalize_rows(rows)
    if not rows:
        return ""
    width = len(rows[0])
    if likely_header_row(rows[0]):
        header, body = rows[0], rows[1:]
    else:
        header, body = [f"col_{i}" for i in range(width)], rows
    lines = [
        "| " + " | ".join(escape_markdown(x) for x in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines += ["| " + " | ".join(escape_markdown(x) for x in row) + " |" for row in body]
    return "\n".join(lines)



# ----------------------------------------------------------------------
# document metadata
# ----------------------------------------------------------------------

def infer_entity(text: str) -> Optional[str]:
    target = compact(text)
    for entity, info in TARGETS.items():
        if any(compact(alias) in target for alias in info["aliases"]):
            return entity
    return None


def infer_fiscal_year(text: str) -> Optional[int]:
    for pattern in [
        r"사업연도\s*(20\d{2})년\s*0?1월\s*0?1일",
        r"사업연도[\s\S]{0,150}?(20\d{2})년",
    ]:
        match = re.search(pattern, clean_text(text), flags=re.IGNORECASE)
        if match and int(match.group(1)) in TARGET_FISCAL_YEARS:
            return int(match.group(1))
    return None


def infer_filing_date(text: str, filename: str) -> Optional[str]:
    match = re.search(r"귀중\s*(20\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일", clean_text(text))
    if not match:
        match = re.search(r"(20\d{2})[.\-_](\d{1,2})[.\-_](\d{1,2})", filename)
    if match:
        y, m, d = map(int, match.groups())
        return f"{y:04d}-{m:02d}-{d:02d}"
    return None


def infer_filing_version(filename: str, text: str) -> str:
    combined = (filename + " " + text[:6000]).lower()
    return "corrected" if any(k in combined for k in ["정정", "기재정정", "첨부정정", "amended", "corrected"]) else "original"


# ----------------------------------------------------------------------
# Table QC / content similarity
# ----------------------------------------------------------------------

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z가-힣])\(?[-+−–]?\d[\d,]*(?:\.\d+)?\)?")


def numeric_tokens(rows: list[list[str]]) -> set[str]:
    text = " ".join(clean_text(c) for r in rows for c in r)
    return {
        x.replace(",", "").replace(" ", "").replace("−", "-").replace("–", "-")
        for x in NUMBER_PATTERN.findall(text)
    }


def text_tokens(rows: list[list[str]]) -> set[str]:
    text = " ".join(clean_text(c).lower() for r in rows for c in r)
    return set(re.findall(r"[가-힣A-Za-z]{2,}", text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def table_content_similarity(a: TableCandidate, b: TableCandidate) -> float:
    return 0.65 * jaccard(numeric_tokens(a.rows), numeric_tokens(b.rows)) + 0.35 * jaccard(text_tokens(a.rows), text_tokens(b.rows))


def compute_table_quality(rows: list[list[str]], parser_report: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    rows = normalize_rows(rows)
    if not rows:
        return {
            "quality_score": 0.0, "qc_score": 10, "qc_priority": "HIGH",
            "qc_flags": ["empty_table"], "numeric_density": 0.0,
            "nonempty_ratio": 0.0,
        }

    n_rows, n_cols = len(rows), len(rows[0])
    cells = [clean_text(c) for r in rows for c in r]
    nonempty = [c for c in cells if c]
    nonempty_ratio = len(nonempty) / len(cells) if cells else 0.0
    numeric_density = sum(looks_numeric(c) for c in nonempty) / len(nonempty) if nonempty else 0.0
    duplicate_ratio = (n_rows - len(set(tuple(r) for r in rows))) / n_rows if n_rows else 0.0
    long_ratio = sum(len(c) >= 140 for c in nonempty) / len(nonempty) if nonempty else 0.0
    empty_cols = sum(not any(row[j] for row in rows) for j in range(n_cols))
    year_signal = any(re.search(r"20(?:2[0-9])|당기|전기|전전기|제\s*\d+\s*기", c) for r in rows[:4] for c in r)

    qc, flags = 0, []
    if n_rows == 1:
        qc += 2; flags.append("single_row")
    if n_cols == 1:
        qc += 4; flags.append("single_column")
    if nonempty_ratio < 0.20:
        qc += 5; flags.append("very_sparse")
    elif nonempty_ratio < 0.40:
        qc += 2; flags.append("sparse")
    if duplicate_ratio >= 0.35:
        qc += 2; flags.append("many_duplicate_rows")
    if long_ratio >= 0.25:
        qc += 2; flags.append("many_long_cells")
    if empty_cols:
        qc += min(3, empty_cols); flags.append("empty_columns")
    if n_cols >= 20:
        qc += 1; flags.append("very_many_columns")

    quality = 0.20
    quality += 0.12 if n_rows >= 2 else 0
    quality += 0.12 if n_cols >= 2 else 0
    quality += 0.26 * min(1.0, nonempty_ratio / 0.75)
    quality += 0.10 * min(1.0, numeric_density / 0.35)
    quality += 0.05 if year_signal else 0

    report = parser_report or {}
    confidence = report.get("confidence")
    try:
        if confidence is not None:
            confidence = float(confidence)
            confidence = confidence / 100.0 if confidence > 1 else confidence
            quality += 0.08 * max(0.0, min(1.0, confidence))
    except Exception:
        pass

    quality -= 0.04 * min(5, qc)
    quality = round(max(0.0, min(1.0, quality)), 4)
    priority = "HIGH" if qc >= QC_HIGH_THRESHOLD else "MEDIUM" if qc >= QC_MEDIUM_THRESHOLD else "LOW"

    return {
        "quality_score": quality,
        "qc_score": qc,
        "qc_priority": priority,
        "qc_flags": flags,
        "numeric_density": round(numeric_density, 4),
        "nonempty_ratio": round(nonempty_ratio, 4),
    }


def make_candidate(page, bbox_pdf, engine, family, strategy, rows, cells=None, source_dpi=None, parser_report=None):
    rows = normalize_rows(rows)
    if not rows:
        return None
    q = compute_table_quality(rows, parser_report)
    return TableCandidate(
        candidate_id=stable_id(page, bbox_pdf, engine, strategy, matrix_to_plain_text(rows), prefix="cand_"),
        page=page, bbox_pdf=bbox_pdf, engine=engine, engine_family=family,
        strategy=strategy, rows=rows, cells=cells or [], source_dpi=source_dpi,
        parser_report=parser_report or {}, quality_score=q["quality_score"],
        qc_score=q["qc_score"], qc_priority=q["qc_priority"], qc_flags=q["qc_flags"],
        numeric_density=q["numeric_density"], nonempty_ratio=q["nonempty_ratio"],
    )


def page_table_likeness_score(page: fitz.Page) -> int:
    """
    Lightweight native-text diagnostic only.
    It does NOT create a table and therefore cannot contaminate the corpus.

    Score signals:
    - common table labels
    - multiple rows containing several numeric tokens
    - repeated x-position anchors across many text lines
    """
    words = page.get_text("words", sort=True)

    if not words:
        return 0

    full_text = clean_text(
        page.get_text("text", sort=True)
    )

    score = 0

    table_terms = [
        "구분",
        "항목",
        "단위",
        "합계",
        "비율",
        "금액",
        "사업부문",
        "소유주식수",
        "성명",
        "직위",
        "기간",
        "당기",
        "전기",
    ]

    term_hits = sum(
        term in full_text
        for term in table_terms
    )

    if term_hits >= 2:
        score += 1

    # Group words into approximate visual lines.
    line_groups = defaultdict(list)

    for word in words:
        if len(word) < 5:
            continue

        x0, y0, x1, y1, value = word[:5]
        value = clean_text(value)

        if not value:
            continue

        line_key = round(
            float(y0) / 3.0
        ) * 3.0

        line_groups[line_key].append(
            (
                float(x0),
                value,
            )
        )

    numeric_rows = 0

    for line_words in line_groups.values():
        numeric_count = sum(
            bool(
                NUMBER_PATTERN.search(
                    value
                )
            )
            for _, value in line_words
        )

        if (
            len(line_words) >= 3
            and numeric_count >= 2
        ):
            numeric_rows += 1

    if numeric_rows >= 3:
        score += 1

    if numeric_rows >= 7:
        score += 1

    # Repeated x anchors are a common table-column signal.
    x_bins = Counter()

    for line_words in line_groups.values():
        used = set()

        for x0, _ in line_words:
            x_bin = int(
                round(
                    x0 / 8.0
                )
            )

            used.add(
                x_bin
            )

        for x_bin in used:
            x_bins[
                x_bin
            ] += 1

    repeated_columns = sum(
        count >= 5
        for count in x_bins.values()
    )

    if repeated_columns >= 3:
        score += 1

    return score


# ----------------------------------------------------------------------
# pdfplumber
# ----------------------------------------------------------------------

def serialize_pdfplumber_cells(table: Any) -> list[dict[str, Any]]:
    out = []
    for i, cell in enumerate(getattr(table, "cells", []) or []):
        try:
            out.append({"cell_index": i, "bbox_pdf": bbox_to_dict(tuple(map(float, cell)))})
        except Exception:
            pass
    return out


def extract_pdfplumber_candidates(
    page: Any,
    page_number: int,
    include_front_rescue: bool = False,
) -> list[TableCandidate]:
    """
    Standard strategies are always used.
    Looser rescue strategies are added only for early pages.

    Rescue candidates receive a small quality penalty so they can recover
    missed tables without automatically beating a cleaner standard candidate.
    """
    out = []

    settings_to_run = dict(
        PDFPLUMBER_SETTINGS
    )

    if include_front_rescue:
        settings_to_run.update(
            PDFPLUMBER_RESCUE_SETTINGS
        )

    for name, settings in settings_to_run.items():
        try:
            tables = page.find_tables(
                table_settings=settings
            )

        except Exception as exc:
            print(
                f"  [pdfplumber:{name}] "
                f"page={page_number} ERROR: {exc}"
            )
            continue

        for table in tables:
            try:
                parser_report = {
                    "front_rescue": (
                        name
                        in PDFPLUMBER_RESCUE_SETTINGS
                    ),
                    "ocr_used": False,
                }

                cand = make_candidate(
                    page_number,
                    tuple(
                        map(
                            float,
                            table.bbox,
                        )
                    ),
                    "pdfplumber",
                    "nurminen",
                    name,
                    table.extract(
                        x_tolerance=3,
                        y_tolerance=3,
                    ),
                    serialize_pdfplumber_cells(
                        table
                    ),
                    None,
                    parser_report,
                )

                if cand:
                    # Conservative penalty for loose rescue strategies.
                    if parser_report[
                        "front_rescue"
                    ]:
                        cand.quality_score = round(
                            max(
                                0.0,
                                cand.quality_score
                                - 0.035,
                            ),
                            4,
                        )

                    out.append(
                        cand
                    )

            except Exception as exc:
                print(
                    f"  [pdfplumber:{name}] "
                    f"extraction ERROR: {exc}"
                )

    return out


# ----------------------------------------------------------------------
# img2table (native PDF only; NO OCR)
# ----------------------------------------------------------------------

def img_bbox_to_pdf(x1, y1, x2, y2, dpi=IMG2TABLE_PDF_DPI):
    scale = 72.0 / float(dpi)
    return x1 * scale, y1 * scale, x2 * scale, y2 * scale


def get_native_words(page: fitz.Page) -> list[dict[str, Any]]:
    """PyMuPDF native words with PDF-point bounding boxes."""
    out = []
    for word in page.get_text("words", sort=True):
        if len(word) < 5:
            continue
        value = clean_text(word[4])
        if not value:
            continue
        out.append({
            "bbox": (float(word[0]), float(word[1]), float(word[2]), float(word[3])),
            "text": value,
            "block": int(word[5]) if len(word) > 5 else 0,
            "line": int(word[6]) if len(word) > 6 else 0,
            "word": int(word[7]) if len(word) > 7 else 0,
        })
    return out


def native_words_in_bbox(words: list[dict[str, Any]], bbox) -> list[dict[str, Any]]:
    selected = []
    for word in words:
        x0, y0, x1, y1 = word["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if bbox[0] - 1.0 <= cx <= bbox[2] + 1.0 and bbox[1] - 1.0 <= cy <= bbox[3] + 1.0:
            selected.append(word)
    return sorted(selected, key=lambda w: (w["block"], w["line"], w["word"], w["bbox"][0]))


def join_native_words(words: list[dict[str, Any]]) -> str:
    if not words:
        return ""
    groups = defaultdict(list)
    for word in words:
        groups[(word["block"], word["line"])].append(word)
    lines = []
    for key in sorted(groups):
        line_words = sorted(groups[key], key=lambda w: (w["word"], w["bbox"][0]))
        lines.append(" ".join(w["text"] for w in line_words))
    return clean_text("\n".join(lines))


def img2table_rows_and_cells(table: Any, page_words: list[dict[str, Any]]):
    """
    Prefer img2table native cell.value. If empty, populate the detected cell
    with PyMuPDF native PDF words by cell bbox. No OCR is used.
    """
    content = getattr(table, "content", None)
    if not content:
        df = getattr(table, "df", None)
        if df is None:
            return [], [], "none"
        rows = normalize_rows(df.astype(str).values.tolist())
        source = "img2table_native_dataframe" if any(any(clean_text(c) for c in r) for r in rows) else "none"
        return rows, [], source

    rows, serialized = [], []
    direct_count = remap_count = 0

    for r, cells in content.items():
        row_values = []
        for c, cell in enumerate(cells):
            bbox = getattr(cell, "bbox", None)
            bbox_pdf = None
            if bbox is not None:
                bbox_pdf = img_bbox_to_pdf(
                    float(bbox.x1), float(bbox.y1), float(bbox.x2), float(bbox.y2)
                )

            direct_value = clean_text(getattr(cell, "value", ""))
            if direct_value:
                value = direct_value
                source = "img2table_native_pdf_text"
                direct_count += 1
            elif bbox_pdf is not None:
                value = join_native_words(native_words_in_bbox(page_words, bbox_pdf))
                source = "pymupdf_native_bbox_remap"
                if value:
                    remap_count += 1
            else:
                value = ""
                source = "empty"

            row_values.append(value)
            serialized.append({
                "row_index": int(r),
                "column_index": int(c),
                "value": value,
                "value_source": source,
                "bbox_pdf": bbox_to_dict(bbox_pdf) if bbox_pdf is not None else None,
            })
        rows.append(row_values)

    if direct_count and remap_count:
        text_source = "img2table_native+pymupdf_bbox_remap"
    elif direct_count:
        text_source = "img2table_native_pdf_text"
    elif remap_count:
        text_source = "pymupdf_native_bbox_remap"
    else:
        text_source = "none"

    return normalize_rows(rows), serialized, text_source


def convert_img2table_result(extracted: dict, strategy: str, mu_doc: fitz.Document) -> dict[int, list[TableCandidate]]:
    out = defaultdict(list)
    for zero_page, tables in extracted.items():
        page = int(zero_page) + 1
        if not (1 <= page <= len(mu_doc)):
            continue
        page_words = get_native_words(mu_doc[page - 1])

        for table in tables:
            try:
                bbox = img_bbox_to_pdf(
                    float(table.bbox.x1), float(table.bbox.y1),
                    float(table.bbox.x2), float(table.bbox.y2)
                )
                rows, cells, text_source = img2table_rows_and_cells(table, page_words)
                if not rows or not any(any(clean_text(c) for c in row) for row in rows):
                    continue
                cand = make_candidate(
                    page, bbox, "img2table", "opencv", strategy, rows, cells,
                    IMG2TABLE_PDF_DPI,
                    {"native_text_source": text_source, "ocr_used": False},
                )
                if cand:
                    out[page].append(cand)
            except Exception as exc:
                print(f"  [img2table:{strategy}] page={page} ERROR: {exc}")
    return out


def extract_img2table_pdf_candidates(pdf_path: Path, mu_doc: fitz.Document) -> dict[int, list[TableCandidate]]:
    """Two img2table strategies, both with ocr=None and native PDF text."""
    doc = Img2TablePDF(
        src=pdf_path,
        pages=None,
        detect_rotation=False,
        pdf_text_extraction=True,
    )

    strict = doc.extract_tables(
        ocr=None,
        implicit_rows=False,
        implicit_columns=False,
        borderless_tables=False,
        min_confidence=IMG2TABLE_MIN_CONFIDENCE,
        max_workers=1,
    )

    flexible = doc.extract_tables(
        ocr=None,
        implicit_rows=True,
        implicit_columns=True,
        borderless_tables=True,
        min_confidence=IMG2TABLE_MIN_CONFIDENCE,
        max_workers=1,
    )

    strict_out = convert_img2table_result(strict, "native_strict_no_ocr", mu_doc)
    flexible_out = convert_img2table_result(flexible, "native_flexible_no_ocr", mu_doc)

    out = defaultdict(list)
    for page in set(strict_out) | set(flexible_out):
        out[page].extend(strict_out.get(page, []))
        out[page].extend(flexible_out.get(page, []))
    return out

# ----------------------------------------------------------------------
# Ensemble
# ----------------------------------------------------------------------

def cluster_candidates(candidates: list[TableCandidate], page: int) -> list[list[TableCandidate]]:
    clusters: list[list[TableCandidate]] = []
    for cand in sorted(candidates, key=lambda x: (x.bbox_pdf[1], x.bbox_pdf[0])):
        assigned = False
        for cluster in clusters:
            if any(same_table_region(cand.bbox_pdf, other.bbox_pdf) for other in cluster):
                cluster.append(cand); assigned = True; break
        if not assigned:
            clusters.append([cand])
    for i, cluster in enumerate(clusters):
        cid = f"p{page:04d}_cluster_{i:03d}"
        for c in cluster:
            c.cluster_id = cid
    return clusters


def engine_prior(c: TableCandidate) -> float:
    # Tiny tie-breaker only. Quality and cross-engine agreement dominate.
    return {"img2table": 0.025, "pdfplumber": 0.020}.get(c.engine, 0.0)


def enrich_cluster_scores(cluster: list[TableCandidate]) -> None:
    family_support = len({c.engine_family for c in cluster})
    candidate_support = len(cluster)
    for cand in cluster:
        sims = [table_content_similarity(cand, other) for other in cluster if other.candidate_id != cand.candidate_id]
        agreement = sum(sims) / len(sims) if sims else 0.0
        family_bonus = 0.12 * max(0, family_support - 1)
        repeated_bonus = 0.02 * min(3, max(0, candidate_support - family_support))
        cand.family_support = family_support
        cand.candidate_support = candidate_support
        cand.content_agreement = round(agreement, 4)
        cand.selection_score = round(cand.quality_score + family_bonus + repeated_bonus + 0.12 * agreement + engine_prior(cand), 4)


def plausible_final_table(c: TableCandidate) -> bool:
    rows = normalize_rows(c.rows)
    if not rows:
        return False
    nr, nc = len(rows), len(rows[0])
    shape_ok = nr >= 2 and nc >= 2
    consensus_exception = c.family_support >= 2 and nr * nc >= 4
    return (shape_ok or consensus_exception) and c.selection_score >= MIN_ACCEPTED_SELECTION_SCORE


def choose_cluster_winner(cluster: list[TableCandidate]) -> Optional[TableCandidate]:
    enrich_cluster_scores(cluster)
    ordered = sorted(cluster, key=lambda c: (-c.selection_score, c.qc_score, -c.nonempty_ratio, -c.numeric_density))
    for cand in ordered:
        if plausible_final_table(cand):
            cand.selected = True
            return cand
    return None


def make_final_table(c: TableCandidate) -> FinalTable:
    return FinalTable(
        table_id=stable_id(c.page, c.bbox_pdf, c.candidate_id, prefix="tbl_"),
        page=c.page, bbox_pdf=c.bbox_pdf, selected_candidate_id=c.candidate_id,
        selected_engine=c.engine, selected_engine_family=c.engine_family,
        selected_strategy=c.strategy, rows=c.rows, cells=c.cells,
        parser_report=c.parser_report, quality_score=c.quality_score,
        selection_score=c.selection_score, qc_score=c.qc_score,
        qc_priority=c.qc_priority, qc_flags=list(c.qc_flags),
        family_support=c.family_support, candidate_support=c.candidate_support,
        content_agreement=c.content_agreement,
    )


def ensemble_page(page: int, candidates: list[TableCandidate]) -> tuple[list[FinalTable], list[list[TableCandidate]]]:
    clusters = cluster_candidates(candidates, page)
    finals = []
    for cluster in clusters:
        winner = choose_cluster_winner(cluster)
        if winner:
            finals.append(make_final_table(winner))
    finals.sort(key=lambda t: (t.bbox_pdf[1], t.bbox_pdf[0]))
    return finals, clusters


def extract_all_tables(
    pdf_path: Path,
    mu_doc: fitz.Document,
    pl_doc: Any,
):
    """
    Final two-engine ensemble:
        1) pdfplumber
        2) img2table native PDF (ocr=None)

    PyMuPDF is NOT a third table engine.
    It is used for native text and bbox-to-cell text mapping.
    """
    try:
        print(
            "  [TABLE] img2table native(no OCR) extraction ..."
        )

        img_pdf = extract_img2table_pdf_candidates(
            pdf_path,
            mu_doc,
        )

        img_candidate_count = sum(
            len(items)
            for items in img_pdf.values()
        )

        img_page_count = sum(
            bool(items)
            for items in img_pdf.values()
        )

        print(
            "  [TABLE] img2table OK: "
            f"{img_candidate_count} candidates "
            f"on {img_page_count} pages"
        )

    except Exception as exc:
        # Fail-soft only for diagnosis. For final dataset, rerun after fixing.
        print()
        print(
            f"  [WARN] img2table extraction failed: {exc}"
        )
        print(
            "  [WARN] Only pdfplumber would remain active. "
            "For the final dataset, fix this error and rerun."
        )
        print()

        img_pdf = defaultdict(
            list
        )

    selected_by_page = {}
    all_candidates = []

    suspicious_no_table_pages = []

    for page_number in range(
        1,
        len(mu_doc) + 1,
    ):
        include_front_rescue = (
            page_number
            <= FRONT_MATTER_RESCUE_PAGES
        )

        candidates = []

        candidates += extract_pdfplumber_candidates(
            pl_doc.pages[
                page_number - 1
            ],
            page_number,
            include_front_rescue=(
                include_front_rescue
            ),
        )

        candidates += img_pdf.get(
            page_number,
            [],
        )

        finals, _ = ensemble_page(
            page_number,
            candidates,
        )

        selected_by_page[
            page_number
        ] = finals

        all_candidates.extend(
            candidates
        )

        # Diagnostic only: identify likely omissions without inventing tables.
        if not finals:
            likeness = page_table_likeness_score(
                mu_doc[
                    page_number - 1
                ]
            )

            if likeness >= 2:
                suspicious_no_table_pages.append(
                    (
                        page_number,
                        likeness,
                    )
                )

                if (
                    page_number
                    <= FRONT_MATTER_RESCUE_PAGES
                ):
                    print(
                        "  [CHECK] "
                        f"page {page_number}: "
                        "table-like layout detected "
                        f"(score={likeness}) but "
                        "no table candidate survived QC"
                    )

        if page_number % 25 == 0:
            selected_so_far = sum(
                len(
                    selected_by_page.get(
                        page,
                        [],
                    )
                )
                for page in range(
                    1,
                    page_number + 1,
                )
            )

            print(
                f"  [TABLE] "
                f"{page_number}/{len(mu_doc)} pages "
                f"| selected={selected_so_far}"
            )

    if suspicious_no_table_pages:
        preview = ", ".join(
            f"p{page}(s={score})"
            for page, score
            in suspicious_no_table_pages[:20]
        )

        print()
        print(
            "  [TABLE QC] "
            f"{len(suspicious_no_table_pages)} pages "
            "look table-like but have no selected table."
        )
        print(
            f"             first cases: {preview}"
        )
        print(
            "             These are REVIEW targets, "
            "not automatically accepted tables."
        )

    return (
        selected_by_page,
        all_candidates,
    )


# ----------------------------------------------------------------------
# Multi-page table linking
# ----------------------------------------------------------------------

def row_signature(row: list[str]) -> set[str]:
    return set(re.findall(r"[가-힣A-Za-z0-9]+", " ".join(clean_text(c).lower() for c in row)))


def header_similarity(a: FinalTable, b: FinalTable) -> float:
    sa, sb = set(), set()
    for row in a.rows[:2]: sa |= row_signature(row)
    for row in b.rows[:2]: sb |= row_signature(row)
    return jaccard(sa, sb)


def column_numeric_profile(rows: list[list[str]]) -> list[float]:
    rows = normalize_rows(rows)
    if not rows:
        return []
    out = []
    for j in range(len(rows[0])):
        vals = [clean_text(r[j]) for r in rows[:20] if clean_text(r[j])]
        out.append(sum(looks_numeric(v) for v in vals) / len(vals) if vals else 0.0)
    return out


def profile_similarity(a: FinalTable, b: FinalTable) -> float:
    pa, pb = column_numeric_profile(a.rows), column_numeric_profile(b.rows)
    if not pa or not pb or len(pa) != len(pb):
        return 0.0
    return max(0.0, 1.0 - sum(abs(x - y) for x, y in zip(pa, pb)) / len(pa))


def detect_multipage_groups(selected_by_page: dict[int, list[FinalTable]], page_heights: dict[int, float]) -> list[dict[str, Any]]:
    links = []
    for page in sorted(selected_by_page):
        if page + 1 not in selected_by_page or not selected_by_page[page] or not selected_by_page[page + 1]:
            continue
        a, b = selected_by_page[page][-1], selected_by_page[page + 1][0]
        near_bottom = a.bbox_pdf[3] >= BOTTOM_REGION_RATIO * page_heights[page]
        near_top = b.bbox_pdf[1] <= TOP_REGION_RATIO * page_heights[page + 1]
        ca = len(a.rows[0]) if a.rows else 0
        cb = len(b.rows[0]) if b.rows else 0
        if not near_bottom or not near_top or ca != cb or ca < 2:
            continue
        hs, ps = header_similarity(a, b), profile_similarity(a, b)
        if hs >= MULTIPAGE_HEADER_SIMILARITY or ps >= MULTIPAGE_PROFILE_SIMILARITY:
            links.append((a.table_id, b.table_id, hs, ps))

    lookup = {t.table_id: t for tables in selected_by_page.values() for t in tables}
    adjacency = defaultdict(set)
    for a, b, _, _ in links:
        adjacency[a].add(b); adjacency[b].add(a)

    rows, visited = [], set()
    for start in adjacency:
        if start in visited:
            continue
        stack, comp = [start], []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur); comp.append(cur); stack.extend(adjacency[cur] - visited)
        comp.sort(key=lambda x: (lookup[x].page, lookup[x].bbox_pdf[1]))
        group_id = stable_id(*comp, prefix="tblgrp_")
        for i, tid in enumerate(comp):
            t = lookup[tid]
            t.table_group_id = group_id
            t.continued_from_previous = i > 0
            t.continues_next = i < len(comp) - 1
            t.repeated_header_on_continuation = i > 0 and header_similarity(lookup[comp[i - 1]], t) >= 0.72
            rows.append({
                "table_group_id": group_id, "table_id": tid, "page": t.page,
                "sequence_index": i, "continued_from_previous": t.continued_from_previous,
                "continues_next": t.continues_next,
                "repeated_header_on_continuation": t.repeated_header_on_continuation,
            })

    for t in lookup.values():
        if t.table_group_id is None:
            t.table_group_id = stable_id(t.table_id, prefix="tblgrp_")
            rows.append({
                "table_group_id": t.table_group_id, "table_id": t.table_id, "page": t.page,
                "sequence_index": 0, "continued_from_previous": False,
                "continues_next": False, "repeated_header_on_continuation": False,
            })
    return rows


# ----------------------------------------------------------------------
# Text extraction
# ----------------------------------------------------------------------

def native_text_quality(text: str) -> float:
    text = clean_text(text)
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad_ratio = sum(c in {"\ufffd", "□", "�"} for c in chars) / len(chars)
    length_score = min(1.0, len(chars) / 250.0)
    corruption = max(0.0, 1.0 - bad_ratio / max(NATIVE_BAD_CHAR_RATIO_LIMIT, 1e-6))
    return round(0.75 * length_score + 0.25 * corruption, 4)


def extract_native_lines(page: fitz.Page) -> list[dict[str, Any]]:
    out = []
    for bi, block in enumerate(page.get_text("dict", sort=True).get("blocks", [])):
        if block.get("type") != 0:
            continue
        for li, line in enumerate(block.get("lines", [])):
            text = clean_text("".join(span.get("text", "") for span in line.get("spans", [])))
            if text:
                out.append({
                    "text": text,
                    "bbox": tuple(map(float, line.get("bbox", block.get("bbox", (0, 0, 0, 0))))),
                    "group_id": f"native_{bi}", "line_index": li, "ocr_confidence": None,
                })
    return out


def validate_native_document(mu_doc: fitz.Document) -> dict[str, Any]:
    total = bad = empty_pages = 0
    for page in mu_doc:
        content = clean_text(page.get_text("text", sort=True))
        compacted = re.sub(r"\s+", "", content)
        total += len(compacted)
        bad += sum(ch in {"�", "□", "\ufffd"} for ch in compacted)
        if not compacted:
            empty_pages += 1
    bad_ratio = bad / total if total else 1.0
    return {
        "valid": total >= 1000 and bad_ratio <= 0.02,
        "total_native_characters": total,
        "bad_character_count": bad,
        "bad_character_ratio": round(bad_ratio, 6),
        "empty_native_text_pages": empty_pages,
    }


def extract_page_lines(page: fitz.Page, table_bboxes: list[tuple[float, float, float, float]]):
    """Native PyMuPDF text only. No OCR fallback."""
    native_text = clean_text(page.get_text("text", sort=True))
    quality = native_text_quality(native_text)
    lines = [
        line for line in extract_native_lines(page)
        if not any(center_inside(line["bbox"], bbox) for bbox in table_bboxes)
    ]
    return lines, "pymupdf_native", quality


def merge_lines_to_blocks(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for line in lines:
        groups[line["group_id"]].append(line)
    out = []
    for group in groups.values():
        group.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        text = clean_text("\n".join(x["text"] for x in group))
        if len(text) < 2:
            continue
        bbox = (
            min(x["bbox"][0] for x in group), min(x["bbox"][1] for x in group),
            max(x["bbox"][2] for x in group), max(x["bbox"][3] for x in group),
        )
        out.append({"text": text, "bbox": bbox})
    return sorted(out, key=lambda x: (x["bbox"][1], x["bbox"][0]))


# ----------------------------------------------------------------------
# Sections / research hints
# ----------------------------------------------------------------------

def detect_major_section(text: str) -> Optional[str]:
    text = clean_text(text)
    for canonical, patterns in MAJOR_SECTION_RULES:
        if any(re.match(p, text, flags=re.IGNORECASE) for p in patterns):
            return canonical
    return None


def looks_like_heading(text: str) -> bool:
    text = clean_text(text)
    if not text or len(text) > 120:
        return False
    if detect_major_section(text):
        return True
    return any(re.match(p, text) for p in [
        r"^\d+(?:-\d+)?\.\s+.+$", r"^[가-하]\.\s*.+$",
        r"^\([0-9가-하]+\)\s*.+$", r"^\[[^\]]+\]$",
    ])


def update_section_path(path: deque[str], heading: str) -> None:
    major = detect_major_section(heading)
    if major:
        path.clear(); path.append(major)
    elif heading and (not path or path[-1] != heading):
        path.append(heading)


def infer_scope_hint(major: str, path: list[str], context: str) -> str:
    if major != FINANCIAL_MAJOR_SECTION:
        return "unknown"
    text = clean_text(" ".join(path) + " " + context).lower()
    if any(x in text for x in [
        "연결재무제표", "연결 재무제표", "연결재무상태표", "연결 재무상태표",
        "연결손익계산서", "연결 손익계산서", "연결포괄손익계산서",
        "연결 현금흐름표", "연결현금흐름표", "연결자본변동표", "(연결)",
    ]):
        return "consolidated"
    for heading in reversed(path):
        h = compact(heading)
        if re.match(r"4[.\-]?재무제표$", h) or re.match(r"5[.\-]?재무제표주석$", h):
            return "separate"
    return "unknown"


def infer_statement_hint(major: str, path: list[str], context: str) -> str:
    text = clean_text(" ".join(path[-5:]) + " " + context).lower()
    for key, code in [
        ("현금흐름표", "CF"), ("자본변동표", "SCE"), ("포괄손익계산서", "CIS"),
        ("손익계산서", "IS"), ("재무상태표", "BS"), ("주석", "NOTE"),
        ("요약재무정보", "SUMMARY"),
    ]:
        if key in text:
            return code
    if major == "II. 사업의 내용": return "BUSINESS"
    if major == "IV. 이사의 경영진단 및 분석의견": return "MD&A"
    if major == "V. 회계감사인의 감사의견 등": return "AUDIT"
    return "OTHER"


def infer_dart_gold_candidate_hint(major: str, scope: str, statement: str) -> bool:
    return major == FINANCIAL_MAJOR_SECTION and scope in {"consolidated", "separate"} and statement in {"BS", "IS", "CIS", "CF", "SCE"}


# ----------------------------------------------------------------------
# Debug / CSV helpers
# ----------------------------------------------------------------------

def save_debug_pdf(source_pdf: Path, tables_by_page: dict[int, list[FinalTable]], path: Path) -> None:
    """Create a PDF copy with selected-table bounding boxes. No raster/OCR."""
    doc = fitz.open(source_pdf)
    try:
        for page_number, tables in tables_by_page.items():
            page = doc[page_number - 1]
            for table in tables:
                rect = fitz.Rect(*table.bbox_pdf)
                page.draw_rect(rect, color=(1, 0, 0), width=0.8)
                page.insert_text(
                    (rect.x0, max(8, rect.y0 - 2)),
                    f"{table.selected_engine}:{table.selected_strategy} {table.selection_score:.2f}",
                    fontsize=5, color=(1, 0, 0),
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(path)
    finally:
        doc.close()


def candidate_csv_row(entity: str, year: int, source_file: str, c: TableCandidate) -> dict[str, Any]:
    rows = normalize_rows(c.rows)
    return {
        "entity": entity, "source_report_year": year, "source_file": source_file,
        "page": c.page, "candidate_id": c.candidate_id, "cluster_id": c.cluster_id,
        "engine": c.engine, "engine_family": c.engine_family, "strategy": c.strategy,
        "native_text_source": clean_text(c.parser_report.get("native_text_source")),
        "bbox_pdf": json.dumps(bbox_to_dict(c.bbox_pdf), ensure_ascii=False),
        "row_count": len(rows), "column_count": len(rows[0]) if rows else 0,
        "quality_score": c.quality_score, "selection_score": c.selection_score,
        "family_support": c.family_support, "candidate_support": c.candidate_support,
        "content_agreement": c.content_agreement, "numeric_density": c.numeric_density,
        "nonempty_ratio": c.nonempty_ratio, "qc_score": c.qc_score,
        "qc_priority": c.qc_priority, "qc_flags": "|".join(c.qc_flags),
        "selected": c.selected, "source_dpi": c.source_dpi,
    }


# ----------------------------------------------------------------------
# Parse one PDF
# ----------------------------------------------------------------------

def parse_one_pdf(pdf_path: Path, save_debug: bool):
    print("\n" + "=" * 96)
    print(f"[PARSE] {pdf_path.name}")
    print("=" * 96)

    mu_doc = fitz.open(pdf_path)
    pl_doc = pdfplumber.open(pdf_path)

    try:
        native_validation = validate_native_document(mu_doc)
        print(
            f"native chars={native_validation['total_native_characters']}, "
            f"bad ratio={native_validation['bad_character_ratio']}"
        )

        if not native_validation["valid"]:
            raise RuntimeError(
                "이 PDF는 native-text-only 기준을 만족하지 않습니다. "
                "현재 버전은 OCR을 의도적으로 사용하지 않습니다. "
                f"validation={native_validation}"
            )

        preview = "\n".join(
            clean_text(mu_doc[i].get_text("text", sort=True))
            for i in range(min(len(mu_doc), 12))
        )

        entity = infer_entity(preview + "\n" + pdf_path.name)
        year = infer_fiscal_year(preview)
        filing_date = infer_filing_date(preview, pdf_path.name)
        filing_version = infer_filing_version(pdf_path.name, preview)

        if entity not in TARGETS:
            raise RuntimeError(f"기업 식별 실패: {entity}")
        if year not in TARGET_FISCAL_YEARS:
            raise RuntimeError(f"사업연도 식별 실패: {year}")
        if not filing_date:
            raise RuntimeError("제출일 식별 실패")

        stock_code = TARGETS[entity]["stock_code"]
        version_id = f"{entity}_{year}_{filing_date}_{filing_version}"
        document_id = stable_id(
            entity, stock_code, year, filing_date, filing_version,
            pdf_path.name, prefix="doc_"
        )

        print(f"entity={entity}, FY={year}, filing={filing_date}, pages={len(mu_doc)}")
        print("[TABLE] pdfplumber + img2table native(no OCR)")

        selected_by_page, all_candidates = extract_all_tables(
            pdf_path, mu_doc, pl_doc
        )

        page_heights = {
            p: float(mu_doc[p - 1].rect.height)
            for p in range(1, len(mu_doc) + 1)
        }

        multipage_rows = detect_multipage_groups(
            selected_by_page, page_heights
        )

        for row in multipage_rows:
            row.update({
                "entity": entity,
                "source_report_year": year,
                "source_file": pdf_path.name,
            })

        page_text_blocks = {}
        page_text_meta = {}
        page_text_qc = []

        for p in range(1, len(mu_doc) + 1):
            bboxes = [
                t.bbox_pdf
                for t in selected_by_page.get(p, [])
            ]

            lines, method, quality = extract_page_lines(
                mu_doc[p - 1], bboxes
            )

            blocks = merge_lines_to_blocks(lines)

            page_text_blocks[p] = blocks
            page_text_meta[p] = {
                "method": method,
                "quality": quality,
            }

            native_page_text = clean_text(
                mu_doc[p - 1].get_text("text", sort=True)
            )

            page_text_qc.append({
                "entity": entity,
                "source_report_year": year,
                "source_file": pdf_path.name,
                "page": p,
                "text_extraction_method": method,
                "native_text_quality": quality,
                "native_character_count": len(re.sub(r"\s+", "", native_page_text)),
                "text_line_count": len(lines),
                "text_block_count": len(blocks),
                "selected_table_count": len(bboxes),
            })

        parsed = []
        table_qc = []

        current_major = "UNKNOWN"
        section_path = deque(maxlen=10)
        recent_text = deque(maxlen=6)

        text_idx = 0
        table_idx = 0

        report_table_dir = TABLE_DIR / f"{entity}_{year}"
        report_table_dir.mkdir(parents=True, exist_ok=True)

        for p in range(1, len(mu_doc) + 1):
            items = [
                {"kind": "text", "y": b["bbox"][1], "payload": b}
                for b in page_text_blocks.get(p, [])
            ]
            items += [
                {"kind": "table", "y": t.bbox_pdf[1], "payload": t}
                for t in selected_by_page.get(p, [])
            ]
            items.sort(key=lambda x: (x["y"], 0 if x["kind"] == "text" else 1))

            for item in items:
                if item["kind"] == "text":
                    b = item["payload"]
                    value = clean_text(b["text"])
                    if not value:
                        continue

                    major = detect_major_section(value)
                    if major:
                        current_major = major
                        update_section_path(section_path, value)
                    elif looks_like_heading(value):
                        update_section_path(section_path, value)

                    context = "\n".join(recent_text)
                    scope = infer_scope_hint(current_major, list(section_path), context)
                    statement = infer_statement_hint(current_major, list(section_path), context)

                    parsed.append(ParsedBlock(
                        block_id=stable_id(document_id, "text", text_idx, p, value, prefix="blk_"),
                        document_id=document_id,
                        entity=entity,
                        stock_code=stock_code,
                        source_report_year=year,
                        filing_date=filing_date,
                        filing_version=filing_version,
                        version_id=version_id,
                        report_type="annual_business_report",
                        major_section=current_major,
                        section_path=list(section_path),
                        modality="text",
                        page=p,
                        bbox_pdf=bbox_to_dict(b["bbox"]),
                        source_file=pdf_path.name,
                        raw_content=value,
                        local_context=context,
                        text_extraction_method="pymupdf_native",
                        native_text_quality=page_text_meta[p]["quality"],
                        table_index=None,
                        table_id=None,
                        table_group_id=None,
                        continued_from_previous=False,
                        continues_next=False,
                        repeated_header_on_continuation=False,
                        table_columns=None,
                        table_rows=None,
                        table_cells=None,
                        table_html=None,
                        table_markdown=None,
                        table_extraction_engine=None,
                        table_extraction_strategy=None,
                        table_family_support=0,
                        table_candidate_support=0,
                        table_content_agreement=0.0,
                        table_selection_score=0.0,
                        is_financial_core=current_major == FINANCIAL_MAJOR_SECTION,
                        scope_hint=scope,
                        statement_hint=statement,
                        dart_gold_candidate_hint=infer_dart_gold_candidate_hint(
                            current_major, scope, statement
                        ),
                        table_qc_score=0,
                        table_qc_priority="N/A",
                        table_qc_flags=[],
                        parsing_review_required=False,
                    ))

                    recent_text.append(value)
                    text_idx += 1

                else:
                    t: FinalTable = item["payload"]
                    rows = normalize_rows(t.rows)
                    if not rows:
                        continue

                    context = "\n".join(recent_text)
                    scope = infer_scope_hint(current_major, list(section_path), context)
                    statement = infer_statement_hint(current_major, list(section_path), context)

                    cols = [f"col_{i}" for i in range(len(rows[0]))]
                    df = matrix_to_dataframe(rows)
                    md = matrix_to_markdown(rows)
                    html = df.to_html(index=False, escape=True)
                    raw = matrix_to_plain_text(rows)

                    base = f"table_{table_idx:04d}_p{p}"
                    csv_path = report_table_dir / f"{base}.csv"
                    html_path = report_table_dir / f"{base}.html"
                    md_path = report_table_dir / f"{base}.md"

                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    html_path.write_text(html, encoding="utf-8")
                    md_path.write_text(md, encoding="utf-8")

                    fin = current_major == FINANCIAL_MAJOR_SECTION
                    primary_fin = fin and statement in {"BS", "IS", "CIS", "CF", "SCE"}

                    review = (
                        t.qc_priority == "HIGH"
                        or (t.family_support == 1 and t.selection_score < 0.62)
                        or (primary_fin and t.qc_priority == "MEDIUM")
                    )

                    native_source = clean_text(
                        t.parser_report.get("native_text_source")
                    ) or (
                        "pdfplumber_pdfminer_native"
                        if t.selected_engine == "pdfplumber"
                        else "native_pdf_text"
                    )

                    table_qc.append({
                        "entity": entity,
                        "source_report_year": year,
                        "source_file": pdf_path.name,
                        "page": p,
                        "table_index": table_idx,
                        "table_id": t.table_id,
                        "table_group_id": t.table_group_id,
                        "major_section": current_major,
                        "scope_hint": scope,
                        "statement_hint": statement,
                        "selected_engine": t.selected_engine,
                        "selected_engine_family": t.selected_engine_family,
                        "selected_strategy": t.selected_strategy,
                        "native_text_source": native_source,
                        "family_support": t.family_support,
                        "candidate_support": t.candidate_support,
                        "content_agreement": t.content_agreement,
                        "quality_score": t.quality_score,
                        "selection_score": t.selection_score,
                        "qc_score": t.qc_score,
                        "qc_priority": t.qc_priority,
                        "qc_flags": "|".join(t.qc_flags),
                        "parsing_review_required": review,
                        "continued_from_previous": t.continued_from_previous,
                        "continues_next": t.continues_next,
                        "repeated_header_on_continuation": t.repeated_header_on_continuation,
                        "row_count": len(rows),
                        "column_count": len(cols),
                        "bbox_pdf": json.dumps(bbox_to_dict(t.bbox_pdf), ensure_ascii=False),
                        "csv_path": str(csv_path.relative_to(BASE_DIR)),
                        "html_path": str(html_path.relative_to(BASE_DIR)),
                        "markdown_path": str(md_path.relative_to(BASE_DIR)),
                    })

                    parsed.append(ParsedBlock(
                        block_id=stable_id(document_id, "table", table_idx, p, t.table_id, raw, prefix="blk_"),
                        document_id=document_id,
                        entity=entity,
                        stock_code=stock_code,
                        source_report_year=year,
                        filing_date=filing_date,
                        filing_version=filing_version,
                        version_id=version_id,
                        report_type="annual_business_report",
                        major_section=current_major,
                        section_path=list(section_path),
                        modality="table",
                        page=p,
                        bbox_pdf=bbox_to_dict(t.bbox_pdf),
                        source_file=pdf_path.name,
                        raw_content=raw,
                        local_context=context,
                        text_extraction_method=None,
                        native_text_quality=None,
                        table_index=table_idx,
                        table_id=t.table_id,
                        table_group_id=t.table_group_id,
                        continued_from_previous=t.continued_from_previous,
                        continues_next=t.continues_next,
                        repeated_header_on_continuation=t.repeated_header_on_continuation,
                        table_columns=cols,
                        table_rows=rows,
                        table_cells=t.cells,
                        table_html=html,
                        table_markdown=md,
                        table_extraction_engine=t.selected_engine,
                        table_extraction_strategy=t.selected_strategy,
                        table_family_support=t.family_support,
                        table_candidate_support=t.candidate_support,
                        table_content_agreement=t.content_agreement,
                        table_selection_score=t.selection_score,
                        is_financial_core=fin,
                        scope_hint=scope,
                        statement_hint=statement,
                        dart_gold_candidate_hint=infer_dart_gold_candidate_hint(
                            current_major, scope, statement
                        ),
                        table_qc_score=t.qc_score,
                        table_qc_priority=t.qc_priority,
                        table_qc_flags=list(t.qc_flags),
                        parsing_review_required=review,
                    ))

                    table_idx += 1

        if save_debug:
            save_debug_pdf(
                pdf_path,
                selected_by_page,
                DEBUG_DIR / f"{entity}_{year}_table_debug.pdf",
            )

        RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
        raw_text_path = RAW_TEXT_DIR / f"{pdf_path.stem}.txt"
        raw_text_path.write_text(
            "\n\n".join(
                b.raw_content
                for b in parsed
                if b.modality == "text"
            ),
            encoding="utf-8",
        )

        candidate_rows = [
            candidate_csv_row(entity, year, pdf_path.name, c)
            for c in all_candidates
        ]

        selected_tables = [
            t
            for tables in selected_by_page.values()
            for t in tables
        ]

        manifest = {
            "document_id": document_id,
            "version_id": version_id,
            "entity": entity,
            "stock_code": stock_code,
            "source_report_year": year,
            "filing_date": filing_date,
            "filing_version": filing_version,
            "report_type": "annual_business_report",
            "source_file": pdf_path.name,
            "page_count": len(mu_doc),
            "native_text_only": True,
            "total_native_characters": native_validation["total_native_characters"],
            "bad_character_ratio": native_validation["bad_character_ratio"],
            "empty_native_text_pages": native_validation["empty_native_text_pages"],
            "num_blocks": len(parsed),
            "num_text_blocks": sum(b.modality == "text" for b in parsed),
            "num_table_blocks": sum(b.modality == "table" for b in parsed),
            "raw_table_candidate_count": len(all_candidates),
            "selected_table_count": len(selected_tables),
            "selected_engine_counts": json.dumps(
                dict(Counter(t.selected_engine for t in selected_tables)),
                ensure_ascii=False,
            ),
            "table_qc_high_count": sum(t.qc_priority == "HIGH" for t in selected_tables),
            "table_qc_medium_count": sum(t.qc_priority == "MEDIUM" for t in selected_tables),
            "two_engine_agreement_count": sum(t.family_support >= 2 for t in selected_tables),
        }

        return parsed, manifest, candidate_rows, table_qc, page_text_qc, multipage_rows

    finally:
        pl_doc.close()
        mu_doc.close()


# ----------------------------------------------------------------------
# Validation / main
# ----------------------------------------------------------------------

def validate_six_reports(manifests: list[dict[str, Any]]) -> None:
    expected = {(e, y) for e in TARGETS for y in TARGET_FISCAL_YEARS}
    actual = {(m["entity"], int(m["source_report_year"])) for m in manifests}
    if expected - actual:
        raise RuntimeError(f"필수 company-year 누락: {sorted(expected - actual)}")
    counts = Counter((m["entity"], int(m["source_report_year"])) for m in manifests)
    dup = {k: v for k, v in counts.items() if v > 1}
    if dup:
        raise RuntimeError(f"동일 company-year 중복: {dup}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-debug-pdfs",
        action="store_true",
        help="선택된 table bbox를 표시한 검수용 PDF 저장",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not PDF_DIR.exists():
        raise SystemExit(f"pdf_data 폴더가 없습니다: {PDF_DIR}")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise SystemExit("pdf_data 안에 PDF가 없습니다.")

    print("=" * 96)
    print("CAUSE-RAG | Native PDF parser")
    print("TEXT  : PyMuPDF native only")
    print("TABLE : pdfplumber + img2table native(no OCR)")
    print("OCR   : disabled")
    print(f"PDFs  : {len(pdf_files)}")
    print("=" * 96)

    all_blocks = []
    manifests = []
    all_candidates = []
    all_table_qc = []
    all_text_qc = []
    all_groups = []

    for pdf_path in pdf_files:
        blocks, manifest, candidates, table_qc, text_qc, groups = parse_one_pdf(
            pdf_path,
            args.save_debug_pdfs,
        )

        all_blocks += blocks
        manifests.append(manifest)
        all_candidates += candidates
        all_table_qc += table_qc
        all_text_qc += text_qc
        all_groups += groups

    validate_six_reports(manifests)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        BLOCKS_PATH,
        (asdict(x) for x in all_blocks),
    )

    pd.DataFrame(manifests).sort_values(
        ["entity", "source_report_year"]
    ).to_csv(
        DOCUMENT_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(all_candidates).to_csv(
        TABLE_CANDIDATES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(all_table_qc).to_csv(
        TABLE_QC_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(all_text_qc).to_csv(
        PAGE_TEXT_QC_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(all_groups).to_csv(
        MULTIPAGE_GROUP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    table_blocks = [
        x
        for x in all_blocks
        if x.modality == "table"
    ]

    dataset_manifest = {
        "dataset_name": "CAUSE-RAG Annual Business Report Native Text-Table Corpus",
        "companies": list(TARGETS),
        "fiscal_years": [2024, 2025],
        "report_type": "annual_business_report",
        "parse_scope": "whole_report",
        "modalities": ["text", "table"],
        "ocr_used": False,
        "document_count": len(manifests),
        "block_count": len(all_blocks),
        "text_block_count": sum(x.modality == "text" for x in all_blocks),
        "table_block_count": len(table_blocks),
        "table_candidate_count": len(all_candidates),
        "table_qc_high_count": sum(x.table_qc_priority == "HIGH" for x in table_blocks),
        "table_qc_medium_count": sum(x.table_qc_priority == "MEDIUM" for x in table_blocks),
        "two_engine_agreement_count": sum(x.table_family_support >= 2 for x in table_blocks),
        "text_method": "PyMuPDF native PDF text only",
        "table_method": {
            "engines": ["pdfplumber", "img2table"],
            "families": {
                "nurminen": ["pdfplumber"],
                "opencv": ["img2table"],
            },
            "img2table_ocr": None,
            "img2table_pdf_text_extraction": True,
            "img2table_native_text_fallback": "PyMuPDF word-to-cell bbox mapping",
            "selection": "spatial cluster + structure QC + numeric-weighted content agreement + two-engine support",
            "multi_page": "page provenance preserved; conservative table_group_id linking",
        },
        "documents": manifests,
    }

    save_json(
        DATASET_MANIFEST_PATH,
        dataset_manifest,
    )

    print("\n" + "=" * 96)
    print("[SUCCESS]")
    print(f"documents={len(manifests)}, blocks={len(all_blocks)}, tables={len(table_blocks)}")
    print(f"raw candidates={len(all_candidates)}")
    print(f"2-engine agreement={sum(x.table_family_support >= 2 for x in table_blocks)}")
    print(f"HIGH QC={sum(x.table_qc_priority == 'HIGH' for x in table_blocks)}")
    print(f"parsed blocks: {BLOCKS_PATH}")
    print(f"candidate audit: {TABLE_CANDIDATES_PATH}")
    print(f"table QC: {TABLE_QC_PATH}")
    print(f"multipage groups: {MULTIPAGE_GROUP_PATH}")
    print("=" * 96)
    print("QC: HIGH -> family_support=1인 재무표 -> DART mismatch 순으로 검수")


if __name__ == "__main__":
    main()
