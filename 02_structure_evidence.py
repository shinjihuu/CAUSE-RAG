#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_structure_evidence.py
================================================================================
CAUSE-RAG / RAGON
Researcher A - FINAL Evidence Structuring

Compatible input
----------------
01_parse_business_reports.py final native-PDF parser
    - Text: PyMuPDF native PDF text
    - Tables: pdfplumber + img2table ensemble
    - OCR/Tesseract: NOT used

Research role
-------------
This script converts parsed Text/Table blocks into structured evidence using the
CAUSE-RAG schema:

    Entity · Metric · Value · Unit · Period · Scope · Version

It is intentionally limited to evidence structuring. It does NOT:
    - generate questions,
    - label conflicts,
    - arbitrate evidence,
    - choose the final answer.

Core methodological rules
--------------------------
1. Parse/retain evidence from the WHOLE business report.
2. Preserve unknown financial metrics instead of filtering to a hand-written
   metric list. This avoids metric-selection bias.
3. Keep source_report_year separate from period_year.
   Example: FY2025 report can contain a comparative FY2024 value.
4. Resolve multi-row table headers conservatively (up to 4 header rows).
5. Resolve fiscal terms such as 제57기 using the report's own term number when
   it can be inferred from front-page native text.
6. Inherit headers/unit/scope/statement across a conservative table_group_id
   for multi-page tables.
7. Propagate all parser ensemble/QC metadata from Script 01.
8. DART alignment candidates are restricted to primary financial-statement
   TABLE evidence. Text duplicates are deliberately excluded from DART Gold
   alignment.
9. DART alignment_key EXCLUDES value. This prevents target-value leakage in
   Script 03.

Input
-----
dataset/01_parsed/parsed_blocks.jsonl

Output
------
dataset/02_structured/
├─ structured_evidence.jsonl
├─ structured_evidence.csv
├─ financial_structured_evidence.csv
├─ nonfinancial_structured_evidence.csv
├─ dart_alignment_candidates.csv
├─ manual_review_candidates.csv
├─ table_structure_diagnostics.csv
├─ evidence_schema.json
└─ structuring_summary.json

Install
-------
pip install -U pandas

Run
---
python 02_structure_evidence.py

Optional
--------
# By default, all financial table metrics are retained, while non-financial
# unknown metrics are kept only when the row has a strong tabular measurement
# structure. This flag makes non-financial table retention more permissive.
python 02_structure_evidence.py --include-unknown-nonfinancial
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ==============================================================================
# 0. PATH
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BASE_DIR
    / "dataset"
    / "01_parsed"
    / "parsed_blocks.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "02_structured"
)

ALL_JSONL = OUTPUT_DIR / "structured_evidence.jsonl"
ALL_CSV = OUTPUT_DIR / "structured_evidence.csv"

FINANCIAL_CSV = OUTPUT_DIR / "financial_structured_evidence.csv"
NONFINANCIAL_CSV = OUTPUT_DIR / "nonfinancial_structured_evidence.csv"
DART_CANDIDATES_CSV = OUTPUT_DIR / "dart_alignment_candidates.csv"
REVIEW_CSV = OUTPUT_DIR / "manual_review_candidates.csv"
TABLE_DIAGNOSTICS_CSV = OUTPUT_DIR / "table_structure_diagnostics.csv"

SCHEMA_PATH = OUTPUT_DIR / "evidence_schema.json"
SUMMARY_PATH = OUTPUT_DIR / "structuring_summary.json"


# ==============================================================================
# 1. RESEARCH CONSTANTS
# ==============================================================================

PRIMARY_FINANCIAL_STATEMENTS = {
    "BS",   # balance sheet / statement of financial position
    "IS",   # income statement
    "CIS",  # comprehensive income statement
    "CF",   # cash flow statement
    "SCE",  # statement of changes in equity
}

VALID_SCOPES = {
    "consolidated",
    "separate",
}

CURRENCY_SCALES = {
    "원": 1,
    "천원": 1_000,
    "백만원": 1_000_000,
    "억원": 100_000_000,
    "조원": 1_000_000_000_000,
}

NON_CURRENCY_UNITS = {
    "%",
    "명",
    "주",
    "대",
    "개",
    "건",
    "톤",
    "천톤",
    "백만톤",
    "MW",
    "GW",
    "MWh",
    "GWh",
}

UNIT_PATTERN = (
    r"조원|억원|백만원|천원|원|"
    r"%|퍼센트|명|주|대|개|건|"
    r"백만톤|천톤|톤|MWh|GWh|MW|GW"
)

NUMBER_WITH_OPTIONAL_UNIT_PATTERN = re.compile(
    rf"(?P<num>\(?\s*[-+−–]?\s*\d[\d,]*(?:\.\d+)?\s*\)?)"
    rf"\s*(?P<unit>{UNIT_PATTERN})?",
    flags=re.IGNORECASE,
)

TEXT_AMOUNT_PATTERN = re.compile(
    rf"(?P<num>\(?\s*[-+−–]?\s*\d[\d,]*(?:\.\d+)?\s*\)?)"
    rf"\s*(?P<unit>{UNIT_PATTERN})",
    flags=re.IGNORECASE,
)

YEAR_PATTERN = re.compile(r"(20\d{2})")
FISCAL_TERM_PATTERN = re.compile(r"제\s*(\d+)\s*기")

GENERIC_COLUMN_PATTERN = re.compile(
    r"^(?:col[_ ]?\d+|column[_ ]?\d+|unnamed.*|\d+)$",
    flags=re.IGNORECASE,
)

NOTE_HEADER_PATTERN = re.compile(
    r"(?:^|\s|/)(?:주석|note|notes)(?:\s|/|$)",
    flags=re.IGNORECASE,
)

# Text evidence is deliberately conservative.
TEXT_METRIC_VALUE_MAX_DISTANCE = 180

# Review thresholds: inherited from the final Script 01 ensemble design.
LOW_CONTENT_AGREEMENT = 0.55
LOW_SELECTION_SCORE = 0.62


# ==============================================================================
# 2. METRIC NORMALIZATION DICTIONARY
# ==============================================================================

# Canonical aliases improve matching, but are NOT a whitelist for financial
# tables. Unmapped financial rows are kept via metric_raw / metric_norm.
METRIC_ALIASES = {
    "revenue": [
        "매출액",
        "영업수익",
        "수익",
        "매출",
    ],
    "operating_profit": [
        "영업이익(손실)",
        "영업이익",
        "영업손익",
        "영업손실",
    ],
    "net_income": [
        "당기순이익(손실)",
        "연결당기순이익",
        "당기순이익",
        "당기순손익",
    ],
    "profit_before_tax": [
        "법인세비용차감전순이익",
        "법인세비용차감전이익",
        "법인세비용차감전순손익",
        "세전이익",
    ],
    "total_assets": [
        "자산총계",
        "총자산",
    ],
    "total_liabilities": [
        "부채총계",
        "총부채",
    ],
    "total_equity": [
        "자본총계",
        "총자본",
    ],
    "cash_and_cash_equivalents": [
        "현금및현금성자산",
        "현금 및 현금성자산",
    ],
    "inventories": [
        "재고자산",
        "재고 자산",
    ],
    "trade_receivables": [
        "매출채권및기타채권",
        "매출채권 및 기타채권",
        "매출채권",
    ],
    "property_plant_equipment": [
        "유형자산",
        "유형 자산",
    ],
    "intangible_assets": [
        "무형자산",
        "무형 자산",
    ],
    "research_and_development": [
        "연구개발비용",
        "연구개발 비용",
        "연구개발비",
        "연구개발투자",
    ],
    "finance_income": [
        "금융수익",
        "금융 수익",
    ],
    "finance_costs": [
        "금융비용",
        "금융 비용",
    ],
    "income_tax_expense": [
        "법인세비용",
        "법인세 비용",
    ],
    "basic_eps": [
        "기본주당순이익",
        "기본주당이익",
        "기본 주당이익",
    ],
    "diluted_eps": [
        "희석주당순이익",
        "희석주당이익",
        "희석 주당이익",
    ],
    "employees": [
        "임직원 수",
        "임직원수",
        "직원 수",
        "직원수",
    ],
    "production_capacity": [
        "생산능력",
        "생산 능력",
    ],
    "production_volume": [
        "생산실적",
        "생산량",
    ],
}

ALIAS_PAIRS = sorted(
    [
        (
            alias.lower(),
            canonical,
        )
        for canonical, aliases in METRIC_ALIASES.items()
        for alias in aliases
    ],
    key=lambda pair: len(pair[0]),
    reverse=True,
)

GENERIC_LABELS = {
    "구분",
    "항목",
    "과목",
    "계정과목",
    "내용",
    "비고",
    "단위",
    "분류",
    "합계",
    "계",
}

TITLE_LIKE_TERMS = {
    "연결재무상태표",
    "재무상태표",
    "연결손익계산서",
    "손익계산서",
    "연결포괄손익계산서",
    "포괄손익계산서",
    "연결현금흐름표",
    "현금흐름표",
    "연결자본변동표",
    "자본변동표",
}


# ==============================================================================
# 3. EVIDENCE MODEL
# ==============================================================================

@dataclass
class EvidenceRecord:
    evidence_id: str
    block_id: str
    document_id: str

    # ------------------------------------------------------------------
    # CAUSE-RAG core schema
    # ------------------------------------------------------------------
    entity: str
    stock_code: str

    metric_raw: str
    metric_path_raw: str
    metric_norm: str
    metric_canonical: Optional[str]

    value_raw: str
    value: float

    unit_raw: Optional[str]
    unit: Optional[str]
    unit_scale: Optional[int]
    value_krw: Optional[float]
    unit_inference_method: str

    period_raw: Optional[str]
    period_year: Optional[int]
    period_inference_method: str
    period_confidence: float

    scope: str
    scope_inference_method: str

    source_report_year: int
    filing_date: str
    filing_version: str
    version_id: str

    # ------------------------------------------------------------------
    # Grounding / provenance
    # ------------------------------------------------------------------
    major_section: str
    section_path: list[str]
    is_financial_core: bool

    modality: str
    statement_type: str
    statement_inference_method: str

    page: int
    bbox_pdf: Optional[dict[str, float]]
    source_file: str

    evidence_text: str

    # ------------------------------------------------------------------
    # Table provenance
    # ------------------------------------------------------------------
    table_index: Optional[int]
    table_id: Optional[str]
    table_group_id: Optional[str]

    row_index: Optional[int]
    column_index: Optional[int]
    column_header: Optional[str]

    continued_from_previous: bool
    continues_next: bool
    repeated_header_on_continuation: bool

    # ------------------------------------------------------------------
    # Parser provenance from Script 01
    # ------------------------------------------------------------------
    parser_engine: str
    parser_strategy: str

    table_family_support: int
    table_candidate_support: int
    table_content_agreement: float
    table_selection_score: float

    table_qc_score: int
    table_qc_priority: str
    table_qc_flags: list[str]
    parsing_review_required: bool

    # ------------------------------------------------------------------
    # Structuring QC
    # ------------------------------------------------------------------
    extraction_method: str
    confidence: float
    quality_flags: list[str]
    needs_review: bool

    # ------------------------------------------------------------------
    # OpenDART alignment eligibility
    # ------------------------------------------------------------------
    dart_alignable: bool
    alignment_metric_key: str
    alignment_key: Optional[str]


# ==============================================================================
# 4. COMMON UTILS
# ==============================================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and pd.isna(value):
        return ""

    text = str(value)
    text = (
        text
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\xad", "")
    )

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def compact(text: str) -> str:
    return re.sub(
        r"\s+",
        "",
        clean_text(text),
    ).lower()


def stable_id(
    *parts: Any,
    prefix: str = "",
) -> str:
    raw = "||".join(
        clean_text(part)
        for part in parts
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return f"{prefix}{digest}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return clean_text(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if clean_text(value) == "":
            return default

        return int(float(value))

    except Exception:
        return default


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if clean_text(value) == "":
            return default

        return float(value)

    except Exception:
        return default


def json_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    text = clean_text(value)

    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    clean_text(item)
                    for item in parsed
                    if clean_text(item)
                ]
        except Exception:
            pass

    return [
        token
        for token in text.split("|")
        if token
    ]


def normalize_rows(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []

    rows = []

    for row in value:
        if not isinstance(row, list):
            continue

        rows.append(
            [
                clean_text(cell)
                for cell in row
            ]
        )

    if not rows:
        return []

    width = max(
        len(row)
        for row in rows
    )

    result = []

    for row in rows:
        if len(row) < width:
            row = (
                row
                + [""] * (
                    width
                    - len(row)
                )
            )

        result.append(
            row[:width]
        )

    while result and not any(
        clean_text(cell)
        for cell in result[-1]
    ):
        result.pop()

    return result


# ==============================================================================
# 5. METRIC NORMALIZATION
# ==============================================================================

def normalize_metric_name(text: str) -> str:
    text = clean_text(text).lower()

    # Leading numbering / bullets.
    text = re.sub(
        r"^\s*(?:[\d①-⑳]+|[가-하])\s*[\.\)]\s*",
        "",
        text,
    )

    # Common note markers.
    text = re.sub(
        r"\(\s*주\s*\d+\s*\)",
        "",
        text,
    )

    text = re.sub(
        r"\b주\s*\d+\b",
        "",
        text,
    )

    text = re.sub(
        r"\bnote\s*\d+\b",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"[\s\.\,\:\;\-\_\(\)\[\]\{\}\/\\ㆍ·]",
        "",
        text,
    )

    return text.strip()


def canonical_metric(text: str) -> Optional[str]:
    lower = clean_text(text).lower()

    for alias, canonical in ALIAS_PAIRS:
        if alias in lower:
            return canonical

    return None


# ==============================================================================
# 6. NUMBER / UNIT
# ==============================================================================

def normalize_unit(
    unit: Optional[str],
) -> Optional[str]:
    if not unit:
        return None

    unit = clean_text(unit)

    if unit == "퍼센트":
        return "%"

    return unit


def parse_number(raw: str) -> Optional[float]:
    text = clean_text(raw)

    if not text:
        return None

    negative_parentheses = (
        text.startswith("(")
        and text.endswith(")")
    )

    if negative_parentheses:
        text = text[1:-1]

    text = (
        text
        .replace(",", "")
        .replace(" ", "")
        .replace("−", "-")
        .replace("–", "-")
    )

    if text in {
        "",
        "-",
        "—",
        "―",
    }:
        return None

    try:
        value = float(text)

    except ValueError:
        return None

    if negative_parentheses:
        value = -abs(value)

    return value


def extract_number_and_explicit_unit(
    cell: str,
) -> tuple[
    Optional[float],
    Optional[str],
    Optional[str],
]:
    text = clean_text(cell)

    if not text:
        return (
            None,
            None,
            None,
        )

    # Pure dates are not evidence values.
    if re.fullmatch(
        r"20\d{2}[./-]\d{1,2}[./-]\d{1,2}",
        text,
    ):
        return (
            None,
            None,
            None,
        )

    match = NUMBER_WITH_OPTIONAL_UNIT_PATTERN.search(
        text
    )

    if not match:
        return (
            None,
            None,
            None,
        )

    raw_number = clean_text(
        match.group("num")
    )

    value = parse_number(
        raw_number
    )

    unit = normalize_unit(
        match.group("unit")
    )

    return (
        value,
        raw_number,
        unit,
    )


def value_to_krw(
    value: float,
    unit: Optional[str],
) -> Optional[float]:
    scale = CURRENCY_SCALES.get(
        unit
    )

    if scale is None:
        return None

    return value * scale


def extract_unit_from_text(
    text: str,
) -> Optional[str]:
    text = clean_text(text)

    if not text:
        return None

    # Strong table-level unit notation.
    match = re.search(
        rf"단위\s*[:：]?\s*\(?\s*({UNIT_PATTERN})\s*\)?",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return normalize_unit(
            match.group(1)
        )

    # Parenthesized unit.
    match = re.search(
        rf"\(\s*({UNIT_PATTERN})\s*\)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return normalize_unit(
            match.group(1)
        )

    return None


# ==============================================================================
# 7. PERIOD / FISCAL TERM
# ==============================================================================

def infer_document_report_term_no(
    document_blocks: list[dict[str, Any]],
) -> tuple[
    Optional[int],
    str,
]:
    """
    Infer the report's own fiscal term number (e.g., 제57기) from front pages.

    Strong patterns are preferred. We never hard-code company-specific term
    numbers. If inference is weak, period mapping can still rely on explicit
    calendar years or local term ordering.
    """
    front_text = "\n".join(
        clean_text(
            block.get("raw_content")
        )
        for block in sorted(
            document_blocks,
            key=lambda item: (
                safe_int(item.get("page")),
                clean_text(item.get("block_id")),
            ),
        )
        if (
            block.get("modality") == "text"
            and safe_int(block.get("page")) <= 20
        )
    )

    strong_patterns = [
        r"사업보고서[\s\S]{0,120}?제\s*(\d+)\s*기",
        r"제\s*(\d+)\s*기[\s\S]{0,120}?사업연도",
    ]

    for pattern in strong_patterns:
        match = re.search(
            pattern,
            front_text,
            flags=re.IGNORECASE,
        )

        if match:
            return (
                int(match.group(1)),
                "front_page_strong_pattern",
            )

    # Conservative fallback: first pages often contain the report-term label.
    fallback_text = "\n".join(
        clean_text(
            block.get("raw_content")
        )
        for block in document_blocks
        if (
            block.get("modality") == "text"
            and safe_int(block.get("page")) <= 5
        )
    )

    terms = [
        int(value)
        for value in FISCAL_TERM_PATTERN.findall(
            fallback_text
        )
    ]

    if terms:
        # Most front-page references point to current report term. Use the mode;
        # ties are resolved by larger term number.
        counts = Counter(terms)
        best = sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                -item[0],
            ),
        )[0][0]

        return (
            best,
            "front_page_term_mode_fallback",
        )

    return (
        None,
        "unresolved",
    )


def build_document_term_info(
    blocks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_document = defaultdict(list)

    for block in blocks:
        by_document[
            block["document_id"]
        ].append(block)

    result = {}

    for document_id, document_blocks in by_document.items():
        report_term_no, method = infer_document_report_term_no(
            document_blocks
        )

        source_year = safe_int(
            document_blocks[0].get(
                "source_report_year"
            )
        )

        result[document_id] = {
            "report_term_no": report_term_no,
            "method": method,
            "source_report_year": source_year,
        }

    return result


def build_fiscal_term_map(
    headers: list[str],
    source_report_year: int,
    report_term_no: Optional[int],
) -> tuple[
    dict[int, int],
    str,
]:
    terms = []

    for header in headers:
        terms.extend(
            int(value)
            for value in FISCAL_TERM_PATTERN.findall(
                clean_text(header)
            )
        )

    unique_terms = sorted(
        set(terms)
    )

    if report_term_no is not None:
        mapping = {}

        for term in unique_terms:
            year = (
                source_report_year
                - (
                    report_term_no
                    - term
                )
            )

            # Reject implausible mapping rather than silently forcing it.
            if 2000 <= year <= source_report_year + 1:
                mapping[term] = year

        return (
            mapping,
            "document_report_term",
        )

    if unique_terms:
        max_term = max(
            unique_terms
        )

        return (
            {
                term: (
                    source_report_year
                    - (
                        max_term
                        - term
                    )
                )
                for term in unique_terms
            },
            "local_max_term_fallback",
        )

    return (
        {},
        "no_term",
    )


def infer_period_from_text(
    text: str,
    source_report_year: int,
    fiscal_term_map: Optional[
        dict[int, int]
    ] = None,
) -> tuple[
    Optional[str],
    Optional[int],
    str,
    float,
]:
    raw = clean_text(text)

    if not raw:
        return (
            None,
            None,
            "unknown",
            0.0,
        )

    years = [
        int(year)
        for year in YEAR_PATTERN.findall(raw)
        if 2000 <= int(year) <= source_report_year + 1
    ]

    if years:
        # Header can include multiple dates from the same year.
        # Use the last year token only after de-duplicating consecutive values.
        return (
            raw,
            years[-1],
            "explicit_year",
            1.0,
        )

    normalized = compact(raw)

    if "전전기" in normalized:
        return (
            raw,
            source_report_year - 2,
            "relative_period",
            0.95,
        )

    if "전기" in normalized:
        return (
            raw,
            source_report_year - 1,
            "relative_period",
            0.95,
        )

    if any(
        token in normalized
        for token in [
            "당기",
            "당년도",
            "당해연도",
            "당해년도",
            "기말",
        ]
    ):
        return (
            raw,
            source_report_year,
            "relative_period",
            0.95,
        )

    match = FISCAL_TERM_PATTERN.search(
        raw
    )

    if (
        match
        and fiscal_term_map
    ):
        term = int(
            match.group(1)
        )

        if term in fiscal_term_map:
            return (
                raw,
                fiscal_term_map[term],
                "fiscal_term_map",
                0.90,
            )

    return (
        raw,
        None,
        "unknown",
        0.0,
    )


# ==============================================================================
# 8. TABLE HEADER ANALYSIS
# ==============================================================================

def is_generic_columns(
    columns: list[str],
) -> bool:
    if not columns:
        return True

    generic = sum(
        (
            not clean_text(column)
            or bool(
                GENERIC_COLUMN_PATTERN.match(
                    clean_text(column)
                )
            )
        )
        for column in columns
    )

    return (
        generic
        >= max(
            1,
            len(columns) - 1,
        )
    )


def is_period_token(text: str) -> bool:
    text = clean_text(text)

    if not text:
        return False

    return bool(
        YEAR_PATTERN.search(text)
        or FISCAL_TERM_PATTERN.search(text)
        or any(
            token in compact(text)
            for token in [
                "당기",
                "전기",
                "전전기",
                "당년도",
                "당해연도",
                "기말",
            ]
        )
    )


def is_pure_period_cell(text: str) -> bool:
    text = clean_text(text)

    if not text:
        return False

    if re.fullmatch(
        r"\(?\s*20\d{2}\s*\)?",
        text,
    ):
        return True

    if re.fullmatch(
        r"\s*제\s*\d+\s*기\s*",
        text,
    ):
        return True

    return False


def looks_measurement_number(text: str) -> bool:
    value, _, _ = extract_number_and_explicit_unit(
        text
    )

    if value is None:
        return False

    if is_pure_period_cell(
        text
    ):
        return False

    return True


def is_preamble_row(
    row: list[str],
) -> bool:
    values = [
        clean_text(cell)
        for cell in row
        if clean_text(cell)
    ]

    if not values:
        return True

    joined = " ".join(values)

    if extract_unit_from_text(
        joined
    ):
        return True

    # A single short label at top is usually table title/caption.
    if len(values) == 1:
        normalized = compact(
            values[0]
        )

        if any(
            compact(term) in normalized
            for term in TITLE_LIKE_TERMS
        ):
            return True

        if (
            len(values[0]) <= 80
            and not looks_measurement_number(
                values[0]
            )
        ):
            return True

    return False


def row_has_body_shape(
    row: list[str],
) -> bool:
    measurement_count = sum(
        looks_measurement_number(cell)
        for cell in row
    )

    textual_count = sum(
        (
            bool(clean_text(cell))
            and not looks_measurement_number(cell)
            and not is_period_token(cell)
        )
        for cell in row
    )

    return (
        measurement_count >= 1
        and textual_count >= 1
    )


def forward_fill_period_row(
    row: list[str],
) -> list[str]:
    """
    Forward-fill only a header row containing period tokens.
    This reconstructs merged period headers without forward-filling arbitrary
    body cells.
    """
    if not any(
        is_period_token(cell)
        for cell in row
    ):
        return list(row)

    result = []
    last_period = ""

    for cell in row:
        cell = clean_text(cell)

        if cell:
            if is_period_token(cell):
                last_period = cell

            result.append(cell)

        else:
            result.append(
                last_period
            )

    return result


def dedup_join(parts: list[str]) -> str:
    result = []

    for part in parts:
        part = clean_text(part)

        if (
            part
            and (
                not result
                or result[-1] != part
            )
        ):
            result.append(part)

    return " / ".join(result)


def build_column_headers(
    header_rows: list[list[str]],
    width: int,
) -> list[str]:
    if not header_rows:
        return [
            f"col_{idx}"
            for idx in range(width)
        ]

    prepared = [
        forward_fill_period_row(row)
        for row in header_rows
    ]

    headers = []

    for column_index in range(width):
        parts = []

        for row in prepared:
            if column_index < len(row):
                value = clean_text(
                    row[column_index]
                )

                if value:
                    parts.append(value)

        header = dedup_join(parts)

        headers.append(
            header
            if header
            else f"col_{column_index}"
        )

    return headers


def infer_header_structure(
    columns: list[str],
    rows: list[list[str]],
) -> dict[str, Any]:
    rows = normalize_rows(rows)

    if not rows:
        return {
            "preamble_rows": [],
            "header_rows": [],
            "body_rows": [],
            "header_depth": 0,
            "column_headers": columns,
            "flags": [
                "empty_table",
            ],
        }

    width = len(rows[0])
    generic = is_generic_columns(
        columns
    )

    preamble_rows = []
    header_rows = []
    flags = []

    index = 0

    # Up to 3 title/unit/caption rows.
    while (
        index < min(3, len(rows))
        and is_preamble_row(
            rows[index]
        )
    ):
        preamble_rows.append(
            rows[index]
        )
        index += 1

    if generic:
        # Up to 4 header rows. Stop at the first strong measurement row.
        while (
            index < len(rows)
            and len(header_rows) < 4
        ):
            row = rows[index]

            if row_has_body_shape(row):
                break

            values = [
                clean_text(cell)
                for cell in row
                if clean_text(cell)
            ]

            if not values:
                header_rows.append(row)
                index += 1
                continue

            has_period = any(
                is_period_token(cell)
                for cell in row
            )

            generic_labels_norm = {
                normalize_metric_name(label)
                for label in GENERIC_LABELS
            }

            has_generic_label = any(
                normalize_metric_name(cell)
                in generic_labels_norm
                for cell in values
            )

            mostly_text = (
                sum(
                    not looks_measurement_number(cell)
                    for cell in values
                )
                / len(values)
                >= 0.60
            )

            if (
                has_period
                or has_generic_label
                or mostly_text
            ):
                header_rows.append(row)
                index += 1
                continue

            break

    body_rows = rows[index:]

    if generic:
        column_headers = build_column_headers(
            header_rows,
            width,
        )
    else:
        column_headers = [
            clean_text(column)
            for column in columns
        ]

    if not header_rows:
        flags.append(
            "header_not_detected"
        )

    return {
        "preamble_rows": preamble_rows,
        "header_rows": header_rows,
        "body_rows": body_rows,
        "header_depth": (
            len(preamble_rows)
            + len(header_rows)
        ),
        "column_headers": column_headers,
        "flags": flags,
    }


# ==============================================================================
# 9. SCOPE / STATEMENT NORMALIZATION
# ==============================================================================

def infer_scope_from_context(
    block: dict[str, Any],
    group_scope: Optional[str] = None,
) -> tuple[str, str]:
    hint = clean_text(
        block.get("scope_hint")
    )

    section_path = [
        clean_text(value)
        for value in (
            block.get("section_path")
            or []
        )
    ]

    # Strong explicit heading evidence first.
    combined = clean_text(
        " ".join(section_path)
        + " "
        + clean_text(
            block.get("local_context")
        )
    ).lower()

    if any(
        token in combined
        for token in [
            "연결재무제표",
            "연결 재무제표",
            "연결재무상태표",
            "연결 재무상태표",
            "연결손익계산서",
            "연결 손익계산서",
            "연결포괄손익계산서",
            "연결현금흐름표",
            "연결 현금흐름표",
            "연결자본변동표",
            "(연결)",
        ]
    ):
        return (
            "consolidated",
            "explicit_consolidated_context",
        )

    # DART annual reports commonly use 4. 재무제표 / 5. 재무제표 주석
    # for separate statements. Inspect newest headings first.
    for heading in reversed(section_path):
        normalized = compact(heading)

        if (
            re.match(
                r"4[\.\-]?재무제표$",
                normalized,
            )
            or re.match(
                r"5[\.\-]?재무제표주석$",
                normalized,
            )
        ):
            return (
                "separate",
                "numbered_separate_heading",
            )

    if hint in VALID_SCOPES:
        return (
            hint,
            "script01_scope_hint",
        )

    if group_scope in VALID_SCOPES:
        return (
            group_scope,
            "table_group_inheritance",
        )

    return (
        "unknown",
        "unresolved",
    )


def infer_statement_from_context(
    block: dict[str, Any],
    group_statement: Optional[str] = None,
) -> tuple[str, str]:
    text = clean_text(
        " ".join(
            block.get("section_path")
            or []
        )
        + " "
        + clean_text(
            block.get("local_context")
        )
        + " "
        + clean_text(
            block.get("raw_content")
        )[:800]
    ).lower()

    ordered = [
        (
            "현금흐름표",
            "CF",
        ),
        (
            "자본변동표",
            "SCE",
        ),
        (
            "포괄손익계산서",
            "CIS",
        ),
        (
            "손익계산서",
            "IS",
        ),
        (
            "재무상태표",
            "BS",
        ),
    ]

    for keyword, code in ordered:
        if keyword in text:
            return (
                code,
                "explicit_statement_context",
            )

    hint = clean_text(
        block.get("statement_hint")
    )

    if hint:
        return (
            hint,
            "script01_statement_hint",
        )

    if group_statement:
        return (
            group_statement,
            "table_group_inheritance",
        )

    return (
        "OTHER",
        "unresolved",
    )


# ==============================================================================
# 10. MULTI-PAGE TABLE GROUP CONTEXT
# ==============================================================================

def choose_group_header_profile(
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = []

    for member in members:
        rows = normalize_rows(
            member.get("table_rows")
        )

        columns = [
            clean_text(column)
            for column in (
                member.get("table_columns")
                or []
            )
        ]

        if not rows:
            continue

        info = infer_header_structure(
            columns,
            rows,
        )

        headers = info[
            "column_headers"
        ]

        resolved_period_headers = sum(
            is_period_token(header)
            for header in headers
        )

        candidates.append(
            {
                "page": safe_int(
                    member.get("page")
                ),
                "width": len(rows[0]),
                "header_rows": info[
                    "header_rows"
                ],
                "column_headers": headers,
                "header_depth": info[
                    "header_depth"
                ],
                "resolved_period_headers": (
                    resolved_period_headers
                ),
            }
        )

    if not candidates:
        return {
            "width": 0,
            "column_headers": [],
            "header_rows": [],
            "source_page": None,
        }

    best = sorted(
        candidates,
        key=lambda item: (
            -item[
                "resolved_period_headers"
            ],
            -len(
                item[
                    "header_rows"
                ]
            ),
            item[
                "page"
            ],
        ),
    )[0]

    return {
        "width": best[
            "width"
        ],
        "column_headers": best[
            "column_headers"
        ],
        "header_rows": best[
            "header_rows"
        ],
        "source_page": best[
            "page"
        ],
    }


def build_table_group_contexts(
    blocks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)

    for block in blocks:
        if block.get("modality") != "table":
            continue

        group_id = (
            clean_text(
                block.get("table_group_id")
            )
            or clean_text(
                block.get("table_id")
            )
            or block[
                "block_id"
            ]
        )

        grouped[group_id].append(
            block
        )

    result = {}

    for group_id, members in grouped.items():
        members = sorted(
            members,
            key=lambda item: (
                safe_int(
                    item.get("page")
                ),
                safe_int(
                    item.get("table_index")
                ),
            ),
        )

        scope_values = [
            clean_text(
                member.get("scope_hint")
            )
            for member in members
            if clean_text(
                member.get("scope_hint")
            ) in VALID_SCOPES
        ]

        statement_values = [
            clean_text(
                member.get("statement_hint")
            )
            for member in members
            if clean_text(
                member.get("statement_hint")
            )
        ]

        unit_candidates = []

        for member in members:
            for text in [
                clean_text(
                    member.get("local_context")
                ),
                clean_text(
                    member.get("raw_content")
                )[:2000],
            ]:
                unit = extract_unit_from_text(
                    text
                )

                if unit:
                    unit_candidates.append(
                        unit
                    )

        unique_units = list(
            dict.fromkeys(
                unit_candidates
            )
        )

        result[group_id] = {
            "group_id": group_id,
            "pages": [
                safe_int(
                    member.get("page")
                )
                for member in members
            ],
            "scope": (
                Counter(
                    scope_values
                ).most_common(1)[0][0]
                if scope_values
                else "unknown"
            ),
            "statement_type": (
                Counter(
                    statement_values
                ).most_common(1)[0][0]
                if statement_values
                else "OTHER"
            ),
            "unit": (
                unique_units[0]
                if len(unique_units) == 1
                else None
            ),
            "mixed_unit_context": (
                len(unique_units) > 1
            ),
            "header_profile": choose_group_header_profile(
                members
            ),
        }

    return result


# ==============================================================================
# 11. TABLE COLUMN / ROW LOGIC
# ==============================================================================

def is_note_column(header: str) -> bool:
    return bool(
        NOTE_HEADER_PATTERN.search(
            clean_text(header)
        )
    )


def numeric_value_for_profile(
    text: str,
) -> Optional[float]:
    value, _, _ = extract_number_and_explicit_unit(
        text
    )

    return value


def detect_note_columns(
    body_rows: list[list[str]],
    headers: list[str],
    primary_financial: bool,
) -> set[int]:
    note_columns = {
        index
        for index, header in enumerate(headers)
        if is_note_column(header)
    }

    if not primary_financial:
        return note_columns

    if not body_rows:
        return note_columns

    width = len(body_rows[0])

    # Structural fallback for statements where '주석' text was lost.
    for column_index in range(1, width - 1):
        values = []

        for row in body_rows[:120]:
            if column_index >= len(row):
                continue

            value = numeric_value_for_profile(
                row[column_index]
            )

            if value is not None:
                values.append(value)

        if len(values) < 5:
            continue

        small_integer_ratio = (
            sum(
                (
                    abs(value) <= 250
                    and float(value).is_integer()
                )
                for value in values
            )
            / len(values)
        )

        # A later column should behave like a real amount column.
        next_values = []

        for row in body_rows[:120]:
            if column_index + 1 >= len(row):
                continue

            value = numeric_value_for_profile(
                row[column_index + 1]
            )

            if value is not None:
                next_values.append(
                    abs(value)
                )

        median_next = (
            pd.Series(
                next_values
            ).median()
            if next_values
            else 0
        )

        if (
            small_integer_ratio >= 0.80
            and median_next >= 1_000
        ):
            note_columns.add(
                column_index
            )

    return note_columns


def row_metric_path(
    row: list[str],
    headers: list[str],
    note_columns: set[int],
) -> tuple[
    Optional[str],
    Optional[str],
    int,
]:
    measurement_columns = []

    for index, cell in enumerate(row):
        if index in note_columns:
            continue

        if looks_measurement_number(
            cell
        ):
            measurement_columns.append(
                index
            )

    if not measurement_columns:
        return (
            None,
            None,
            -1,
        )

    first_measurement = min(
        measurement_columns
    )

    generic_norm = {
        normalize_metric_name(label)
        for label in GENERIC_LABELS
    }

    textual_parts = []

    for index in range(
        first_measurement
    ):
        if index in note_columns:
            continue

        cell = clean_text(
            row[index]
        )

        if not cell:
            continue

        if (
            looks_measurement_number(cell)
            or is_period_token(cell)
        ):
            continue

        normalized = normalize_metric_name(
            cell
        )

        if (
            not normalized
            or normalized in generic_norm
        ):
            continue

        textual_parts.append(
            cell
        )

    if not textual_parts:
        return (
            None,
            None,
            first_measurement,
        )

    metric_raw = textual_parts[-1]
    metric_path = " > ".join(
        textual_parts
    )

    return (
        metric_raw,
        metric_path,
        first_measurement,
    )


def resolve_unit_for_value(
    cell: str,
    row: list[str],
    column_header: str,
    block: dict[str, Any],
    group_context: dict[str, Any],
) -> tuple[
    Optional[str],
    Optional[str],
    str,
]:
    _, _, explicit = extract_number_and_explicit_unit(
        cell
    )

    if explicit:
        return (
            explicit,
            explicit,
            "explicit_cell",
        )

    header_unit = extract_unit_from_text(
        column_header
    )

    if header_unit:
        return (
            header_unit,
            header_unit,
            "column_header",
        )

    row_unit = extract_unit_from_text(
        " ".join(row)
    )

    if row_unit:
        return (
            row_unit,
            row_unit,
            "row_context",
        )

    local_text = (
        clean_text(
            block.get("local_context")
        )
        + "\n"
        + clean_text(
            block.get("raw_content")
        )[:1500]
    )

    table_unit = extract_unit_from_text(
        local_text
    )

    if table_unit:
        return (
            table_unit,
            table_unit,
            "table_context",
        )

    group_unit = group_context.get(
        "unit"
    )

    if group_unit:
        return (
            group_unit,
            group_unit,
            "table_group_context",
        )

    return (
        None,
        None,
        "unknown",
    )


def parser_risk_flags(
    block: dict[str, Any],
) -> list[str]:
    flags = []

    family_support = safe_int(
        block.get("table_family_support")
    )

    content_agreement = safe_float(
        block.get("table_content_agreement")
    )

    selection_score = safe_float(
        block.get("table_selection_score")
    )

    qc_priority = clean_text(
        block.get("table_qc_priority")
    )

    if qc_priority == "HIGH":
        flags.append(
            "parser_qc_high"
        )

    elif qc_priority == "MEDIUM":
        flags.append(
            "parser_qc_medium"
        )

    if family_support <= 1:
        flags.append(
            "single_parser_family"
        )

    if (
        content_agreement > 0
        and content_agreement < LOW_CONTENT_AGREEMENT
    ):
        flags.append(
            "low_parser_content_agreement"
        )

    if (
        selection_score > 0
        and selection_score < LOW_SELECTION_SCORE
    ):
        flags.append(
            "low_table_selection_score"
        )

    if as_bool(
        block.get("parsing_review_required")
    ):
        flags.append(
            "script01_parser_review"
        )

    return flags


# ==============================================================================
# 12. TABLE STRUCTURING
# ==============================================================================

def structure_table_block(
    block: dict[str, Any],
    group_contexts: dict[str, dict[str, Any]],
    document_term_info: dict[str, dict[str, Any]],
    include_unknown_nonfinancial: bool,
) -> tuple[
    list[EvidenceRecord],
    dict[str, Any],
]:
    rows = normalize_rows(
        block.get("table_rows")
    )

    if not rows:
        return (
            [],
            {
                "block_id": block.get("block_id"),
                "status": "empty_table",
            },
        )

    columns = [
        clean_text(column)
        for column in (
            block.get("table_columns")
            or []
        )
    ]

    if not columns:
        columns = [
            f"col_{index}"
            for index in range(
                len(rows[0])
            )
        ]

    local_header = infer_header_structure(
        columns,
        rows,
    )

    table_group_id = (
        clean_text(
            block.get("table_group_id")
        )
        or clean_text(
            block.get("table_id")
        )
        or block["block_id"]
    )

    group_context = group_contexts.get(
        table_group_id,
        {},
    )

    group_header = group_context.get(
        "header_profile",
        {},
    )

    use_inherited_header = False
    column_headers = list(
        local_header[
            "column_headers"
        ]
    )

    body_rows = list(
        local_header[
            "body_rows"
        ]
    )

    header_depth = safe_int(
        local_header[
            "header_depth"
        ]
    )

    local_has_period_header = any(
        is_period_token(header)
        for header in column_headers
    )

    inherited_headers = group_header.get(
        "column_headers"
    ) or []

    # Continuation page may start directly with data. Inherit the group header
    # only if table width agrees exactly.
    if (
        as_bool(
            block.get("continued_from_previous")
        )
        and not local_has_period_header
        and inherited_headers
        and len(inherited_headers) == len(rows[0])
    ):
        column_headers = [
            clean_text(header)
            for header in inherited_headers
        ]

        use_inherited_header = True

        # If local parser treated first data rows as headers because the table
        # continued without headings, restore all original rows as body rows.
        if local_header[
            "header_depth"
        ] > 0:
            body_rows = rows
            header_depth = 0

    source_report_year = safe_int(
        block.get("source_report_year")
    )

    document_info = document_term_info.get(
        block["document_id"],
        {},
    )

    report_term_no = document_info.get(
        "report_term_no"
    )

    fiscal_term_map, term_map_method = build_fiscal_term_map(
        column_headers,
        source_report_year,
        report_term_no,
    )

    scope, scope_method = infer_scope_from_context(
        block,
        group_context.get("scope"),
    )

    statement_type, statement_method = infer_statement_from_context(
        block,
        group_context.get("statement_type"),
    )

    is_financial_core = as_bool(
        block.get("is_financial_core")
    )

    primary_financial = (
        is_financial_core
        and statement_type in PRIMARY_FINANCIAL_STATEMENTS
    )

    note_columns = detect_note_columns(
        body_rows,
        column_headers,
        primary_financial,
    )

    parser_flags = parser_risk_flags(
        block
    )

    evidence = []

    skipped_rows = 0
    skipped_values = 0

    for body_index, row in enumerate(
        body_rows
    ):
        metric_raw, metric_path_raw, first_measurement = row_metric_path(
            row,
            column_headers,
            note_columns,
        )

        if metric_raw is None:
            skipped_rows += 1
            continue

        metric_norm = normalize_metric_name(
            metric_raw
        )

        if not metric_norm:
            skipped_rows += 1
            continue

        metric_canonical = canonical_metric(
            metric_raw
        )

        # Whole-report policy:
        # - Financial tables: retain unmapped metrics.
        # - Nonfinancial tables: known metric OR explicit measurement structure.
        if (
            not is_financial_core
            and metric_canonical is None
            and not include_unknown_nonfinancial
        ):
            numeric_after_metric = sum(
                looks_measurement_number(
                    cell
                )
                for index, cell in enumerate(row)
                if (
                    index >= first_measurement
                    and index not in note_columns
                )
            )

            if numeric_after_metric < 2:
                skipped_rows += 1
                continue

        absolute_row_index = (
            header_depth
            + body_index
        )

        for column_index, cell in enumerate(
            row
        ):
            if column_index < first_measurement:
                continue

            if column_index in note_columns:
                continue

            value, raw_number, _ = extract_number_and_explicit_unit(
                cell
            )

            if value is None:
                skipped_values += 1
                continue

            # Header-like numbers are not measurements.
            if is_pure_period_cell(
                cell
            ):
                skipped_values += 1
                continue

            column_header = (
                column_headers[
                    column_index
                ]
                if column_index < len(
                    column_headers
                )
                else f"col_{column_index}"
            )

            (
                period_raw,
                period_year,
                period_method,
                period_confidence,
            ) = infer_period_from_text(
                column_header,
                source_report_year,
                fiscal_term_map,
            )

            # A single-value-column primary statement can be current period,
            # but only as a low-confidence fallback.
            measurement_columns = [
                index
                for index in range(
                    first_measurement,
                    len(column_headers),
                )
                if index not in note_columns
            ]

            if (
                period_year is None
                and primary_financial
                and len(measurement_columns) == 1
            ):
                period_raw = column_header
                period_year = source_report_year
                period_method = "single_amount_column_fallback"
                period_confidence = 0.60

            (
                unit_raw,
                unit,
                unit_method,
            ) = resolve_unit_for_value(
                cell=cell,
                row=row,
                column_header=column_header,
                block=block,
                group_context=group_context,
            )

            quality_flags = list(
                parser_flags
            )

            if period_year is None:
                quality_flags.append(
                    "period_unknown"
                )

            if unit is None:
                quality_flags.append(
                    "unit_unknown"
                )

            if (
                is_financial_core
                and scope not in VALID_SCOPES
            ):
                quality_flags.append(
                    "scope_unknown"
                )

            if use_inherited_header:
                quality_flags.append(
                    "inherited_multipage_header"
                )

            if group_context.get(
                "mixed_unit_context"
            ):
                quality_flags.append(
                    "mixed_unit_context"
                )

            if (
                term_map_method
                == "local_max_term_fallback"
                and FISCAL_TERM_PATTERN.search(
                    column_header
                )
            ):
                quality_flags.append(
                    "local_fiscal_term_mapping"
                )

            confidence = 0.90

            if period_year is None:
                confidence -= 0.18
            else:
                confidence += (
                    0.04
                    * period_confidence
                )

            if unit is None:
                confidence -= 0.12

            if (
                is_financial_core
                and scope not in VALID_SCOPES
            ):
                confidence -= 0.12

            qc_priority = clean_text(
                block.get("table_qc_priority")
            )

            if qc_priority == "MEDIUM":
                confidence -= 0.05

            elif qc_priority == "HIGH":
                confidence -= 0.16

            family_support = safe_int(
                block.get("table_family_support")
            )

            if family_support >= 2:
                confidence += 0.04

            content_agreement = safe_float(
                block.get("table_content_agreement")
            )

            if content_agreement >= 0.80:
                confidence += 0.03

            if use_inherited_header:
                confidence -= 0.04

            confidence = round(
                max(
                    0.0,
                    min(
                        1.0,
                        confidence,
                    ),
                ),
                3,
            )

            value_krw = value_to_krw(
                value,
                unit,
            )

            dart_alignable = (
                primary_financial
                and scope in VALID_SCOPES
                and period_year is not None
                and value_krw is not None
            )

            alignment_metric_key = (
                metric_canonical
                if metric_canonical
                else metric_norm
            )

            alignment_key = None

            if dart_alignable:
                # IMPORTANT: value is intentionally excluded.
                alignment_key = "||".join(
                    [
                        clean_text(
                            block.get("entity")
                        ),
                        str(
                            source_report_year
                        ),
                        str(
                            period_year
                        ),
                        scope,
                        statement_type,
                        alignment_metric_key,
                    ]
                )

            evidence_text = (
                f"{metric_path_raw} | "
                f"{column_header}: {cell}"
            )

            evidence_id = stable_id(
                block["block_id"],
                absolute_row_index,
                column_index,
                metric_norm,
                clean_text(cell),
                prefix="ev_",
            )

            needs_review = (
                as_bool(
                    block.get("parsing_review_required")
                )
                or confidence < 0.80
                or (
                    primary_financial
                    and (
                        period_year is None
                        or unit is None
                        or scope not in VALID_SCOPES
                    )
                )
                or "parser_qc_high" in quality_flags
            )

            evidence.append(
                EvidenceRecord(
                    evidence_id=evidence_id,
                    block_id=block["block_id"],
                    document_id=block["document_id"],

                    entity=clean_text(
                        block.get("entity")
                    ),
                    stock_code=clean_text(
                        block.get("stock_code")
                    ),

                    metric_raw=metric_raw,
                    metric_path_raw=(
                        metric_path_raw
                        or metric_raw
                    ),
                    metric_norm=metric_norm,
                    metric_canonical=metric_canonical,

                    value_raw=clean_text(cell),
                    value=value,

                    unit_raw=unit_raw,
                    unit=unit,
                    unit_scale=CURRENCY_SCALES.get(
                        unit
                    ),
                    value_krw=value_krw,
                    unit_inference_method=unit_method,

                    period_raw=period_raw,
                    period_year=period_year,
                    period_inference_method=period_method,
                    period_confidence=period_confidence,

                    scope=scope,
                    scope_inference_method=scope_method,

                    source_report_year=source_report_year,
                    filing_date=clean_text(
                        block.get("filing_date")
                    ),
                    filing_version=clean_text(
                        block.get("filing_version")
                    ),
                    version_id=clean_text(
                        block.get("version_id")
                    ),

                    major_section=clean_text(
                        block.get("major_section")
                    ),
                    section_path=[
                        clean_text(value)
                        for value in (
                            block.get("section_path")
                            or []
                        )
                    ],
                    is_financial_core=is_financial_core,

                    modality="table",
                    statement_type=statement_type,
                    statement_inference_method=(
                        statement_method
                    ),

                    page=safe_int(
                        block.get("page")
                    ),
                    bbox_pdf=block.get("bbox_pdf"),
                    source_file=clean_text(
                        block.get("source_file")
                    ),

                    evidence_text=evidence_text,

                    table_index=(
                        safe_int(
                            block.get("table_index"),
                            default=-1,
                        )
                        if block.get("table_index")
                        is not None
                        else None
                    ),
                    table_id=clean_text(
                        block.get("table_id")
                    ) or None,
                    table_group_id=(
                        table_group_id
                        or None
                    ),

                    row_index=absolute_row_index,
                    column_index=column_index,
                    column_header=column_header,

                    continued_from_previous=as_bool(
                        block.get("continued_from_previous")
                    ),
                    continues_next=as_bool(
                        block.get("continues_next")
                    ),
                    repeated_header_on_continuation=as_bool(
                        block.get(
                            "repeated_header_on_continuation"
                        )
                    ),

                    parser_engine=clean_text(
                        block.get("table_extraction_engine")
                    ) or "unknown",
                    parser_strategy=clean_text(
                        block.get("table_extraction_strategy")
                    ) or "unknown",

                    table_family_support=family_support,
                    table_candidate_support=safe_int(
                        block.get("table_candidate_support")
                    ),
                    table_content_agreement=content_agreement,
                    table_selection_score=safe_float(
                        block.get("table_selection_score")
                    ),

                    table_qc_score=safe_int(
                        block.get("table_qc_score")
                    ),
                    table_qc_priority=(
                        clean_text(
                            block.get("table_qc_priority")
                        )
                        or "LOW"
                    ),
                    table_qc_flags=json_list(
                        block.get("table_qc_flags")
                    ),
                    parsing_review_required=as_bool(
                        block.get("parsing_review_required")
                    ),

                    extraction_method=(
                        "table_metric_value_"
                        "multirow_header_v2"
                    ),
                    confidence=confidence,
                    quality_flags=list(
                        dict.fromkeys(
                            quality_flags
                        )
                    ),
                    needs_review=needs_review,

                    dart_alignable=dart_alignable,
                    alignment_metric_key=(
                        alignment_metric_key
                    ),
                    alignment_key=alignment_key,
                )
            )

    diagnostic_flags = list(
        local_header.get(
            "flags",
            [],
        )
    )

    if use_inherited_header:
        diagnostic_flags.append(
            "group_header_inherited"
        )

    if not any(
        is_period_token(header)
        for header in column_headers
    ):
        diagnostic_flags.append(
            "no_period_header_signal"
        )

    if group_context.get(
        "mixed_unit_context"
    ):
        diagnostic_flags.append(
            "mixed_group_units"
        )

    diagnostic = {
        "block_id": block.get("block_id"),
        "document_id": block.get("document_id"),
        "entity": block.get("entity"),
        "source_report_year": source_report_year,
        "source_file": block.get("source_file"),
        "page": block.get("page"),
        "table_index": block.get("table_index"),
        "table_id": block.get("table_id"),
        "table_group_id": table_group_id,
        "continued_from_previous": as_bool(
            block.get("continued_from_previous")
        ),
        "repeated_header_on_continuation": as_bool(
            block.get("repeated_header_on_continuation")
        ),
        "row_count_raw": len(rows),
        "column_count_raw": len(rows[0]),
        "preamble_row_count": len(
            local_header[
                "preamble_rows"
            ]
        ),
        "header_row_count": len(
            local_header[
                "header_rows"
            ]
        ),
        "body_row_count": len(
            body_rows
        ),
        "column_headers": json.dumps(
            column_headers,
            ensure_ascii=False,
        ),
        "header_inherited": use_inherited_header,
        "report_term_no": report_term_no,
        "report_term_method": document_info.get(
            "method",
            "unresolved",
        ),
        "fiscal_term_map": json.dumps(
            fiscal_term_map,
            ensure_ascii=False,
        ),
        "fiscal_term_map_method": term_map_method,
        "scope": scope,
        "scope_method": scope_method,
        "statement_type": statement_type,
        "statement_method": statement_method,
        "note_columns": json.dumps(
            sorted(
                note_columns
            ),
            ensure_ascii=False,
        ),
        "group_unit": group_context.get(
            "unit"
        ),
        "parser_engine": block.get(
            "table_extraction_engine"
        ),
        "parser_strategy": block.get(
            "table_extraction_strategy"
        ),
        "parser_family_support": block.get(
            "table_family_support"
        ),
        "parser_content_agreement": block.get(
            "table_content_agreement"
        ),
        "parser_selection_score": block.get(
            "table_selection_score"
        ),
        "parser_qc_priority": block.get(
            "table_qc_priority"
        ),
        "structured_evidence_count": len(
            evidence
        ),
        "dart_alignable_count": sum(
            item.dart_alignable
            for item in evidence
        ),
        "skipped_rows": skipped_rows,
        "skipped_values": skipped_values,
        "diagnostic_flags": "|".join(
            dict.fromkeys(
                diagnostic_flags
            )
        ),
    }

    return (
        evidence,
        diagnostic,
    )


# ==============================================================================
# 13. TEXT STRUCTURING
# ==============================================================================

def find_metric_mentions(
    text: str,
) -> list[
    tuple[
        str,
        str,
        int,
        int,
    ]
]:
    lower = text.lower()
    mentions = []

    occupied = []

    for alias, canonical in ALIAS_PAIRS:
        start = 0

        while True:
            position = lower.find(
                alias,
                start,
            )

            if position < 0:
                break

            end = position + len(alias)

            # Avoid nested alias duplicates such as 매출 inside 매출액.
            if not any(
                (
                    position >= left
                    and end <= right
                )
                for left, right in occupied
            ):
                mentions.append(
                    (
                        text[position:end],
                        canonical,
                        position,
                        end,
                    )
                )

                occupied.append(
                    (
                        position,
                        end,
                    )
                )

            start = end

    mentions.sort(
        key=lambda item: item[2]
    )

    return mentions


def sentence_window(
    text: str,
    left: int,
    right: int,
    radius: int = 180,
) -> str:
    start = max(
        0,
        left - radius,
    )

    end = min(
        len(text),
        right + radius,
    )

    return clean_text(
        text[start:end]
    )


def structure_text_block(
    block: dict[str, Any],
    document_term_info: dict[str, dict[str, Any]],
) -> list[EvidenceRecord]:
    text = clean_text(
        block.get("raw_content")
    )

    if not text:
        return []

    mentions = find_metric_mentions(
        text
    )

    amounts = list(
        TEXT_AMOUNT_PATTERN.finditer(
            text
        )
    )

    if (
        not mentions
        or not amounts
    ):
        return []

    source_report_year = safe_int(
        block.get("source_report_year")
    )

    document_info = document_term_info.get(
        block["document_id"],
        {},
    )

    report_term_no = document_info.get(
        "report_term_no"
    )

    local_terms = [
        int(value)
        for value in FISCAL_TERM_PATTERN.findall(
            text
        )
    ]

    if report_term_no is not None:
        term_map = {
            term: (
                source_report_year
                - (
                    report_term_no
                    - term
                )
            )
            for term in set(
                local_terms
            )
        }
    else:
        term_map = {}

    scope, scope_method = infer_scope_from_context(
        block
    )

    statement_type, statement_method = infer_statement_from_context(
        block
    )

    is_financial_core = as_bool(
        block.get("is_financial_core")
    )

    evidence = []
    used_amount_spans = set()

    for (
        metric_raw,
        canonical,
        metric_start,
        metric_end,
    ) in mentions:
        candidates = []

        for amount in amounts:
            distance = min(
                abs(
                    amount.start()
                    - metric_end
                ),
                abs(
                    metric_start
                    - amount.end()
                ),
            )

            if distance <= TEXT_METRIC_VALUE_MAX_DISTANCE:
                candidates.append(
                    (
                        distance,
                        amount,
                    )
                )

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: item[0]
        )

        distance, amount = candidates[0]

        amount_span = (
            amount.start(),
            amount.end(),
        )

        if amount_span in used_amount_spans:
            continue

        used_amount_spans.add(
            amount_span
        )

        value = parse_number(
            amount.group("num")
        )

        if value is None:
            continue

        unit_raw = clean_text(
            amount.group("unit")
        )

        unit = normalize_unit(
            unit_raw
        )

        evidence_text = sentence_window(
            text,
            min(
                metric_start,
                amount.start(),
            ),
            max(
                metric_end,
                amount.end(),
            ),
        )

        (
            period_raw,
            period_year,
            period_method,
            period_confidence,
        ) = infer_period_from_text(
            evidence_text,
            source_report_year,
            term_map,
        )

        confidence = 0.74

        if distance <= 60:
            confidence += 0.08

        if period_year is not None:
            confidence += 0.05

        # Native text only, no OCR penalty needed.
        if clean_text(
            block.get("text_extraction_method")
        ) == "pymupdf_native":
            confidence += 0.04

        confidence = round(
            min(
                1.0,
                confidence,
            ),
            3,
        )

        quality_flags = []

        if period_year is None:
            quality_flags.append(
                "period_unknown"
            )

        if scope not in VALID_SCOPES and is_financial_core:
            quality_flags.append(
                "scope_unknown"
            )

        metric_norm = normalize_metric_name(
            metric_raw
        )

        evidence.append(
            EvidenceRecord(
                evidence_id=stable_id(
                    block["block_id"],
                    metric_raw,
                    amount.group(0),
                    metric_start,
                    prefix="ev_",
                ),
                block_id=block["block_id"],
                document_id=block["document_id"],

                entity=clean_text(
                    block.get("entity")
                ),
                stock_code=clean_text(
                    block.get("stock_code")
                ),

                metric_raw=metric_raw,
                metric_path_raw=metric_raw,
                metric_norm=metric_norm,
                metric_canonical=canonical,

                value_raw=clean_text(
                    amount.group(0)
                ),
                value=value,

                unit_raw=unit_raw,
                unit=unit,
                unit_scale=CURRENCY_SCALES.get(
                    unit
                ),
                value_krw=value_to_krw(
                    value,
                    unit,
                ),
                unit_inference_method="explicit_text",

                period_raw=period_raw,
                period_year=period_year,
                period_inference_method=period_method,
                period_confidence=period_confidence,

                scope=scope,
                scope_inference_method=scope_method,

                source_report_year=source_report_year,
                filing_date=clean_text(
                    block.get("filing_date")
                ),
                filing_version=clean_text(
                    block.get("filing_version")
                ),
                version_id=clean_text(
                    block.get("version_id")
                ),

                major_section=clean_text(
                    block.get("major_section")
                ),
                section_path=[
                    clean_text(value)
                    for value in (
                        block.get("section_path")
                        or []
                    )
                ],
                is_financial_core=is_financial_core,

                modality="text",
                statement_type=statement_type,
                statement_inference_method=statement_method,

                page=safe_int(
                    block.get("page")
                ),
                bbox_pdf=block.get("bbox_pdf"),
                source_file=clean_text(
                    block.get("source_file")
                ),

                evidence_text=evidence_text,

                table_index=None,
                table_id=None,
                table_group_id=None,

                row_index=None,
                column_index=None,
                column_header=None,

                continued_from_previous=False,
                continues_next=False,
                repeated_header_on_continuation=False,

                parser_engine="pymupdf_native",
                parser_strategy="native_text",

                table_family_support=0,
                table_candidate_support=0,
                table_content_agreement=0.0,
                table_selection_score=0.0,

                table_qc_score=0,
                table_qc_priority="N/A",
                table_qc_flags=[],
                parsing_review_required=False,

                extraction_method=(
                    "text_known_metric_nearest_explicit_amount"
                ),
                confidence=confidence,
                quality_flags=quality_flags,
                # Text evidence is useful for conflict research but is kept out
                # of automatic DART Gold alignment and remains reviewable.
                needs_review=True,

                dart_alignable=False,
                alignment_metric_key=(
                    canonical
                    if canonical
                    else metric_norm
                ),
                alignment_key=None,
            )
        )

    return evidence


# ==============================================================================
# 14. DEDUPLICATION / CSV SERIALIZATION
# ==============================================================================

def deduplicate_evidence(
    evidence: list[EvidenceRecord],
) -> list[EvidenceRecord]:
    seen = set()
    result = []

    for item in evidence:
        # source_report_year/version is intentionally in the key so that the
        # same FY2024 amount appearing in the FY2024 report and as prior-year
        # comparative in FY2025 remains TWO source evidences.
        key = (
            item.version_id,
            item.modality,
            item.page,
            item.table_id,
            item.row_index,
            item.column_index,
            item.metric_norm,
            round(
                item.value,
                8,
            ),
            item.unit,
            item.period_year,
            item.scope,
            item.statement_type,
            item.evidence_text,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def evidence_dataframe(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Make CSV machine-readable rather than relying on Python repr strings.
    for column in [
        "section_path",
        "table_qc_flags",
        "quality_flags",
    ]:
        if column in df.columns:
            df[column] = df[column].map(
                lambda value: json.dumps(
                    value,
                    ensure_ascii=False,
                )
                if isinstance(value, list)
                else clean_text(value)
            )

    if "bbox_pdf" in df.columns:
        df["bbox_pdf"] = df[
            "bbox_pdf"
        ].map(
            lambda value: json.dumps(
                value,
                ensure_ascii=False,
            )
            if isinstance(value, dict)
            else clean_text(value)
        )

    return df


# ==============================================================================
# 15. SCHEMA METADATA
# ==============================================================================

EVIDENCE_SCHEMA = {
    "research": "CAUSE-RAG",
    "core_conflict_axes": [
        "entity",
        "metric",
        "value",
        "unit",
        "period",
        "scope",
        "version",
    ],
    "version_fields": {
        "source_report_year": (
            "사업보고서 자체의 사업연도"
        ),
        "period_year": (
            "해당 evidence value가 의미하는 실제 기준연도"
        ),
        "filing_date": (
            "DART 제출일"
        ),
        "filing_version": (
            "original/corrected"
        ),
        "version_id": (
            "source report version identifier"
        ),
    },
    "metric_policy": {
        "financial_tables": (
            "Unmapped financial metrics are retained. "
            "metric_canonical is optional."
        ),
        "nonfinancial_tables": (
            "Known metrics are retained; unknown rows require a strong "
            "multi-value table structure unless CLI permissive flag is used."
        ),
        "text": (
            "Conservative known-metric + explicit-unit extraction only."
        ),
    },
    "parser_policy": {
        "ocr_used": False,
        "text": "PyMuPDF native",
        "table": "pdfplumber + img2table native ensemble",
        "parser_quality_propagated": True,
    },
    "multipage_policy": {
        "physical_merge": False,
        "table_group_id": (
            "Same logical table may span pages while page provenance remains."
        ),
        "header_inheritance": (
            "Only continuation members with exact matching width and no local "
            "period header inherit the group's strongest header profile."
        ),
    },
    "dart_alignment_policy": {
        "eligible_modality": "table",
        "eligible_section": "III. 재무에 관한 사항",
        "eligible_statements": sorted(
            PRIMARY_FINANCIAL_STATEMENTS
        ),
        "requires": [
            "scope in {consolidated,separate}",
            "period_year resolved",
            "currency unit resolved to KRW-scale",
        ],
        "alignment_key_excludes_value": True,
        "reason": (
            "Prevent DART target value from leaking into PDF evidence candidate "
            "selection in Script 03."
        ),
    },
}


# ==============================================================================
# 16. CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--include-unknown-nonfinancial",
        action="store_true",
        help=(
            "Keep unknown non-financial table metrics more permissively. "
            "Financial unknown metrics are always retained."
        ),
    )

    return parser.parse_args()


# ==============================================================================
# 17. MAIN
# ==============================================================================

def main() -> None:
    args = parse_args()

    if not INPUT_PATH.exists():
        raise SystemExit(
            "\n[ERROR] parsed_blocks.jsonl이 없습니다.\n"
            "먼저 최종 01_parse_business_reports.py를 실행하세요.\n"
            f"Expected: {INPUT_PATH}\n"
        )

    blocks = read_jsonl(
        INPUT_PATH
    )

    if not blocks:
        raise SystemExit(
            "[ERROR] parsed_blocks.jsonl이 비어 있습니다."
        )

    required_fields = {
        "block_id",
        "document_id",
        "entity",
        "stock_code",
        "source_report_year",
        "filing_date",
        "filing_version",
        "version_id",
        "major_section",
        "modality",
        "page",
        "raw_content",
    }

    missing_required = sorted(
        required_fields
        - set(
            blocks[0].keys()
        )
    )

    if missing_required:
        raise RuntimeError(
            "Script 01 output schema mismatch. Missing: "
            + ", ".join(
                missing_required
            )
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "="
        * 100
    )

    print(
        "CAUSE-RAG | FINAL Evidence Structuring"
    )

    print(
        "="
        * 100
    )

    print(
        f"Input blocks : {len(blocks)}"
    )

    print(
        "OCR          : not used"
    )

    print(
        "DART key     : value-excluded / non-leaky"
    )

    # ------------------------------------------------------------------
    # Global contexts
    # ------------------------------------------------------------------

    document_term_info = build_document_term_info(
        blocks
    )

    group_contexts = build_table_group_contexts(
        blocks
    )

    print(
        f"Documents    : {len(document_term_info)}"
    )

    for document_id, info in sorted(
        document_term_info.items(),
        key=lambda item: (
            item[1][
                "source_report_year"
            ],
            item[0],
        ),
    ):
        print(
            "  report term: "
            f"{document_id[:12]}... "
            f"FY={info['source_report_year']} "
            f"term={info['report_term_no']} "
            f"({info['method']})"
        )

    # ------------------------------------------------------------------
    # Evidence extraction
    # ------------------------------------------------------------------

    evidence: list[EvidenceRecord] = []
    diagnostics = []

    for index, block in enumerate(
        blocks,
        start=1,
    ):
        modality = clean_text(
            block.get("modality")
        )

        if modality == "table":
            table_evidence, diagnostic = structure_table_block(
                block=block,
                group_contexts=group_contexts,
                document_term_info=document_term_info,
                include_unknown_nonfinancial=(
                    args.include_unknown_nonfinancial
                ),
            )

            evidence.extend(
                table_evidence
            )

            diagnostics.append(
                diagnostic
            )

        elif modality == "text":
            evidence.extend(
                structure_text_block(
                    block=block,
                    document_term_info=(
                        document_term_info
                    ),
                )
            )

        if index % 1000 == 0:
            print(
                f"[PROGRESS] "
                f"{index}/{len(blocks)} blocks -> "
                f"{len(evidence)} evidence"
            )

    evidence = deduplicate_evidence(
        evidence
    )

    rows = [
        asdict(item)
        for item in evidence
    ]

    # JSONL keeps list/dict fields natively.
    write_jsonl(
        ALL_JSONL,
        rows,
    )

    df = evidence_dataframe(
        rows
    )

    df.to_csv(
        ALL_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Subsets
    # ------------------------------------------------------------------

    if df.empty:
        financial = df.copy()
        nonfinancial = df.copy()
        dart = df.copy()
        review = df.copy()

    else:
        financial = df[
            df[
                "is_financial_core"
            ] == True
        ].copy()

        nonfinancial = df[
            df[
                "is_financial_core"
            ] == False
        ].copy()

        dart = df[
            df[
                "dart_alignable"
            ] == True
        ].copy()

        review = df[
            df[
                "needs_review"
            ] == True
        ].copy()

    financial.to_csv(
        FINANCIAL_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    nonfinancial.to_csv(
        NONFINANCIAL_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    dart.to_csv(
        DART_CANDIDATES_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # Review order: parser-risk first, then low confidence.
    if not review.empty:
        priority_map = {
            "HIGH": 0,
            "MEDIUM": 1,
            "LOW": 2,
            "N/A": 3,
        }

        review[
            "_parser_priority"
        ] = (
            review[
                "table_qc_priority"
            ]
            .map(
                priority_map
            )
            .fillna(4)
        )

        review = (
            review
            .sort_values(
                [
                    "_parser_priority",
                    "confidence",
                    "entity",
                    "source_report_year",
                    "page",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            )
            .drop(
                columns=[
                    "_parser_priority"
                ]
            )
        )

    review.to_csv(
        REVIEW_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        diagnostics
    ).to_csv(
        TABLE_DIAGNOSTICS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        SCHEMA_PATH,
        EVIDENCE_SCHEMA,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    modality_counts = Counter(
        item.modality
        for item in evidence
    )

    parser_engine_counts = Counter(
        item.parser_engine
        for item in evidence
    )

    period_method_counts = Counter(
        item.period_inference_method
        for item in evidence
    )

    unit_method_counts = Counter(
        item.unit_inference_method
        for item in evidence
    )

    scope_counts = Counter(
        item.scope
        for item in evidence
    )

    summary = {
        "input_block_count": len(
            blocks
        ),
        "evidence_count": len(
            evidence
        ),
        "financial_evidence_count": len(
            financial
        ),
        "nonfinancial_evidence_count": len(
            nonfinancial
        ),
        "dart_alignable_count": len(
            dart
        ),
        "manual_review_count": len(
            review
        ),
        "by_modality": dict(
            modality_counts
        ),
        "by_parser_engine": dict(
            parser_engine_counts
        ),
        "by_period_inference_method": dict(
            period_method_counts
        ),
        "by_unit_inference_method": dict(
            unit_method_counts
        ),
        "by_scope": dict(
            scope_counts
        ),
        "unknown_period_count": sum(
            item.period_year is None
            for item in evidence
        ),
        "unknown_unit_count": sum(
            item.unit is None
            for item in evidence
        ),
        "unknown_financial_scope_count": sum(
            (
                item.is_financial_core
                and item.scope not in VALID_SCOPES
            )
            for item in evidence
        ),
        "high_parser_risk_evidence_count": sum(
            item.table_qc_priority == "HIGH"
            for item in evidence
        ),
        "two_family_table_evidence_count": sum(
            (
                item.modality == "table"
                and item.table_family_support >= 2
            )
            for item in evidence
        ),
        "alignment_key_excludes_value": True,
        "ocr_used": False,
        "document_report_terms": document_term_info,
    }

    save_json(
        SUMMARY_PATH,
        summary,
    )

    print()
    print(
        "="
        * 100
    )

    print(
        "[SUCCESS]"
    )

    print(
        f"Evidence            : {len(evidence)}"
    )

    print(
        f"  financial         : {len(financial)}"
    )

    print(
        f"  non-financial     : {len(nonfinancial)}"
    )

    print(
        f"DART alignable      : {len(dart)}"
    )

    print(
        f"Manual review       : {len(review)}"
    )

    print(
        f"Unknown period      : {summary['unknown_period_count']}"
    )

    print(
        f"Unknown unit        : {summary['unknown_unit_count']}"
    )

    print(
        f"Unknown fin. scope  : {summary['unknown_financial_scope_count']}"
    )

    print()

    print(
        f"All evidence        : {ALL_CSV}"
    )

    print(
        f"DART candidates     : {DART_CANDIDATES_CSV}"
    )

    print(
        f"Review candidates   : {REVIEW_CSV}"
    )

    print(
        f"Table diagnostics   : {TABLE_DIAGNOSTICS_CSV}"
    )

    print(
        "="
        * 100
    )

    print(
        "\n[NEXT QC]\n"
        "1. table_structure_diagnostics.csv에서 no_period_header_signal 확인\n"
        "2. manual_review_candidates.csv에서 HIGH parser risk 확인\n"
        "3. dart_alignment_candidates.csv의 period/scope/unit 분포 확인\n"
        "4. 그 다음에만 03_build_dart_gold.py를 실행하세요.\n"
    )


if __name__ == "__main__":
    main()
