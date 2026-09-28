#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_build_dart_gold.py
================================================================================
CAUSE-RAG / RAGON
Researcher A - FINAL OpenDART Gold Builder

Compatible with
---------------
01_parse_business_reports.py
02_structure_evidence.py

Research purpose
----------------
Build a manually verifiable Gold structured-evidence dataset by aligning
PDF-derived structured evidence with OpenDART structured financial statements.

IMPORTANT METHODOLOGY
---------------------
This script deliberately separates:

    [REFERENCE CONSTRUCTION]
        OpenDART structured financial-statement data

    [NON-LEAKY ALIGNMENT]
        PDF candidate selection WITHOUT using the DART value

    [VALUE COMPARISON]
        Only AFTER the PDF candidate has been selected

This avoids target leakage such as:
    "choose whichever PDF value is closest to the DART answer."

CAUSE-RAG Gold policy
---------------------
Gold is NOT:
    "Whatever the API returned."

Gold is:
    Exact filing version
    + OpenDART structured XBRL-derived financial reference
    + PDF/source-version consistency check
    + manual verification

Final Gold is generated only from rows marked:
    manual_verified = Y

OpenDART APIs used
------------------
1. corpCode.xml
   - stock_code -> corp_code

2. list.json
   - identify the exact business-report filing from the PDF filing date

3. fnlttSinglAcntAll.json
   - full CFS/OFS financial statements
   - reprt_code = 11011

4. fnlttXbrl.xml
   - used only as an exact-filing diagnostic fallback when the
     year-based financial API returns a different rcept_no
   - this script DOES NOT silently replace the reference with a different version

Why exact rcept_no validation matters
-------------------------------------
fnlttSinglAcntAll is queried by:
    corp_code + bsns_year + report code + CFS/OFS

It is not queried by exact rcept_no.

If a correction filing exists later, the API response can potentially refer to
a different filing version from the PDF currently used in the experiment.

Therefore:
    expected rcept_no from list.json
        must equal
    rcept_no in fnlttSinglAcntAll rows

Otherwise:
    STRICT VERSION MISMATCH
    -> exact XBRL is downloaded for diagnosis
    -> automatic Gold construction stops

Value comparison
----------------
The PDF may display values in:
    원 / 천원 / 백만원 / 억원 / 조원

02_structure_evidence.py converts them to value_krw.

Comparison statuses:
    MATCH_EXACT
        exact KRW equality

    MATCH_ROUNDING
        difference is explainable by displayed PDF precision/unit

    VALUE_MISMATCH
        difference exceeds rounding tolerance

    NO_PDF_CANDIDATE
        no non-leaky candidate found

    AMBIGUOUS_NONLEAKY
        multiple similarly plausible candidates remain;
        value is NOT used to choose one

Manual review priority
----------------------
P0_VERSION
P1_ALIGNMENT
P1_PARSING
P1_STRUCTURING
P2_VALUE_MISMATCH
P2_MISSING
P3_ROUNDING
P4_MATCH

Input
-----
dataset/01_parsed/document_manifest.csv

dataset/02_structured/dart_alignment_candidates.csv

API key
-------
Preferred:
    environment variable DART_API_KEY

Windows CMD:
    set DART_API_KEY=YOUR_KEY

PowerShell:
    $env:DART_API_KEY="YOUR_KEY"

Fallback:
    ./dart_api_key.txt

Install
-------
pip install -U requests pandas

Run
---
1. Build DART reference + alignment review:
    python 03_build_dart_gold.py

2. Review:
    dataset/03_dart_gold/gold_review_template.csv

   At minimum fill:
       manual_verified = Y
       manual_note     = optional

   Recommended manual_label values:
       VERIFIED
       PDF_PARSER_ERROR
       STRUCTURING_ERROR
       SOURCE_VERSION_ISSUE
       AMBIGUOUS
       EXCLUDE

3. Finalize Gold:
    python 03_build_dart_gold.py --finalize

Output
------
dataset/03_dart_gold/
├─ raw_api/
│  ├─ corp_codes/
│  ├─ disclosures/
│  ├─ financial/
│  └─ exact_xbrl_on_mismatch/
│
├─ exact_filing_manifest.csv
├─ dart_reference_evidence.jsonl
├─ dart_reference_evidence.csv
├─ dart_pdf_alignment.csv
├─ gold_review_template.csv
├─ review_queue.csv
├─ version_mismatch.csv
├─ gold_evidence.jsonl
├─ gold_evidence.csv
├─ gold_core.csv
└─ gold_manifest.json
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

import pandas as pd
import requests


# ==============================================================================
# 0. PATH
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_MANIFEST_PATH = (
    BASE_DIR
    / "dataset"
    / "01_parsed"
    / "document_manifest.csv"
)

PDF_EVIDENCE_PATH = (
    BASE_DIR
    / "dataset"
    / "02_structured"
    / "dart_alignment_candidates.csv"
)

SCRIPT02_PATH = (
    BASE_DIR
    / "02_structure_evidence.py"
)

OUTPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "03_dart_gold"
)

RAW_API_DIR = OUTPUT_DIR / "raw_api"

RAW_CORP_DIR = RAW_API_DIR / "corp_codes"
RAW_DISCLOSURE_DIR = RAW_API_DIR / "disclosures"
RAW_FINANCIAL_DIR = RAW_API_DIR / "financial"
RAW_XBRL_MISMATCH_DIR = (
    RAW_API_DIR
    / "exact_xbrl_on_mismatch"
)

CORP_CODE_CACHE_CSV = (
    RAW_CORP_DIR
    / "corp_codes.csv"
)

EXACT_FILING_MANIFEST_PATH = (
    OUTPUT_DIR
    / "exact_filing_manifest.csv"
)

REFERENCE_JSONL_PATH = (
    OUTPUT_DIR
    / "dart_reference_evidence.jsonl"
)

REFERENCE_CSV_PATH = (
    OUTPUT_DIR
    / "dart_reference_evidence.csv"
)

ALIGNMENT_CSV_PATH = (
    OUTPUT_DIR
    / "dart_pdf_alignment.csv"
)

GOLD_REVIEW_PATH = (
    OUTPUT_DIR
    / "gold_review_template.csv"
)

REVIEW_QUEUE_PATH = (
    OUTPUT_DIR
    / "review_queue.csv"
)

VERSION_MISMATCH_PATH = (
    OUTPUT_DIR
    / "version_mismatch.csv"
)

FINAL_GOLD_JSONL_PATH = (
    OUTPUT_DIR
    / "gold_evidence.jsonl"
)

FINAL_GOLD_CSV_PATH = (
    OUTPUT_DIR
    / "gold_evidence.csv"
)

FINAL_GOLD_CORE_PATH = (
    OUTPUT_DIR
    / "gold_core.csv"
)

GOLD_MANIFEST_PATH = (
    OUTPUT_DIR
    / "gold_manifest.json"
)


# ==============================================================================
# 1. OPENDART
# ==============================================================================

CORP_CODE_URL = (
    "https://opendart.fss.or.kr/api/corpCode.xml"
)

DISCLOSURE_LIST_URL = (
    "https://opendart.fss.or.kr/api/list.json"
)

FINANCIAL_ALL_URL = (
    "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
)

XBRL_FILE_URL = (
    "https://opendart.fss.or.kr/api/fnlttXbrl.xml"
)

BUSINESS_REPORT_CODE = "11011"

SCOPE_REQUESTS = {
    "consolidated": "CFS",
    "separate": "OFS",
}

FS_DIV_TO_SCOPE = {
    "CFS": "consolidated",
    "OFS": "separate",
}

PRIMARY_STATEMENTS = {
    "BS",
    "IS",
    "CIS",
    "CF",
    "SCE",
}

DART_SUCCESS = "000"
DART_NO_DATA = "013"

RETRYABLE_DART_STATUS = {
    "020",  # request limit
    "800",  # service inspection
    "900",  # undefined temporary-ish error
}

REQUEST_TIMEOUT = 45
REQUEST_RETRIES = 4
REQUEST_SLEEP_SECONDS = 0.12


# ==============================================================================
# 2. ALIGNMENT PARAMETERS
# ==============================================================================

FUZZY_METRIC_THRESHOLD = 0.90

# If the two highest candidates differ by less than this score,
# do NOT use the DART value to break the tie.
AMBIGUITY_MARGIN = 0.025

# Candidate must have at least this metric-level score.
MIN_NONLEAKY_SCORE = 0.58

# Used to detect post-alignment many-to-one mappings.
MANY_TO_ONE_REVIEW = True


# ==============================================================================
# 3. TARGET DATASET
# ==============================================================================

EXPECTED_COMPANY_YEARS = {
    ("삼성전자", 2024),
    ("삼성전자", 2025),
    ("SK하이닉스", 2024),
    ("SK하이닉스", 2025),
    ("현대자동차", 2024),
    ("현대자동차", 2025),
}

EXPECTED_STOCK_CODES = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    "현대자동차": "005380",
}


# ==============================================================================
# 4. COMMON UTILS
# ==============================================================================

def clean_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    if (
        isinstance(value, float)
        and pd.isna(value)
    ):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


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


def write_jsonl(
    path: Path,
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
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


def as_bool(
    value: Any,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value

    return clean_text(
        value
    ).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def safe_float(
    value: Any,
) -> Optional[float]:
    text = clean_text(
        value
    )

    if not text:
        return None

    try:
        number = float(
            text
        )

    except Exception:
        return None

    if math.isnan(
        number
    ):
        return None

    return number


def safe_int(
    value: Any,
) -> Optional[int]:
    number = safe_float(
        value
    )

    if number is None:
        return None

    return int(
        number
    )


def clamp01(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def normalize_stock_code(
    value: Any,
) -> str:
    text = clean_text(
        value
    )

    if not text:
        return ""

    text = re.sub(
        r"\D",
        "",
        text,
    )

    return text.zfill(
        6
    )


def parse_jsonish_list(
    value: Any,
) -> list[str]:
    if isinstance(
        value,
        list,
    ):
        return [
            clean_text(item)
            for item in value
            if clean_text(
                item
            )
        ]

    text = clean_text(
        value
    )

    if not text:
        return []

    if text.startswith(
        "["
    ):
        try:
            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                list,
            ):
                return [
                    clean_text(item)
                    for item in parsed
                    if clean_text(
                        item
                    )
                ]

        except Exception:
            pass

    return [
        clean_text(item)
        for item in text.split(
            "|"
        )
        if clean_text(
            item
        )
    ]


# ==============================================================================
# 5. LOAD SCRIPT 02 NORMALIZERS
# ==============================================================================

def load_script02_module():
    """
    Reuse the exact metric normalization used during PDF structuring.

    This prevents a subtle evaluation bug where:
        Script 02 normalizes PDF metric one way,
        Script 03 normalizes DART metric another way.
    """
    if not SCRIPT02_PATH.exists():
        raise RuntimeError(
            "02_structure_evidence.py가 없습니다. "
            "DART metric normalization을 02와 동일하게 유지할 수 없습니다."
        )

    spec = importlib.util.spec_from_file_location(
        "cause_rag_structure_v2",
        SCRIPT02_PATH,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "02_structure_evidence.py import 실패"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    # Required for dataclasses in dynamically imported Script 02.
    sys.modules[spec.name] = module

    spec.loader.exec_module(
        module
    )

    for required in [
        "normalize_metric_name",
        "canonical_metric",
    ]:
        if not hasattr(
            module,
            required,
        ):
            raise RuntimeError(
                f"02_structure_evidence.py에 "
                f"{required} 함수가 없습니다."
            )

    return module


STRUCTURE_MODULE = None


def normalize_metric_name(
    text: str,
) -> str:
    return STRUCTURE_MODULE.normalize_metric_name(
        text
    )


def canonical_metric(
    text: str,
) -> Optional[str]:
    result = STRUCTURE_MODULE.canonical_metric(
        text
    )

    return (
        clean_text(result)
        or None
    )


# ==============================================================================
# 6. API KEY / SESSION
# ==============================================================================

def get_api_key() -> str:
    key = clean_text(
        os.getenv(
            "DART_API_KEY"
        )
    )

    if key:
        return key

    key_file = (
        BASE_DIR
        / "dart_api_key.txt"
    )

    if key_file.exists():
        key = clean_text(
            key_file.read_text(
                encoding="utf-8"
            )
        )

        if key:
            return key

    raise SystemExit(
        "\n[ERROR] OpenDART API key가 필요합니다.\n\n"
        "Windows CMD:\n"
        "    set DART_API_KEY=YOUR_KEY\n\n"
        "PowerShell:\n"
        '    $env:DART_API_KEY="YOUR_KEY"\n\n'
        "또는 프로젝트 폴더에 dart_api_key.txt를 두세요.\n"
    )


def build_session() -> requests.Session:
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "CAUSE-RAG-RAGON/"
                "1.0 research dataset builder"
            ),
        }
    )

    return session


# ==============================================================================
# 7. DART REQUESTS
# ==============================================================================

def dart_request_json(
    session: requests.Session,
    url: str,
    params: dict[
        str,
        Any,
    ],
    cache_path: Optional[
        Path
    ] = None,
) -> dict[
    str,
    Any,
]:
    """
    If cache exists, use it.
    API key is never written to cache.
    """
    if (
        cache_path is not None
        and cache_path.exists()
    ):
        return json.loads(
            cache_path.read_text(
                encoding="utf-8"
            )
        )

    last_error = None

    for attempt in range(
        1,
        REQUEST_RETRIES + 1,
    ):
        try:
            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            status = clean_text(
                data.get(
                    "status"
                )
            )

            if status in {
                DART_SUCCESS,
                DART_NO_DATA,
            }:
                if cache_path is not None:
                    save_json(
                        cache_path,
                        data,
                    )

                return data

            message = clean_text(
                data.get(
                    "message"
                )
            )

            if (
                status
                in RETRYABLE_DART_STATUS
                and attempt
                < REQUEST_RETRIES
            ):
                time.sleep(
                    1.5
                    * attempt
                )

                continue

            raise RuntimeError(
                f"OpenDART status={status}, "
                f"message={message}"
            )

        except Exception as exc:
            last_error = exc

            if attempt < REQUEST_RETRIES:
                time.sleep(
                    1.5
                    * attempt
                )

    raise RuntimeError(
        f"OpenDART request failed: "
        f"{last_error}"
    )


# ==============================================================================
# 8. CORP CODE
# ==============================================================================

def load_corp_codes(
    session: requests.Session,
    api_key: str,
) -> pd.DataFrame:
    RAW_CORP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CORP_CODE_CACHE_CSV.exists():
        return pd.read_csv(
            CORP_CODE_CACHE_CSV,
            dtype=str,
        ).fillna(
            ""
        )

    response = session.get(
        CORP_CODE_URL,
        params={
            "crtfc_key": api_key,
        },
        timeout=60,
    )

    response.raise_for_status()

    content = response.content

    if not content.startswith(
        b"PK"
    ):
        # OpenDART error responses can be XML.
        decoded = content.decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            "corpCode.xml ZIP 다운로드 실패:\n"
            + decoded[:1000]
        )

    archive = zipfile.ZipFile(
        io.BytesIO(
            content
        )
    )

    xml_files = [
        name
        for name
        in archive.namelist()
        if name.lower().endswith(
            ".xml"
        )
    ]

    if not xml_files:
        raise RuntimeError(
            "corpCode ZIP 안에 XML이 없습니다."
        )

    root = ET.fromstring(
        archive.read(
            xml_files[0]
        )
    )

    rows = []

    for node in root.findall(
        "list"
    ):
        rows.append(
            {
                "corp_code": clean_text(
                    node.findtext(
                        "corp_code"
                    )
                ),
                "corp_name": clean_text(
                    node.findtext(
                        "corp_name"
                    )
                ),
                "corp_eng_name": clean_text(
                    node.findtext(
                        "corp_eng_name"
                    )
                ),
                "stock_code": normalize_stock_code(
                    node.findtext(
                        "stock_code"
                    )
                ),
                "modify_date": clean_text(
                    node.findtext(
                        "modify_date"
                    )
                ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        CORP_CODE_CACHE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    return dataframe


def resolve_corp_code(
    corp_codes: pd.DataFrame,
    entity: str,
    stock_code: str,
) -> tuple[
    str,
    str,
]:
    stock_code = normalize_stock_code(
        stock_code
    )

    matched = corp_codes[
        corp_codes[
            "stock_code"
        ]
        == stock_code
    ]

    if len(
        matched
    ) != 1:
        raise RuntimeError(
            "OpenDART corp_code를 유일하게 찾지 못했습니다: "
            f"entity={entity}, "
            f"stock_code={stock_code}, "
            f"matches={len(matched)}"
        )

    row = matched.iloc[
        0
    ]

    return (
        clean_text(
            row[
                "corp_code"
            ]
        ),
        clean_text(
            row[
                "corp_name"
            ]
        ),
    )


# ==============================================================================
# 9. INPUT VALIDATION
# ==============================================================================

def load_document_manifest() -> pd.DataFrame:
    if not DOCUMENT_MANIFEST_PATH.exists():
        raise SystemExit(
            "\n[ERROR] document_manifest.csv가 없습니다.\n"
            "먼저 01_parse_business_reports.py를 완료하세요.\n"
        )

    dataframe = pd.read_csv(
        DOCUMENT_MANIFEST_PATH,
        dtype=str,
    ).fillna(
        ""
    )

    required = {
        "entity",
        "stock_code",
        "source_report_year",
        "filing_date",
        "filing_version",
        "version_id",
        "source_file",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "01 manifest schema mismatch. Missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if len(
        dataframe
    ) != 6:
        raise RuntimeError(
            "현재 연구 설계는 6개 사업보고서를 기대합니다. "
            f"manifest rows={len(dataframe)}"
        )

    actual_pairs = {
        (
            clean_text(
                row[
                    "entity"
                ]
            ),
            int(
                row[
                    "source_report_year"
                ]
            ),
        )
        for _, row in dataframe.iterrows()
    }

    missing_pairs = (
        EXPECTED_COMPANY_YEARS
        - actual_pairs
    )

    extra_pairs = (
        actual_pairs
        - EXPECTED_COMPANY_YEARS
    )

    if (
        missing_pairs
        or extra_pairs
    ):
        raise RuntimeError(
            "company-year 구성 불일치.\n"
            f"missing={sorted(missing_pairs)}\n"
            f"extra={sorted(extra_pairs)}"
        )

    for _, row in dataframe.iterrows():
        entity = clean_text(
            row[
                "entity"
            ]
        )

        expected_stock = EXPECTED_STOCK_CODES.get(
            entity
        )

        actual_stock = normalize_stock_code(
            row[
                "stock_code"
            ]
        )

        if (
            expected_stock
            and actual_stock
            != expected_stock
        ):
            raise RuntimeError(
                f"stock_code mismatch: "
                f"{entity}: "
                f"{actual_stock} != "
                f"{expected_stock}"
            )

    return dataframe


def load_pdf_evidence() -> pd.DataFrame:
    if not PDF_EVIDENCE_PATH.exists():
        raise SystemExit(
            "\n[ERROR] dart_alignment_candidates.csv가 없습니다.\n"
            "먼저 02_structure_evidence.py를 완료하세요.\n"
        )

    dataframe = pd.read_csv(
        PDF_EVIDENCE_PATH,
        dtype=str,
    ).fillna(
        ""
    )

    required = {
        "evidence_id",
        "entity",
        "stock_code",
        "source_report_year",
        "period_year",
        "scope",
        "statement_type",
        "metric_raw",
        "metric_norm",
        "metric_canonical",
        "value_krw",
        "alignment_metric_key",
        "alignment_key",
        "confidence",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "02 output schema mismatch. Missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if dataframe.empty:
        raise RuntimeError(
            "02의 DART alignment candidate가 0개입니다. "
            "Period/Unit/Scope structuring을 먼저 점검하세요."
        )

    # Numeric helper columns.
    for column in [
        "source_report_year",
        "period_year",
        "value",
        "value_krw",
        "unit_scale",
        "confidence",
        "period_confidence",
        "table_family_support",
        "table_candidate_support",
        "table_content_agreement",
        "table_selection_score",
        "table_qc_score",
    ]:
        if column in dataframe.columns:
            dataframe[
                f"_num_{column}"
            ] = pd.to_numeric(
                dataframe[
                    column
                ],
                errors="coerce",
            )

    return dataframe


# ==============================================================================
# 10. EXACT BUSINESS-REPORT FILING
# ==============================================================================

def report_is_correction(
    report_name: str,
) -> bool:
    normalized = clean_text(
        report_name
    )

    return (
        "정정"
        in normalized
    )


def find_exact_business_report(
    session: requests.Session,
    api_key: str,
    corp_code: str,
    entity: str,
    source_report_year: int,
    filing_date: str,
    filing_version: str,
) -> dict[
    str,
    Any,
]:
    """
    Match the exact filing DATE already identified from the PDF.

    We do not silently use another date, because version identity is part
    of CAUSE-RAG's evidence schema.
    """
    date_compact = filing_date.replace(
        "-",
        "",
    )

    cache_path = (
        RAW_DISCLOSURE_DIR
        / (
            f"{entity}_"
            f"{source_report_year}_"
            f"{date_compact}.json"
        )
    )

    data = dart_request_json(
        session=session,
        url=DISCLOSURE_LIST_URL,
        params={
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bgn_de": date_compact,
            "end_de": date_compact,
            "last_reprt_at": "N",
            "pblntf_ty": "A",
            "sort": "date",
            "sort_mth": "asc",
            "page_no": 1,
            "page_count": 100,
        },
        cache_path=cache_path,
    )

    if clean_text(
        data.get(
            "status"
        )
    ) == DART_NO_DATA:
        raise RuntimeError(
            "PDF 제출일과 동일한 DART 공시를 찾지 못했습니다: "
            f"{entity}, FY{source_report_year}, "
            f"{filing_date}"
        )

    candidates = []

    for row in data.get(
        "list",
        []
    ):
        if clean_text(
            row.get(
                "rcept_dt"
            )
        ) != date_compact:
            continue

        report_name = clean_text(
            row.get(
                "report_nm"
            )
        )

        if "사업보고서" not in report_name:
            continue

        candidates.append(
            row
        )

    if not candidates:
        raise RuntimeError(
            "동일 제출일의 사업보고서를 찾지 못했습니다: "
            f"{entity}, {filing_date}"
        )

    normalized_version = clean_text(
        filing_version
    ).lower()

    if normalized_version == "corrected":
        version_filtered = [
            row
            for row in candidates
            if report_is_correction(
                row.get(
                    "report_nm",
                    "",
                )
            )
        ]

    else:
        version_filtered = [
            row
            for row in candidates
            if not report_is_correction(
                row.get(
                    "report_nm",
                    "",
                )
            )
        ]

    if version_filtered:
        candidates = (
            version_filtered
        )

    if len(
        candidates
    ) != 1:
        diagnostic = [
            {
                "rcept_no": clean_text(
                    row.get(
                        "rcept_no"
                    )
                ),
                "rcept_dt": clean_text(
                    row.get(
                        "rcept_dt"
                    )
                ),
                "report_nm": clean_text(
                    row.get(
                        "report_nm"
                    )
                ),
            }
            for row in candidates
        ]

        raise RuntimeError(
            "정확한 PDF 버전에 대응하는 사업보고서를 "
            "유일하게 선택할 수 없습니다.\n"
            f"entity={entity}, "
            f"year={source_report_year}, "
            f"date={filing_date}, "
            f"version={filing_version}\n"
            f"candidates={diagnostic}"
        )

    selected = candidates[
        0
    ]

    rcept_no = clean_text(
        selected.get(
            "rcept_no"
        )
    )

    if not re.fullmatch(
        r"\d{14}",
        rcept_no,
    ):
        raise RuntimeError(
            "DART rcept_no 형식이 예상과 다릅니다: "
            f"{rcept_no}"
        )

    return selected


# ==============================================================================
# 11. FINANCIAL API
# ==============================================================================

def fetch_financial_scope(
    session: requests.Session,
    api_key: str,
    corp_code: str,
    entity: str,
    source_report_year: int,
    fs_div: str,
    expected_rcept_no: str,
) -> dict[
    str,
    Any,
]:
    cache_path = (
        RAW_FINANCIAL_DIR
        / (
            f"{entity}_"
            f"{source_report_year}_"
            f"{fs_div}_"
            f"{expected_rcept_no}.json"
        )
    )

    return dart_request_json(
        session=session,
        url=FINANCIAL_ALL_URL,
        params={
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bsns_year": str(
                source_report_year
            ),
            "reprt_code": (
                BUSINESS_REPORT_CODE
            ),
            "fs_div": fs_div,
        },
        cache_path=cache_path,
    )


def parse_dart_amount(
    value: Any,
) -> Optional[float]:
    text = clean_text(
        value
    )

    if not text:
        return None

    negative_parentheses = (
        text.startswith(
            "("
        )
        and text.endswith(
            ")"
        )
    )

    if negative_parentheses:
        text = text[
            1:-1
        ]

    text = (
        text
        .replace(
            ",",
            "",
        )
        .replace(
            "−",
            "-",
        )
        .replace(
            "–",
            "-",
        )
        .replace(
            " ",
            "",
        )
    )

    if text in {
        "",
        "-",
        "—",
        "―",
    }:
        return None

    try:
        result = float(
            text
        )

    except ValueError:
        return None

    if negative_parentheses:
        result = -abs(
            result
        )

    return result


ANNUAL_TERM_FIELDS = [
    {
        "reporting_role": "current",
        "term_name_field": "thstrm_nm",
        "amount_field": "thstrm_amount",
        "period_offset": 0,
    },
    {
        "reporting_role": "prior",
        "term_name_field": "frmtrm_nm",
        "amount_field": "frmtrm_amount",
        "period_offset": -1,
    },
    {
        "reporting_role": "prior2",
        "term_name_field": "bfefrmtrm_nm",
        "amount_field": "bfefrmtrm_amount",
        "period_offset": -2,
    },
]


def expand_dart_reference_rows(
    api_rows: list[
        dict[
            str,
            Any,
        ]
    ],
    document_row: dict[
        str,
        Any,
    ],
    exact_filing: dict[
        str,
        Any,
    ],
    corp_code: str,
    dart_corp_name: str,
    fs_div: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    source_report_year = int(
        document_row[
            "source_report_year"
        ]
    )

    expected_rcept_no = clean_text(
        exact_filing[
            "rcept_no"
        ]
    )

    scope = FS_DIV_TO_SCOPE[
        fs_div
    ]

    reference = []

    for api_row in api_rows:
        row_rcept_no = clean_text(
            api_row.get(
                "rcept_no"
            )
        )

        if (
            row_rcept_no
            != expected_rcept_no
        ):
            raise RuntimeError(
                "STRICT_VERSION_MISMATCH_INTERNAL"
            )

        statement_type = clean_text(
            api_row.get(
                "sj_div"
            )
        )

        if statement_type not in PRIMARY_STATEMENTS:
            continue

        account_id = clean_text(
            api_row.get(
                "account_id"
            )
        )

        account_name = clean_text(
            api_row.get(
                "account_nm"
            )
        )

        account_detail = clean_text(
            api_row.get(
                "account_detail"
            )
        )

        metric_norm = normalize_metric_name(
            account_name
        )

        metric_canonical = canonical_metric(
            account_name
        )

        alignment_metric_key = (
            metric_canonical
            if metric_canonical
            else metric_norm
        )

        currency = clean_text(
            api_row.get(
                "currency"
            )
        ).upper()

        for term in ANNUAL_TERM_FIELDS:
            amount_raw = clean_text(
                api_row.get(
                    term[
                        "amount_field"
                    ]
                )
            )

            amount = parse_dart_amount(
                amount_raw
            )

            if amount is None:
                continue

            period_year = (
                source_report_year
                + int(
                    term[
                        "period_offset"
                    ]
                )
            )

            period_raw = clean_text(
                api_row.get(
                    term[
                        "term_name_field"
                    ]
                )
            )

            alignment_key = "||".join(
                [
                    clean_text(
                        document_row[
                            "entity"
                        ]
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

            reference_id = stable_id(
                expected_rcept_no,
                scope,
                statement_type,
                account_id,
                account_name,
                account_detail,
                term[
                    "reporting_role"
                ],
                period_year,
                amount_raw,
                prefix="dart_",
            )

            reference.append(
                {
                    "dart_reference_id": (
                        reference_id
                    ),

                    # --------------------------------------
                    # Core evidence schema
                    # --------------------------------------
                    "entity": clean_text(
                        document_row[
                            "entity"
                        ]
                    ),
                    "stock_code": normalize_stock_code(
                        document_row[
                            "stock_code"
                        ]
                    ),

                    "metric_raw": (
                        account_name
                    ),
                    "metric_norm": (
                        metric_norm
                    ),
                    "metric_canonical": (
                        metric_canonical
                        or ""
                    ),
                    "alignment_metric_key": (
                        alignment_metric_key
                    ),

                    "account_id": (
                        account_id
                    ),
                    "account_detail": (
                        account_detail
                    ),

                    "value_raw": (
                        amount_raw
                    ),
                    "value": (
                        amount
                    ),

                    "dart_currency": (
                        currency
                    ),
                    "unit": (
                        "원"
                        if currency
                        == "KRW"
                        else currency
                    ),
                    "unit_scale": (
                        1
                        if currency
                        == "KRW"
                        else ""
                    ),
                    "value_krw": (
                        amount
                        if currency
                        == "KRW"
                        else ""
                    ),

                    "period_raw": (
                        period_raw
                    ),
                    "period_year": (
                        period_year
                    ),

                    "scope": (
                        scope
                    ),

                    "source_report_year": (
                        source_report_year
                    ),

                    "filing_date": clean_text(
                        document_row[
                            "filing_date"
                        ]
                    ),
                    "filing_version": clean_text(
                        document_row[
                            "filing_version"
                        ]
                    ),
                    "version_id": clean_text(
                        document_row[
                            "version_id"
                        ]
                    ),

                    # --------------------------------------
                    # DART provenance
                    # --------------------------------------
                    "corp_code": (
                        corp_code
                    ),
                    "dart_corp_name": (
                        dart_corp_name
                    ),

                    "rcept_no": (
                        expected_rcept_no
                    ),
                    "report_name": clean_text(
                        exact_filing.get(
                            "report_nm"
                        )
                    ),
                    "rcept_date": clean_text(
                        exact_filing.get(
                            "rcept_dt"
                        )
                    ),

                    "reprt_code": (
                        BUSINESS_REPORT_CODE
                    ),
                    "fs_div": (
                        fs_div
                    ),

                    "statement_type": (
                        statement_type
                    ),
                    "statement_name": clean_text(
                        api_row.get(
                            "sj_nm"
                        )
                    ),

                    "reporting_role": (
                        term[
                            "reporting_role"
                        ]
                    ),

                    "ord": clean_text(
                        api_row.get(
                            "ord"
                        )
                    ),

                    "source_pdf_file": clean_text(
                        document_row[
                            "source_file"
                        ]
                    ),

                    "reference_source": (
                        "OpenDART "
                        "fnlttSinglAcntAll"
                    ),

                    # --------------------------------------
                    # Non-leaky key
                    # --------------------------------------
                    "alignment_key": (
                        alignment_key
                    ),
                }
            )

    return reference


# ==============================================================================
# 12. STRICT VERSION CHECK
# ==============================================================================

def download_exact_xbrl_for_diagnosis(
    session: requests.Session,
    api_key: str,
    entity: str,
    source_report_year: int,
    expected_rcept_no: str,
) -> str:
    """
    Diagnostic only.

    It preserves the exact-filing XBRL ZIP when the year-based financial API
    points at a different filing version.
    """
    RAW_XBRL_MISMATCH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        RAW_XBRL_MISMATCH_DIR
        / (
            f"{entity}_"
            f"{source_report_year}_"
            f"{expected_rcept_no}.zip"
        )
    )

    if destination.exists():
        return str(
            destination.relative_to(
                BASE_DIR
            )
        )

    response = session.get(
        XBRL_FILE_URL,
        params={
            "crtfc_key": api_key,
            "rcept_no": (
                expected_rcept_no
            ),
            "reprt_code": (
                BUSINESS_REPORT_CODE
            ),
        },
        timeout=60,
    )

    response.raise_for_status()

    if response.content.startswith(
        b"PK"
    ):
        destination.write_bytes(
            response.content
        )

        return str(
            destination.relative_to(
                BASE_DIR
            )
        )

    # Save error body separately for diagnosis.
    error_path = destination.with_suffix(
        ".xml"
    )

    error_path.write_bytes(
        response.content
    )

    return str(
        error_path.relative_to(
            BASE_DIR
        )
    )


def validate_financial_response_version(
    session: requests.Session,
    api_key: str,
    response: dict[
        str,
        Any,
    ],
    entity: str,
    source_report_year: int,
    fs_div: str,
    expected_rcept_no: str,
) -> Optional[
    dict[
        str,
        Any,
    ]
]:
    if clean_text(
        response.get(
            "status"
        )
    ) == DART_NO_DATA:
        return None

    rows = response.get(
        "list",
        []
    )

    actual_rcept_numbers = sorted(
        {
            clean_text(
                row.get(
                    "rcept_no"
                )
            )
            for row in rows
            if clean_text(
                row.get(
                    "rcept_no"
                )
            )
        }
    )

    if (
        len(
            actual_rcept_numbers
        )
        == 1
        and actual_rcept_numbers[
            0
        ]
        == expected_rcept_no
    ):
        return None

    exact_xbrl_path = ""

    try:
        exact_xbrl_path = (
            download_exact_xbrl_for_diagnosis(
                session=session,
                api_key=api_key,
                entity=entity,
                source_report_year=(
                    source_report_year
                ),
                expected_rcept_no=(
                    expected_rcept_no
                ),
            )
        )

    except Exception as exc:
        exact_xbrl_path = (
            "DOWNLOAD_FAILED: "
            + clean_text(
                exc
            )
        )

    return {
        "entity": (
            entity
        ),
        "source_report_year": (
            source_report_year
        ),
        "fs_div": (
            fs_div
        ),
        "expected_rcept_no": (
            expected_rcept_no
        ),
        "api_rcept_numbers": (
            "|".join(
                actual_rcept_numbers
            )
        ),
        "exact_xbrl_diagnostic_path": (
            exact_xbrl_path
        ),
        "action": (
            "STOP_AUTO_GOLD"
        ),
        "reason": (
            "fnlttSinglAcntAll filing version "
            "does not match the source PDF filing"
        ),
    }


# ==============================================================================
# 13. BUILD DART REFERENCE
# ==============================================================================

def build_dart_reference(
    session: requests.Session,
    api_key: str,
    document_manifest: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    corp_codes = load_corp_codes(
        session=session,
        api_key=api_key,
    )

    reference_rows = []
    exact_filing_rows = []
    version_mismatches = []

    RAW_DISCLOSURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_FINANCIAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sorted_manifest = (
        document_manifest
        .copy()
        .sort_values(
            [
                "entity",
                "source_report_year",
            ]
        )
    )

    for _, manifest_row in sorted_manifest.iterrows():
        row = manifest_row.to_dict()

        entity = clean_text(
            row[
                "entity"
            ]
        )

        stock_code = normalize_stock_code(
            row[
                "stock_code"
            ]
        )

        source_report_year = int(
            row[
                "source_report_year"
            ]
        )

        filing_date = clean_text(
            row[
                "filing_date"
            ]
        )

        filing_version = clean_text(
            row[
                "filing_version"
            ]
        )

        corp_code, dart_corp_name = resolve_corp_code(
            corp_codes,
            entity,
            stock_code,
        )

        exact_filing = (
            find_exact_business_report(
                session=session,
                api_key=api_key,
                corp_code=corp_code,
                entity=entity,
                source_report_year=(
                    source_report_year
                ),
                filing_date=filing_date,
                filing_version=(
                    filing_version
                ),
            )
        )

        expected_rcept_no = clean_text(
            exact_filing[
                "rcept_no"
            ]
        )

        print(
            f"[DART] {entity} "
            f"FY{source_report_year} "
            f"-> {expected_rcept_no}"
        )

        exact_filing_rows.append(
            {
                "entity": (
                    entity
                ),
                "stock_code": (
                    stock_code
                ),
                "corp_code": (
                    corp_code
                ),
                "dart_corp_name": (
                    dart_corp_name
                ),
                "source_report_year": (
                    source_report_year
                ),
                "source_pdf_file": clean_text(
                    row[
                        "source_file"
                    ]
                ),
                "source_filing_date": (
                    filing_date
                ),
                "source_filing_version": (
                    filing_version
                ),
                "version_id": clean_text(
                    row[
                        "version_id"
                    ]
                ),
                "rcept_no": (
                    expected_rcept_no
                ),
                "rcept_date": clean_text(
                    exact_filing.get(
                        "rcept_dt"
                    )
                ),
                "report_name": clean_text(
                    exact_filing.get(
                        "report_nm"
                    )
                ),
            }
        )

        for scope, fs_div in SCOPE_REQUESTS.items():
            response = fetch_financial_scope(
                session=session,
                api_key=api_key,
                corp_code=corp_code,
                entity=entity,
                source_report_year=(
                    source_report_year
                ),
                fs_div=fs_div,
                expected_rcept_no=(
                    expected_rcept_no
                ),
            )

            mismatch = (
                validate_financial_response_version(
                    session=session,
                    api_key=api_key,
                    response=response,
                    entity=entity,
                    source_report_year=(
                        source_report_year
                    ),
                    fs_div=fs_div,
                    expected_rcept_no=(
                        expected_rcept_no
                    ),
                )
            )

            if mismatch is not None:
                version_mismatches.append(
                    mismatch
                )

                continue

            if clean_text(
                response.get(
                    "status"
                )
            ) == DART_NO_DATA:
                print(
                    f"       {fs_div}: no data"
                )

                continue

            expanded = (
                expand_dart_reference_rows(
                    api_rows=response.get(
                        "list",
                        []
                    ),
                    document_row=row,
                    exact_filing=(
                        exact_filing
                    ),
                    corp_code=corp_code,
                    dart_corp_name=(
                        dart_corp_name
                    ),
                    fs_div=fs_div,
                )
            )

            reference_rows.extend(
                expanded
            )

            print(
                f"       {fs_div}: "
                f"{len(expanded)} reference rows"
            )

            time.sleep(
                REQUEST_SLEEP_SECONDS
            )

    reference_df = pd.DataFrame(
        reference_rows
    )

    exact_filing_df = pd.DataFrame(
        exact_filing_rows
    )

    version_mismatch_df = (
        pd.DataFrame(
            version_mismatches
        )
    )

    if not reference_df.empty:
        reference_df = (
            reference_df
            .drop_duplicates(
                subset=[
                    "dart_reference_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    return (
        reference_df,
        exact_filing_df,
        version_mismatch_df,
    )


# ==============================================================================
# 14. NON-LEAKY METRIC MATCHING
# ==============================================================================

def sequence_similarity(
    a: str,
    b: str,
) -> float:
    a = clean_text(
        a
    )

    b = clean_text(
        b
    )

    if (
        not a
        or not b
    ):
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def token_set(
    text: str,
) -> set[str]:
    return set(
        token
        for token in re.findall(
            r"[가-힣A-Za-z0-9]+",
            clean_text(
                text
            ).lower(),
        )
        if token
    )


def token_jaccard(
    a: str,
    b: str,
) -> float:
    a_tokens = token_set(
        a
    )

    b_tokens = token_set(
        b
    )

    if (
        not a_tokens
        and not b_tokens
    ):
        return 0.0

    if (
        not a_tokens
        or not b_tokens
    ):
        return 0.0

    return (
        len(
            a_tokens
            & b_tokens
        )
        / len(
            a_tokens
            | b_tokens
        )
    )


def metric_match_score(
    dart_row: pd.Series,
    pdf_row: pd.Series,
) -> tuple[
    float,
    str,
]:
    dart_norm = clean_text(
        dart_row[
            "metric_norm"
        ]
    )

    pdf_norm = clean_text(
        pdf_row[
            "metric_norm"
        ]
    )

    dart_canonical = clean_text(
        dart_row.get(
            "metric_canonical"
        )
    )

    pdf_canonical = clean_text(
        pdf_row.get(
            "metric_canonical"
        )
    )

    if (
        dart_norm
        and pdf_norm
        and dart_norm
        == pdf_norm
    ):
        return (
            1.0,
            "metric_norm_exact",
        )

    if (
        dart_canonical
        and pdf_canonical
        and dart_canonical
        == pdf_canonical
    ):
        return (
            0.97,
            "metric_canonical_exact",
        )

    fuzzy = sequence_similarity(
        dart_norm,
        pdf_norm,
    )

    if fuzzy >= FUZZY_METRIC_THRESHOLD:
        return (
            0.90
            + 0.07
            * (
                fuzzy
                - FUZZY_METRIC_THRESHOLD
            )
            / (
                1.0
                - FUZZY_METRIC_THRESHOLD
            ),
            "metric_fuzzy",
        )

    return (
        0.0,
        "no_metric_match",
    )


def detail_path_score(
    dart_row: pd.Series,
    pdf_row: pd.Series,
) -> float:
    """
    Helpful mainly for SCE or custom accounts.

    No numeric value is used.
    """
    dart_detail = clean_text(
        dart_row.get(
            "account_detail"
        )
    )

    if not dart_detail:
        return 0.0

    pdf_path = clean_text(
        pdf_row.get(
            "metric_path_raw"
        )
    )

    if not pdf_path:
        pdf_path = clean_text(
            pdf_row.get(
                "metric_raw"
            )
        )

    sequence = sequence_similarity(
        normalize_metric_name(
            dart_detail
        ),
        normalize_metric_name(
            pdf_path
        ),
    )

    token_overlap = token_jaccard(
        dart_detail,
        pdf_path,
    )

    return clamp01(
        0.55
        * sequence
        + 0.45
        * token_overlap
    )


def pdf_parser_quality(
    row: pd.Series,
) -> float:
    confidence = safe_float(
        row.get(
            "confidence"
        )
    )

    if confidence is None:
        confidence = 0.70

    confidence = clamp01(
        confidence
    )

    family_support = safe_int(
        row.get(
            "table_family_support"
        )
    )

    family_component = (
        1.0
        if (
            family_support
            is not None
            and family_support
            >= 2
        )
        else 0.68
    )

    agreement = safe_float(
        row.get(
            "table_content_agreement"
        )
    )

    if (
        agreement is None
        or agreement <= 0
    ):
        agreement = 0.68

    agreement = clamp01(
        agreement
    )

    selection = safe_float(
        row.get(
            "table_selection_score"
        )
    )

    if (
        selection is None
        or selection <= 0
    ):
        selection = 0.68

    selection = clamp01(
        selection
    )

    qc_priority = clean_text(
        row.get(
            "table_qc_priority"
        )
    ).upper()

    qc_component = {
        "LOW": 1.0,
        "MEDIUM": 0.72,
        "HIGH": 0.35,
        "N/A": 0.75,
        "": 0.75,
    }.get(
        qc_priority,
        0.65,
    )

    review_penalty = (
        0.10
        if as_bool(
            row.get(
                "parsing_review_required"
            )
        )
        else 0.0
    )

    score = (
        0.35
        * confidence
        + 0.20
        * family_component
        + 0.15
        * agreement
        + 0.15
        * selection
        + 0.15
        * qc_component
        - review_penalty
    )

    return clamp01(
        score
    )


def structuring_quality(
    row: pd.Series,
) -> float:
    period_confidence = safe_float(
        row.get(
            "period_confidence"
        )
    )

    if period_confidence is None:
        period_confidence = 0.60

    scope_method = clean_text(
        row.get(
            "scope_inference_method"
        )
    ).lower()

    statement_method = clean_text(
        row.get(
            "statement_inference_method"
        )
    ).lower()

    scope_component = (
        1.0
        if (
            scope_method
            and "unknown"
            not in scope_method
        )
        else 0.65
    )

    statement_component = (
        1.0
        if (
            statement_method
            and "unknown"
            not in statement_method
        )
        else 0.70
    )

    return clamp01(
        0.55
        * clamp01(
            period_confidence
        )
        + 0.25
        * scope_component
        + 0.20
        * statement_component
    )


def same_nonmetric_dimensions(
    dart_row: pd.Series,
    pdf_row: pd.Series,
) -> bool:
    try:
        return (
            clean_text(
                dart_row[
                    "entity"
                ]
            )
            == clean_text(
                pdf_row[
                    "entity"
                ]
            )
            and int(
                dart_row[
                    "source_report_year"
                ]
            )
            == int(
                float(
                    pdf_row[
                        "source_report_year"
                    ]
                )
            )
            and int(
                dart_row[
                    "period_year"
                ]
            )
            == int(
                float(
                    pdf_row[
                        "period_year"
                    ]
                )
            )
            and clean_text(
                dart_row[
                    "scope"
                ]
            )
            == clean_text(
                pdf_row[
                    "scope"
                ]
            )
            and clean_text(
                dart_row[
                    "statement_type"
                ]
            )
            == clean_text(
                pdf_row[
                    "statement_type"
                ]
            )
        )

    except Exception:
        return False


def candidate_pool_for_dart_row(
    dart_row: pd.Series,
    pdf_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter by NON-VALUE dimensions only.
    """
    source_year = int(
        dart_row[
            "source_report_year"
        ]
    )

    period_year = int(
        dart_row[
            "period_year"
        ]
    )

    base = pdf_df[
        (
            pdf_df[
                "entity"
            ]
            == clean_text(
                dart_row[
                    "entity"
                ]
            )
        )
        &
        (
            pdf_df[
                "_num_source_report_year"
            ]
            == source_year
        )
        &
        (
            pdf_df[
                "_num_period_year"
            ]
            == period_year
        )
        &
        (
            pdf_df[
                "scope"
            ]
            == clean_text(
                dart_row[
                    "scope"
                ]
            )
        )
        &
        (
            pdf_df[
                "statement_type"
            ]
            == clean_text(
                dart_row[
                    "statement_type"
                ]
            )
        )
    ].copy()

    return base


@dataclass
class NonLeakyMatch:
    status: str

    selected_index: Optional[int]

    candidate_count: int

    metric_match_type: str
    metric_match_score: float

    detail_path_score: float
    parser_quality: float
    structuring_quality: float

    nonleaky_score: float

    runner_up_score: Optional[float]
    score_margin: Optional[float]


def select_pdf_candidate_nonleaky(
    dart_row: pd.Series,
    pdf_df: pd.DataFrame,
) -> NonLeakyMatch:
    """
    CRITICAL:
    This function MUST NOT use:
        DART value
        DART value_krw
        absolute difference
        relative difference

    It only uses:
        entity
        report year
        period
        scope
        statement
        metric name/canonical form
        account detail vs PDF metric path
        parser / structuring confidence
    """
    pool = candidate_pool_for_dart_row(
        dart_row,
        pdf_df,
    )

    if pool.empty:
        return NonLeakyMatch(
            status="NO_PDF_CANDIDATE",
            selected_index=None,
            candidate_count=0,
            metric_match_type="none",
            metric_match_score=0.0,
            detail_path_score=0.0,
            parser_quality=0.0,
            structuring_quality=0.0,
            nonleaky_score=0.0,
            runner_up_score=None,
            score_margin=None,
        )

    scored = []

    for index, pdf_row in pool.iterrows():
        metric_score, match_type = metric_match_score(
            dart_row,
            pdf_row,
        )

        if metric_score <= 0:
            continue

        path_score = detail_path_score(
            dart_row,
            pdf_row,
        )

        parser_score = pdf_parser_quality(
            pdf_row
        )

        structure_score = structuring_quality(
            pdf_row
        )

        # Metric identity dominates.
        # Value is intentionally absent.
        final_score = (
            0.68
            * metric_score
            + 0.08
            * path_score
            + 0.14
            * parser_score
            + 0.10
            * structure_score
        )

        scored.append(
            {
                "index": index,
                "metric_match_type": (
                    match_type
                ),
                "metric_match_score": (
                    metric_score
                ),
                "detail_path_score": (
                    path_score
                ),
                "parser_quality": (
                    parser_score
                ),
                "structuring_quality": (
                    structure_score
                ),
                "nonleaky_score": (
                    final_score
                ),
            }
        )

    if not scored:
        return NonLeakyMatch(
            status="NO_PDF_CANDIDATE",
            selected_index=None,
            candidate_count=0,
            metric_match_type="none",
            metric_match_score=0.0,
            detail_path_score=0.0,
            parser_quality=0.0,
            structuring_quality=0.0,
            nonleaky_score=0.0,
            runner_up_score=None,
            score_margin=None,
        )

    scored.sort(
        key=lambda row: (
            -row[
                "nonleaky_score"
            ],
            -row[
                "metric_match_score"
            ],
            -row[
                "detail_path_score"
            ],
            -row[
                "parser_quality"
            ],
            -row[
                "structuring_quality"
            ],
            clean_text(
                pdf_df.loc[
                    row[
                        "index"
                    ],
                    "evidence_id",
                ]
            ),
        )
    )

    top = scored[
        0
    ]

    runner_up = (
        scored[
            1
        ]
        if len(
            scored
        )
        > 1
        else None
    )

    runner_up_score = (
        runner_up[
            "nonleaky_score"
        ]
        if runner_up
        else None
    )

    margin = (
        top[
            "nonleaky_score"
        ]
        - runner_up_score
        if runner_up_score
        is not None
        else None
    )

    if (
        top[
            "nonleaky_score"
        ]
        < MIN_NONLEAKY_SCORE
    ):
        return NonLeakyMatch(
            status=(
                "NO_PDF_CANDIDATE"
            ),
            selected_index=None,
            candidate_count=len(
                scored
            ),
            metric_match_type=top[
                "metric_match_type"
            ],
            metric_match_score=top[
                "metric_match_score"
            ],
            detail_path_score=top[
                "detail_path_score"
            ],
            parser_quality=top[
                "parser_quality"
            ],
            structuring_quality=top[
                "structuring_quality"
            ],
            nonleaky_score=top[
                "nonleaky_score"
            ],
            runner_up_score=(
                runner_up_score
            ),
            score_margin=margin,
        )

    # Important:
    # do not use target value to decide between near-tied candidates.
    if (
        runner_up is not None
        and margin is not None
        and margin
        < AMBIGUITY_MARGIN
    ):
        top_evidence = clean_text(
            pdf_df.loc[
                top[
                    "index"
                ],
                "evidence_id",
            ]
        )

        second_evidence = clean_text(
            pdf_df.loc[
                runner_up[
                    "index"
                ],
                "evidence_id",
            ]
        )

        if (
            top_evidence
            != second_evidence
        ):
            return NonLeakyMatch(
                status=(
                    "AMBIGUOUS_NONLEAKY"
                ),
                selected_index=None,
                candidate_count=len(
                    scored
                ),
                metric_match_type=top[
                    "metric_match_type"
                ],
                metric_match_score=top[
                    "metric_match_score"
                ],
                detail_path_score=top[
                    "detail_path_score"
                ],
                parser_quality=top[
                    "parser_quality"
                ],
                structuring_quality=top[
                    "structuring_quality"
                ],
                nonleaky_score=top[
                    "nonleaky_score"
                ],
                runner_up_score=(
                    runner_up_score
                ),
                score_margin=margin,
            )

    return NonLeakyMatch(
        status="CANDIDATE_SELECTED",
        selected_index=int(
            top[
                "index"
            ]
        ),
        candidate_count=len(
            scored
        ),
        metric_match_type=top[
            "metric_match_type"
        ],
        metric_match_score=top[
            "metric_match_score"
        ],
        detail_path_score=top[
            "detail_path_score"
        ],
        parser_quality=top[
            "parser_quality"
        ],
        structuring_quality=top[
            "structuring_quality"
        ],
        nonleaky_score=top[
            "nonleaky_score"
        ],
        runner_up_score=(
            runner_up_score
        ),
        score_margin=margin,
    )


# ==============================================================================
# 15. VALUE COMPARISON - ONLY AFTER NON-LEAKY SELECTION
# ==============================================================================

def decimal_places_from_value_raw(
    value_raw: str,
) -> int:
    text = clean_text(
        value_raw
    )

    if not text:
        return 0

    # Get the first numeric token only.
    match = re.search(
        r"\(?\s*[-+−–]?\s*\d[\d,]*(?:\.(\d+))?",
        text,
    )

    if not match:
        return 0

    decimals = match.group(
        1
    )

    return (
        len(
            decimals
        )
        if decimals
        else 0
    )


def rounding_tolerance_krw(
    pdf_row: pd.Series,
) -> float:
    """
    Example:
        PDF unit = 백만원
        value_raw = 123
        display resolution = 1,000,000 KRW
        rounding tolerance  = 500,000 KRW

        PDF unit = 백만원
        value_raw = 123.4
        display resolution = 100,000 KRW
        tolerance = 50,000 KRW
    """
    unit_scale = safe_float(
        pdf_row.get(
            "unit_scale"
        )
    )

    if (
        unit_scale is None
        or unit_scale <= 0
    ):
        return 0.0

    decimal_places = (
        decimal_places_from_value_raw(
            clean_text(
                pdf_row.get(
                    "value_raw"
                )
            )
        )
    )

    resolution = (
        unit_scale
        * (
            10
            ** (
                -decimal_places
            )
        )
    )

    return (
        0.5
        * resolution
        + 1e-9
    )


def compare_values_after_selection(
    dart_row: pd.Series,
    pdf_row: pd.Series,
) -> dict[
    str,
    Any,
]:
    dart_value = safe_float(
        dart_row.get(
            "value_krw"
        )
    )

    pdf_value = safe_float(
        pdf_row.get(
            "value_krw"
        )
    )

    if dart_value is None:
        return {
            "value_status": (
                "NON_KRW_DART"
            ),
            "absolute_delta_krw": "",
            "relative_delta": "",
            "rounding_tolerance_krw": "",
        }

    if pdf_value is None:
        return {
            "value_status": (
                "PDF_VALUE_MISSING"
            ),
            "absolute_delta_krw": "",
            "relative_delta": "",
            "rounding_tolerance_krw": "",
        }

    delta = abs(
        dart_value
        - pdf_value
    )

    relative_delta = (
        delta
        / abs(
            dart_value
        )
        if dart_value != 0
        else (
            0.0
            if delta == 0
            else ""
        )
    )

    tolerance = (
        rounding_tolerance_krw(
            pdf_row
        )
    )

    if delta == 0:
        status = (
            "MATCH_EXACT"
        )

    elif (
        tolerance > 0
        and delta <= tolerance
    ):
        status = (
            "MATCH_ROUNDING"
        )

    else:
        status = (
            "VALUE_MISMATCH"
        )

    return {
        "value_status": status,
        "absolute_delta_krw": (
            delta
        ),
        "relative_delta": (
            relative_delta
        ),
        "rounding_tolerance_krw": (
            tolerance
        ),
    }


# ==============================================================================
# 16. REVIEW PRIORITY
# ==============================================================================

def parser_risk(
    pdf_row: Optional[
        pd.Series
    ],
) -> bool:
    if pdf_row is None:
        return False

    if as_bool(
        pdf_row.get(
            "parsing_review_required"
        )
    ):
        return True

    qc_priority = clean_text(
        pdf_row.get(
            "table_qc_priority"
        )
    ).upper()

    if qc_priority == "HIGH":
        return True

    family_support = safe_int(
        pdf_row.get(
            "table_family_support"
        )
    )

    if (
        family_support is not None
        and family_support <= 1
    ):
        agreement = safe_float(
            pdf_row.get(
                "table_content_agreement"
            )
        )

        if (
            agreement is None
            or agreement < 0.60
        ):
            return True

    return False


def structuring_risk(
    pdf_row: Optional[
        pd.Series
    ],
) -> bool:
    if pdf_row is None:
        return False

    period_confidence = safe_float(
        pdf_row.get(
            "period_confidence"
        )
    )

    if (
        period_confidence
        is not None
        and period_confidence
        < 0.80
    ):
        return True

    quality_flags = parse_jsonish_list(
        pdf_row.get(
            "quality_flags"
        )
    )

    risky_tokens = {
        "period_unknown",
        "unit_unknown",
        "scope_unknown",
        "statement_unknown",
        "header_inherited",
    }

    if any(
        flag in risky_tokens
        for flag in quality_flags
    ):
        return True

    return False


def assign_review_priority(
    alignment_status: str,
    value_status: str,
    pdf_row: Optional[
        pd.Series
    ],
) -> tuple[
    str,
    str,
]:
    if alignment_status == "AMBIGUOUS_NONLEAKY":
        return (
            "P1_ALIGNMENT",
            (
                "비값(non-value) 정보만으로 PDF 후보가 "
                "유일하게 결정되지 않음. "
                "DART 값을 보고 후보를 고르면 안 됨."
            ),
        )

    if alignment_status == "NO_PDF_CANDIDATE":
        return (
            "P2_MISSING",
            (
                "PDF structuring candidate 없음. "
                "01 table parsing 또는 02 metric/period/unit/scope "
                "구조화 누락을 확인."
            ),
        )

    if value_status == "VALUE_MISMATCH":
        if parser_risk(
            pdf_row
        ):
            return (
                "P1_PARSING",
                (
                    "DART와 값이 다르고 parser risk가 존재함. "
                    "실제 conflict 판단 전에 PDF 원본/table CSV를 확인."
                ),
            )

        if structuring_risk(
            pdf_row
        ):
            return (
                "P1_STRUCTURING",
                (
                    "DART와 값이 다르고 Period/Unit/Scope/Statement "
                    "구조화 위험이 존재함."
                ),
            )

        return (
            "P2_VALUE_MISMATCH",
            (
                "비누설 metric alignment와 parser QC는 비교적 양호하지만 "
                "DART와 값이 다름. PDF 원문/버전/단위 확인 필요."
            ),
        )

    if value_status == "MATCH_ROUNDING":
        return (
            "P3_ROUNDING",
            (
                "DART 원 단위 값과 PDF 표시단위 값의 차이가 "
                "표시 정밀도 반올림 범위 내에 있음."
            ),
        )

    if value_status == "MATCH_EXACT":
        if (
            parser_risk(
                pdf_row
            )
            or structuring_risk(
                pdf_row
            )
        ):
            return (
                "P3_MATCH_REVIEW",
                (
                    "값은 일치하지만 parser/structuring QC 때문에 "
                    "Gold 포함 전 확인 권장."
                ),
            )

        return (
            "P4_MATCH",
            (
                "비누설 alignment 후 DART KRW 값과 정확히 일치."
            ),
        )

    return (
        "P2_REVIEW",
        "수동 검수 필요",
    )


# ==============================================================================
# 17. ALIGN DART REFERENCE TO PDF EVIDENCE
# ==============================================================================

def safe_pdf_value(
    row: Optional[
        pd.Series
    ],
    column: str,
) -> Any:
    if row is None:
        return ""

    return row.get(
        column,
        "",
    )


def align_reference_to_pdf(
    reference_df: pd.DataFrame,
    pdf_df: pd.DataFrame,
) -> pd.DataFrame:
    aligned_rows = []

    for index, dart_row in reference_df.iterrows():
        match = (
            select_pdf_candidate_nonleaky(
                dart_row,
                pdf_df,
            )
        )

        pdf_row = None

        if match.selected_index is not None:
            pdf_row = pdf_df.loc[
                match.selected_index
            ]

        if (
            match.status
            == "CANDIDATE_SELECTED"
            and pdf_row
            is not None
        ):
            value_result = (
                compare_values_after_selection(
                    dart_row,
                    pdf_row,
                )
            )

            alignment_status = (
                "SELECTED_NONLEAKY"
            )

        else:
            value_result = {
                "value_status": (
                    ""
                ),
                "absolute_delta_krw": (
                    ""
                ),
                "relative_delta": (
                    ""
                ),
                "rounding_tolerance_krw": (
                    ""
                ),
            }

            alignment_status = (
                match.status
            )

        review_priority, review_reason = assign_review_priority(
            alignment_status=(
                alignment_status
            ),
            value_status=(
                value_result[
                    "value_status"
                ]
            ),
            pdf_row=pdf_row,
        )

        row = dict(
            dart_row
        )

        row.update(
            {
                # ----------------------------------
                # Non-leaky alignment diagnostics
                # ----------------------------------
                "alignment_policy": (
                    "VALUE_EXCLUDED_FROM_SELECTION"
                ),

                "alignment_status": (
                    alignment_status
                ),

                "pdf_candidate_count": (
                    match.candidate_count
                ),

                "metric_match_type": (
                    match.metric_match_type
                ),

                "metric_match_score": round(
                    match.metric_match_score,
                    6,
                ),

                "detail_path_score": round(
                    match.detail_path_score,
                    6,
                ),

                "parser_quality_score": round(
                    match.parser_quality,
                    6,
                ),

                "structuring_quality_score": round(
                    match.structuring_quality,
                    6,
                ),

                "nonleaky_alignment_score": round(
                    match.nonleaky_score,
                    6,
                ),

                "runner_up_nonleaky_score": (
                    round(
                        match.runner_up_score,
                        6,
                    )
                    if match.runner_up_score
                    is not None
                    else ""
                ),

                "nonleaky_score_margin": (
                    round(
                        match.score_margin,
                        6,
                    )
                    if match.score_margin
                    is not None
                    else ""
                ),

                # ----------------------------------
                # Selected PDF provenance
                # ----------------------------------
                "matched_pdf_evidence_id": (
                    safe_pdf_value(
                        pdf_row,
                        "evidence_id",
                    )
                ),

                "matched_pdf_block_id": (
                    safe_pdf_value(
                        pdf_row,
                        "block_id",
                    )
                ),

                "matched_pdf_page": (
                    safe_pdf_value(
                        pdf_row,
                        "page",
                    )
                ),

                "matched_pdf_table_index": (
                    safe_pdf_value(
                        pdf_row,
                        "table_index",
                    )
                ),

                "matched_pdf_table_id": (
                    safe_pdf_value(
                        pdf_row,
                        "table_id",
                    )
                ),

                "matched_pdf_table_group_id": (
                    safe_pdf_value(
                        pdf_row,
                        "table_group_id",
                    )
                ),

                "pdf_metric_raw": (
                    safe_pdf_value(
                        pdf_row,
                        "metric_raw",
                    )
                ),

                "pdf_metric_path_raw": (
                    safe_pdf_value(
                        pdf_row,
                        "metric_path_raw",
                    )
                ),

                "pdf_value_raw": (
                    safe_pdf_value(
                        pdf_row,
                        "value_raw",
                    )
                ),

                "pdf_value_krw": (
                    safe_pdf_value(
                        pdf_row,
                        "value_krw",
                    )
                ),

                "pdf_unit": (
                    safe_pdf_value(
                        pdf_row,
                        "unit",
                    )
                ),

                "pdf_unit_scale": (
                    safe_pdf_value(
                        pdf_row,
                        "unit_scale",
                    )
                ),

                "pdf_period_raw": (
                    safe_pdf_value(
                        pdf_row,
                        "period_raw",
                    )
                ),

                "pdf_parser_engine": (
                    safe_pdf_value(
                        pdf_row,
                        "parser_engine",
                    )
                ),

                "pdf_parser_strategy": (
                    safe_pdf_value(
                        pdf_row,
                        "parser_strategy",
                    )
                ),

                "pdf_table_family_support": (
                    safe_pdf_value(
                        pdf_row,
                        "table_family_support",
                    )
                ),

                "pdf_table_content_agreement": (
                    safe_pdf_value(
                        pdf_row,
                        "table_content_agreement",
                    )
                ),

                "pdf_table_selection_score": (
                    safe_pdf_value(
                        pdf_row,
                        "table_selection_score",
                    )
                ),

                "pdf_table_qc_priority": (
                    safe_pdf_value(
                        pdf_row,
                        "table_qc_priority",
                    )
                ),

                "pdf_table_qc_score": (
                    safe_pdf_value(
                        pdf_row,
                        "table_qc_score",
                    )
                ),

                "pdf_parsing_review_required": (
                    safe_pdf_value(
                        pdf_row,
                        "parsing_review_required",
                    )
                ),

                "pdf_period_inference_method": (
                    safe_pdf_value(
                        pdf_row,
                        "period_inference_method",
                    )
                ),

                "pdf_period_confidence": (
                    safe_pdf_value(
                        pdf_row,
                        "period_confidence",
                    )
                ),

                "pdf_unit_inference_method": (
                    safe_pdf_value(
                        pdf_row,
                        "unit_inference_method",
                    )
                ),

                "pdf_scope_inference_method": (
                    safe_pdf_value(
                        pdf_row,
                        "scope_inference_method",
                    )
                ),

                "pdf_statement_inference_method": (
                    safe_pdf_value(
                        pdf_row,
                        "statement_inference_method",
                    )
                ),

                "pdf_quality_flags": (
                    safe_pdf_value(
                        pdf_row,
                        "quality_flags",
                    )
                ),

                # ----------------------------------
                # Value comparison, after selection
                # ----------------------------------
                **value_result,

                # ----------------------------------
                # Review
                # ----------------------------------
                "review_priority": (
                    review_priority
                ),

                "review_reason": (
                    review_reason
                ),

                "manual_verified": "",
                "manual_label": "",
                "manual_note": "",
            }
        )

        aligned_rows.append(
            row
        )

        if (
            index + 1
        ) % 1000 == 0:
            print(
                f"[ALIGN] "
                f"{index + 1}/"
                f"{len(reference_df)}"
            )

    aligned = pd.DataFrame(
        aligned_rows
    )

    if (
        aligned.empty
        or not MANY_TO_ONE_REVIEW
    ):
        return aligned

    # ------------------------------------------------------------------
    # Many DART rows mapped to one PDF evidence.
    # This can happen in SCE/custom accounts and must be manually checked.
    # ------------------------------------------------------------------

    selected = aligned[
        aligned[
            "matched_pdf_evidence_id"
        ]
        != ""
    ].copy()

    counts = (
        selected[
            "matched_pdf_evidence_id"
        ]
        .value_counts()
    )

    duplicated_ids = set(
        counts[
            counts > 1
        ].index
    )

    if duplicated_ids:
        mask = aligned[
            "matched_pdf_evidence_id"
        ].isin(
            duplicated_ids
        )

        aligned.loc[
            mask,
            "many_to_one_alignment",
        ] = True

        for row_index in aligned[
            mask
        ].index:
            current_priority = clean_text(
                aligned.at[
                    row_index,
                    "review_priority",
                ]
            )

            if current_priority.startswith(
                "P4"
            ):
                aligned.at[
                    row_index,
                    "review_priority",
                ] = (
                    "P1_ALIGNMENT"
                )

                aligned.at[
                    row_index,
                    "review_reason",
                ] = (
                    "하나의 PDF Evidence가 여러 DART reference row에 "
                    "대응됨. account_detail/metric_path 확인 필요."
                )

    if "many_to_one_alignment" not in aligned.columns:
        aligned[
            "many_to_one_alignment"
        ] = False

    aligned[
        "many_to_one_alignment"
    ] = aligned[
        "many_to_one_alignment"
    ].fillna(
        False
    )

    return aligned


# ==============================================================================
# 18. REVIEW QUEUE
# ==============================================================================

REVIEW_PRIORITY_ORDER = {
    "P0_VERSION": 0,
    "P1_ALIGNMENT": 1,
    "P1_PARSING": 2,
    "P1_STRUCTURING": 3,
    "P2_VALUE_MISMATCH": 4,
    "P2_MISSING": 5,
    "P2_REVIEW": 6,
    "P3_ROUNDING": 7,
    "P3_MATCH_REVIEW": 8,
    "P4_MATCH": 9,
}


def build_review_queue(
    aligned: pd.DataFrame,
) -> pd.DataFrame:
    if aligned.empty:
        return aligned.copy()

    review = aligned.copy()

    review[
        "_priority_order"
    ] = (
        review[
            "review_priority"
        ]
        .map(
            REVIEW_PRIORITY_ORDER
        )
        .fillna(
            99
        )
    )

    if "absolute_delta_krw" in review.columns:
        review[
            "_abs_delta"
        ] = pd.to_numeric(
            review[
                "absolute_delta_krw"
            ],
            errors="coerce",
        ).fillna(
            -1
        )

    else:
        review[
            "_abs_delta"
        ] = -1

    review = (
        review
        .sort_values(
            [
                "_priority_order",
                "_abs_delta",
                "entity",
                "source_report_year",
                "scope",
                "period_year",
                "statement_type",
                "metric_raw",
            ],
            ascending=[
                True,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
            ],
        )
        .drop(
            columns=[
                "_priority_order",
                "_abs_delta",
            ]
        )
    )

    return review


# ==============================================================================
# 19. BUILD MODE
# ==============================================================================

def build_mode() -> None:
    global STRUCTURE_MODULE

    STRUCTURE_MODULE = load_script02_module()

    document_manifest = (
        load_document_manifest()
    )

    pdf_evidence = (
        load_pdf_evidence()
    )

    api_key = get_api_key()
    session = build_session()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "="
        * 100
    )

    print(
        "CAUSE-RAG | "
        "OpenDART Gold Builder"
    )

    print(
        "="
        * 100
    )

    print(
        "Reference    : "
        "OpenDART fnlttSinglAcntAll"
    )

    print(
        "Report code  : "
        "11011 (사업보고서)"
    )

    print(
        "Scopes       : "
        "CFS + OFS"
    )

    print(
        "Alignment    : "
        "VALUE EXCLUDED"
    )

    print(
        "Final Gold   : "
        "manual_verified=Y only"
    )

    print(
        "="
        * 100
    )

    (
        reference_df,
        exact_filing_df,
        version_mismatch_df,
    ) = build_dart_reference(
        session=session,
        api_key=api_key,
        document_manifest=(
            document_manifest
        ),
    )

    exact_filing_df.to_csv(
        EXACT_FILING_MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    version_mismatch_df.to_csv(
        VERSION_MISMATCH_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if not version_mismatch_df.empty:
        save_json(
            GOLD_MANIFEST_PATH,
            {
                "status": (
                    "STOPPED_VERSION_MISMATCH"
                ),
                "version_mismatch_count": (
                    len(
                        version_mismatch_df
                    )
                ),
                "methodology": (
                    "Exact source filing rcept_no must match "
                    "fnlttSinglAcntAll rcept_no. "
                    "No silent version substitution is allowed."
                ),
                "version_mismatch_file": str(
                    VERSION_MISMATCH_PATH.relative_to(
                        BASE_DIR
                    )
                ),
            },
        )

        raise RuntimeError(
            "\n"
            "============================================================\n"
            "STRICT VERSION MISMATCH\n"
            "============================================================\n"
            "OpenDART 전체재무제표 API가 현재 PDF와 다른 "
            "rcept_no를 반환했습니다.\n"
            "자동으로 다른 버전을 Gold로 사용하지 않았습니다.\n\n"
            f"확인: {VERSION_MISMATCH_PATH}\n"
            f"정확한 filing XBRL도 "
            f"{RAW_XBRL_MISMATCH_DIR}에 진단용으로 저장했습니다.\n"
            "============================================================\n"
        )

    if reference_df.empty:
        raise RuntimeError(
            "DART reference evidence가 0개입니다."
        )

    # ------------------------------------------------------------------
    # Save reference before evaluating parser
    # ------------------------------------------------------------------

    reference_df.to_csv(
        REFERENCE_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    write_jsonl(
        REFERENCE_JSONL_PATH,
        reference_df.to_dict(
            orient="records"
        ),
    )

    # ------------------------------------------------------------------
    # Non-leaky alignment
    # ------------------------------------------------------------------

    print(
        "\n[ALIGN] "
        "DART value를 사용하지 않고 PDF Evidence 선택"
    )

    aligned = align_reference_to_pdf(
        reference_df=reference_df,
        pdf_df=pdf_evidence,
    )

    aligned.to_csv(
        ALIGNMENT_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Gold review template = same rows with manual columns.
    aligned.to_csv(
        GOLD_REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    review_queue = build_review_queue(
        aligned
    )

    review_queue.to_csv(
        REVIEW_QUEUE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    alignment_counts = (
        dict(
            Counter(
                aligned[
                    "alignment_status"
                ].astype(
                    str
                )
            )
        )
        if not aligned.empty
        else {}
    )

    value_counts = (
        dict(
            Counter(
                aligned[
                    "value_status"
                ].astype(
                    str
                )
            )
        )
        if (
            not aligned.empty
            and "value_status"
            in aligned.columns
        )
        else {}
    )

    review_counts = (
        dict(
            Counter(
                aligned[
                    "review_priority"
                ].astype(
                    str
                )
            )
        )
        if (
            not aligned.empty
            and "review_priority"
            in aligned.columns
        )
        else {}
    )

    manifest = {
        "status": (
            "REFERENCE_AND_REVIEW_READY"
        ),

        "dataset_name": (
            "CAUSE-RAG OpenDART "
            "Structured Evidence Gold"
        ),

        "source_documents": (
            6
        ),

        "reference_count": (
            len(
                reference_df
            )
        ),

        "pdf_alignment_candidate_count": (
            len(
                pdf_evidence
            )
        ),

        "exact_filing_count": (
            len(
                exact_filing_df
            )
        ),

        "version_mismatch_count": 0,

        "alignment_policy": {
            "selection_uses_dart_value": (
                False
            ),
            "selection_dimensions": [
                "entity",
                "source_report_year",
                "period_year",
                "scope",
                "statement_type",
                "metric_norm/canonical",
                "account_detail vs metric_path",
                "parser quality",
                "structuring quality",
            ],
            "value_comparison_occurs": (
                "after candidate selection"
            ),
            "ambiguity_policy": (
                "never resolve near-ties using DART value"
            ),
        },

        "gold_policy": (
            "Exact filing OpenDART structured reference "
            "+ manual verification"
        ),

        "alignment_status_counts": (
            alignment_counts
        ),

        "value_status_counts": (
            value_counts
        ),

        "review_priority_counts": (
            review_counts
        ),

        "files": {
            "exact_filing_manifest": str(
                EXACT_FILING_MANIFEST_PATH.relative_to(
                    BASE_DIR
                )
            ),
            "dart_reference": str(
                REFERENCE_CSV_PATH.relative_to(
                    BASE_DIR
                )
            ),
            "alignment": str(
                ALIGNMENT_CSV_PATH.relative_to(
                    BASE_DIR
                )
            ),
            "gold_review_template": str(
                GOLD_REVIEW_PATH.relative_to(
                    BASE_DIR
                )
            ),
            "review_queue": str(
                REVIEW_QUEUE_PATH.relative_to(
                    BASE_DIR
                )
            ),
        },
    }

    save_json(
        GOLD_MANIFEST_PATH,
        manifest,
    )

    print(
        "\n"
        + "="
        * 100
    )

    print(
        "[SUCCESS]"
    )

    print(
        f"DART reference rows : "
        f"{len(reference_df)}"
    )

    print(
        f"PDF candidates      : "
        f"{len(pdf_evidence)}"
    )

    print(
        f"alignment status    : "
        f"{alignment_counts}"
    )

    print(
        f"value status        : "
        f"{value_counts}"
    )

    print(
        f"review priority     : "
        f"{review_counts}"
    )

    print()

    print(
        f"Review template     : "
        f"{GOLD_REVIEW_PATH}"
    )

    print(
        f"Priority queue      : "
        f"{REVIEW_QUEUE_PATH}"
    )

    print(
        "="
        * 100
    )

    print(
        "\n다음 단계:\n"
        "1. review_queue.csv를 P1부터 확인\n"
        "2. gold_review_template.csv에서 실제 PDF와 DART reference를 검수\n"
        "3. 확정 행에 manual_verified=Y 입력\n"
        "4. python 03_build_dart_gold.py --finalize\n"
    )


# ==============================================================================
# 20. FINALIZE GOLD
# ==============================================================================

MANUAL_YES = {
    "y",
    "yes",
    "true",
    "1",
    "확인",
    "검수완료",
    "verified",
}

MANUAL_EXCLUDE_LABELS = {
    "exclude",
    "제외",
    "not_gold",
}


def finalize_mode() -> None:
    if not GOLD_REVIEW_PATH.exists():
        raise SystemExit(
            "\n[ERROR] gold_review_template.csv가 없습니다.\n"
            "먼저:\n"
            "    python 03_build_dart_gold.py\n"
            "를 실행하세요.\n"
        )

    review = pd.read_csv(
        GOLD_REVIEW_PATH,
        dtype=str,
    ).fillna(
        ""
    )

    if "manual_verified" not in review.columns:
        raise RuntimeError(
            "manual_verified column이 없습니다."
        )

    verified_mask = (
        review[
            "manual_verified"
        ]
        .str.strip()
        .str.lower()
        .isin(
            MANUAL_YES
        )
    )

    if "manual_label" in review.columns:
        exclude_mask = (
            review[
                "manual_label"
            ]
            .str.strip()
            .str.lower()
            .isin(
                MANUAL_EXCLUDE_LABELS
            )
        )

    else:
        exclude_mask = pd.Series(
            False,
            index=review.index,
        )

    gold = review[
        verified_mask
        & ~exclude_mask
    ].copy()

    if gold.empty:
        raise SystemExit(
            "\n[ERROR] manual_verified=Y인 Gold 행이 없습니다.\n"
        )

    if gold[
        "dart_reference_id"
    ].duplicated().any():
        duplicated = gold[
            gold[
                "dart_reference_id"
            ].duplicated(
                keep=False
            )
        ][
            [
                "dart_reference_id",
                "entity",
                "source_report_year",
                "metric_raw",
            ]
        ]

        raise RuntimeError(
            "Gold dart_reference_id 중복:\n"
            + duplicated.to_string(
                index=False
            )
        )

    gold[
        "gold_status"
    ] = (
        "MANUALLY_VERIFIED"
    )

    gold[
        "gold_source_policy"
    ] = (
        "OpenDART exact filing reference "
        "+ manual PDF verification"
    )

    # ------------------------------------------------------------------
    # Full Gold
    # ------------------------------------------------------------------

    gold.to_csv(
        FINAL_GOLD_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    write_jsonl(
        FINAL_GOLD_JSONL_PATH,
        gold.to_dict(
            orient="records"
        ),
    )

    # ------------------------------------------------------------------
    # Core Gold for later CAUSE-RAG experiments
    # ------------------------------------------------------------------

    core_columns = [
        "dart_reference_id",

        "entity",
        "stock_code",

        "metric_raw",
        "metric_norm",
        "metric_canonical",
        "account_id",
        "account_detail",

        "value_raw",
        "value",
        "dart_currency",
        "unit",
        "unit_scale",
        "value_krw",

        "period_raw",
        "period_year",

        "scope",

        "source_report_year",
        "filing_date",
        "filing_version",
        "version_id",

        "statement_type",
        "statement_name",
        "reporting_role",

        "rcept_no",
        "report_name",
        "source_pdf_file",

        "alignment_key",

        "manual_verified",
        "manual_label",
        "manual_note",

        "gold_status",
        "gold_source_policy",
    ]

    existing_core_columns = [
        column
        for column in core_columns
        if column in gold.columns
    ]

    gold[
        existing_core_columns
    ].to_csv(
        FINAL_GOLD_CORE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Update manifest
    # ------------------------------------------------------------------

    if GOLD_MANIFEST_PATH.exists():
        try:
            manifest = json.loads(
                GOLD_MANIFEST_PATH.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            manifest = {}

    else:
        manifest = {}

    manifest.update(
        {
            "status": (
                "FINAL_GOLD_READY"
            ),

            "final_gold_count": (
                len(
                    gold
                )
            ),

            "final_gold_files": {
                "full_csv": str(
                    FINAL_GOLD_CSV_PATH.relative_to(
                        BASE_DIR
                    )
                ),
                "full_jsonl": str(
                    FINAL_GOLD_JSONL_PATH.relative_to(
                        BASE_DIR
                    )
                ),
                "core_csv": str(
                    FINAL_GOLD_CORE_PATH.relative_to(
                        BASE_DIR
                    )
                ),
            },
        }
    )

    save_json(
        GOLD_MANIFEST_PATH,
        manifest,
    )

    print(
        "="
        * 100
    )

    print(
        "[FINAL GOLD READY]"
    )

    print(
        f"verified rows : "
        f"{len(gold)}"
    )

    print(
        f"full CSV      : "
        f"{FINAL_GOLD_CSV_PATH}"
    )

    print(
        f"core CSV      : "
        f"{FINAL_GOLD_CORE_PATH}"
    )

    print(
        "="
        * 100
    )


# ==============================================================================
# 21. SYNTHETIC LEAKAGE TEST
# ==============================================================================

def run_internal_alignment_test() -> None:
    """
    A tiny sanity check:
    candidate selection must prefer metric/parser evidence,
    not the numerically closer target.
    """
    dart = pd.Series(
        {
            "entity": "TEST",
            "source_report_year": 2025,
            "period_year": 2025,
            "scope": "consolidated",
            "statement_type": "BS",
            "metric_norm": "자산총계",
            "metric_canonical": "total_assets",
            "account_detail": "",
            "value_krw": 1000.0,
        }
    )

    pdf = pd.DataFrame(
        [
            {
                "evidence_id": "good_nonleaky",
                "entity": "TEST",
                "source_report_year": "2025",
                "period_year": "2025",
                "scope": "consolidated",
                "statement_type": "BS",
                "metric_norm": "자산총계",
                "metric_canonical": "total_assets",
                "metric_path_raw": "자산총계",
                "value_krw": "999999999",
                "confidence": "0.99",
                "table_family_support": "2",
                "table_content_agreement": "0.95",
                "table_selection_score": "0.95",
                "table_qc_priority": "LOW",
                "parsing_review_required": "False",
                "period_confidence": "1.0",
                "scope_inference_method": "explicit",
                "statement_inference_method": "explicit",
            },
            {
                "evidence_id": "bad_metric_but_value_close",
                "entity": "TEST",
                "source_report_year": "2025",
                "period_year": "2025",
                "scope": "consolidated",
                "statement_type": "BS",
                "metric_norm": "부채총계",
                "metric_canonical": "total_liabilities",
                "metric_path_raw": "부채총계",
                "value_krw": "1000",
                "confidence": "1.0",
                "table_family_support": "2",
                "table_content_agreement": "1.0",
                "table_selection_score": "1.0",
                "table_qc_priority": "LOW",
                "parsing_review_required": "False",
                "period_confidence": "1.0",
                "scope_inference_method": "explicit",
                "statement_inference_method": "explicit",
            },
        ]
    )

    pdf[
        "_num_source_report_year"
    ] = pd.to_numeric(
        pdf[
            "source_report_year"
        ]
    )

    pdf[
        "_num_period_year"
    ] = pd.to_numeric(
        pdf[
            "period_year"
        ]
    )

    match = (
        select_pdf_candidate_nonleaky(
            dart,
            pdf,
        )
    )

    if (
        match.selected_index
        is None
        or pdf.loc[
            match.selected_index,
            "evidence_id",
        ]
        != "good_nonleaky"
    ):
        raise RuntimeError(
            "INTERNAL TEST FAILED: "
            "alignment may be using target value leakage."
        )


# ==============================================================================
# 22. CLI / MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "gold_review_template.csv에서 "
            "manual_verified=Y인 행만 최종 Gold로 저장"
        ),
    )

    return parser.parse_args()


def main() -> None:
    global STRUCTURE_MODULE

    args = parse_args()

    STRUCTURE_MODULE = load_script02_module()

    # Always run the non-leakage sanity test before API work.
    run_internal_alignment_test()

    if args.finalize:
        finalize_mode()

    else:
        build_mode()


if __name__ == "__main__":
    main()
