#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CAUSE-RAG 통합 실행기: 질문 하나를 입력하면 최종 답변까지 한 번에 수행한다.

    python main.py --question "2025년 연결 기준 삼성전자의 영업이익은?"

파이프라인
----------
  [준비 단계 - 입력이 바뀐 경우에만 실행]
    01 PDF 파싱  →  02 근거 구조화  →  05 검색 인덱스(SQLite) 구축
  [질문 단계 - 질문마다 실행]
    05 하이브리드 검색·비전 재순위화(Top-5)
      → 04 질문·근거 정규화(LLM 구조화 + Python 규칙 계산)
      → 06 충돌 판별(no / apparent / genuine)
      → 07 VLM 최종 답변

재사용(= 같은 일을 두 번 하지 않는 규칙)
----------------------------------------
  * 01: 산출물이 PDF보다 새로우면 건너뜀            (--force-parse 로 강제 재실행)
  * 02: 산출물이 01 결과보다 새로우면 건너뜀        (--force-structure 로 강제 재실행)
  * 05 인덱스: 02 결과·PDF·임베딩 설정이 같으면 건너뜀 (--rebuild-index 로 강제 재구축)
  * 05 비전 재순위화: 같은 (질문, 페이지)는 디스크 캐시 재사용 (--no-vision-cache 로 끔)
  * 04 LLM 추출: 같은 요청은 dataset/04_normalized/cache 재사용   (--no-cache 로 끔)
  * 여러 질문(--question 반복 / --question-file)은 인덱스·모델 객체를 한 번만 로드해 공유

03(OpenAI/DART 기반 Gold 구축)은 질문 답변 경로에 쓰이지 않으므로 기본 실행에서 제외한다.
필요할 때만 --with-gold 로 실행한다(수동 검증 후 `python 03_build_dart_gold.py --finalize`).

질문마다 중간 산출물은 dataset/runs/<실행시각>/q01/ 아래에 분리 저장되어 서로 덮어쓰지 않는다.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from cause_rag_arbitration import run_arbitration

CAUSE_RAG_UPGRADE_VERSION = "cause-rag-source-identity-guards-20261105-v5"

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdf_data"
DATASET_DIR = BASE_DIR / "dataset"

# 각 단계 스크립트가 만드는 대표 산출물(README 3.1 기준)
PARSED_BLOCKS = DATASET_DIR / "01_parsed" / "parsed_blocks.jsonl"
DOCUMENT_MANIFEST = DATASET_DIR / "01_parsed" / "document_manifest.csv"
STRUCTURED_EVIDENCE = DATASET_DIR / "02_structured" / "structured_evidence.jsonl"
SHARED_CACHE_04 = DATASET_DIR / "04_normalized" / "cache"  # 실행 간 공유되는 LLM 캐시
RUNS_DIR = DATASET_DIR / "runs"


def _scope_from_pdf_heading(line: str) -> str | None:
    """Recognize *section headings*, not loose financial terms in tables."""
    compact = re.sub(r"\s+", "", line)
    summary = re.fullmatch(
        r"\(\d+\)요약재무정보\((연결|별도)재무제표기준\)", compact
    )
    if summary:
        return "CFS" if summary.group(1) == "연결" else "OFS"
    # Samsung's annual reports use these exact titles on adjacent pages.
    # They are headings, not mentions of 연결/별도 in ordinary body text.
    simple_summary = re.fullmatch(
        r"[가-힣][.)]?요약(연결|별도)재무정보", compact
    )
    if simple_summary:
        return "CFS" if simple_summary.group(1) == "연결" else "OFS"
    if re.fullmatch(r"\d+(?:[-.]\d+)?\.?연결재무제표", compact):
        return "CFS"
    # DART's '4. 재무제표' section contains separate financial statements.
    if re.fullmatch(r"\d+(?:[-.]\d+)?\.?재무제표", compact):
        return "OFS"
    return None


@lru_cache(maxsize=512)
def _scope_from_nearby_pdf_sections(pdf_file: str, page_number: int) -> str | None:
    """Verify scope using nearby source-PDF section titles.

    Stage02 sometimes carries 'consolidated' across a section boundary:
    the subsequent '4. 재무제표' is separate, even if section_path is stale.
    Three local pages suffice for this guard and are cached across questions.
    No network access, LLM call, or PDF modification occurs.
    """
    if page_number < 1:
        return None
    try:
        import fitz

        with fitz.open(pdf_file) as source:
            if page_number > len(source):
                return None
            headings: list[str] = []
            for page_index in range(max(0, page_number - 3), page_number):
                for line in source[page_index].get_text("text").splitlines():
                    scope = _scope_from_pdf_heading(line)
                    if scope:
                        headings.append(scope)
            return headings[-1] if headings else None
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _source_pdf_scope(meta: dict[str, Any]) -> str | None:
    """Read only the PDF referenced by this evidence, if available locally."""
    try:
        page_number = int(meta.get("page") or 0)
    except (TypeError, ValueError):
        return None
    if page_number < 1:
        return None
    pdf_path = Path(str(meta.get("pdf_path") or ""))
    if not pdf_path.is_file():
        filename = Path(str(meta.get("source_file") or "")).name
        pdf_path = PDF_DIR / filename
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return None
    return _scope_from_nearby_pdf_sections(str(pdf_path.resolve()), page_number)


# ==============================================================================
# 유틸
# ==============================================================================


def load_stage(filename: str, module_name: str) -> Any:
    """파일명이 숫자로 시작하는 단계 스크립트를 모듈로 로드한다(이미 로드했으면 재사용)."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = BASE_DIR / filename
    if not path.exists():
        raise SystemExit(f"[ERROR] 단계 스크립트가 없습니다: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[ERROR] 모듈을 로드할 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclass/pydantic이 모듈을 찾을 수 있도록 먼저 등록
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def patched_argv(*argv: str) -> Iterator[None]:
    """argparse를 쓰는 단계 스크립트의 main()을 프로세스 안에서 호출하기 위한 임시 argv."""
    previous = sys.argv
    sys.argv = [previous[0], *argv]
    try:
        yield
    finally:
        sys.argv = previous


def is_stale(outputs: list[Path], inputs: list[Path]) -> bool:
    """산출물이 없거나, 입력 중 하나라도 가장 오래된 산출물보다 새로우면 True."""
    if any(not path.exists() for path in outputs):
        return True
    newest_input = max(path.stat().st_mtime_ns for path in inputs)
    oldest_output = min(path.stat().st_mtime_ns for path in outputs)
    return newest_input > oldest_output


def banner(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


class Timer:
    def __init__(self) -> None:
        self.laps: dict[str, float] = {}

    @contextlib.contextmanager
    def lap(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.laps[name] = round(time.perf_counter() - start, 2)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


# ==============================================================================
# 준비 단계: 01 / 02
# ==============================================================================


def ensure_parsed(args: argparse.Namespace) -> None:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"[ERROR] 원본 PDF가 없습니다: {PDF_DIR}/*.pdf")
    if not args.force_parse and not is_stale([PARSED_BLOCKS, DOCUMENT_MANIFEST], pdfs + [BASE_DIR / '01_parse_audit_reports.py']):
        print("[01] PDF 파싱: 최신 상태 → 건너뜀 (강제 재실행: --force-parse)")
        return
    banner("[01] PDF 본문·표 파싱")
    stage = load_stage("01_parse_audit_reports.py", "cause_rag_stage01")
    with patched_argv(*(["--save-debug-pdfs"] if args.save_debug_pdfs else [])):
        stage.main()


def ensure_structured(args: argparse.Namespace) -> None:
    if not args.force_structure and not is_stale([STRUCTURED_EVIDENCE], [PARSED_BLOCKS, BASE_DIR / '02_structure_evidence.py']):
        print("[02] 근거 구조화: 최신 상태 → 건너뜀 (강제 재실행: --force-structure)")
        return
    banner("[02] 근거 구조화")
    stage = load_stage("02_structure_evidence.py", "cause_rag_stage02")
    flags = ["--include-unknown-nonfinancial"] if args.include_unknown_nonfinancial else []
    with patched_argv(*flags):
        stage.main()


def apply_source_provenance_guard(bundle: dict[str, Any]) -> dict[str, Any]:
    """
    04단계의 정규화 결과를 원본 근거 메타데이터로 재검증합니다.

    1. 원문 재무제표의 기간을 우선합니다.
    2. 요약재무정보의 연결/별도 구분을 복원합니다.
    3. 증감액을 절대금액과 구분합니다.
    """
    import re

    question = bundle.get("question") or {}
    evidences = bundle.get("evidences") or []

    raw_question = str(
        bundle.get("raw_question")
        or question.get("raw_content")
        or ""
    )

    def source_scope(raw: Any) -> str | None:
        """Stage02 emits 'consolidated'/'separate', not just CFS/OFS."""
        value = re.sub(r"\s+", "", str(raw or "")).casefold()
        if value in {"cfs", "consolidated", "연결", "연결재무제표", "연결기준"}:
            return "CFS"
        if value in {"ofs", "separate", "standalone", "individual", "별도", "개별", "별도재무제표"}:
            return "OFS"
        return None

    def verified_summary_scope(meta: dict[str, Any]) -> str | None:
        """Summary financial tables identify scope in their section heading."""
        statement = str(meta.get("statement_type") or meta.get("statement") or "").upper()
        if statement != "SUMMARY":
            return None
        sections = meta.get("section_path") or []
        if isinstance(sections, str):
            sections = [sections]
        for item in reversed(sections):
            title = re.sub(r"\s+", "", str(item))
            if "요약" in title and "별도" in title:
                return "OFS"
            if "요약" in title and "연결" in title:
                return "CFS"
        return None

    # Cash-flow movements must not be compared with balance-sheet balances.
    balance_sheet_stock_metrics = {
        "current_assets", "noncurrent_assets", "total_assets",
        "non_current_assets",
        "current_liabilities", "noncurrent_liabilities", "total_liabilities",
        "non_current_liabilities",
        "total_equity", "retained_earnings", "inventories",
        "trade_receivables", "property_plant_equipment", "intangible_assets",
        "short_term_financial_products",
    }

    def is_change_column(ev: dict) -> bool:
        meta = ev.get("source_metadata") or {}

        column = re.sub(
            r"\s+", "", str(meta.get("column_header") or "")
        )
        raw = str(ev.get("raw_content") or "")

        return (
            "증감" in column
            or bool(
                re.search(
                    r"\|\s*(?:전년\s*대비\s*)?증감(?:률)?\s*:",
                    raw,
                )
            )
        )

    # --------------------------------------------------
    # 1. 질문의 수치 유형 판별
    # --------------------------------------------------
    # Question value-type must be fixed before evidence comparison. Otherwise
    # an UNKNOWN amount question can incorrectly accept CF changes as stocks.
    q_value_type = str(
        question.get("value_type") or "UNKNOWN"
    ).upper()

    change_words = (
        "증감", "증가", "감소", "변화",
        "차이", "전년 대비", "성장률",
    )

    amount_words = (
        "얼마", "금액", "총계",
        "매출액", "영업이익",
        "수치", "원 단위", "알려주세요", "확인할 수 있", "무엇인가요",
    )

    if q_value_type == "UNKNOWN":
        if any(word in raw_question for word in change_words):
            question["value_type"] = "CHANGE"
            question["value_type_normalization_method"] = "QUESTION_CHANGE_RULE"
        elif (
            any(word in raw_question for word in amount_words)
            or bool(question.get("metric_key"))
        ):
            question["value_type"] = "ABSOLUTE"
            question["value_type_normalization_method"] = "QUESTION_AMOUNT_RULE"

    # --------------------------------------------------
    # 2. 각 근거의 원본 조건 복원
    # --------------------------------------------------
    corrections: list[dict[str, Any]] = []
    monitored = (
        "period_start", "period_end", "period_type", "scope",
        "statement", "value_type", "metric", "metric_key",
    )
    for ev in evidences:
        before = {key: ev.get(key) for key in monitored}
        meta = ev.get("source_metadata") or {}

        # A. 기간: 제77기 → 2024년 등 원본 매핑 우선
        period_year = str(meta.get("period_year") or "")
        period_raw = str(meta.get("period_raw") or "")

        reliable_fiscal_period = (
            meta.get("period_inference_method") == "fiscal_term_map"
            and bool(re.fullmatch(r"20\d{2}", period_year))
            and bool(re.search(r"제\s*\d+\s*기", period_raw))
        )
        # Stage02's explicit year belongs to the SOURCE TABLE COLUMN. The
        # normalization LLM occasionally replaces a 2023 comparative column
        # with the requested 2025 year; the row/column header must win.
        explicit_column_year = (
            meta.get("period_inference_method") == "explicit_year"
            and bool(re.fullmatch(r"20\d{2}", period_year))
            and bool(re.search(rf"(?<!\d){re.escape(period_year)}\s*년", period_raw))
            and bool(re.search(
                rf"(?<!\d){re.escape(period_year)}\s*년",
                str(meta.get("column_header") or period_raw),
            ))
        )

        current_start = str(ev.get("period_start") or "")
        current_end = str(ev.get("period_end") or "")

        # An annual fiscal-term column is a stronger source signal than the
        # LLM's possibly invented year. Include year-end point-in-time dates,
        # but never overwrite quarter/month-specific values.
        annual_shape = (
            bool(re.fullmatch(r"20\d{2}", current_start))
            and bool(re.fullmatch(r"20\d{2}", current_end))
        ) or (
            bool(re.fullmatch(r"20\d{2}_12_31", current_end))
            and bool(re.fullmatch(r"20\d{2}_(?:01_01|12_31)", current_start))
        )
        if (reliable_fiscal_period or explicit_column_year) and annual_shape:
            period_type = str(ev.get("period_type") or "").upper()
            if period_type == "POINT_IN_TIME":
                corrected = f"{period_year}_12_31"
                ev["period_start"] = corrected
                ev["period_end"] = corrected
            else:
                ev["period_start"] = period_year
                ev["period_end"] = period_year
            ev["period_normalization_method"] = (
                "SOURCE_FISCAL_TERM_OVERRIDE"
            )

        # B. 연결/별도: verify the actual PDF's nearest section heading first.
        # Stage02 scope can be stale across a '연결재무제표' -> '재무제표'
        # transition (confirmed on Hyundai's 2025 PDF, page 218).
        native_scope = _source_pdf_scope(meta)
        page_title_scope = source_scope(meta.get("source_title_verified_scope"))
        canonical_scope = source_scope(meta.get("scope"))
        if native_scope is not None:
            ev["scope"] = native_scope
            ev["scope_normalization_method"] = "PDF_NEARBY_SECTION_OVERRIDE"
        elif page_title_scope is not None:
            # Populated by Stage05 ONLY when the source page's native PDF text
            # itself contains a recognized, explicit financial-summary title.
            ev["scope"] = page_title_scope
            ev["scope_normalization_method"] = "PDF_SUMMARY_PAGE_TITLE_OVERRIDE"
        elif canonical_scope is not None:
            ev["scope"] = canonical_scope
            ev["scope_normalization_method"] = (
                "SOURCE_SCOPE_OVERRIDE"
            )
        else:
            heading_scope = verified_summary_scope(meta)
            if heading_scope is not None:
                ev["scope"] = heading_scope
                ev["scope_normalization_method"] = "SOURCE_SECTION_OVERRIDE"
            else:
                ev["scope"] = "UNKNOWN"
                ev["scope_normalization_method"] = "UNVERIFIED_SOURCE_SCOPE"

        # E. Exact source row label guards against broad LLM / Stage02 metric
        # aliases. These are different financial line items, not contradictory
        # figures for the same entity-period-scope-metric tuple.
        row_label = re.sub(
            r"\s+|[\[\]ㆍ·]", "", str(meta.get("metric_raw") or "")
        )
        source_line_item = None
        if row_label.startswith("매출원가"):
            source_line_item = "cost_of_revenue"
        elif row_label.startswith("기타유동자산"):
            source_line_item = "other_current_assets"
        elif row_label.startswith("기타비유동자산"):
            source_line_item = "other_noncurrent_assets"
        elif row_label.startswith("기타유동부채"):
            source_line_item = "other_current_liabilities"
        elif row_label.startswith("기타비유동부채"):
            source_line_item = "other_noncurrent_liabilities"
        if source_line_item is not None:
            ev["metric"] = source_line_item
            ev["metric_key"] = source_line_item
            ev["metric_normalization_method"] = "SOURCE_EXACT_ROW_METRIC_OVERRIDE"

        # C. 수치 유형: '증감' 열은 절대금액이 아님
        if is_change_column(ev):
            ev["value_type"] = "CHANGE"
            ev["value_type_normalization_method"] = (
                "SOURCE_COLUMN_OVERRIDE"
            )

        # D. Cash-flow working-capital movements are not BS point-in-time
        # balances, even when the extracted metric string matches exactly.
        statement = str(meta.get("statement_type") or meta.get("statement") or "").upper()
        if statement in {"BS", "IS", "CF", "SUMMARY"}:
            ev["statement"] = statement
            ev["statement_normalization_method"] = "SOURCE_STATEMENT_OVERRIDE"
        if (
            statement == "CF"
            and str(ev.get("metric_key") or ev.get("metric") or "").lower()
            in balance_sheet_stock_metrics
        ):
            ev["value_type"] = "CHANGE"
            ev["value_type_normalization_method"] = "SOURCE_CF_WORKING_CAPITAL_GUARD"

        changed = {
            key: {"before": before[key], "after": ev.get(key)}
            for key in monitored if before[key] != ev.get(key)
        }
        if changed:
            corrections.append({
                "evidence_id": ev.get("record_id"),
                "changed_fields": changed,
            })

    bundle["source_guard_audit"] = {
        "version": CAUSE_RAG_UPGRADE_VERSION,
        "evidences_checked": len(evidences),
        "evidences_corrected": len(corrections),
        "corrections": corrections,
    }
    return bundle

def reconcile_display_precision(
    result: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """
    표시 정밀도 차이를 검증한다.
    2개 이상의 근거도 처리하며 원본 수치는 변경하지 않는다.
    """
    import re
    from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

    if result.get("overall_status") != "genuine_conflict":
        return result

    records = result.get("final_context_for_vlm") or []
    if len(records) < 2:
        return result

    source = {
        str(item.get("record_id") or
            (item.get("source_metadata") or {}).get("evidence_id")): item
        for item in bundle.get("evidences", [])
    }
    linked = [
        (record, source.get(str(record.get("evidence_id"))))
        for record in records
    ]
    if any(src is None for _, src in linked):
        return result

    keys = (
        "entity", "metric", "period_start", "period_end",
        "period_type", "scope", "value_type", "unit_normalized",
    )
    reference = records[0].get("structured", {})
    if any(
        any(rec.get("structured", {}).get(k) != reference.get(k)
            for k in keys)
        for rec in records[1:]
    ):
        return result

    def displayed_step(item):
        metadata = item.get("source_metadata") or {}
        raw = str(item.get("value_text_raw") or "")
        unit = re.sub(
            r"\s+", "", str(item.get("unit_raw")
                               or metadata.get("unit_raw")
                               or metadata.get("unit") or "")
        )

        # '66조 1,930억원' 등의 최소 표시 단위는 억원.
        amount = re.search(
            r"([+-]?\d[\d,]*(?:\.(\d+))?)\s*억\s*원?",
            raw,
        )
        if amount:
            places = len(amount.group(2) or "")
            return Decimal(100000000) / (Decimal(10) ** places)

        amount = re.search(
            r"([+-]?\d[\d,]*(?:\.(\d+))?)\s*조\s*원?",
            raw,
        )
        if amount:
            places = len(amount.group(2) or "")
            return Decimal(1000000000000) / (Decimal(10) ** places)

        scales = {
            "원": Decimal(1),
            "천원": Decimal(1000),
            "만원": Decimal(10000),
            "백만원": Decimal(1000000),
            "억원": Decimal(100000000),
            "조원": Decimal(1000000000000),
        }
        scale = scales.get(unit)
        if scale is None:
            return None

        value_raw = str(
            item.get("value_raw")
            or metadata.get("value_raw")
            or ""
        )
        match = re.search(
            r"[+-]?\d[\d,]*(?:\.(\d+))?", value_raw
        )
        if not match:
            return None
        places = len(match.group(1) or "")
        return scale / (Decimal(10) ** places)

    candidates = []
    try:
        for record, original in linked:
            fields = record["structured"]
            value = Decimal(str(fields["value_normalized"]))
            step = displayed_step(original)
            if step is None or step <= 0:
                return result
            candidates.append((step, value, record))
    except (InvalidOperation, ValueError, KeyError, TypeError):
        return result

    candidates.sort(key=lambda entry: entry[0])
    finest_step, base_value, best_record = candidates[0]

    # 가장 정밀한 근거가 두 개 이상이면 값이 같아야 함.
    if any(
        step == finest_step and value != base_value
        for step, value, _ in candidates
    ):
        return result

    distinct_values = {value for _, value, _ in candidates}
    if len(distinct_values) <= 1:
        return result

    for step, displayed_value, _ in candidates:
        rounded = (
            (base_value / step).quantize(
                Decimal(1), rounding=ROUND_HALF_UP
            ) * step
        )
        if rounded != displayed_value:
            return result

    # 모든 차이가 표시 정밀도로 설명될 때만 Apparent로 변경.
    for record in records:
        record["conflict_label"] = "apparent_conflict"

    result["overall_status"] = "apparent_conflict"
    result["apparent_conflict"] = records
    result["genuine_conflict"] = []
    result["no_conflict"] = []
    result["summary"]["apparent_conflict_count"] = len(records)
    result["summary"]["genuine_conflict_count"] = 0
    result["evidence_comparison"]["group_status"] = "apparent_conflict"
    result["conflict_causes"] = ["display_rounding"]

    coarser = next(
        (row for row in candidates if row[0] > finest_step),
        candidates[-1],
    )
    result["precision_resolution"] = {
        "reason": "compatible_with_rounding",
        "exact_evidence_id": best_record["evidence_id"],
        "exact_value_krw": str(base_value),
        "rounded_evidence_id": coarser[2]["evidence_id"],
        "rounded_value_krw": str(coarser[1]),
        "rounding_step_krw": str(coarser[0]),
        "difference_krw": str(abs(base_value - coarser[1])),
        "compatible_evidence_ids": [
            row[2]["evidence_id"] for row in candidates
        ],
    }
    return result



# ==============================================================================
# 질문 단계: 05 → 04 → 06 → 07
# ==============================================================================


class Pipeline:
    """무거운 객체(임베더·비전·검색 인덱스·단계 모듈)를 한 번만 만들고 질문마다 재사용한다."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stage05 = load_stage("05_retrieve_evidence.py", "cause_rag_stage05")
        self.stage04 = self.stage05.load_script04()  # 05와 같은 04 모듈 인스턴스를 공유
        if self.stage04 is None:
            raise SystemExit("[ERROR] 04_normalize_retrieved_evidence.py가 없습니다.")
        self.stage06 = load_stage("06_conflict_detect.py", "cause_rag_stage06")
        self.stage07 = load_stage("07_vlm_answer.py", "cause_rag_stage07")

        # 05 옵션은 05 자신의 파서 기본값을 그대로 쓰고 필요한 것만 덮어쓴다(기본값 중복 방지).
        self.cfg = self.stage05.parse_args([])
        self.cfg.mode = "run"
        self.cfg.skip_vision_rerank = args.skip_vision_rerank
        self.cfg.no_vision_cache = args.no_vision_cache
        self.cfg.force_reembed = args.rebuild_index
        if args.vision_model:
            self.cfg.vision_model = args.vision_model
        if args.embedding_model:
            self.cfg.embedding_model = args.embedding_model
        if args.normalization_model:
            self.cfg.normalization_model = args.normalization_model

        banner("[05] 검색 인덱스 준비 (입력이 같으면 재구축 생략)")
        self.session = self.stage05.RetrievalSession(self.cfg, build=True)

    def answer(self, index: int, question: str, run_dir: Path) -> dict[str, Any]:
        args, cfg = self.args, self.cfg
        qdir = run_dir / f"q{index:02d}"
        retrieval_dir = qdir / "05_retrieval"
        retrieval_input = retrieval_dir / "retrieval_input.json"
        conflict_path = qdir / "06_detect" / "conflict_result.json"
        answer_path = qdir / "07_answer" / "answer_result.json"
        timer = Timer()

        banner(f"[Q{index:02d}] {question}")

        with timer.lap("05_retrieval"):
            results = self.session.query(
                question,
                retrieval_input,
                retrieval_dir / "multimodal_context.json",
                retrieval_dir / "retrieval_diagnostics.json",
            )
        for item in results:
            meta = item["metadata"]
            print(
                f"  #{meta['retrieval_rank']} {item['evidence_id']} "
                f"score={meta['retrieval_score']:.4f} {meta.get('source_file')} "
                f"p.{meta.get('page')} [{meta.get('modality')}]"
            )

        with timer.lap("04_normalize"):
            bundle = self.stage04.run_normalization(
                argparse.Namespace(
                    input=retrieval_input,
                    output_dir=qdir / "04_normalized",
                    cache_dir=SHARED_CACHE_04,
                    model=cfg.normalization_model,
                    workers=cfg.normalization_workers,
                    timeout=cfg.timeout,
                    max_attempts=cfg.max_attempts,
                    max_output_tokens=cfg.normalization_max_output_tokens,
                    max_question_chars=6000,
                    max_evidence_chars=20000,
                    allow_variable_k=False,
                    no_cache=args.no_cache,
                )
            )

        # 04 결과를 원본 공시 메타데이터로 검증
        bundle = apply_source_provenance_guard(bundle)
        for rec in bundle.get("evidences", []):
            semantic = rec.get("semantic_condition_key")
            if isinstance(semantic, dict):
                for key in (
                    "period_start", "period_end",
                    "period_type", "scope", "statement", "metric"
                ):
                    semantic[key] = rec.get(key)
            rec["question_match"] = (
                self.stage04.deterministic_question_match(
                    rec, bundle["question"]
                )
            )

        # 수정 전 normalized_bundle.json은 그대로 보존
        # 수정 후 결과를 별도로 저장하여 비교 가능하게 함
        save_json(
            qdir / "04_normalized" / "normalized_bundle_source_guarded.json",
            bundle,
        )

        with timer.lap("06_conflict"):
            question_obj, evidences = self.stage06.parse_bundle(bundle)
            conflict = self.stage06.run_conflict_disambiguator(question_obj, evidences)
            # 추가: 표시 정밀도로 설명 가능한 수치 차이 검증
            conflict = reconcile_display_precision(conflict, bundle)
            conflict = run_arbitration(conflict, bundle)
            self.stage06.save_result(conflict, conflict_path)

        with timer.lap("07_answer"):
            answer = self.stage07.run(
                input_path=conflict_path,
                output_path=answer_path,
                db_path=cfg.db.resolve(),
                model=args.answer_model,
                dry_run=args.dry_run,
                use_page_text=not args.no_page_text,
                use_images=not args.no_images,
                use_page_image=args.page_image,
                result=conflict,  # 방금 만든 결과를 그대로 전달(파일 재로딩 없음)
            )

        final = {
            "pipeline_version": CAUSE_RAG_UPGRADE_VERSION,
            "question": question,
            "answer": answer.get("answer"),
            "overall_status": conflict["overall_status"],
            "called_vlm": answer.get("called_vlm"),
            "conflict_summary": conflict["summary"],
            "arbitration": conflict.get("arbitration"),
            "precision_resolution": conflict.get("precision_resolution"),
            "evidence_ids_in_prompt": answer.get("evidence_ids_in_prompt"),
            "evidence_ids_genuine_conflict": answer.get("evidence_ids_genuine_conflict"),
            "evidence_ids_excluded_inapplicable": answer.get("evidence_ids_excluded_inapplicable"),
            "usage": answer.get("usage"),
            "timings_sec": timer.laps,
            "outputs": {
                "retrieval": str(retrieval_input),
                "normalized": str(qdir / "04_normalized" / "normalized_bundle.json"),
                "normalized_source_guarded": str(qdir / "04_normalized" / "normalized_bundle_source_guarded.json"),
                "conflict": str(conflict_path),
                "answer": str(answer_path),
            },
        }
        save_json(qdir / "final_result.json", final)
        print_final(index, final)
        return final


def print_final(index: int, final: dict[str, Any]) -> None:
    summary = final["conflict_summary"]
    print("\n" + "-" * 96)
    print(
        f"[Q{index:02d}] 상태={final['overall_status']} | "
        f"사용 근거={len(final['evidence_ids_in_prompt'] or [])}건 | "
        f"상충 근거={summary['genuine_conflict_count']}건"
    )
    print("-" * 96)
    if final["answer"]:
        print(final["answer"])
    else:
        print("(--dry-run: VLM을 호출하지 않았습니다. 프롬프트는 answer_result.json의 prompt_text 참고)")
    print("-" * 96)
    print(f"단계별 소요(초): {final['timings_sec']}")
    print(f"결과 폴더: {Path(final['outputs']['answer']).parent.parent}")


# ==============================================================================
# 인자 / 진입점
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAUSE-RAG 통합 실행기: 질문 → 검색 → 정규화 → 충돌 판별 → 최종 답변",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    q = parser.add_argument_group("입력")
    q.add_argument("-q", "--question", action="append", default=[], help="질문(여러 번 지정 가능)")
    q.add_argument("--question-file", type=Path, help="한 줄에 질문 하나(빈 줄·# 주석 무시)")
    q.add_argument("--run-dir", type=Path, help="결과 폴더(기본: dataset/runs/<실행시각>)")

    r = parser.add_argument_group("재실행 제어")
    r.add_argument("--force-parse", action="store_true", help="01을 입력과 무관하게 재실행")
    r.add_argument("--force-structure", action="store_true", help="02를 입력과 무관하게 재실행")
    r.add_argument("--rebuild-index", action="store_true", help="05 인덱스를 전부 다시 임베딩")
    r.add_argument("--no-cache", action="store_true", help="04 LLM 추출 캐시 비사용")
    r.add_argument("--no-vision-cache", action="store_true", help="05 비전 재순위화 캐시 비사용")

    o = parser.add_argument_group("단계 옵션")
    o.add_argument("--save-debug-pdfs", action="store_true", help="[01] 표 영역 검수용 PDF 저장")
    o.add_argument("--include-unknown-nonfinancial", action="store_true", help="[02] 미등록 비재무 지표 보존")
    o.add_argument("--skip-vision-rerank", action="store_true", help="[05] 비전 재순위화 생략")
    o.add_argument("--vision-model", help="[05] 기본: OPENAI_VISION_MODEL 또는 gpt-4o-mini")
    o.add_argument("--embedding-model", help="[05] 기본: OPENAI_EMBEDDING_MODEL 또는 text-embedding-3-large")
    o.add_argument("--normalization-model", help="[04] 기본: OPENAI_MODEL 또는 gpt-4o-mini")
    o.add_argument("--answer-model", default=os.environ.get("VLM_MODEL", "gpt-4o-mini"), help="[07] 최종 답변 모델")
    o.add_argument("--dry-run", action="store_true", help="[07] VLM 호출 없이 프롬프트만 생성")
    o.add_argument("--no-page-text", action="store_true", help="[07] DB 페이지 원문 제외")
    o.add_argument("--no-images", action="store_true", help="[07] 이미지 제외(텍스트만)")
    o.add_argument("--page-image", action="store_true", help="[07] crop 외에 페이지 전체 이미지도 첨부")

    g = parser.add_argument_group("선택 실행")
    g.add_argument("--with-gold", action="store_true", help="[03] DART Gold 구축도 함께 실행(DART_API_KEY 필요)")
    return parser.parse_args()


def collect_questions(args: argparse.Namespace) -> list[str]:
    questions = [q.strip() for q in args.question if q and q.strip()]
    if args.question_file:
        for line in args.question_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)
    if not questions:
        raise SystemExit('[ERROR] 질문이 없습니다. 예: python main.py --question "2025년 연결 기준 삼성전자의 영업이익은?"')
    return questions


def check_environment(args: argparse.Namespace) -> None:
    """긴 작업(01·02 파싱 등)을 시작하기 전에 필수 환경변수를 먼저 확인한다."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[ERROR] OPENAI_API_KEY 환경변수가 필요합니다(05 임베딩·04 정규화·07 답변).")
    if args.with_gold and not os.environ.get("DART_API_KEY"):
        raise SystemExit("[ERROR] --with-gold 는 DART_API_KEY 환경변수가 필요합니다.")


def main() -> int:
    args = parse_args()
    questions = collect_questions(args)
    check_environment(args)
    started = time.perf_counter()

    banner("CAUSE-RAG 통합 파이프라인 | 01 → 02 → 05 → 04 → 06 → 07")
    print(f"질문 {len(questions)}개")

    ensure_parsed(args)
    ensure_structured(args)

    run_dir = (args.run_dir or RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(args)

    finals: list[dict[str, Any]] = []
    failures = 0
    for index, question in enumerate(questions, 1):
        try:
            finals.append(pipeline.answer(index, question, run_dir))
        except Exception as exc:  # 한 질문의 실패가 나머지 질문을 막지 않도록 기록 후 계속
            failures += 1
            traceback.print_exc()
            finals.append({"question": question, "error": f"{type(exc).__name__}: {exc}"})
            if len(questions) == 1:
                raise

    save_json(
        run_dir / "run_summary.json",
        {
            "pipeline_version": CAUSE_RAG_UPGRADE_VERSION,
            "questions": len(questions),
            "failures": failures,
            "elapsed_sec": round(time.perf_counter() - started, 2),
            "results": finals,
        },
    )

    if args.with_gold:
        banner("[03] DART Gold 구축 (선택)")
        stage03 = load_stage("03_build_dart_gold.py", "cause_rag_stage03")
        with patched_argv():
            stage03.main()

    banner(f"완료: {len(questions) - failures}/{len(questions)}개 성공 | {run_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
