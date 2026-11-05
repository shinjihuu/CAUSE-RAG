"""
Conflict Disambiguator (normalized_bundle.json 기반)

[입력]  CAUSE-RAG/dataset/04_normalized/normalized_bundle.json
[출력]  CAUSE-RAG/dataset/06_detect/conflict_result.json

[1단계] 질문 vs 근거 비교 (근거 단위)
  - 비교 키: entity, metric, period_start, period_end, period_type,
             scope, unit_normalized, value_normalized
  - period_start / period_end 는 정확 일치가 아니라 '포함 관계'로 비교한다.
    (근거 기간이 질문 기간 범위 안에 들어가면 일치. 예: 근거 2025_12 ⊆ 질문 2025)
  - 질문 값이 null / "UNKNOWN" / "" 이면 해당 키는 비교하지 않는다(skipped).
  - 질문에는 값이 있는데 근거 값이 비어있으면(null/UNKNOWN) -> 충돌(missing_in_evidence).
    단, scope는 사유를 scope_unresolved로 구분 기록 (04 단계에서 scope 보강 후 되살릴 근거 식별용).
    scope 미확정만이 유일한 충돌 사유이면 revivable_if_scope_resolved=true.
  - 하나라도 불일치 -> genuine_conflict (제외)
  - 전부 일치      -> kept 후보

[2단계] kept 근거끼리 비교
  - (unit_normalized, value_normalized)가 모두 같으면 no_conflict
  - 서로 다르면 apparent_conflict
  - 그 외 키(period/scope/unit/value)별로 근거 간 값 분포도 함께 기록

[전달]  RAG 필요 요소 / 추가 요소는 판단에 쓰지 않고 input 값을 그대로 output에 실어 보낸다.
"""

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# =====================================================================
# 0. 경로 / 상수
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent  # CAUSE-RAG/
INPUT_PATH = BASE_DIR / "dataset" / "04_normalized" / "normalized_bundle.json"
OUTPUT_DIR = BASE_DIR / "dataset" / "06_detect"
OUTPUT_PATH = OUTPUT_DIR / "conflict_result.json"

# 질문 vs 근거 비교 키
COMPARISON_KEYS = [
    "entity", "metric",
    "period_start", "period_end", "period_type",
    "scope", "value_type", "unit_normalized", "value_normalized",
]

# 근거끼리 비교할 때 분포를 기록할 키
EVIDENCE_COMPARE_KEYS = [
    "period_start", "period_end", "period_type",
    "scope", "unit_normalized", "value_normalized",
]

# RAG 적용 시 필요 요소: 출력키 -> (source_metadata 내 키)
RAG_META_KEYS = ["document_id", "source_file", "evidence_crop_path",
                 "page_image_path", "pdf_path", "page"]

# 추가 요소(확장 고려): source_metadata 내 키
EXTRA_META_KEYS = ["statement_type", "filing_date", "source_report_year",
                   "version", "version_id", "major_section", "section_path", "page"]

EMPTY_TOKENS = {"", "UNKNOWN"}


# =====================================================================
# 1. 유틸
# =====================================================================

def is_empty(value: Any) -> bool:
    """None / 빈 문자열 / 'UNKNOWN'(대소문자 무관) 이면 '값 없음'으로 간주"""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().upper() in EMPTY_TOKENS:
        return True
    return False


def _to_decimal(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def values_equal(key: str, a: Any, b: Any) -> bool:
    """키별 정확 일치 비교. value_normalized는 숫자(Decimal)로 비교, 나머지는 문자열(대소문자 무시)."""
    if key == "value_normalized":
        da, db = _to_decimal(a), _to_decimal(b)
        if da is not None and db is not None:
            return da == db
    return str(a).strip().casefold() == str(b).strip().casefold()


_PERIOD_RE = re.compile(r"^\s*(\d{4})(?:[-_./]?(\d{1,2}))?(?:[-_./]?(\d{1,2}))?\s*$")
_QUARTER_RE = re.compile(r"^\s*(\d{4})[-_ ]?Q([1-4])\s*$", re.IGNORECASE)

PERIOD_KEYS = ("period_start", "period_end")


def parse_period(value: Any):
    """YYYY/YYYY_MM/YYYY_MM_DD/YYYY_Qn을 실제 날짜 구간으로 변환."""
    import calendar
    from datetime import date

    s = str(value or "").strip().replace("-", "_")
    q = re.fullmatch(r"(\d{4})_?Q([1-4])", s, re.I)
    if q:
        year, quarter = map(int, q.groups())
        first_month = 3 * quarter - 2
        last_month = first_month + 2
        return (
            date(year, first_month, 1),
            date(year, last_month, calendar.monthrange(year, last_month)[1]),
        )

    match = re.fullmatch(
        r"(\d{4})(?:[_./](\d{1,2}))?(?:[_./](\d{1,2}))?", s
    )
    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else None
    day = int(match.group(3)) if match.group(3) else None

    try:
        if month is None:
            return date(year, 1, 1), date(year, 12, 31)
        if day is None:
            return (
                date(year, month, 1),
                date(year, month, calendar.monthrange(year, month)[1]),
            )
        instant = date(year, month, day)
        return instant, instant
    except ValueError:
        return None



def period_contained(
    key: str, q_fields: Dict[str, Any], e_val: Any
) -> Optional[bool]:
    """
    POINT_IN_TIME: 같은 기준일이어야 함.
    DURATION: 시작일과 종료일이 각각 같아야 함.
    연간 매출액 질문에 분기 매출액이 통과하는 것을 차단.
    """
    e_range = parse_period(e_val)
    q_start = q_fields.get("period_start")
    q_end = q_fields.get("period_end") or q_start
    start_range = parse_period(q_start)
    end_range = parse_period(q_end)

    if e_range is None or start_range is None or end_range is None:
        return None

    period_type = str(
        q_fields.get("period_type") or "UNKNOWN"
    ).upper()

    if period_type == "POINT_IN_TIME":
        return e_range[1] == end_range[1]

    if key == "period_start":
        return e_range[0] == start_range[0]

    if key == "period_end":
        return e_range[1] == end_range[1]

    return None



def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, str):  # 이중 인코딩 대비
        data = json.loads(data)
    return data


# =====================================================================
# 2. 데이터 구조
# =====================================================================

@dataclass
class Question:
    raw_question: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)  # COMPARISON_KEYS만


@dataclass
class Evidence:
    evidence_id: str = ""
    raw_text: str = ""
    fields: Dict[str, Any] = field(default_factory=dict)   # COMPARISON_KEYS만 (판단용)
    rag: Dict[str, Any] = field(default_factory=dict)      # RAG 필요 요소 (그대로 전달)
    extra: Dict[str, Any] = field(default_factory=dict)    # 추가 요소 (그대로 전달)


# =====================================================================
# 3. 입력 파싱 (normalized_bundle.json -> 필요한 요소만 추출)
# =====================================================================

def parse_bundle(bundle: Dict[str, Any]) -> Tuple[Question, List[Evidence]]:
    if "question" not in bundle:
        raise ValueError("입력에 'question' 키가 없습니다.")
    if "evidences" not in bundle or not isinstance(bundle["evidences"], list):
        raise ValueError("입력에 'evidences' 리스트가 없습니다.")

    q = bundle["question"]
    question = Question(
        raw_question=bundle.get("raw_question") or q.get("raw_content", ""),
        fields={k: q.get(k) for k in COMPARISON_KEYS},
    )

    evidences: List[Evidence] = []
    for ev in bundle["evidences"]:
        meta = ev.get("source_metadata", {}) or {}
        ev_id = ev.get("record_id") or meta.get("evidence_id") or ""

        evidences.append(Evidence(
            evidence_id=str(ev_id),
            raw_text=ev.get("raw_content", ""),
            fields={k: ev.get(k) for k in COMPARISON_KEYS},
            rag={
                "evidence_id": meta.get("evidence_id") or ev_id,
                "record_id": ev.get("record_id"),
                "raw_content": ev.get("raw_content"),
                **{k: meta.get(k) for k in RAG_META_KEYS},
            },
            extra={k: meta.get(k) for k in EXTRA_META_KEYS},
        ))
    return question, evidences


# =====================================================================
# 4. [1단계] 질문 vs 근거 비교
# =====================================================================

def match_question_evidence(question: Question, evidence: Evidence) -> Dict[str, Any]:
    checked, skipped, mismatched, details = [], [], [], []

    for key in COMPARISON_KEYS:
        q_val = question.fields.get(key)
        e_val = evidence.fields.get(key)

        # 값/단위는 질문과 비교하지 않고,
        # 조건에 맞는 근거들 사이에서 비교한다.
        if key in ("unit_normalized", "value_normalized"):
            skipped.append(key)
            details.append({
                "key": key,
                "question": q_val,
                "evidence": e_val,
                "result": "skipped",
                "reason": "inter_evidence_only",
            })
            continue

        # 질문 값이 없거나 UNKNOWN -> 비교 제외
        if is_empty(q_val):
            skipped.append(key)
            details.append({"key": key, "question": q_val, "evidence": e_val,
                            "result": "skipped"})
            continue

        checked.append(key)

        # 질문엔 있는데 근거에 없음 -> 충돌
        if is_empty(e_val):
            mismatched.append(key)
            reason = "scope_unresolved" if key == "scope" else "missing_in_evidence"
            details.append({"key": key, "question": q_val, "evidence": e_val,
                            "result": "mismatch", "reason": reason})
        elif key in PERIOD_KEYS:
            # 기간은 정확 일치가 아니라 '포함 관계'로 판단 (근거 기간 ⊆ 질문 기간)
            contained = period_contained(key, question.fields, e_val)
            if contained is None:  # 파싱 불가 -> 문자열 정확 일치로 fallback
                contained = values_equal(key, q_val, e_val)
                rule = "exact_fallback"
            else:
                rule = "containment"

            if contained:
                details.append({"key": key, "question": q_val, "evidence": e_val,
                                "result": "match", "rule": rule})
            else:
                mismatched.append(key)
                details.append({"key": key, "question": q_val, "evidence": e_val,
                                "result": "mismatch", "reason": "value_mismatch",
                                "rule": rule})
        elif values_equal(key, q_val, e_val):
            details.append({"key": key, "question": q_val, "evidence": e_val,
                            "result": "match"})
        else:
            mismatched.append(key)
            details.append({"key": key, "question": q_val, "evidence": e_val,
                            "result": "mismatch", "reason": "value_mismatch"})

    return {
        "checked_keys": checked,
        "skipped_keys": skipped,
        "mismatched_keys": mismatched,
        "question_vs_evidence": details,
        "is_conflict": len(mismatched) > 0,
    }


def filter_evidences(question: Question, evidences: List[Evidence]) -> Dict[str, List[Dict[str, Any]]]:
    kept, excluded = [], []
    for ev in evidences:
        res = match_question_evidence(question, ev)
        record = {"evidence": ev, **res}
        (excluded if res["is_conflict"] else kept).append(record)
    return {"kept": kept, "excluded": excluded}


# =====================================================================
# 5. [2단계] kept 근거끼리 비교
# =====================================================================

def _norm_for_group(key: str, value: Any) -> str:
    if is_empty(value):
        return "null"
    if key == "value_normalized":
        d = _to_decimal(value)
        if d is not None:
            return format(d.normalize(), "f")
    return str(value).strip()


def compare_among_kept(kept: List[Dict[str, Any]]) -> Dict[str, Any]:
    """kept 근거 간 키별 값 분포 + 최종 그룹 판정"""
    if not kept:
        return {"group_status": "no_candidate", "by_key": {}}

    by_key: Dict[str, Any] = {}
    for key in EVIDENCE_COMPARE_KEYS:
        groups: Dict[str, List[str]] = {}
        for r in kept:
            ev = r["evidence"]
            groups.setdefault(_norm_for_group(key, ev.fields.get(key)), []).append(ev.evidence_id)
        by_key[key] = {"all_same": len(groups) == 1, "groups": groups}

    # 판정 기준: (unit_normalized, value_normalized) 조합이 모두 같은가
    uv = {(_norm_for_group("unit_normalized", r["evidence"].fields.get("unit_normalized")),
           _norm_for_group("value_normalized", r["evidence"].fields.get("value_normalized")))
          for r in kept}
    incomplete = any(
        is_empty(r["evidence"].fields.get(key))
        for r in kept
        for key in ("unit_normalized", "value_normalized")
    )

    group_status = (
        "insufficient" if incomplete else
        "no_conflict" if len(uv) <= 1 else
        "genuine_conflict"
    )

    return {"group_status": group_status, "by_key": by_key}


# =====================================================================
# 6. 출력 레코드
# =====================================================================

def _to_output_record(ev: Evidence, label: str, res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "evidence_id": ev.evidence_id,
        "conflict_label": label,                                   # 판정 결과 (기존 유지)
        "checked_keys": res["checked_keys"],
        "skipped_keys": res["skipped_keys"],
        "mismatched_keys": res["mismatched_keys"],
        "exclusion_reasons": [d["reason"] for d in res["question_vs_evidence"]
                              if d["result"] == "mismatch"],
        # scope 미확정이 유일한 충돌 사유 -> 04 단계에서 scope 보강 시 되살릴 후보
        "revivable_if_scope_resolved": (res["mismatched_keys"] == ["scope"]
                                        and label == "excluded_inapplicable"),
        "comparison": res["question_vs_evidence"],                 # 근거에서 비교한 내용
        "structured": ev.fields,                                   # 비교에 쓴 구조화 값
        "raw_text": ev.raw_text,
        "rag": ev.rag,                                             # RAG 필요 요소 (input 그대로)
        "extra": ev.extra,                                         # 추가 요소 (input 그대로)
    }


# =====================================================================
# 7. 전체 파이프라인
# =====================================================================

def run_conflict_disambiguator(
    question: Question, evidences: List[Evidence]
) -> Dict[str, Any]:
    filtered = filter_evidences(question, evidences)

    # 필수 수치가 없는 후보 때문에 정상 후보까지 제거하지 않음.
    valid, incomplete = [], []
    for row in filtered["kept"]:
        fields = row["evidence"].fields
        if (
            is_empty(fields.get("value_normalized"))
            or is_empty(fields.get("unit_normalized"))
        ):
            incomplete.append(row)
        else:
            valid.append(row)

    comparison = compare_among_kept(valid)
    group_status = comparison["group_status"]

    no_conflict = []
    apparent_conflict = []
    genuine_conflict = []
    excluded_inapplicable = []

    for row in valid:
        record = _to_output_record(
            row["evidence"], group_status, row
        )
        if group_status == "genuine_conflict":
            genuine_conflict.append(record)
        else:
            no_conflict.append(record)

    for row in incomplete:
        excluded_inapplicable.append(
            _to_output_record(
                row["evidence"], "insufficient", row
            )
        )

    for row in filtered["excluded"]:
        excluded_inapplicable.append(
            _to_output_record(
                row["evidence"], "excluded_inapplicable", row
            )
        )

    if genuine_conflict:
        status = "genuine_conflict"
    elif no_conflict:
        status = "no_conflict"
    else:
        status = "no_valid_candidate"

    return {
        "overall_status": status,
        "question": {
            "raw_question": question.raw_question,
            "compare_fields": question.fields,
        },
        "summary": {
            "no_conflict_count": len(no_conflict),
            "apparent_conflict_count": 0,
            "genuine_conflict_count": len(genuine_conflict),
            "excluded_inapplicable_count": len(excluded_inapplicable),
            "insufficient_count": len(incomplete),
            "scope_unresolved_count": sum(
                "scope_unresolved" in r["exclusion_reasons"]
                for r in excluded_inapplicable
            ),
            "revivable_evidence_ids": [
                r["evidence_id"]
                for r in excluded_inapplicable
                if r["revivable_if_scope_resolved"]
            ],
        },
        "evidence_comparison": comparison,
        "no_conflict": no_conflict,
        "apparent_conflict": apparent_conflict,
        "genuine_conflict": genuine_conflict,
        "excluded_inapplicable": excluded_inapplicable,
        "final_context_for_vlm": (
            no_conflict + apparent_conflict + genuine_conflict
        ),
    }



# =====================================================================
# 8. 진입점
# =====================================================================

def run_from_file(path: Union[str, Path] = INPUT_PATH) -> Dict[str, Any]:
    question, evidences = parse_bundle(load_json(path))
    return run_conflict_disambiguator(question, evidences)


def save_result(result: Dict[str, Any], path: Union[str, Path] = OUTPUT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    result = run_from_file(INPUT_PATH)
    save_result(result, OUTPUT_PATH)
    print(f"saved -> {OUTPUT_PATH}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("overall_status:", result["overall_status"])
