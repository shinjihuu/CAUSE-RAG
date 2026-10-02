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
    "scope", "unit_normalized", "value_normalized",
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


def parse_period(value: Any) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    기간 문자열을 (하한, 상한) 월 단위 튜플 ((년, 월), (년, 월))로 변환.
      "2025"      -> ((2025, 1),  (2025, 12))
      "2025_12"   -> ((2025, 12), (2025, 12))
      "2025_Q1"   -> ((2025, 1),  (2025, 3))
    파싱 불가하면 None.
    """
    s = str(value).strip()

    m = _QUARTER_RE.match(s)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        return (y, 3 * q - 2), (y, 3 * q)

    m = _PERIOD_RE.match(s)
    if m:
        y = int(m.group(1))
        if m.group(2) is None:
            return (y, 1), (y, 12)
        mon = int(m.group(2))
        if 1 <= mon <= 12:
            return (y, mon), (y, mon)
    return None


def period_contained(key: str, q_fields: Dict[str, Any], e_val: Any) -> Optional[bool]:
    """
    근거의 period 값(e_val)이 질문 기간 [period_start, period_end] 범위 안에 포함되는지 판단.
    - 질문의 하한: period_start의 하한 / 상한: period_end의 상한 (period_end 없으면 period_start의 상한)
    - 파싱 불가 -> None (호출 측에서 문자열 정확 일치로 fallback)
    """
    e_rng = parse_period(e_val)
    q_start = q_fields.get("period_start")
    q_end = q_fields.get("period_end")

    q_start_rng = parse_period(q_start) if not is_empty(q_start) else None
    q_end_rng = parse_period(q_end) if not is_empty(q_end) else None

    if e_rng is None:
        return None
    if (not is_empty(q_start) and q_start_rng is None) or (not is_empty(q_end) and q_end_rng is None):
        return None

    lower = q_start_rng[0] if q_start_rng else None
    if q_end_rng:
        upper = q_end_rng[1]
    elif q_start_rng:
        upper = q_start_rng[1]
    else:
        upper = None

    # period_start 근거는 시작점(하한)을, period_end 근거는 끝점(상한)을 범위에 대입
    e_lo, e_hi = e_rng
    if lower is not None and e_lo < lower:
        return False
    if upper is not None and e_hi > upper:
        return False
    return True


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
    group_status = "no_conflict" if len(uv) <= 1 else "apparent_conflict"

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
                                        and label == "genuine_conflict"),
        "comparison": res["question_vs_evidence"],                 # 근거에서 비교한 내용
        "structured": ev.fields,                                   # 비교에 쓴 구조화 값
        "raw_text": ev.raw_text,
        "rag": ev.rag,                                             # RAG 필요 요소 (input 그대로)
        "extra": ev.extra,                                         # 추가 요소 (input 그대로)
    }


# =====================================================================
# 7. 전체 파이프라인
# =====================================================================

def run_conflict_disambiguator(question: Question, evidences: List[Evidence]) -> Dict[str, Any]:
    filtered = filter_evidences(question, evidences)
    kept_cmp = compare_among_kept(filtered["kept"])
    group_status = kept_cmp["group_status"]

    no_conflict, apparent_conflict, genuine_conflict = [], [], []

    for r in filtered["kept"]:
        rec = _to_output_record(r["evidence"], group_status, r)
        (no_conflict if group_status == "no_conflict" else apparent_conflict).append(rec)

    for r in filtered["excluded"]:
        genuine_conflict.append(_to_output_record(r["evidence"], "genuine_conflict", r))

    if apparent_conflict:
        overall_status = "apparent_conflict"
    elif no_conflict:
        overall_status = "no_conflict"
    else:
        overall_status = "no_valid_candidate"

    return {
        "overall_status": overall_status,
        "question": {
            "raw_question": question.raw_question,
            "compare_fields": question.fields,
        },
        "summary": {
            "no_conflict_count": len(no_conflict),
            "apparent_conflict_count": len(apparent_conflict),
            "genuine_conflict_count": len(genuine_conflict),
            "scope_unresolved_count": sum(
                "scope_unresolved" in r["exclusion_reasons"] for r in genuine_conflict),
            "revivable_evidence_ids": [
                r["evidence_id"] for r in genuine_conflict if r["revivable_if_scope_resolved"]],
        },
        "evidence_comparison": kept_cmp,   # kept 근거끼리 비교 내용
        "no_conflict": no_conflict,
        "apparent_conflict": apparent_conflict,
        "genuine_conflict": genuine_conflict,
        "final_context_for_vlm": no_conflict + apparent_conflict,
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