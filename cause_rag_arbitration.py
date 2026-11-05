"""CAUSE-RAG evidence arbitration - cautious, deterministic research baseline.

Drop-in companion for main.py::run_arbitration(conflict, normalized_bundle).

Input
-----
* result: output of 06_conflict_detect.py, optionally post-processed by
  main.py::reconcile_display_precision.
* bundle: output of 04_normalize_retrieved_evidence.py AFTER the existing
  main.py::apply_source_provenance_guard.

Output
------
The same `result` mapping, enhanced with `arbitration` and, crucially,
`final_context_for_vlm` restricted to selected/integrated evidence. When the
action is `abstain`, the final context is empty, so the existing stage 07
will NOT call a language model. Audit lists (`genuine_conflict`,
`excluded_inapplicable`, etc.) are preserved.

Policy (experimental, NOT independently validated)
--------------------------------------------------
* SELECT: only when one evidence-value group clearly dominates and passes
  minimum provenance, applicability, extraction-quality and margin checks.
* INTEGRATE: only if values agree after normalization or an explicit
  precision reconciliation certifies their compatibility.
* ABSTAIN: no trustworthy candidate, unresolved genuine disagreement,
  malformed metadata or conflicting normalized values.
* EXCLUDE is a per-evidence disposition, not a top-level answer action:
  stage 06 already excludes inapplicable/insufficient records.

Weights, limits and thresholds are pre-registered heuristic defaults to
calibrate on a separate DEV set. Do not tune them on TEST and do not report
these constants as empirically optimized.

Standard library only; Python 3.10+.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any


POLICY_VERSION = "cause-rag-arbitration-v2.0-provisional"
WEIGHTS = {
    "applicability": 0.45,
    "provenance": 0.30,
    "extraction_quality": 0.25,
}

# Conservative defaults; evaluate/tune on DEV (not TEST).
SELECTION_MARGIN = 0.20
MIN_SELECT_SCORE = 0.78
MIN_APPLICABILITY_SCORE = 0.80
MIN_PROVENANCE_SCORE = 0.85
MIN_EXTRACTION_QUALITY_SCORE = 0.85

_BASIS_SCORES = {
    "EXPLICIT": 1.00,
    "METADATA": 0.95,
    "RULE": 0.90,
    "INFERRED": 0.82,
    "LLM": 0.78,
    "QUESTION_INHERITED": 0.40,
    "UNKNOWN": 0.60,
}


def clamp(value: Any, default: float = 0.5) -> float:
    """Parse a confidence safely, rejecting NaN/Infinity and bogus types."""
    if value is None or isinstance(value, bool):
        return default
    try:
        numeric = float(value)
    except (ValueError, TypeError, OverflowError):
        return default
    if not math.isfinite(numeric):
        return default
    return max(0.0, min(1.0, numeric))


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip().upper() not in ("", "UNKNOWN", "NULL")


def _safe_numeric_value(value: Any) -> Decimal | None:
    if not _nonempty(value):
        return None
    try:
        number = Decimal(str(value).strip().replace(",", ""))
        return number if number.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _evidence_key(record: dict) -> tuple[str, str] | None:
    fields = _as_dict(record.get("structured"))
    unit = fields.get("unit_normalized")
    value = _safe_numeric_value(fields.get("value_normalized"))
    if not _nonempty(unit) or value is None:
        return None
    # Decimal numeric equality intentionally handles 1.0 versus 1.00.
    return (str(unit).strip().casefold(), format(value.normalize(), "f"))


def _is_relevant(record: dict, source: dict) -> tuple[bool, str]:
    """Fail closed on an explicit semantic mismatch or invalid source."""
    if _evidence_key(record) is None:
        return False, "missing_or_unparseable_normalized_value_or_unit"
    for item in _as_list(record.get("comparison")):
        if _as_dict(item).get("result") == "mismatch":
            return False, "question_condition_mismatch"
    matches = _as_dict(source.get("question_match"))
    for key in ("entity", "metric", "period", "scope"):
        if matches.get(key) is False:
            return False, f"explicit_question_match_failure:{key}"
    status = str(source.get("extraction_status") or "").upper()
    if status == "NOT_FOUND":
        return False, "extraction_not_found"
    return True, "valid_candidate"


def _applicability(record: dict, source: dict) -> tuple[float, dict]:
    """Compare condition agreement *and reliability*, avoiding a constant 1.

    Stage 06 already discards mismatched candidates, so plain match count is
    nearly always 100%. Basis reliability and independent question-match
    signals provide useful remaining distinctions without inventing a match.
    """
    checked = [x for x in _as_list(record.get("comparison"))
               if _as_dict(x).get("result") not in (None, "skipped")]
    match_ratio = (sum(_as_dict(x).get("result") == "match" for x in checked)
                   / len(checked)) if checked else 0.65

    basis_items = []
    for key in ("entity_basis", "metric_basis", "period_basis", "scope_basis"):
        value = str(source.get(key) or "").upper()
        if value:
            basis_items.append(_BASIS_SCORES.get(value, 0.75))
    basis_reliability = (sum(basis_items) / len(basis_items)) if basis_items else 0.75

    question_match = _as_dict(source.get("question_match"))
    observed = [question_match[key] for key in ("entity", "metric", "period", "scope")
                if question_match.get(key) is not None]
    hint_reliability = (sum(1.0 if x is True else 0.0 for x in observed) / len(observed)
                        if observed else 0.80)

    score = 0.50 * match_ratio + 0.35 * basis_reliability + 0.15 * hint_reliability
    return clamp(score), {
        "question_condition_match": round(match_ratio, 4),
        "field_basis_reliability": round(basis_reliability, 4),
        "question_match_reliability": round(hint_reliability, 4),
    }


def _provenance(source: dict) -> tuple[float, dict]:
    meta = _as_dict(source.get("source_metadata"))
    version = str(meta.get("filing_version") or meta.get("version")
                  or source.get("version") or "").upper()
    # Version is a provisional preference, not proof that a correction is
    # genuinely official or supersedes the exact filing. That verification
    # belongs to stage03 and human-audited Gold.
    if "CORRECT" in version or "정정" in version:
        base = 0.95
        code = "corrected_label"
    elif "ORIGINAL" in version or "원본" in version:
        base = 0.85
        code = "original_label"
    else:
        base = 0.65
        code = "unverified_version"

    has_file = _nonempty(meta.get("source_file"))
    has_page = meta.get("page") is not None and str(meta.get("page")) != ""
    if has_file and has_page:
        base += 0.03
    if _nonempty(meta.get("pdf_path")) and has_file:
        base += 0.01
    return clamp(base), {
        "version_class": code,
        "has_source_file_and_page": bool(has_file and has_page),
    }


def _extraction_quality(source: dict) -> tuple[float, dict]:
    meta = _as_dict(source.get("source_metadata"))
    llm = source.get("confidence")
    parser = meta.get("confidence")
    if llm is not None and parser is not None:
        quality = 0.45 * clamp(llm, 0.60) + 0.55 * clamp(parser, 0.60)
        confidence_basis = "llm_and_parser"
    elif parser is not None:
        quality = clamp(parser, 0.60)
        confidence_basis = "parser_only"
    elif llm is not None:
        quality = clamp(llm, 0.60)
        confidence_basis = "llm_only"
    else:
        quality = 0.60
        confidence_basis = "unknown"

    if meta.get("parsing_review_required") in (True, "True", "true", 1):
        quality *= 0.70

    flags = _as_list(meta.get("quality_flags"))
    flags += _as_list(meta.get("table_qc_flags"))
    if flags:
        quality *= max(0.60, 1.0 - 0.05 * len(set(map(str, flags))))

    selection_score = meta.get("table_selection_score")
    if selection_score is not None and _nonempty(selection_score):
        if clamp(selection_score) < 0.62:
            quality *= 0.85

    field_conflicts = source.get("field_conflicts")
    if isinstance(field_conflicts, list) and field_conflicts:
        quality *= 0.75
    elif isinstance(field_conflicts, dict) and field_conflicts:
        quality *= 0.75

    extraction_status = str(source.get("extraction_status") or "").upper()
    if extraction_status == "AMBIGUOUS":
        quality *= 0.70
    elif extraction_status == "NOT_FOUND":
        quality *= 0.30

    return clamp(quality), {
        "confidence_basis": confidence_basis,
        "parser_review_flag": bool(meta.get("parsing_review_required")),
        "quality_flag_count": len(set(map(str, flags))),
        "extraction_status": extraction_status or "UNSPECIFIED",
    }


def score_candidate(record: dict, source: dict) -> dict:
    """Public scorer, backward-compatible with the original module."""
    if not _as_dict(record).get("evidence_id"):
        raise ValueError("candidate evidence_id is required")
    relevance, disposition_reason = _is_relevant(record, _as_dict(source))
    applicability, applicability_meta = _applicability(record, _as_dict(source))
    provenance, provenance_meta = _provenance(_as_dict(source))
    extraction_quality, quality_meta = _extraction_quality(_as_dict(source))
    score = (WEIGHTS["applicability"] * applicability
             + WEIGHTS["provenance"] * provenance
             + WEIGHTS["extraction_quality"] * extraction_quality)

    return {
        "evidence_id": str(record["evidence_id"]),
        "applicability_score": round(applicability, 4),
        "provenance_score": round(provenance, 4),
        "extraction_quality_score": round(extraction_quality, 4),
        "evidence_score": round(score, 4),
        "eligible_for_arbitration": relevance,
        "eligibility_reason": disposition_reason,
        "normalized_value_key": _evidence_key(record),
        "score_details": {
            **applicability_meta,
            **provenance_meta,
            **quality_meta,
        },
    }


def _build_lookup(bundle: dict) -> tuple[dict[str, dict], list[str]]:
    lookup = {}
    duplicates = []
    for item in _as_list(bundle.get("evidences")):
        if not isinstance(item, dict):
            continue
        meta = _as_dict(item.get("source_metadata"))
        ev_id = item.get("record_id") or meta.get("evidence_id")
        if not _nonempty(ev_id):
            continue
        ev_id = str(ev_id)
        if ev_id in lookup:
            duplicates.append(ev_id)
        else:
            lookup[ev_id] = item
    return lookup, sorted(set(duplicates))


def _unique_ids(records: list) -> list[str]:
    return list(dict.fromkeys(str(rec["evidence_id"]) for rec in records
                              if isinstance(rec, dict) and _nonempty(rec.get("evidence_id"))))


def run_arbitration(result: dict, bundle: dict) -> dict:
    """Attach an auditable arbitration decision and enforce context filtering.

    Keeps the original `overall_status` unchanged for fair conflict-status
    evaluation. Input result is updated in-place, as in the original API.
    """
    if not isinstance(result, dict) or not isinstance(bundle, dict):
        raise TypeError("result and bundle must both be dictionaries")

    status = str(result.get("overall_status") or "")
    records = _as_list(result.get("final_context_for_vlm"))
    excluded_records = _as_list(result.get("excluded_inapplicable"))
    source_by_id, duplicate_source_ids = _build_lookup(bundle)

    # Stage06 emits full audit lists, but only these records may be used as
    # answer candidates. Excluded rows are preserved, never reintroduced.
    original_ids = _unique_ids(records)
    excluded_ids = _unique_ids(excluded_records)
    duplicate_candidate_ids = [ev_id for ev_id, count in Counter(
        str(rec.get("evidence_id")) for rec in records if isinstance(rec, dict)
        and _nonempty(rec.get("evidence_id"))
    ).items() if count > 1]
    scored = []
    for record in records:
        if not isinstance(record, dict) or not _nonempty(record.get("evidence_id")):
            continue
        ev_id = str(record["evidence_id"])
        # Missing normalized metadata is never silently replaced by the
        # question. It leads to abstention unless another valid source exists.
        entry = score_candidate(record, source_by_id.get(ev_id, {}))
        if ev_id not in source_by_id:
            entry["eligible_for_arbitration"] = False
            entry["eligibility_reason"] = "missing_normalized_source_record"
        scored.append(entry)
    scored.sort(key=lambda x: (-x["evidence_score"], x["evidence_id"]))

    valid = [row for row in scored if row["eligible_for_arbitration"]]
    invalid_ids = [row["evidence_id"] for row in scored if not row["eligible_for_arbitration"]]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for candidate in valid:
        groups[tuple(candidate["normalized_value_key"])].append(candidate)
    # A group represents one normalized value (unit + Decimal numeric value).
    # Comparing groups instead of top two rows prevents duplicate agreement
    # evidence from hiding a genuine disagreement.
    ranked_groups = sorted(
        groups.items(),
        key=lambda pair: (-pair[1][0]["evidence_score"], pair[0]),
    )

    action = "abstain"
    selected_ids: list[str] = []
    reason = "질문 조건에 부합하는 검증 가능한 근거가 부족함"
    margin: float | None = None
    if duplicate_source_ids or duplicate_candidate_ids:
        reason = "중복 evidence_id로 후보의 출처를 안전하게 식별할 수 없음"
    elif status == "no_conflict" and valid:
        if len(groups) == 1:
            action = "integrate"
            selected_ids = [item["evidence_id"] for item in valid]
            reason = "적합한 근거들의 정규화 단위와 값이 모두 일치함"
        else:
            reason = "no_conflict 판정과 실제 정규화 값 분포가 일치하지 않음"

    elif status == "apparent_conflict" and valid:
        precision = _as_dict(result.get("precision_resolution"))
        exact = precision.get("exact_evidence_id")
        compatible = set(map(str, _as_list(precision.get("compatible_evidence_ids"))))
        if exact and str(exact) in {row["evidence_id"] for row in valid} and (
            precision.get("reason") == "compatible_with_rounding"
            and compatible.issuperset(row["evidence_id"] for row in valid)
        ):
            action = "select"
            selected_ids = [str(exact)]
            reason = "원본 수치와 반올림된 표시값의 호환성이 검증되어 정밀한 수치를 선택함"
        elif len(groups) == 1:
            action = "integrate"
            selected_ids = [item["evidence_id"] for item in valid]
            reason = "단위 정규화 후 모든 유효 근거가 동일한 값을 나타냄"
        else:
            reason = "표면적 차이의 원인을 충분히 검증하지 못해 판단을 보류함"

    elif status == "genuine_conflict" and len(ranked_groups) >= 2:
        first_group = ranked_groups[0][1]
        second_group = ranked_groups[1][1]
        first, runner = first_group[0], second_group[0]
        margin = first["evidence_score"] - runner["evidence_score"]
        if (
            margin + 1e-9 >= SELECTION_MARGIN
            and first["evidence_score"] >= MIN_SELECT_SCORE
            and first["applicability_score"] >= MIN_APPLICABILITY_SCORE
            and first["provenance_score"] >= MIN_PROVENANCE_SCORE
            and first["extraction_quality_score"] >= MIN_EXTRACTION_QUALITY_SCORE
        ):
            action = "select"
            selected_ids = [first["evidence_id"]]
            reason = "독립적인 값 그룹 간 신뢰도 차이가 기준을 충족하여 우선 근거를 선택함"
        else:
            reason = "동일 조건의 상충 수치에 대한 신뢰도 차이가 선택 임계값에 미달함"

    elif status == "genuine_conflict":
        reason = "실질 충돌 상태이나 두 개 이상의 비교 가능한 값 그룹이 존재하지 않음"
    elif status == "no_valid_candidate":
        reason = "질문에 적합한 수치 근거가 없거나 필수 정보가 누락됨"
    elif status not in {"no_conflict", "apparent_conflict", "genuine_conflict"}:
        reason = f"지원되지 않는 충돌 상태({status or 'missing'})로 판단을 보류함"

    # Enforce the action, don't merely give a prompt hint to the VLM.
    allowed = set(selected_ids)
    result["final_context_for_vlm"] = [
        record for record in records
        if isinstance(record, dict) and str(record.get("evidence_id")) in allowed
    ]
    result["arbitration"] = {
        "action": action,
        "selected_evidence_ids": selected_ids,
        "excluded_evidence_ids": excluded_ids + [x for x in invalid_ids if x not in excluded_ids],
        "unselected_conflicting_evidence_ids": [
            ev_id for ev_id in original_ids if ev_id not in allowed and ev_id not in invalid_ids
        ],
        "original_context_evidence_ids": original_ids,
        "reason": reason,
        "candidate_scores": scored,
        "evidence_value_groups": [
            {"unit_normalized": key[0], "value_normalized": key[1],
             "evidence_ids": [c["evidence_id"] for c in rows],
             "best_score": rows[0]["evidence_score"]}
            for key, rows in ranked_groups
        ],
        "score_margin": round(margin, 4) if margin is not None else None,
        "weights": dict(WEIGHTS),
        "thresholds": {
            "selection_margin": SELECTION_MARGIN,
            "min_select_score": MIN_SELECT_SCORE,
            "min_applicability_score": MIN_APPLICABILITY_SCORE,
            "min_provenance_score": MIN_PROVENANCE_SCORE,
            "min_extraction_quality_score": MIN_EXTRACTION_QUALITY_SCORE,
        },
        "selection_margin": SELECTION_MARGIN,  # legacy compatibility
        "policy_version": POLICY_VERSION,
        "overall_status_before_arbitration": status,
        "experimental_heuristic": True,
    }
    return result
