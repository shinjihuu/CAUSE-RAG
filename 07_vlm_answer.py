"""
VLM 답변 생성기 (conflict_detector 결과 -> GPT-4o(mini) 답변)

[입력]  CAUSE-RAG/dataset/06_detect/conflict_result.json
[DB]    CAUSE-RAG/dataset/05_retrieval/cause_rag_multimodal.db  (SQLite, 읽기 전용으로만 사용)
[출력]  CAUSE-RAG/dataset/07_answer/answer_result.json

동작
  1. conflict_result.json의 overall_status 에 따라 충돌 경우별 지시문을 프롬프트에 추가
       - no_conflict        : 값이 모두 일치
       - apparent_conflict  : 조건은 일치하나 값이 다름 -> raw_text 비교 후 선택/차이 명시
       - no_valid_candidate : 프롬프트 구성 없이 '근거 부족' 응답 (API 호출 안 함)
  2. genuine_conflict 근거는 프롬프트에 절대 넣지 않는다 (final_context_for_vlm 만 사용).
  3. 근거마다 (a) 질문과 비교한 구조화 내용, (b) 출처, (c) raw_text,
     (d) DB의 페이지 원문(native_text), (e) crop 이미지를 함께 넣는다.
  4. 질문은 raw 질문 문자열만 전달한다 (구조화된 질문은 VLM에 보내지 않음).

실행
  python vlm_answer.py              # 실제 호출 (환경변수 OPENAI_API_KEY 필요)
  python vlm_answer.py --dry-run    # API 호출 없이 프롬프트만 생성/저장
"""

import argparse
import base64
import json
import os
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

# =====================================================================
# 0. 경로 / 설정
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent  # CAUSE-RAG/
INPUT_PATH = BASE_DIR / "dataset" / "06_detect" / "conflict_result.json"
DB_PATH = BASE_DIR / "dataset" / "05_retrieval" / "cause_rag_multimodal.db"
OUTPUT_DIR = BASE_DIR / "dataset" / "07_answer"
OUTPUT_PATH = OUTPUT_DIR / "answer_result.json"

MODEL = os.environ.get("VLM_MODEL", "gpt-4o-mini")
TEMPERATURE = 0
MAX_TOKENS = 1200
IMAGE_DETAIL = "auto"            # "low" | "high" | "auto"
PAGE_TEXT_MAX_CHARS = 3000       # DB 페이지 원문 최대 길이

NO_EVIDENCE_ANSWER = "질문 조건에 부합하는 근거를 찾지 못해 답변할 수 없습니다."

KEY_LABELS = {
    "entity": "기업(entity)",
    "metric": "지표(metric)",
    "period_start": "기간 시작(period_start)",
    "period_end": "기간 종료(period_end)",
    "period_type": "기간 유형(period_type)",
    "scope": "범위(scope, 연결/별도)",
    "unit_normalized": "정규화 단위(unit_normalized)",
    "value_normalized": "정규화 값(value_normalized)",
}
# 질문에 값이 없어 비교가 생략되었더라도, 충돌 판단에 필요하므로 항상 보여줄 키
ALWAYS_SHOW_KEYS = ["unit_normalized", "value_normalized"]


# =====================================================================
# 1. 프롬프트 텍스트
# =====================================================================

SYSTEM_PROMPT = """당신은 기업 공시 문서(사업보고서 등)를 근거로 질문에 답하는 재무 질의응답 어시스턴트입니다.

[기본 원칙]
1. 제공된 근거(구조화 정보, 원문 텍스트, 페이지 텍스트, 이미지)만 사용하세요. 근거에 없는 내용은 추측하지 말고 "근거에서 확인되지 않습니다"라고 답하세요.
2. 근거의 값이 질문이 묻는 대상 전체의 값인지, 일부(사업부문·세그먼트·자회사·증감액·비율 등)의 값인지 구분하세요. 질문이 묻는 대상과 다른 값은 답으로 사용하지 마세요.
3. 텍스트와 이미지의 숫자가 서로 다르면 어느 쪽이 맞는지 이미지로 확인하고, 불일치가 있었다는 사실을 짧게 언급하세요.
4. 단위를 반드시 명시하세요(조원, 억원, 백만원 등). 단순 단위 환산 외의 계산(합산, 평균 등)은 근거가 직접 제시하지 않는 한 하지 마세요.
5. 연결/별도 기준, 기간(회계연도)이 질문과 일치하는지 근거에서 확인하고, 확인되지 않으면 그 점을 명시하세요.
6. 답변에는 사용한 근거의 출처(파일명, 페이지)를 함께 적으세요.

[출력 형식]
- 답변: 질문에 대한 직접적인 답 (1~3문장)
- 근거: 사용한 근거 번호, 출처(파일명/페이지), 해당 값
- 유의사항: 값 차이·불확실성·확인되지 않은 조건이 있을 때만 작성 (없으면 생략)"""

CASE_INSTRUCTIONS = {
    "no_conflict": "아래 근거들은 질문 조건과 값이 모두 일치합니다. 이를 바탕으로 답변하세요.",
    "apparent_conflict": (
        "아래 근거들은 질문 조건은 모두 일치하지만 값이 서로 다릅니다. "
        "raw_text를 비교해 차이의 원인(표기, 반올림, 정정, 잠정/확정 등)을 판단하고, "
        "근거에 따라 하나를 선택하거나 차이를 명시하여 답변하세요. "
        "근거 없이 임의로 값을 선택하거나 평균 내지 마세요."
    ),
}


# =====================================================================
# 2. 유틸
# =====================================================================

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, str):
        data = json.loads(data)
    return data


def resolve_path(p: Optional[str]) -> Optional[Path]:
    """
    JSON에 저장된 경로(Windows 절대경로 등)를 현재 실행 환경(CAUSE-RAG 기준)의 실제 경로로 해석.
      1) 그대로 존재하면 사용
      2) 경로에 'CAUSE-RAG' 가 있으면 그 이하를 BASE_DIR 기준으로 재조립
      3) 'dataset' / 'pdf_data' 이하를 BASE_DIR 기준으로 재조립
    """
    if not p:
        return None
    try:
        cand = Path(p)
        if cand.exists():
            return cand
    except OSError:
        pass

    parts = PureWindowsPath(p).parts
    for anchor, skip in (("CAUSE-RAG", 1), ("dataset", 0), ("pdf_data", 0)):
        if anchor in parts:
            i = parts.index(anchor)
            cand = BASE_DIR.joinpath(*parts[i + skip:])
            if cand.exists():
                return cand
    return None


def format_krw(value: Any) -> Optional[str]:
    """정규화 KRW 정수 값을 '12조 8,527억 원' 형태로 변환 (가독성용 보조 표기)"""
    try:
        v = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if v != v.to_integral_value():
        return None
    n = int(v)
    sign = "-" if n < 0 else ""
    n = abs(n)
    jo, rest = divmod(n, 10 ** 12)
    eok, rest = divmod(rest, 10 ** 8)
    man, won = divmod(rest, 10 ** 4)
    parts = []
    if jo:
        parts.append(f"{jo:,}조")
    if eok:
        parts.append(f"{eok:,}억")
    if man:
        parts.append(f"{man:,}만")
    if won:
        parts.append(f"{won:,}")
    if not parts:
        return "0원"
    return sign + " ".join(parts) + " 원"


def format_value(value: Any, unit: Any) -> str:
    s = str(value)
    try:
        s = f"{int(Decimal(str(value))):,}"
    except (InvalidOperation, ValueError):
        pass
    if str(unit).upper() == "KRW":
        pretty = format_krw(value)
        return f"{s} KRW" + (f" (= {pretty})" if pretty else "")
    return f"{s} {unit}" if unit else s


def encode_image(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# =====================================================================
# 3. RAG DB 조회 (SQLite, SELECT 전용)
# =====================================================================

class RetrievalDB:
    def __init__(self, db_path: Path):
        self.conn: Optional[sqlite3.Connection] = None
        self.warning: Optional[str] = None
        if not db_path.exists():
            self.warning = f"DB 파일이 없습니다: {db_path}"
            return
        try:
            self.conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        except sqlite3.Error:
            try:
                self.conn = sqlite3.connect(str(db_path))
            except sqlite3.Error as e:
                self.warning = f"DB 연결 실패: {e}"

    def close(self) -> None:
        if self.conn:
            self.conn.close()

    def get_evidence(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        if not self.conn:
            return None
        try:
            row = self.conn.execute(
                "SELECT raw_content, modality, document_id, page "
                "FROM evidence_index WHERE evidence_id = ?", (evidence_id,)
            ).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        return {"raw_content": row[0], "modality": row[1], "document_id": row[2], "page": row[3]}

    def get_page_text(self, document_id: Any, page: Any) -> Optional[str]:
        if not self.conn or document_id is None or page is None:
            return None
        try:
            row = self.conn.execute(
                "SELECT native_text FROM page_index WHERE document_id = ? AND page = ?",
                (document_id, page),
            ).fetchone()
        except sqlite3.Error:
            return None
        return row[0] if row and row[0] else None


# =====================================================================
# 4. 근거 블록 구성
# =====================================================================

def build_structured_lines(rec: Dict[str, Any]) -> List[str]:
    """질문과 '비교한 항목'만 구조화 내용으로 표시 (+ 값/단위는 충돌 판단용으로 항상 표시)"""
    structured = rec.get("structured", {})
    lines, shown = [], set()

    for d in rec.get("comparison", []):
        key = d["key"]
        if d["result"] == "skipped":
            continue
        shown.add(key)
        lines.append(f"- {KEY_LABELS.get(key, key)}: {d['evidence']}  (질문 조건과 일치)")

    for key in ALWAYS_SHOW_KEYS:
        if key in shown:
            continue
        if key == "value_normalized" and structured.get(key) is not None:
            lines.append(f"- {KEY_LABELS[key]}: "
                         f"{format_value(structured[key], structured.get('unit_normalized'))}")
        elif key == "unit_normalized" and structured.get(key) is not None:
            if "value_normalized" not in structured:
                lines.append(f"- {KEY_LABELS[key]}: {structured[key]}")
    return lines


def build_evidence_block(
    idx: int,
    rec: Dict[str, Any],
    db: RetrievalDB,
    seen_pages: Dict[Tuple[Any, Any], int],
    use_page_text: bool,
    use_images: bool,
    use_page_image: bool,
) -> Tuple[str, List[Path], Dict[str, Any]]:
    """근거 1건 -> (프롬프트 텍스트, 첨부 이미지 경로 목록, 디버그 정보)"""
    rag = rec.get("rag", {})
    extra = rec.get("extra", {})
    ev_id = rec.get("evidence_id")

    db_ev = db.get_evidence(ev_id) if ev_id else None
    raw_text = rec.get("raw_text") or rag.get("raw_content") or (db_ev or {}).get("raw_content") or ""

    doc_id = rag.get("document_id") or (db_ev or {}).get("document_id")
    page = rag.get("page") if rag.get("page") is not None else (db_ev or {}).get("page")

    lines = [f"[근거 {idx}] evidence_id: {ev_id}"]

    lines.append("■ 구조화 정보 (질문과 비교한 항목)")
    lines.extend(build_structured_lines(rec) or ["- (비교 항목 없음)"])

    lines.append("■ 출처")
    src = [f"파일: {rag.get('source_file')}", f"페이지: {page}"]
    for label, key in (("제출일", "filing_date"), ("보고서 연도", "source_report_year"),
                       ("버전", "version"), ("공시 구분", "statement_type"),
                       ("섹션", "major_section")):
        if extra.get(key) not in (None, ""):
            src.append(f"{label}: {extra[key]}")
    lines.append("- " + " / ".join(src))

    lines.append("■ 근거 원문(raw_text)")
    lines.append(raw_text)

    page_text_used = False
    if use_page_text:
        page_key = (doc_id, page)
        if page_key in seen_pages:
            lines.append(f"■ 해당 페이지 원문 텍스트: 근거 {seen_pages[page_key]}과 동일 페이지 (중복 생략)")
        else:
            text = db.get_page_text(doc_id, page)
            if text:
                seen_pages[page_key] = idx
                if len(text) > PAGE_TEXT_MAX_CHARS:
                    text = text[:PAGE_TEXT_MAX_CHARS] + " ...(이하 생략)"
                lines.append("■ 해당 페이지 원문 텍스트 (PDF 직접 추출)")
                lines.append(text)
                page_text_used = True

    images: List[Path] = []
    if use_images:
        crop = resolve_path(rag.get("evidence_crop_path"))
        page_img = resolve_path(rag.get("page_image_path"))
        if crop:
            images.append(crop)
            if use_page_image and page_img:   # 요청 시 페이지 전체 이미지도 추가
                images.append(page_img)
        elif page_img:                        # crop이 없으면 페이지 이미지로 fallback
            images.append(page_img)
        if not images:
            lines.append("■ 이미지: (이미지 파일을 찾지 못했습니다)")

    debug = {
        "evidence_id": ev_id,
        "document_id": doc_id,
        "page": page,
        "db_evidence_found": db_ev is not None,
        "page_text_used": page_text_used,
        "images": [str(p) for p in images],
    }
    return "\n".join(lines), images, debug


# =====================================================================
# 5. 메시지 구성
# =====================================================================

def build_messages(
    result: Dict[str, Any],
    db: RetrievalDB,
    use_page_text: bool = True,
    use_images: bool = True,
    use_page_image: bool = False,
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    """
    Returns (messages, 사람이 읽을 수 있는 프롬프트 텍스트, 근거별 디버그 정보)
    - final_context_for_vlm 만 사용 -> genuine_conflict 는 절대 포함되지 않는다.
    """
    status = result["overall_status"]
    records = result.get("final_context_for_vlm", [])
    raw_question = result["question"]["raw_question"]

    header = (f"{CASE_INSTRUCTIONS[status]}\n\n"
              f"[질문]\n{raw_question}\n\n"
              f"[근거 {len(records)}건]")

    parts: List[Dict[str, Any]] = [{"type": "text", "text": header}]
    readable = [header]
    debug_list: List[Dict[str, Any]] = []
    seen_pages: Dict[Tuple[Any, Any], int] = {}

    for i, rec in enumerate(records, start=1):
        text, images, debug = build_evidence_block(
            i, rec, db, seen_pages, use_page_text, use_images, use_page_image)
        parts.append({"type": "text", "text": text})
        readable.append(text)
        for img in images:
            parts.append({"type": "text", "text": f"[근거 {i} 이미지: {img.name}]"})
            parts.append({"type": "image_url",
                          "image_url": {"url": encode_image(img), "detail": IMAGE_DETAIL}})
            readable.append(f"[근거 {i} 이미지: {img.name}]")
        debug_list.append(debug)

    footer = "위 근거만을 사용하여 질문에 답하세요. 출력 형식은 시스템 지침을 따르세요."
    parts.append({"type": "text", "text": footer})
    readable.append(footer)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": parts},
    ]
    return messages, "\n\n".join(readable), debug_list


# =====================================================================
# 6. VLM 호출
# =====================================================================

def call_vlm(messages: List[Dict[str, Any]], model: str = MODEL) -> Tuple[str, Dict[str, Any]]:
    """OpenAI Chat Completions 호출. API 키는 환경변수 OPENAI_API_KEY 사용."""
    from openai import OpenAI  # 지연 import (dry-run 시 패키지 불필요)

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    usage = resp.usage
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return resp.choices[0].message.content, usage_dict


# =====================================================================
# 7. 전체 실행
# =====================================================================

def run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    db_path: Path = DB_PATH,
    model: str = MODEL,
    dry_run: bool = False,
    use_page_text: bool = True,
    use_images: bool = True,
    use_page_image: bool = False,
) -> Dict[str, Any]:
    result = load_json(input_path)
    status = result["overall_status"]
    records = result.get("final_context_for_vlm", [])

    out: Dict[str, Any] = {
        "raw_question": result["question"]["raw_question"],
        "overall_status": status,
        "model": model,
        "evidence_ids_in_prompt": [r.get("evidence_id") for r in records],
        "evidence_ids_excluded_genuine": [r.get("evidence_id") for r in result.get("genuine_conflict", [])],
    }

    if status not in CASE_INSTRUCTIONS or not records:
        out.update({"called_vlm": False, "answer": NO_EVIDENCE_ANSWER,
                    "prompt_text": None, "evidence_debug": []})
    else:
        db = RetrievalDB(db_path)
        try:
            messages, readable, debug_list = build_messages(
                result, db, use_page_text, use_images, use_page_image)
        finally:
            db.close()

        out.update({
            "db_warning": db.warning,
            "system_prompt": SYSTEM_PROMPT,
            "prompt_text": readable,          # 이미지는 base64 대신 파일명만 표시
            "evidence_debug": debug_list,
        })

        if dry_run:
            out.update({"called_vlm": False, "answer": None})
        else:
            answer, usage = call_vlm(messages, model)
            out.update({"called_vlm": True, "answer": answer, "usage": usage})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=INPUT_PATH)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 프롬프트만 생성")
    ap.add_argument("--no-page-text", action="store_true", help="DB 페이지 원문 제외")
    ap.add_argument("--no-images", action="store_true", help="이미지 제외(텍스트만)")
    ap.add_argument("--page-image", action="store_true", help="crop 외에 페이지 전체 이미지도 첨부")
    args = ap.parse_args()

    res = run(args.input, args.output, args.db, args.model, args.dry_run,
              not args.no_page_text, not args.no_images, args.page_image)

    print(f"saved -> {args.output}")
    print("status:", res["overall_status"], "| called_vlm:", res["called_vlm"])
    if res.get("prompt_text") and args.dry_run:
        print("=" * 70)
        print(res["prompt_text"])
    if res.get("answer"):
        print("=" * 70)
        print(res["answer"])