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
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdf_data"
DATASET_DIR = BASE_DIR / "dataset"

# 각 단계 스크립트가 만드는 대표 산출물(README 3.1 기준)
PARSED_BLOCKS = DATASET_DIR / "01_parsed" / "parsed_blocks.jsonl"
DOCUMENT_MANIFEST = DATASET_DIR / "01_parsed" / "document_manifest.csv"
STRUCTURED_EVIDENCE = DATASET_DIR / "02_structured" / "structured_evidence.jsonl"
SHARED_CACHE_04 = DATASET_DIR / "04_normalized" / "cache"  # 실행 간 공유되는 LLM 캐시
RUNS_DIR = DATASET_DIR / "runs"


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
    if not args.force_parse and not is_stale([PARSED_BLOCKS, DOCUMENT_MANIFEST], pdfs):
        print("[01] PDF 파싱: 최신 상태 → 건너뜀 (강제 재실행: --force-parse)")
        return
    banner("[01] PDF 본문·표 파싱")
    stage = load_stage("01_parse_audit_reports.py", "cause_rag_stage01")
    with patched_argv(*(["--save-debug-pdfs"] if args.save_debug_pdfs else [])):
        stage.main()


def ensure_structured(args: argparse.Namespace) -> None:
    if not args.force_structure and not is_stale([STRUCTURED_EVIDENCE], [PARSED_BLOCKS]):
        print("[02] 근거 구조화: 최신 상태 → 건너뜀 (강제 재실행: --force-structure)")
        return
    banner("[02] 근거 구조화")
    stage = load_stage("02_structure_evidence.py", "cause_rag_stage02")
    flags = ["--include-unknown-nonfinancial"] if args.include_unknown_nonfinancial else []
    with patched_argv(*flags):
        stage.main()


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

        with timer.lap("06_conflict"):
            question_obj, evidences = self.stage06.parse_bundle(bundle)
            conflict = self.stage06.run_conflict_disambiguator(question_obj, evidences)
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
            "question": question,
            "answer": answer.get("answer"),
            "overall_status": conflict["overall_status"],
            "called_vlm": answer.get("called_vlm"),
            "conflict_summary": conflict["summary"],
            "evidence_ids_in_prompt": answer.get("evidence_ids_in_prompt"),
            "evidence_ids_excluded_genuine": answer.get("evidence_ids_excluded_genuine"),
            "usage": answer.get("usage"),
            "timings_sec": timer.laps,
            "outputs": {
                "retrieval": str(retrieval_input),
                "normalized": str(qdir / "04_normalized" / "normalized_bundle.json"),
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
        f"제외(genuine)={summary['genuine_conflict_count']}건"
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