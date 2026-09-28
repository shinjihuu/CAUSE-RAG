#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CAUSE-RAG Stage 05: PDF-grounded multimodal hybrid retrieval.

Pipeline
--------
1. Script 02 rows -> evidence dense/BM25/metadata retrieval.
2. Original PDFs -> page dense/BM25 candidate retrieval.
3. Only shortlisted original pages -> OpenAI vision relevance reranking.
4. Evidence score + linked page score -> fused Top-5.
5. Preserve raw evidence, page image, bbox crop, and provenance for Script 04
   and later conflict/VLM stages.

Gold data (dataset/03_dart_gold) is deliberately never indexed.

Install: pip install -U openai pydantic numpy pymupdf

Windows CMD:
  set OPENAI_API_KEY=YOUR_KEY
  python 05_retrieve_evidence.py --mode run ^
    --question "2025년 연결 기준 삼성전자의 영업이익은?"

Parsed-only ablation:
  python 05_retrieve_evidence.py --mode run --skip-vision-rerank ^
    --question "질문"
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, Sequence

try:
    import numpy as np
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:
    raise SystemExit(
        "[ERROR] dependency missing: pip install -U openai pydantic numpy pymupdf"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE_DIR / "dataset" / "02_structured" / "structured_evidence.jsonl"
DEFAULT_PDF_DIR = BASE_DIR / "pdf_data"
DEFAULT_OUT = BASE_DIR / "dataset" / "05_retrieval"
DEFAULT_DB = DEFAULT_OUT / "cause_rag_multimodal.db"
DEFAULT_RETRIEVAL = DEFAULT_OUT / "retrieval_input.json"
DEFAULT_CONTEXT = DEFAULT_OUT / "multimodal_context.json"
DEFAULT_DIAGNOSTICS = DEFAULT_OUT / "retrieval_diagnostics.json"
DEFAULT_IMAGES = DEFAULT_OUT / "page_images"
DEFAULT_NORMALIZED = BASE_DIR / "dataset" / "04_normalized"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_VISION_MAX_OUTPUT_TOKENS = 1800
DEFAULT_NORMALIZATION_MAX_OUTPUT_TOKENS = 1600
SCRIPT04_CANDIDATES = (
    BASE_DIR / "04_normalize_retrieved_evidence.py",
    BASE_DIR / "04_normalize_retrieved_evidence(1).py",
)
SCRIPT04 = next(
    (path for path in SCRIPT04_CANDIDATES if path.exists()), SCRIPT04_CANDIDATES[0]
)
SCHEMA_VERSION = "cause-rag-multimodal-retrieval-v2.0"
DB_VERSION = "2"
VISION_PROMPT_VERSION = "cause-rag-page-rerank-v1.0"
CONTENT_KEYS = ("evidence_text", "raw_content", "content", "text", "page_content")


@dataclass(frozen=True)
class EvidenceDoc:
    evidence_id: str
    raw_content: str
    retrieval_text: str
    tokens: list[str]
    fields: dict[str, Any]
    metadata: dict[str, Any]
    content_hash: str


@dataclass(frozen=True)
class PageDoc:
    page_id: str
    document_id: str
    entity: str
    source_file: str
    page: int
    pdf_path: str
    pdf_sha256: str
    native_text: str
    retrieval_text: str
    tokens: list[str]
    linked_ids: list[str]
    metadata: dict[str, Any]
    content_hash: str


@dataclass(frozen=True)
class PageCandidate:
    row: sqlite3.Row
    base_score: float
    dense_score: float
    lexical_score: float
    prior_score: float
    image_path: Path


class VisionPageScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_label: str
    relevance: float = Field(ge=0.0, le=1.0)
    contains_answer_evidence: bool
    modalities: list[Literal["text", "table", "chart", "image", "other"]]
    visual_evidence_text: str
    rationale: str


class VisionBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pages: list[VisionPageScore]


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed_texts(self, texts: Sequence[str]) -> list[np.ndarray]: ...


class VisionReranker(Protocol):
    model: str

    def rerank(
        self, question: str, candidates: Sequence[PageCandidate]
    ) -> dict[str, dict[str, Any]]: ...


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def clean_content(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def compact_key(value: Any) -> str:
    text = clean_text(value).lower().replace("㈜", "").replace("주식회사", "")
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def stable_hash(*parts: Any, length: int = 24) -> str:
    value = "\x1f".join(clean_text(part) for part in parts)
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be object: {path}:{number}")
                rows.append(row)
        return rows
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, list):
        return value
    for key in ("evidences", "records", "data", "items"):
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return value[key]
    raise ValueError(f"unsupported JSON structure: {path}")


def first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None and clean_text(row[key]):
            return row[key]
    return None


def safe_int(value: Any) -> int | None:
    try:
        return int(float(clean_text(value).replace(",", "")))
    except (TypeError, ValueError):
        return None


def parse_bbox(value: Any) -> dict[str, float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        value = dict(zip(("x0", "y0", "x1", "y1"), value))
    if not isinstance(value, dict):
        return None
    try:
        result = {key: float(value[key]) for key in ("x0", "y0", "x1", "y1")}
    except (KeyError, TypeError, ValueError):
        return None
    return (
        result if result["x1"] > result["x0"] and result["y1"] > result["y0"] else None
    )


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    return text[:head] + "\n[...TRUNCATED...]\n" + text[-(limit - head) :]


def batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def require_fitz() -> Any:
    try:
        import fitz

        return fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF required: pip install -U pymupdf") from exc


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("invalid zero embedding")
    return vector / norm


def blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unblob(value: bytes, dimensions: int) -> np.ndarray:
    vector = np.frombuffer(value, dtype=np.float32)
    if vector.size != dimensions:
        raise ValueError(f"embedding dimension mismatch: {vector.size} != {dimensions}")
    return vector


def score01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if math.isclose(low, high, abs_tol=1e-12):
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)


TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[가-힣]+")
STOPWORDS = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "에서",
    "으로",
    "로",
    "과",
    "와",
    "및",
    "기준",
    "얼마",
    "무엇",
    "인가",
    "알려줘",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "is",
    "what",
    "how",
}


def tokenize(text: str) -> list[str]:
    output = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        token = match.group()
        if token in STOPWORDS:
            continue
        output.append(token)
        if re.fullmatch(r"[가-힣]{3,}", token):
            output.extend(token[i : i + 2] for i in range(len(token) - 1))
    return output


def bm25(query: Sequence[str], documents: Sequence[Sequence[str]]) -> np.ndarray:
    scores = np.zeros(len(documents), dtype=np.float64)
    if not documents or not query:
        return scores
    lengths = np.array([len(doc) for doc in documents], dtype=np.float64)
    average = max(float(lengths.mean()), 1.0)
    frequencies = [Counter(doc) for doc in documents]
    document_frequency = Counter()
    for frequency in frequencies:
        document_frequency.update(frequency.keys())
    for token in set(query):
        df = document_frequency.get(token, 0)
        if not df:
            continue
        idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
        for index, frequency in enumerate(frequencies):
            tf = frequency.get(token, 0)
            if tf:
                scores[index] += (
                    idf
                    * tf
                    * 2.5
                    / (tf + 1.5 * (0.25 + 0.75 * lengths[index] / average))
                )
    return scores


def normalized_filename(value: str) -> str:
    return compact_key(Path(value).name)


def discover_pdfs(explicit: Sequence[Path], pdf_dir: Path) -> list[Path]:
    paths = []
    for path in explicit:
        path = path.resolve()
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"PDF not found: {path}")
        paths.append(path)
    if pdf_dir.exists():
        paths.extend(path.resolve() for path in pdf_dir.rglob("*.pdf"))
    paths = sorted(set(paths), key=lambda path: str(path).casefold())
    if not paths:
        raise FileNotFoundError(
            "원본 PDF가 없습니다. pdf_data 또는 --pdf를 지정하세요."
        )
    return paths


def pdf_lookup(paths: Sequence[Path]) -> dict[str, Path]:
    lookup = {}
    for path in paths:
        for key in (path.name.casefold(), normalized_filename(path.name)):
            if key in lookup and lookup[key] != path:
                raise ValueError(f"ambiguous PDF filename: {lookup[key]} / {path}")
            lookup[key] = path
    return lookup


def match_pdf(source_file: str, lookup: dict[str, Path]) -> Path | None:
    return lookup.get(Path(source_file).name.casefold()) or lookup.get(
        normalized_filename(source_file)
    )


def image_path(
    image_dir: Path, document_id: str, page: int, pdf_hash: str, dpi: int
) -> Path:
    return (
        image_dir
        / document_id
        / f"page_{page:04d}_{stable_hash(pdf_hash, dpi, length=10)}.jpg"
    )


def render_page(pdf_path: Path, page: int, output: Path, dpi: int) -> Path:
    if output.exists() and output.stat().st_size:
        return output
    fitz = require_fitz()
    output.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as document:
        if not 1 <= page <= len(document):
            raise ValueError(
                f"page out of range: {pdf_path.name} {page}/{len(document)}"
            )
        pixmap = document[page - 1].get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False, colorspace=fitz.csRGB
        )
        pixmap.save(output)
    return output


def render_crop(
    pdf_path: Path, page: int, bbox: dict[str, float] | None, output: Path, dpi: int
) -> Path | None:
    if not bbox:
        return None
    if output.exists() and output.stat().st_size:
        return output
    fitz = require_fitz()
    output.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as document:
        pdf_page = document[page - 1]
        rect = (
            fitz.Rect(bbox["x0"] - 6, bbox["y0"] - 6, bbox["x1"] + 6, bbox["y1"] + 6)
            & pdf_page.rect
        )
        if rect.is_empty:
            return None
        pixmap = pdf_page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72),
            clip=rect,
            alpha=False,
            colorspace=fitz.csRGB,
        )
        pixmap.save(output)
    return output


def evidence_text(row: dict[str, Any], raw: str) -> str:
    fields = [
        ("Entity", first(row, "entity", "company", "corp_name")),
        ("Metric", first(row, "metric_raw", "metric_path_raw", "metric")),
        (
            "Metric canonical",
            first(row, "metric_canonical", "alignment_metric_key", "metric_norm"),
        ),
        ("Value", first(row, "value_raw", "value")),
        ("Unit", first(row, "unit_raw", "unit")),
        ("Period", first(row, "period_raw", "period_year", "period")),
        ("Scope", first(row, "scope", "scope_raw")),
        ("Statement", first(row, "statement_type", "statement")),
        ("Modality", first(row, "modality")),
    ]
    return "\n".join(
        [f"{key}: {clean_text(value)}" for key, value in fields if value is not None]
        + [f"Evidence:\n{raw}"]
    )


def load_evidence(
    sources: Sequence[Path], lookup: dict[str, Path], limit: int, allow_missing: bool
) -> tuple[list[EvidenceDoc], dict[str, Any]]:
    documents, seen, missing = [], {}, Counter()
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        for number, row in enumerate(read_records(source), 1):
            raw = clean_content(first(row, *CONTENT_KEYS))
            if not raw:
                continue
            source_file = clean_text(first(row, "source_file", "source_document"))
            pdf = match_pdf(source_file, lookup)
            if not pdf:
                missing[source_file or "<empty>"] += 1
            evidence_id = clean_text(
                first(row, "evidence_id", "block_id", "id")
            ) or "ev_" + stable_hash(source, number, raw)
            text = clip(evidence_text(row, raw), limit)
            fields = {
                "modality": clean_text(first(row, "modality")) or "unknown",
                "entity": clean_text(first(row, "entity", "company", "corp_name")),
                "metric_raw": clean_text(
                    first(row, "metric_raw", "metric_path_raw", "metric")
                ),
                "metric_canonical": clean_text(
                    first(
                        row, "metric_canonical", "alignment_metric_key", "metric_norm"
                    )
                ),
                "period_raw": clean_text(first(row, "period_raw", "period")),
                "period_year": clean_text(
                    first(row, "period_year", "source_report_year")
                ),
                "scope": clean_text(first(row, "scope", "scope_raw")),
                "statement": clean_text(first(row, "statement_type", "statement")),
                "version": clean_text(first(row, "filing_version", "version")),
                "document_id": clean_text(first(row, "document_id", "version_id"))
                or "doc_" + stable_hash(source_file),
                "source_file": source_file,
                "page": safe_int(first(row, "page")),
                "bbox_pdf": parse_bbox(first(row, "bbox_pdf", "bbox")),
                "pdf_path": str(pdf) if pdf else "",
                "source_path": str(source),
            }
            excluded = set(CONTENT_KEYS)
            metadata = {key: value for key, value in row.items() if key not in excluded}
            metadata.update(
                {
                    "retrieval_source_path": str(source),
                    "retrieval_source_row": number,
                    "pdf_linked": bool(pdf),
                }
            )
            document = EvidenceDoc(
                evidence_id,
                raw,
                text,
                tokenize(text),
                fields,
                metadata,
                hashlib.sha256(text.encode()).hexdigest(),
            )
            if (
                evidence_id in seen
                and seen[evidence_id].content_hash != document.content_hash
            ):
                raise ValueError(
                    f"duplicate evidence_id with different content: {evidence_id}"
                )
            if evidence_id not in seen:
                seen[evidence_id] = document
                documents.append(document)
    if not documents:
        raise ValueError("no evidence to index")
    if missing and not allow_missing:
        detail = "\n".join(
            f"- {key}: {count}" for key, count in missing.most_common(20)
        )
        raise ValueError(
            f"구조화 근거와 PDF 파일명을 연결하지 못했습니다:\n{detail}\n텍스트 ablation에만 --allow-missing-pdf를 사용하세요."
        )
    return documents, {
        "evidence_count": len(documents),
        "pdf_linked_count": sum(bool(d.fields["pdf_path"]) for d in documents),
        "missing_pdf_rows": sum(missing.values()),
    }


def infer_entity(filename: str) -> str:
    match = re.search(r"\[([^\]]+)\]", filename)
    return (
        clean_text(match.group(1))
        if match
        else clean_text(
            re.split(r"사업보고서|지속가능|감사보고서", Path(filename).stem)[0]
        )
    )


def load_pages(
    pdfs: Sequence[Path], evidence: Sequence[EvidenceDoc], limit: int
) -> tuple[list[PageDoc], dict[str, Any]]:
    fitz = require_fitz()
    linked = defaultdict(list)
    for item in evidence:
        if item.fields["pdf_path"] and item.fields["page"]:
            linked[
                (str(Path(item.fields["pdf_path"]).resolve()), item.fields["page"])
            ].append(item)
    pages, summaries = [], []
    for pdf in pdfs:
        pdf, digest = pdf.resolve(), sha256_file(pdf)
        document_id = "pdf_" + stable_hash(pdf.name, digest)
        with fitz.open(pdf) as document:
            entity_values = [
                item.fields["entity"]
                for (path, _), items in linked.items()
                if path == str(pdf)
                for item in items
                if item.fields["entity"]
            ]
            entity = (
                Counter(entity_values).most_common(1)[0][0]
                if entity_values
                else infer_entity(pdf.name)
            )
            for page_number, page in enumerate(document, 1):
                native = clean_content(page.get_text("text", sort=True))
                items = linked.get((str(pdf), page_number), [])
                structured = "\n".join(
                    clip(item.retrieval_text, 1400) for item in items[:12]
                )
                text = clip(
                    f"Entity: {entity}\nSource: {pdf.name}\nPage: {page_number}\nNative page text:\n{native}\nLinked structured evidence:\n{structured}",
                    limit,
                )
                content_hash = hashlib.sha256(
                    (digest + "\x1f" + text).encode()
                ).hexdigest()
                pages.append(
                    PageDoc(
                        "pg_" + stable_hash(digest, page_number),
                        document_id,
                        entity,
                        pdf.name,
                        page_number,
                        str(pdf),
                        digest,
                        native,
                        text,
                        tokenize(text),
                        [item.evidence_id for item in items],
                        {
                            "width": float(page.rect.width),
                            "height": float(page.rect.height),
                            "rotation": int(page.rotation),
                            "native_text_chars": len(native),
                            "embedded_images": len(page.get_images(full=True)),
                        },
                        content_hash,
                    )
                )
            summaries.append(
                {
                    "source_file": pdf.name,
                    "pdf_path": str(pdf),
                    "pdf_sha256": digest,
                    "document_id": document_id,
                    "page_count": len(document),
                }
            )
    return pages, {"page_count": len(pages), "pdfs": summaries}


class OpenAIEmbedder:
    def __init__(self, model: str, dimensions: int, timeout: float, attempts: int):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
        from openai import OpenAI

        self.model, self.dimensions, self.attempts = model, dimensions, attempts
        self.client = OpenAI(timeout=timeout, max_retries=0)

    def embed_texts(self, texts: Sequence[str]) -> list[np.ndarray]:
        error = None
        for attempt in range(self.attempts):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=list(texts),
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
                output = [
                    normalize_vector(np.asarray(item.embedding, dtype=np.float32))
                    for item in sorted(response.data, key=lambda x: x.index)
                ]
                if len(output) != len(texts) or any(
                    len(vector) != self.dimensions for vector in output
                ):
                    raise RuntimeError("invalid embedding response")
                return output
            except Exception as exc:
                error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(min(8, 2**attempt + random.random()))
        raise RuntimeError(f"embedding API failed: {error}") from error


class OpenAIVisionReranker:
    def __init__(
        self,
        model: str,
        detail: str,
        batch_size: int,
        timeout: float,
        attempts: int,
        fallback: bool,
        max_output_tokens: int,
    ):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY 환경변수가 없습니다.")
        from openai import OpenAI

        self.model, self.detail, self.batch_size = model, detail, batch_size
        self.attempts, self.fallback = attempts, fallback
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(timeout=timeout, max_retries=0)

    def _batch(
        self, question: str, batch: Sequence[PageCandidate]
    ) -> dict[str, dict[str, Any]]:
        labels = [f"PAGE_{index:02d}" for index in range(1, len(batch) + 1)]
        content = [
            {
                "type": "input_text",
                "text": f"Question: {question}\nEvaluate every labeled original PDF page for visible answer evidence. Check table headers, units, period, scope, charts, captions and footnotes. Do not infer invisible facts. Return exactly one result per label.",
            }
        ]
        for label, candidate in zip(labels, batch):
            encoded = base64.b64encode(candidate.image_path.read_bytes()).decode(
                "ascii"
            )
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"{label}: source={candidate.row['source_file']}, page={candidate.row['page']}, entity={candidate.row['entity']}",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded}",
                        "detail": self.detail,
                    },
                ]
            )
        error = None
        for attempt in range(self.attempts):
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "Conservative multimodal retrieval reranker for Korean "
                                "corporate reports. Rerank only; do not answer."
                            ),
                        },
                        {"role": "user", "content": content},
                    ],
                    text_format=VisionBatchResult,
                    max_output_tokens=self.max_output_tokens,
                    temperature=0,
                    store=False,
                )
                parsed = response.output_parsed
                if parsed is None:
                    raise RuntimeError("empty structured output")
                by_label = {item.page_label: item for item in parsed.pages}
                if set(by_label) != set(labels):
                    raise RuntimeError(
                        f"page label mismatch: {set(labels) - set(by_label)}"
                    )
                return {
                    str(candidate.row["page_id"]): by_label[label].model_dump(
                        mode="json"
                    )
                    for label, candidate in zip(labels, batch)
                }
            except Exception as exc:
                error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(min(10, 2**attempt + random.random()))
        if self.fallback:
            return {
                str(candidate.row["page_id"]): {
                    "page_label": label,
                    "relevance": candidate.base_score,
                    "contains_answer_evidence": False,
                    "modalities": [],
                    "visual_evidence_text": "",
                    "rationale": f"VISION_FALLBACK: {error}",
                }
                for label, candidate in zip(labels, batch)
            }
        raise RuntimeError(f"vision reranking failed: {error}") from error

    def rerank(
        self, question: str, candidates: Sequence[PageCandidate]
    ) -> dict[str, dict[str, Any]]:
        output = {}
        for batch in batches(list(candidates), self.batch_size):
            output.update(self._batch(question, batch))
        return output


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def set_meta(connection: sqlite3.Connection, key: str, value: Any) -> None:
    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    connection.execute(
        "INSERT INTO index_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_meta(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM index_meta WHERE key=?", (key,)
    ).fetchone()
    return str(row["value"]) if row else None


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS index_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
    )
    row = connection.execute(
        "SELECT value FROM index_meta WHERE key='db_version'"
    ).fetchone()
    if row and str(row["value"]) != DB_VERSION:
        connection.executescript(
            "DROP TABLE IF EXISTS evidence_index; DROP TABLE IF EXISTS page_index; DELETE FROM index_meta;"
        )
    connection.executescript("""
    CREATE TABLE IF NOT EXISTS evidence_index(
      evidence_id TEXT PRIMARY KEY, raw_content TEXT NOT NULL, retrieval_text TEXT NOT NULL,
      tokens_json TEXT NOT NULL, modality TEXT NOT NULL, entity TEXT NOT NULL,
      metric_raw TEXT NOT NULL, metric_canonical TEXT NOT NULL, period_raw TEXT NOT NULL,
      period_year TEXT NOT NULL, scope TEXT NOT NULL, statement TEXT NOT NULL,
      version TEXT NOT NULL, document_id TEXT NOT NULL, source_file TEXT NOT NULL,
      page INTEGER, bbox_json TEXT, pdf_path TEXT NOT NULL, metadata_json TEXT NOT NULL,
      source_path TEXT NOT NULL, content_hash TEXT NOT NULL, embedding_model TEXT NOT NULL,
      embedding_dim INTEGER NOT NULL, embedding BLOB NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
    CREATE INDEX IF NOT EXISTS idx_evidence_page ON evidence_index(source_file,page);
    CREATE INDEX IF NOT EXISTS idx_evidence_fields ON evidence_index(entity,metric_canonical);
    CREATE TABLE IF NOT EXISTS page_index(
      page_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, entity TEXT NOT NULL,
      source_file TEXT NOT NULL, page INTEGER NOT NULL, pdf_path TEXT NOT NULL,
      pdf_sha256 TEXT NOT NULL, native_text TEXT NOT NULL, retrieval_text TEXT NOT NULL,
      tokens_json TEXT NOT NULL, linked_ids_json TEXT NOT NULL, metadata_json TEXT NOT NULL,
      content_hash TEXT NOT NULL, embedding_model TEXT NOT NULL, embedding_dim INTEGER NOT NULL,
      embedding BLOB NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(pdf_path,page));
    CREATE INDEX IF NOT EXISTS idx_page_source ON page_index(source_file,page);
    """)
    set_meta(connection, "db_version", DB_VERSION)
    connection.commit()


def sync_evidence(
    connection: sqlite3.Connection,
    documents: Sequence[EvidenceDoc],
    embedder: Embedder,
    batch_size: int,
    force: bool,
) -> dict[str, int]:
    existing = {
        str(row["evidence_id"]): row
        for row in connection.execute(
            "SELECT evidence_id,content_hash,embedding_model,embedding_dim FROM evidence_index"
        )
    }
    pending, unchanged = [], 0
    for document in documents:
        old = existing.get(document.evidence_id)
        current = (
            old
            and old["content_hash"] == document.content_hash
            and old["embedding_model"] == embedder.model
            and int(old["embedding_dim"]) == embedder.dimensions
        )
        if current and not force:
            unchanged += 1
            f = document.fields
            connection.execute(
                """UPDATE evidence_index SET modality=?,entity=?,metric_raw=?,metric_canonical=?,period_raw=?,period_year=?,scope=?,statement=?,version=?,document_id=?,source_file=?,page=?,bbox_json=?,pdf_path=?,metadata_json=?,source_path=?,updated_at=CURRENT_TIMESTAMP WHERE evidence_id=?""",
                (
                    f["modality"],
                    f["entity"],
                    f["metric_raw"],
                    f["metric_canonical"],
                    f["period_raw"],
                    f["period_year"],
                    f["scope"],
                    f["statement"],
                    f["version"],
                    f["document_id"],
                    f["source_file"],
                    f["page"],
                    json.dumps(f["bbox_pdf"], ensure_ascii=False)
                    if f["bbox_pdf"]
                    else None,
                    f["pdf_path"],
                    json.dumps(document.metadata, ensure_ascii=False),
                    f["source_path"],
                    document.evidence_id,
                ),
            )
        else:
            pending.append(document)
    updated = 0
    for batch in batches(pending, batch_size):
        vectors = embedder.embed_texts([item.retrieval_text for item in batch])
        for item, vector in zip(batch, vectors):
            f = item.fields
            connection.execute(
                """INSERT INTO evidence_index VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(evidence_id) DO UPDATE SET raw_content=excluded.raw_content,retrieval_text=excluded.retrieval_text,tokens_json=excluded.tokens_json,modality=excluded.modality,entity=excluded.entity,metric_raw=excluded.metric_raw,metric_canonical=excluded.metric_canonical,period_raw=excluded.period_raw,period_year=excluded.period_year,scope=excluded.scope,statement=excluded.statement,version=excluded.version,document_id=excluded.document_id,source_file=excluded.source_file,page=excluded.page,bbox_json=excluded.bbox_json,pdf_path=excluded.pdf_path,metadata_json=excluded.metadata_json,source_path=excluded.source_path,content_hash=excluded.content_hash,embedding_model=excluded.embedding_model,embedding_dim=excluded.embedding_dim,embedding=excluded.embedding,updated_at=CURRENT_TIMESTAMP""",
                (
                    item.evidence_id,
                    item.raw_content,
                    item.retrieval_text,
                    json.dumps(item.tokens, ensure_ascii=False),
                    f["modality"],
                    f["entity"],
                    f["metric_raw"],
                    f["metric_canonical"],
                    f["period_raw"],
                    f["period_year"],
                    f["scope"],
                    f["statement"],
                    f["version"],
                    f["document_id"],
                    f["source_file"],
                    f["page"],
                    json.dumps(f["bbox_pdf"], ensure_ascii=False)
                    if f["bbox_pdf"]
                    else None,
                    f["pdf_path"],
                    json.dumps(item.metadata, ensure_ascii=False),
                    f["source_path"],
                    item.content_hash,
                    embedder.model,
                    embedder.dimensions,
                    blob(vector),
                ),
            )
        updated += len(batch)
        connection.commit()
        print(f"  evidence embeddings: {updated}/{len(pending)}")
    stale = set(existing) - {item.evidence_id for item in documents}
    connection.executemany(
        "DELETE FROM evidence_index WHERE evidence_id=?", [(item,) for item in stale]
    )
    return {"updated": updated, "unchanged": unchanged, "deleted": len(stale)}


def sync_pages(
    connection: sqlite3.Connection,
    documents: Sequence[PageDoc],
    embedder: Embedder,
    batch_size: int,
    force: bool,
) -> dict[str, int]:
    existing = {
        str(row["page_id"]): row
        for row in connection.execute(
            "SELECT page_id,content_hash,embedding_model,embedding_dim FROM page_index"
        )
    }
    pending, unchanged = [], 0
    for item in documents:
        old = existing.get(item.page_id)
        current = (
            old
            and old["content_hash"] == item.content_hash
            and old["embedding_model"] == embedder.model
            and int(old["embedding_dim"]) == embedder.dimensions
        )
        if current and not force:
            unchanged += 1
            connection.execute(
                """UPDATE page_index SET document_id=?,entity=?,source_file=?,page=?,pdf_path=?,pdf_sha256=?,native_text=?,linked_ids_json=?,metadata_json=?,updated_at=CURRENT_TIMESTAMP WHERE page_id=?""",
                (
                    item.document_id,
                    item.entity,
                    item.source_file,
                    item.page,
                    item.pdf_path,
                    item.pdf_sha256,
                    item.native_text,
                    json.dumps(item.linked_ids, ensure_ascii=False),
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.page_id,
                ),
            )
        else:
            pending.append(item)
    updated = 0
    for batch in batches(pending, batch_size):
        vectors = embedder.embed_texts([item.retrieval_text for item in batch])
        for item, vector in zip(batch, vectors):
            connection.execute(
                """INSERT INTO page_index VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(page_id) DO UPDATE SET document_id=excluded.document_id,entity=excluded.entity,source_file=excluded.source_file,page=excluded.page,pdf_path=excluded.pdf_path,pdf_sha256=excluded.pdf_sha256,native_text=excluded.native_text,retrieval_text=excluded.retrieval_text,tokens_json=excluded.tokens_json,linked_ids_json=excluded.linked_ids_json,metadata_json=excluded.metadata_json,content_hash=excluded.content_hash,embedding_model=excluded.embedding_model,embedding_dim=excluded.embedding_dim,embedding=excluded.embedding,updated_at=CURRENT_TIMESTAMP""",
                (
                    item.page_id,
                    item.document_id,
                    item.entity,
                    item.source_file,
                    item.page,
                    item.pdf_path,
                    item.pdf_sha256,
                    item.native_text,
                    item.retrieval_text,
                    json.dumps(item.tokens, ensure_ascii=False),
                    json.dumps(item.linked_ids, ensure_ascii=False),
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.content_hash,
                    embedder.model,
                    embedder.dimensions,
                    blob(vector),
                ),
            )
        updated += len(batch)
        connection.commit()
        print(f"  page embeddings: {updated}/{len(pending)}")
    stale = set(existing) - {item.page_id for item in documents}
    connection.executemany(
        "DELETE FROM page_index WHERE page_id=?", [(item,) for item in stale]
    )
    return {"updated": updated, "unchanged": unchanged, "deleted": len(stale)}


def build_index(
    sources: Sequence[Path],
    pdfs: Sequence[Path],
    db: Path,
    embedder: Embedder,
    batch_size: int,
    limit: int,
    force: bool,
    allow_missing: bool,
) -> dict[str, Any]:
    evidence, evidence_summary = load_evidence(
        sources, pdf_lookup(pdfs), limit, allow_missing
    )
    pages, page_summary = load_pages(pdfs, evidence, limit)
    connection = connect_db(db)
    try:
        initialize_schema(connection)
        set_meta(connection, "state", "building")
        connection.commit()
        evidence_sync = sync_evidence(connection, evidence, embedder, batch_size, force)
        page_sync = sync_pages(connection, pages, embedder, batch_size, force)
        set_meta(connection, "schema_version", SCHEMA_VERSION)
        set_meta(connection, "embedding_model", embedder.model)
        set_meta(connection, "embedding_dim", embedder.dimensions)
        set_meta(connection, "state", "ready")
        connection.commit()
        return {
            "schema_version": SCHEMA_VERSION,
            "embedding_model": embedder.model,
            "embedding_dimensions": embedder.dimensions,
            "evidence": {**evidence_summary, **evidence_sync},
            "pages": {**page_summary, **page_sync},
            "gold_data_indexed": False,
        }
    except Exception:
        set_meta(connection, "state", "failed")
        connection.commit()
        raise
    finally:
        connection.close()


def validate_index(connection: sqlite3.Connection, embedder: Embedder) -> None:
    if get_meta(connection, "state") != "ready":
        raise ValueError("retrieval DB is not ready; run build")
    if get_meta(connection, "embedding_model") != embedder.model or get_meta(
        connection, "embedding_dim"
    ) != str(embedder.dimensions):
        raise ValueError("DB/query embedding configuration mismatch; rebuild index")


def load_script04() -> Any | None:
    if not SCRIPT04.exists():
        return None
    spec = importlib.util.spec_from_file_location("cause_rag_script04_rules", SCRIPT04)
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scope_fallback(text: str) -> str:
    key = compact_key(text)
    if any(value in key for value in ("연결", "cfs", "consolidated")):
        return "CFS"
    if any(value in key for value in ("별도", "개별", "ofs", "separate")):
        return "OFS"
    return "UNKNOWN"


def year_in(text: str) -> str | None:
    match = re.search(r"20\d{2}", text)
    return match.group() if match else None


def question_hints(
    question: str, rows: Sequence[sqlite3.Row], module: Any | None
) -> dict[str, Any]:
    if module:
        metric = module.canonical_metric_from_text(question)
        scope = module.normalize_scope(question, "UNKNOWN")[0]
        period = module.normalize_period(
            question, None, None, "UNKNOWN", "UNKNOWN"
        ).get("period_start")
    else:
        metric, scope, period = None, scope_fallback(question), year_in(question)
    key = compact_key(question)
    entities = sorted(
        {clean_text(row["entity"]) for row in rows if clean_text(row["entity"])},
        key=len,
        reverse=True,
    )
    entity = next((item for item in entities if compact_key(item) in key), None)
    return {
        "entity": entity,
        "entity_key": compact_key(entity) or None,
        "metric": metric,
        "scope": scope,
        "period_start": period,
        "period_year": year_in(str(period)) if period else None,
    }


def expanded_query(question: str, hints: dict[str, Any]) -> str:
    output = [f"Question: {question}"]
    for label, key in (
        ("Entity", "entity"),
        ("Metric canonical", "metric"),
        ("Period", "period_start"),
        ("Scope", "scope"),
    ):
        if hints.get(key) and hints[key] != "UNKNOWN":
            output.append(f"{label}: {hints[key]}")
    return "\n".join(output)


def metadata_score(
    row: sqlite3.Row, hints: dict[str, Any], module: Any | None
) -> tuple[float, dict[str, bool | None]]:
    details = {"entity": None, "metric": None, "period": None, "scope": None}
    checks = []
    if hints.get("entity_key"):
        details["entity"] = compact_key(row["entity"]) == hints["entity_key"]
        checks.append((0.45, details["entity"]))
    if hints.get("metric"):
        canonical = clean_text(row["metric_canonical"])
        metric = (
            canonical.lower()
            if re.fullmatch(r"[a-z][a-z0-9_]*", canonical.lower())
            else (
                module.canonical_metric_from_text(row["metric_raw"], canonical)
                if module
                else canonical
            )
        )
        details["metric"] = metric == hints["metric"]
        checks.append((0.45, details["metric"]))
    if hints.get("period_year"):
        details["period"] = (
            clean_text(row["period_year"]) or year_in(clean_text(row["period_raw"]))
        ) == hints["period_year"]
        checks.append((0.05, details["period"]))
    if hints.get("scope") and hints["scope"] != "UNKNOWN":
        scope = (
            module.normalize_scope(row["scope"], "UNKNOWN")[0]
            if module
            else scope_fallback(row["scope"])
        )
        details["scope"] = scope == hints["scope"]
        checks.append((0.05, details["scope"]))
    denominator = sum(weight for weight, _ in checks)
    return (
        sum(weight for weight, matched in checks if matched) / denominator
        if denominator
        else 0.0
    ), details


def page_key(source: Any, page: Any) -> tuple[str, int] | None:
    page = safe_int(page)
    return (Path(clean_text(source)).name.casefold(), page) if source and page else None


def select_pages(
    rows: Sequence[sqlite3.Row],
    page_scores: np.ndarray,
    page_dense_scores: np.ndarray,
    page_lexical_scores: np.ndarray,
    evidence_rows: Sequence[sqlite3.Row],
    evidence_scores: np.ndarray,
    candidate_k: int,
    images: Path,
    dpi: int,
) -> list[PageCandidate]:
    evidence_prior = defaultdict(float)
    for index in np.argsort(-evidence_scores)[: max(candidate_k * 4, 40)]:
        key = page_key(
            evidence_rows[int(index)]["source_file"], evidence_rows[int(index)]["page"]
        )
        if key:
            evidence_prior[key] = max(
                evidence_prior[key], float(evidence_scores[int(index)])
            )
    ranked = []
    for index, row in enumerate(rows):
        ranked.append(
            (
                max(
                    float(page_scores[index]),
                    evidence_prior.get(page_key(row["source_file"], row["page"]), 0),
                ),
                index,
            )
        )
    ranked.sort(key=lambda pair: (-pair[0], str(rows[pair[1]]["page_id"])))
    output = []
    for prior, index in ranked[:candidate_k]:
        row = rows[index]
        path = image_path(
            images,
            str(row["document_id"]),
            int(row["page"]),
            str(row["pdf_sha256"]),
            dpi,
        )
        render_page(Path(str(row["pdf_path"])), int(row["page"]), path, dpi)
        output.append(
            PageCandidate(
                row=row,
                base_score=float(page_scores[index]),
                dense_score=float(page_dense_scores[index]),
                lexical_score=float(page_lexical_scores[index]),
                prior_score=prior,
                image_path=path,
            )
        )
    return output


def diverse(
    candidates: Sequence[dict[str, Any]], top_k: int, maximum: int
) -> list[dict[str, Any]]:
    selected, used, counts = [], set(), Counter()
    for item in candidates:
        document = clean_text(item["metadata"].get("document_id")) or "UNKNOWN"
        if item["evidence_id"] in used or counts[document] >= maximum:
            continue
        selected.append(item)
        used.add(item["evidence_id"])
        counts[document] += 1
        if len(selected) == top_k:
            return selected
    for item in candidates:
        if item["evidence_id"] not in used:
            selected.append(item)
            used.add(item["evidence_id"])
            if len(selected) == top_k:
                break
    return selected


def retrieve(
    db: Path,
    question: str,
    embedder: Embedder,
    vision: VisionReranker | None,
    top_k: int,
    ew: tuple[float, float, float],
    pw: tuple[float, float],
    fw: tuple[float, float],
    vision_weight: float,
    candidate_k: int,
    max_per_document: int,
    images: Path,
    dpi: int,
    allow_page_only: bool,
    page_only_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    connection = connect_db(db)
    try:
        initialize_schema(connection)
        validate_index(connection, embedder)
        evidence_rows = connection.execute(
            "SELECT * FROM evidence_index ORDER BY evidence_id"
        ).fetchall()
        original_pages = connection.execute(
            "SELECT * FROM page_index ORDER BY page_id"
        ).fetchall()
        if len(evidence_rows) < top_k or not original_pages:
            raise ValueError(
                f"insufficient index: evidence={len(evidence_rows)}, pages={len(original_pages)}"
            )
        module = load_script04()
        hints = question_hints(question, evidence_rows, module)
        query_text = expanded_query(question, hints)
        query_vector = embedder.embed_texts([query_text])[0]
        query_tokens = tokenize(query_text)

        evidence_dense_raw = (
            np.vstack(
                [unblob(row["embedding"], embedder.dimensions) for row in evidence_rows]
            )
            @ query_vector
        )
        evidence_lexical_raw = bm25(
            query_tokens, [json.loads(row["tokens_json"]) for row in evidence_rows]
        )
        evidence_metadata = np.zeros(len(evidence_rows))
        match_details = []
        for index, row in enumerate(evidence_rows):
            evidence_metadata[index], details = metadata_score(row, hints, module)
            match_details.append(details)
        evidence_base = (
            ew[0] * score01(evidence_dense_raw)
            + ew[1] * score01(evidence_lexical_raw)
            + ew[2] * evidence_metadata
        )

        page_dense_raw = (
            np.vstack(
                [
                    unblob(row["embedding"], embedder.dimensions)
                    for row in original_pages
                ]
            )
            @ query_vector
        )
        page_lexical_raw = bm25(
            query_tokens, [json.loads(row["tokens_json"]) for row in original_pages]
        )
        page_base = pw[0] * score01(page_dense_raw) + pw[1] * score01(page_lexical_raw)

        candidates = select_pages(
            rows=original_pages,
            page_scores=page_base,
            page_dense_scores=page_dense_raw,
            page_lexical_scores=page_lexical_raw,
            evidence_rows=evidence_rows,
            evidence_scores=evidence_base,
            candidate_k=candidate_k,
            images=images,
            dpi=dpi,
        )

        visual = (
            vision.rerank(question, candidates)
            if vision
            else {
                str(candidate.row["page_id"]): {
                    "page_label": "BASELINE",
                    "relevance": candidate.base_score,
                    "contains_answer_evidence": False,
                    "modalities": [],
                    "visual_evidence_text": "",
                    "rationale": "vision disabled",
                }
                for candidate in candidates
            }
        )
        page_details = {}
        for candidate in candidates:
            row, result = candidate.row, visual[str(candidate.row["page_id"])]
            key = page_key(row["source_file"], row["page"])
            visual_score = float(result.get("relevance", candidate.base_score))
            fused = (
                candidate.base_score
                if not vision
                else (1 - vision_weight) * candidate.base_score
                + vision_weight * visual_score
            )
            page_details[key] = {
                "page_id": str(row["page_id"]),
                "document_id": str(row["document_id"]),
                "source_file": str(row["source_file"]),
                "page": int(row["page"]),
                "pdf_path": str(row["pdf_path"]),
                "pdf_sha256": str(row["pdf_sha256"]),
                "native_text": str(row["native_text"]),
                "page_image_path": str(candidate.image_path),
                "linked_ids": json.loads(row["linked_ids_json"]),
                "page_dense_cosine": candidate.dense_score,
                "page_lexical_bm25": candidate.lexical_score,
                "page_base_score": candidate.base_score,
                "candidate_prior_score": candidate.prior_score,
                "vision": result,
                "multimodal_page_score": fused,
            }

        all_results = []
        for index, row in enumerate(evidence_rows):
            detail = page_details.get(page_key(row["source_file"], row["page"]))
            page_score = float(detail["multimodal_page_score"]) if detail else 0.0
            final_score = fw[0] * float(evidence_base[index]) + fw[1] * page_score
            metadata = json.loads(row["metadata_json"])
            metadata.update(
                {
                    "retrieval_unit_type": "structured_evidence",
                    "entity": str(row["entity"]),
                    "metric_raw": str(row["metric_raw"]),
                    "metric_canonical": str(row["metric_canonical"]),
                    "period_raw": str(row["period_raw"]),
                    "period_year": str(row["period_year"]),
                    "scope": str(row["scope"]),
                    "statement": str(row["statement"]),
                    "version": str(row["version"]),
                    "modality": str(row["modality"]),
                    "document_id": str(row["document_id"]),
                    "source_file": str(row["source_file"]),
                    "page": row["page"],
                    "bbox_pdf": json.loads(row["bbox_json"])
                    if row["bbox_json"]
                    else None,
                    "pdf_path": str(row["pdf_path"]),
                    "page_image_path": detail["page_image_path"] if detail else None,
                    "vision": detail["vision"] if detail else None,
                    "evidence_dense_cosine": round(float(evidence_dense_raw[index]), 8),
                    "evidence_lexical_bm25": round(
                        float(evidence_lexical_raw[index]), 8
                    ),
                    "metadata_match_score": round(float(evidence_metadata[index]), 8),
                    "metadata_match_details": match_details[index],
                    "evidence_base_score": round(float(evidence_base[index]), 8),
                    "multimodal_page_score": round(page_score, 8),
                    "retrieval_score": round(final_score, 8),
                }
            )
            all_results.append(
                {
                    "evidence_id": str(row["evidence_id"]),
                    "content": str(row["raw_content"]),
                    "raw_content": str(row["raw_content"]),
                    "metadata": metadata,
                }
            )

        if allow_page_only:
            for key, detail in page_details.items():
                result = detail["vision"]
                modalities = set(result.get("modalities") or [])
                visible = clean_content(result.get("visual_evidence_text"))
                if (
                    detail["multimodal_page_score"] < page_only_threshold
                    or not result.get("contains_answer_evidence")
                    or not visible
                    or "chart" not in modalities
                ):
                    continue
                raw = f"[VLM visual evidence]\n{visible}\n\n[Native PDF text]\n{clip(detail['native_text'], 5000)}"
                all_results.append(
                    {
                        "evidence_id": "visual_" + detail["page_id"],
                        "content": raw,
                        "raw_content": raw,
                        "metadata": {
                            "retrieval_unit_type": "visual_page_evidence",
                            "entity": "",
                            "metric_raw": "",
                            "metric_canonical": "",
                            "period_raw": "",
                            "period_year": "",
                            "scope": "UNKNOWN",
                            "statement": "UNKNOWN",
                            "version": "UNKNOWN",
                            "modality": "chart",
                            "document_id": detail["document_id"],
                            "source_file": detail["source_file"],
                            "page": detail["page"],
                            "bbox_pdf": None,
                            "pdf_path": detail["pdf_path"],
                            "page_image_path": detail["page_image_path"],
                            "vision": result,
                            "evidence_base_score": 0.0,
                            "multimodal_page_score": round(
                                float(detail["multimodal_page_score"]), 8
                            ),
                            "retrieval_score": round(
                                float(detail["multimodal_page_score"]), 8
                            ),
                        },
                    }
                )

        all_results.sort(
            key=lambda item: (
                -float(item["metadata"]["retrieval_score"]),
                str(item["evidence_id"]),
            )
        )
        selected = diverse(all_results, top_k, max_per_document)
        if len(selected) != top_k:
            raise ValueError(f"cannot construct Top-{top_k}: {len(selected)}")
        page_row_lookup = {
            page_key(row["source_file"], row["page"]): row for row in original_pages
        }
        for rank, item in enumerate(selected, 1):
            metadata = item["metadata"]
            metadata["retrieval_rank"] = rank
            row = page_row_lookup.get(
                page_key(metadata.get("source_file"), metadata.get("page"))
            )
            if row:
                path = image_path(
                    images,
                    str(row["document_id"]),
                    int(row["page"]),
                    str(row["pdf_sha256"]),
                    dpi,
                )
                render_page(Path(str(row["pdf_path"])), int(row["page"]), path, dpi)
                metadata["page_image_path"] = str(path)
                crop = render_crop(
                    Path(str(row["pdf_path"])),
                    int(row["page"]),
                    metadata.get("bbox_pdf"),
                    path.parent
                    / f"crop_{int(row['page']):04d}_{stable_hash(item['evidence_id'], length=10)}.jpg",
                    dpi,
                )
                metadata["evidence_crop_path"] = str(crop) if crop else None
            metadata["embedding_model"], metadata["embedding_dimensions"] = (
                embedder.model,
                embedder.dimensions,
            )

        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_stage": "05_multimodal_retrieval",
            "question": question,
            "expanded_query_text": query_text,
            "question_hints": hints,
            "database_counts": {
                "evidence": len(evidence_rows),
                "pages": len(original_pages),
            },
            "weights": {
                "evidence": {"dense": ew[0], "lexical": ew[1], "metadata": ew[2]},
                "page": {"dense": pw[0], "lexical": pw[1]},
                "fusion": {
                    "evidence": fw[0],
                    "page": fw[1],
                    "vision_within_page": vision_weight,
                },
            },
            "vision": {
                "enabled": vision is not None,
                "model": vision.model if vision else None,
                "prompt_version": VISION_PROMPT_VERSION if vision else None,
                "candidate_count": len(candidates),
                "pages": [
                    {
                        key: detail[key]
                        for key in (
                            "page_id",
                            "source_file",
                            "page",
                            "page_base_score",
                            "multimodal_page_score",
                            "vision",
                        )
                    }
                    for detail in sorted(
                        page_details.values(), key=lambda x: -x["multimodal_page_score"]
                    )
                ],
            },
            "selected": [
                {
                    "rank": item["metadata"]["retrieval_rank"],
                    "evidence_id": item["evidence_id"],
                    "retrieval_score": item["metadata"]["retrieval_score"],
                    "retrieval_unit_type": item["metadata"]["retrieval_unit_type"],
                    "source_file": item["metadata"].get("source_file"),
                    "page": item["metadata"].get("page"),
                    "modality": item["metadata"].get("modality"),
                    "page_image_path": item["metadata"].get("page_image_path"),
                    "evidence_crop_path": item["metadata"].get("evidence_crop_path"),
                }
                for item in selected
            ],
            "gold_data_indexed": False,
        }
        return selected, diagnostics
    finally:
        connection.close()


def save_outputs(
    retrieval_path: Path,
    context_path: Path,
    diagnostics_path: Path,
    question: str,
    results: Sequence[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    save_json(
        retrieval_path,
        {
            "schema_version": SCHEMA_VERSION,
            "pipeline_stage": "05_multimodal_retrieval",
            "question": question,
            "evidences": list(results),
            "retrieval_config": {
                "top_k": len(results),
                "multimodal": diagnostics["vision"]["enabled"],
                "vision_model": diagnostics["vision"]["model"],
                "weights": diagnostics["weights"],
                "gold_data_indexed": False,
            },
        },
    )
    save_json(
        context_path,
        {
            "schema_version": SCHEMA_VERSION,
            "raw_question": question,
            "instruction": "Later stages use raw question, filtered raw/normalized evidence, and linked original PDF page/crop images.",
            "items": [
                {
                    "rank": item["metadata"]["retrieval_rank"],
                    "evidence_id": item["evidence_id"],
                    "raw_content": item["raw_content"],
                    "source_file": item["metadata"].get("source_file"),
                    "page": item["metadata"].get("page"),
                    "modality": item["metadata"].get("modality"),
                    "pdf_path": item["metadata"].get("pdf_path"),
                    "page_image_path": item["metadata"].get("page_image_path"),
                    "evidence_crop_path": item["metadata"].get("evidence_crop_path"),
                    "bbox_pdf": item["metadata"].get("bbox_pdf"),
                    "retrieval_score": item["metadata"].get("retrieval_score"),
                    "vision": item["metadata"].get("vision"),
                }
                for item in results
            ],
        },
    )
    save_json(diagnostics_path, diagnostics)


def run_stage04(
    retrieval_input: Path,
    model: str,
    output: Path,
    workers: int,
    timeout: float,
    max_output_tokens: int,
    no_cache: bool,
) -> None:
    if not SCRIPT04.exists():
        raise FileNotFoundError(f"Script 04 not found: {SCRIPT04}")
    command = [
        sys.executable,
        str(SCRIPT04),
        "--input",
        str(retrieval_input),
        "--output-dir",
        str(output),
        "--model",
        model,
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--max-output-tokens",
        str(max_output_tokens),
    ]
    if no_cache:
        command.append("--no-cache")
    subprocess.run(command, cwd=BASE_DIR, check=True)


class HashingEmbedder:
    model, dimensions = "hashing-test-embedder", 128

    def embed_texts(self, texts: Sequence[str]) -> list[np.ndarray]:
        output = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for token in tokenize(text):
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "little") % self.dimensions] += (
                    1 if digest[4] % 2 == 0 else -1
                )
            if not np.any(vector):
                vector[0] = 1
            output.append(normalize_vector(vector))
        return output


class MockVision:
    model = "mock-vision"

    def rerank(
        self, question: str, candidates: Sequence[PageCandidate]
    ) -> dict[str, dict[str, Any]]:
        output = {}
        for candidate in candidates:
            relevant = (
                "영업이익" in candidate.row["native_text"]
                and "삼성전자" in candidate.row["native_text"]
            )
            output[str(candidate.row["page_id"])] = {
                "page_label": "MOCK",
                "relevance": 0.98 if relevant else 0.2,
                "contains_answer_evidence": relevant,
                "modalities": ["table"] if relevant else ["text"],
                "visual_evidence_text": "삼성전자 2025년 연결 영업이익 36조원"
                if relevant
                else "",
                "rationale": "offline test",
            }
        return output


def self_test() -> None:
    fitz = require_fitz()
    with tempfile.TemporaryDirectory(prefix="cause_rag_mm_test_") as temp_name:
        root = Path(temp_name)
        pdf = root / "[삼성전자]사업보고서(2026.03.10).pdf"
        document = fitz.open()
        page_texts = [
            "삼성전자 기업 개요",
            "삼성전자 연결 손익계산서 2025년 영업이익 36조원",
            "삼성전자 별도 손익계산서 2025년 영업이익 20조원",
            "삼성전자 2024년 연결 영업이익 10조원",
            "삼성전자 매출액 2025년 300조원",
            "삼성전자 영업이익 추이 차트 2025년 36조원",
        ]
        for text in page_texts:
            page = document.new_page(width=595, height=842)
            page.insert_text((50, 90), text, fontsize=12)
        document.save(pdf)
        document.close()
        specs = [
            (2, "영업이익", "36조원", "CFS", 2025),
            (3, "영업이익", "20조원", "OFS", 2025),
            (4, "영업이익", "10조원", "CFS", 2024),
            (5, "매출액", "300조원", "CFS", 2025),
            (2, "당기순이익", "25조원", "CFS", 2025),
            (3, "매출액", "180조원", "OFS", 2025),
        ]
        source = root / "structured.jsonl"
        with source.open("w", encoding="utf-8") as stream:
            for index, (page, metric, value, scope, year) in enumerate(specs, 1):
                row = {
                    "evidence_id": f"E{index:03d}",
                    "document_id": "samsung",
                    "entity": "삼성전자",
                    "metric_raw": metric,
                    "metric_canonical": "operating_profit"
                    if metric == "영업이익"
                    else "revenue",
                    "value_raw": value,
                    "unit_raw": "조원",
                    "period_raw": str(year),
                    "period_year": year,
                    "scope": scope,
                    "statement_type": "IS",
                    "modality": "table",
                    "page": page,
                    "bbox_pdf": {"x0": 40, "y0": 50, "x1": 550, "y1": 150},
                    "source_file": pdf.name,
                    "evidence_text": f"{year}년 {scope} {metric} {value}",
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        db, embedder = root / "test.db", HashingEmbedder()
        summary = build_index([source], [pdf], db, embedder, 3, 4000, False, False)
        assert (
            summary["evidence"]["evidence_count"] == 6
            and summary["pages"]["page_count"] == 6
        )
        results, diagnostics = retrieve(
            db,
            "2025년 연결 기준 삼성전자 영업이익은?",
            embedder,
            MockVision(),
            5,
            (0.6, 0.25, 0.15),
            (0.7, 0.3),
            (0.72, 0.28),
            0.65,
            6,
            3,
            root / "images",
            120,
            True,
            0.6,
        )
        assert len(results) == 5 and all(
            Path(item["metadata"]["page_image_path"]).exists() for item in results
        )
        save_outputs(
            root / "retrieval.json",
            root / "context.json",
            root / "diagnostics.json",
            "test",
            results,
            diagnostics,
        )
        assert (
            len(
                json.loads((root / "retrieval.json").read_text(encoding="utf-8"))[
                    "evidences"
                ]
            )
            == 5
        )
    print("[SELF-TEST SUCCESS] evidence + PDF page + image rerank + fusion + Top-5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAUSE-RAG PDF-grounded multimodal hybrid retrieval"
    )
    parser.add_argument("--mode", choices=("build", "query", "run"), default="run")
    parser.add_argument(
        "--source", action="append", type=Path, help="Script 02 JSONL/JSON; repeatable"
    )
    parser.add_argument(
        "--pdf", action="append", type=Path, default=[], help="original PDF; repeatable"
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument(
        "--allow-missing-pdf", action="store_true", help="parsed-only ablation only"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--question")
    parser.add_argument("--question-file", type=Path)
    parser.add_argument("--retrieval-output", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--context-output", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--diagnostics-output", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
    )
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--max-embedding-chars", type=int, default=7000)
    parser.add_argument("--force-reembed", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--vision-model",
        default=os.environ.get("OPENAI_VISION_MODEL", DEFAULT_OPENAI_MODEL),
    )
    parser.add_argument("--skip-vision-rerank", action="store_true")
    parser.add_argument("--vision-candidate-k", type=int, default=12)
    parser.add_argument("--vision-batch-size", type=int, default=4)
    parser.add_argument(
        "--vision-max-output-tokens",
        type=int,
        default=DEFAULT_VISION_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--vision-detail", choices=("low", "high", "auto"), default="high"
    )
    parser.add_argument("--allow-vision-fallback", action="store_true")
    parser.add_argument("--render-dpi", type=int, default=144)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--evidence-dense-weight", type=float, default=0.60)
    parser.add_argument("--evidence-lexical-weight", type=float, default=0.25)
    parser.add_argument("--evidence-metadata-weight", type=float, default=0.15)
    parser.add_argument("--page-dense-weight", type=float, default=0.70)
    parser.add_argument("--page-lexical-weight", type=float, default=0.30)
    parser.add_argument("--evidence-fusion-weight", type=float, default=0.72)
    parser.add_argument("--page-fusion-weight", type=float, default=0.28)
    parser.add_argument("--vision-weight", type=float, default=0.65)
    parser.add_argument("--max-per-document", type=int, default=3)
    parser.add_argument(
        "--allow-page-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--page-only-threshold", type=float, default=0.72)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--normalization-model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
    )
    parser.add_argument(
        "--normalization-output-dir", type=Path, default=DEFAULT_NORMALIZED
    )
    parser.add_argument("--normalization-workers", type=int, default=5)
    parser.add_argument(
        "--normalization-max-output-tokens",
        type=int,
        default=DEFAULT_NORMALIZATION_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--normalization-no-cache", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def resolve_question(args: argparse.Namespace) -> str:
    if args.question and args.question_file:
        raise ValueError("--question and --question-file are mutually exclusive")
    if args.question_file:
        question = args.question_file.read_text(encoding="utf-8-sig")
    else:
        question = args.question or ""
    question = clean_text(question)
    if not question:
        raise ValueError("query/run mode requires a question")
    return question


def validate_args(args: argparse.Namespace) -> None:
    integers = (
        args.dimensions,
        args.embedding_batch_size,
        args.vision_candidate_k,
        args.vision_batch_size,
        args.vision_max_output_tokens,
        args.render_dpi,
        args.top_k,
        args.max_per_document,
        args.normalization_max_output_tokens,
    )
    if any(value <= 0 for value in integers):
        raise ValueError("count/dimension/DPI arguments must be positive")
    if args.max_embedding_chars < 500:
        raise ValueError("--max-embedding-chars must be >= 500")
    if not 1 <= args.normalization_workers <= 5:
        raise ValueError("--normalization-workers must be in [1,5]")
    if (
        args.vision_max_output_tokens < 512
        or args.normalization_max_output_tokens < 512
    ):
        raise ValueError("structured-output token limits must be >= 512")
    if args.mode == "run" and not args.retrieval_only and args.top_k != 5:
        raise ValueError("run -> Script 04 contract requires --top-k 5")
    totals = {
        "evidence": args.evidence_dense_weight
        + args.evidence_lexical_weight
        + args.evidence_metadata_weight,
        "page": args.page_dense_weight + args.page_lexical_weight,
        "fusion": args.evidence_fusion_weight + args.page_fusion_weight,
    }
    for name, total in totals.items():
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"{name} weights must sum to 1: {total}")
    if not 0 <= args.vision_weight <= 1 or not 0 <= args.page_only_threshold <= 1:
        raise ValueError("vision/page threshold must be in [0,1]")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    validate_args(args)
    sources = [path.resolve() for path in (args.source or [DEFAULT_SOURCE])]
    pdfs = discover_pdfs(args.pdf, args.pdf_dir.resolve())
    question = resolve_question(args) if args.mode in {"query", "run"} else None
    embedder = OpenAIEmbedder(
        args.embedding_model, args.dimensions, args.timeout, args.max_attempts
    )
    print("=" * 92)
    print("CAUSE-RAG | Stage 05 PDF-grounded Multimodal Hybrid Retrieval")
    print("=" * 92)
    print(f"Mode={args.mode} | PDFs={len(pdfs)} | DB={args.db.resolve()}")
    print(
        f"Embedding={args.embedding_model}/{args.dimensions} | Vision={'disabled' if args.skip_vision_rerank else args.vision_model}"
    )
    print("Gold index=disabled")
    build_summary = None
    if args.mode == "build" or (args.mode == "run" and not args.skip_build):
        print("[INDEX] structured evidence + original PDF pages")
        build_summary = build_index(
            sources,
            pdfs,
            args.db.resolve(),
            embedder,
            args.embedding_batch_size,
            args.max_embedding_chars,
            args.force_reembed,
            args.allow_missing_pdf,
        )
        print(
            f"[INDEX SUCCESS] evidence={build_summary['evidence']['evidence_count']} pages={build_summary['pages']['page_count']}"
        )
    if args.mode == "build":
        save_json(args.diagnostics_output.resolve(), {"build": build_summary})
        return
    assert question is not None
    vision = (
        None
        if args.skip_vision_rerank
        else OpenAIVisionReranker(
            model=args.vision_model,
            detail=args.vision_detail,
            batch_size=args.vision_batch_size,
            timeout=args.timeout,
            attempts=args.max_attempts,
            fallback=args.allow_vision_fallback,
            max_output_tokens=args.vision_max_output_tokens,
        )
    )
    results, diagnostics = retrieve(
        args.db.resolve(),
        question,
        embedder,
        vision,
        args.top_k,
        (
            args.evidence_dense_weight,
            args.evidence_lexical_weight,
            args.evidence_metadata_weight,
        ),
        (args.page_dense_weight, args.page_lexical_weight),
        (args.evidence_fusion_weight, args.page_fusion_weight),
        args.vision_weight,
        args.vision_candidate_k,
        args.max_per_document,
        args.image_dir.resolve(),
        args.render_dpi,
        args.allow_page_only,
        args.page_only_threshold,
    )
    diagnostics["build"] = build_summary
    save_outputs(
        args.retrieval_output.resolve(),
        args.context_output.resolve(),
        args.diagnostics_output.resolve(),
        question,
        results,
        diagnostics,
    )
    for item in results:
        m = item["metadata"]
        print(
            f"#{m['retrieval_rank']} {item['evidence_id']} score={m['retrieval_score']:.6f} {m.get('source_file')} p.{m.get('page')} [{m.get('modality')}]"
        )
    print(f"[RETRIEVAL SUCCESS] {args.retrieval_output.resolve()}")
    print(f"[MULTIMODAL CONTEXT] {args.context_output.resolve()}")
    if args.mode == "run" and not args.retrieval_only:
        run_stage04(
            retrieval_input=args.retrieval_output.resolve(),
            model=args.normalization_model,
            output=args.normalization_output_dir.resolve(),
            workers=args.normalization_workers,
            timeout=args.timeout,
            max_output_tokens=args.normalization_max_output_tokens,
            no_cache=args.normalization_no_cache,
        )
        print("[END-TO-END SUCCESS] multimodal Top-5 -> Script 04")


if __name__ == "__main__":
    main()
