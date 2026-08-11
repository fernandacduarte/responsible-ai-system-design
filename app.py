"""
Interactive demo for the anti-spoiler reading companion.

A thin web layer over the `antispoiler` package — the counterpart to the
notebook. Where the notebook is the *evaluation* harness (batch question set,
LLM judge, metrics), this is the *demonstration*: it models the real product
interaction, a (selected_text, intention, reader_position) triple.

The reader sees the book rendered only up to their position (the slider), so
they can only select text they've "read"; selecting a span and clicking an
intention calls `antispoiler.respond.respond`, which keeps every retrieval
bounded by that same position. Moving the slider makes spoilers appear/vanish —
the anti-spoiler mechanism, made visible.

Run (native arm64 env; see antispoiler/README.md):
    conda run -n antispoiler-arm uvicorn app:app --reload --port 8000
then open http://127.0.0.1:8000

First request is slow: it downloads the embedding model and indexes the book
once, at startup.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from antispoiler import config
from antispoiler.book import Chunk, fetch_and_chunk
from antispoiler.discuss import (
    MAX_HISTORY_MESSAGES,
    MAX_MESSAGE_CHARS,
    MAX_ORIGINAL_ANSWER_CHARS,
    discuss_response,
)
from antispoiler.index import build_index
from antispoiler.llm_client import LLMClient, make_validator
from antispoiler.respond import INTENTIONS, respond_with_evidence

from validator import CONF_THRESHOLD, dictionary
from validator.service import VALIDATED_FEATURES, validate_response

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - dependency surfaced by the upload endpoint
    PdfReader = None  # type: ignore[assignment]

app = FastAPI(title="Anti-spoiler reading companion (demo)")

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_PDF_BYTES = 120 * 1024 * 1024
MAX_PDF_PAGES = 300


@dataclass
class ActiveDocument:
    document_id: str
    title: str
    author: str
    source: str
    position_unit: str
    default_position: int
    max_position: int
    chunks: list[Chunk]
    index: Any
    filename: str | None = None
    pdf_bytes: bytes | None = None


def _safe_filename(name: str | None) -> str:
    clean = os.path.basename((name or "").replace("\\", "/")).strip()
    clean = re.sub(r'[\r\n"]+', "_", clean)
    return clean[:160] or "uploaded.pdf"


def _json_error(message: str, status_code: int = 400):
    return JSONResponse({"error": message}, status_code=status_code)


def _split_long_paragraph(text: str, max_chars: int = config.MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces = re.split(r"(?<=[.!?])\s+", text)
    if len(pieces) == 1:
        pieces = text.split()

    out: list[str] = []
    buf = ""
    sep = " " if pieces and "\n" not in pieces[0] else "\n"
    for piece in pieces:
        if not piece:
            continue
        candidate = piece if not buf else f"{buf}{sep}{piece}"
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            buf = piece
    if buf:
        out.append(buf)
    return out


def _normalise_pdf_text(text: str) -> str:
    text = (text or "").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_pdf_pages(page_texts: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_index, raw_text in enumerate(page_texts, start=1):
        text = _normalise_pdf_text(raw_text)
        if not text:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        expanded: list[str] = []
        for paragraph in paragraphs or [text]:
            compact = re.sub(r"\s+", " ", paragraph).strip()
            expanded.extend(_split_long_paragraph(compact))

        merged: list[str] = []
        buf = ""
        for paragraph in expanded:
            if not buf:
                buf = paragraph
            elif len(buf) < config.MIN_CHARS:
                buf = f"{buf}\n\n{paragraph}"
            else:
                merged.append(buf)
                buf = paragraph
        if buf:
            merged.append(buf)

        for paragraph_index, paragraph in enumerate(merged, start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"p{page_index:03d}_c{paragraph_index:02d}",
                    chapter_index=page_index,
                    chapter_label=f"Page {page_index}",
                    paragraph_index=paragraph_index,
                    text=paragraph,
                )
            )
    return chunks


def _extract_pdf_document(pdf_bytes: bytes, filename: str) -> ActiveDocument:
    if PdfReader is None:
        raise RuntimeError("PDF support is not installed. Run pip install -r requirements.txt.")
    if not pdf_bytes:
        raise ValueError("No PDF data was received.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("The PDF is too large for this demo. Use a file up to 25 MB.")

    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")
    page_count = len(reader.pages)
    if page_count < 1:
        raise ValueError("The PDF has no pages.")
    if page_count > MAX_PDF_PAGES:
        raise ValueError(f"The PDF has {page_count} pages; this demo supports up to {MAX_PDF_PAGES}.")

    page_texts = [(page.extract_text() or "") for page in reader.pages]
    if not "".join(page_texts).strip():
        raise ValueError("No selectable text was found. Scanned/OCR-only PDFs are not supported yet.")

    chunks = _chunk_pdf_pages(page_texts)
    if not chunks:
        raise ValueError("No usable text chunks could be extracted from this PDF.")

    metadata = reader.metadata
    title = str(getattr(metadata, "title", "") or "").strip()
    author = str(getattr(metadata, "author", "") or "").strip()
    if not title:
        title = os.path.splitext(filename)[0] or "Uploaded PDF"

    index = build_index(chunks)
    return ActiveDocument(
        document_id=f"pdf-{uuid.uuid4().hex[:12]}",
        title=title,
        author=author,
        source="pdf",
        position_unit="page",
        default_position=max(1, min(config.READER_POSITION, page_count)),
        max_position=page_count,
        chunks=chunks,
        index=index,
        filename=filename,
        pdf_bytes=pdf_bytes,
    )


def _document_config(doc: ActiveDocument) -> dict:
    return {
        "document_id": doc.document_id,
        "title": doc.title,
        "author": doc.author,
        "source": doc.source,
        "filename": doc.filename,
        "position_unit": doc.position_unit,
        "max_position": doc.max_position,
        "max_chapter": doc.max_position,  # compatibility for older UI code
        "default_position": doc.default_position,
        "intentions": INTENTIONS,
        "validated_features": sorted(VALIDATED_FEATURES),
        "conf_threshold": CONF_THRESHOLD,
    }


def _group_document_text(doc: ActiveDocument, upto: int) -> dict:
    upto = max(1, min(int(upto), doc.max_position))
    sections: list[dict] = []
    cur: dict | None = None
    for c in doc.chunks:
        if c.chapter_index > upto:
            break
        if cur is None or cur["index"] != c.chapter_index:
            cur = {"index": c.chapter_index, "label": c.chapter_label, "paragraphs": []}
            sections.append(cur)
        cur["paragraphs"].append(c.text)
    return {
        "document_id": doc.document_id,
        "upto": upto,
        "max_position": doc.max_position,
        "max_chapter": doc.max_position,
        "position_unit": doc.position_unit,
        "sections": sections,
        "chapters": sections,  # compatibility for older UI code
    }

# Mode banner (prod = Anthropic Haiku+Sonnet; dev = one cheap model via OpenRouter).
print(f"Mode: {config.APP_MODE.upper()}  |  backend={config.BACKEND}  "
      f"generator={config.ANSWERER_MODEL}  validator={config.VALIDATOR_MODEL}")
if config.APP_MODE == "dev":
    print("  ⚠  dev mode: one cheap model via OpenRouter — for iteration only, NOT the "
          "characterized setup (validator==generator breaks D13).")

# Built once at import time. Heavy (model download + embedding) but one-off.
print("Loading book and building index (first run downloads the embedding model)…")
BUILTIN_CHUNKS = fetch_and_chunk()
BUILTIN_INDEX = build_index(BUILTIN_CHUNKS)
BUILTIN_DOC = ActiveDocument(
    document_id="builtin-pride-prejudice",
    title=config.BOOK_TITLE,
    author=config.BOOK_AUTHOR,
    source="book",
    position_unit="chapter",
    default_position=config.READER_POSITION,
    max_position=max(c.chapter_index for c in BUILTIN_CHUNKS),
    chunks=BUILTIN_CHUNKS,
    index=BUILTIN_INDEX,
)
ACTIVE_DOC = BUILTIN_DOC
LLM = LLMClient(model=config.ANSWERER_MODEL)          # generator
VALIDATOR = make_validator()                          # validator LLM 3 (config.VALIDATOR_MODEL); validator != generator (D13)
print(f"Ready: {len(BUILTIN_CHUNKS)} chunks across {BUILTIN_DOC.max_position} chapters.")
print(f"Validator: model={VALIDATOR.model}  tau={CONF_THRESHOLD}  features={sorted(VALIDATED_FEATURES)}")
_dict_ok, _dict_detail = dictionary.available()  # warms the WordNet corpus; surfaces setup issues now
print(f"Dictionary: {'OK' if _dict_ok else 'UNAVAILABLE'} — {_dict_detail}")


class RespondRequest(BaseModel):
    selected_text: str
    intention: str
    reader_position: int
    document_id: str | None = None


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class FollowUpRequest(BaseModel):
    selected_text: str
    intention: str
    original_answer: str
    validation: dict[str, Any] | None = None
    messages: list[ConversationMessage] = Field(default_factory=list)
    reader_position: int
    document_id: str | None = None


@app.get("/")
def home():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/config")
def app_config():
    return _document_config(ACTIVE_DOC)


@app.get("/document/text")
def document_text(upto: int = Query(config.READER_POSITION)):
    """Readable text for the active document, grouped by chapter/page."""
    return _group_document_text(ACTIVE_DOC, upto)


@app.get("/document/pdf")
def active_pdf():
    doc = ACTIVE_DOC
    if doc.source != "pdf" or doc.pdf_bytes is None:
        return _json_error("No uploaded PDF is active.", 404)
    filename = _safe_filename(doc.filename)
    return Response(
        doc.pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.post("/document/pdf")
async def upload_pdf(request: Request, filename: str | None = Query(None)):
    """Load a local PDF as the active in-memory reading document."""
    global ACTIVE_DOC

    clean_filename = _safe_filename(filename)
    pdf_bytes = await request.body()
    try:
        ACTIVE_DOC = _extract_pdf_document(pdf_bytes, clean_filename)
    except ValueError as e:
        return _json_error(str(e), 400)
    except RuntimeError as e:
        return _json_error(str(e), 500)
    except Exception as e:
        print(f"[pdf] failed: {type(e).__name__}: {e}")
        return _json_error(f"Could not read this PDF ({type(e).__name__}).", 400)

    print(
        f"Loaded PDF: {ACTIVE_DOC.filename} — "
        f"{len(ACTIVE_DOC.chunks)} chunks across {ACTIVE_DOC.max_position} pages."
    )
    return _document_config(ACTIVE_DOC)


@app.post("/document/reset")
def reset_document():
    global ACTIVE_DOC
    ACTIVE_DOC = BUILTIN_DOC
    return _document_config(ACTIVE_DOC)


@app.get("/book")
def book(upto: int = Query(config.READER_POSITION)):
    """Chapters 1..upto, grouped — the text the reader is allowed to select from."""
    return _group_document_text(ACTIVE_DOC, upto)


@app.post("/respond")
def do_respond(req: RespondRequest):
    doc = ACTIVE_DOC
    if req.document_id and req.document_id != doc.document_id:
        return _json_error(
            "The active document changed. Select the text again in the current document.",
            409,
        )
    if req.intention not in INTENTIONS:
        return JSONResponse(
            {"error": f"unknown intention {req.intention!r}; expected {INTENTIONS}"},
            status_code=400,
        )
    if not req.selected_text.strip():
        return JSONResponse({"error": "no text selected"}, status_code=400)
    pos = max(1, min(int(req.reader_position), doc.max_position))

    # Generate (LLM 1/1.2), keeping the retrieved grounding chunks for the validator.
    # A generation failure must not 500 the request — degrade to an honest message
    # (e.g. a cheap dev model returning an empty response).
    try:
        out = respond_with_evidence(
            LLM,
            doc.index,
            req.selected_text,
            req.intention,
            pos,
            title=doc.title,
            author=doc.author,
        )
    except Exception as e:
        print(f"[generator] failed: {type(e).__name__}: {e}")
        return {
            "answer": f"The assistant couldn't generate a response this time ({type(e).__name__}).",
            "intention": req.intention,
            "reader_position": pos,
            "document_id": doc.document_id,
            "validation": {
                "enabled": False,
                "reason": "generation_error",
                "note": f"Generation failed ({type(e).__name__}); there's nothing to validate.",
            },
        }
    answer = out["answer"]

    # Validate (LLM 3) — blocking; the frontend shows a spinner meanwhile.
    # selected_text is the grounding source for paraphrase (D15).
    validation = validate_response(
        VALIDATOR, req.intention, answer, out["chunks"], req.selected_text
    )

    return {
        "answer": answer,
        "intention": req.intention,
        "reader_position": pos,
        "document_id": doc.document_id,
        "validation": validation,
    }


@app.post("/follow-up")
def do_follow_up(req: FollowUpRequest):
    """Continue a discussion about one response, under its original spoiler bound."""
    doc = ACTIVE_DOC
    if req.document_id and req.document_id != doc.document_id:
        return _json_error(
            "The active document changed. Start a new discussion from the current response.",
            409,
        )
    if req.intention not in INTENTIONS:
        return _json_error(
            f"Unknown intention {req.intention!r}; expected one of {INTENTIONS}.", 400
        )
    if not req.selected_text.strip():
        return _json_error("The original selected text is missing.", 400)
    if not req.original_answer.strip():
        return _json_error("The original response is missing.", 400)
    if len(req.original_answer) > MAX_ORIGINAL_ANSWER_CHARS:
        return _json_error("The original response is too long to discuss.", 400)
    if not req.messages or req.messages[-1].role != "user":
        return _json_error("Ask a follow-up question first.", 400)
    if len(req.messages) > MAX_HISTORY_MESSAGES:
        return _json_error("This discussion is too long. Start a new discussion.", 400)
    if any(not m.content.strip() or len(m.content) > MAX_MESSAGE_CHARS for m in req.messages):
        return _json_error(
            f"Each conversation message must contain 1-{MAX_MESSAGE_CHARS} characters.",
            400,
        )

    pos = max(1, min(int(req.reader_position), doc.max_position))
    try:
        answer = discuss_response(
            LLM,
            doc.index,
            selected_text=req.selected_text.strip(),
            intention=req.intention,
            original_answer=req.original_answer,
            validation=req.validation,
            messages=[
                m.model_dump() if hasattr(m, "model_dump") else m.dict()
                for m in req.messages
            ],
            reader_position=pos,
            title=doc.title,
            author=doc.author,
        )
    except ValueError as e:
        return _json_error(str(e), 400)
    except Exception as e:
        print(f"[follow-up] failed: {type(e).__name__}: {e}")
        return _json_error(
            f"The assistant couldn't answer this follow-up ({type(e).__name__}).", 502
        )

    return {
        "answer": answer,
        "reader_position": pos,
        "document_id": doc.document_id,
    }
