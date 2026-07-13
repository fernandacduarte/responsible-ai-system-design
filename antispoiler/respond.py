"""
Intention dispatcher for the interactive demo.

The eval harness (evaluate.py) drives a single generic QA prompt over a question
set. The *product*, by contrast, takes a (selected_text, intention,
reader_position) triple — the reader highlights a span and picks what they want
done with it. This module is the one function that maps that triple onto the
existing retrieval + LLM layer:

    respond(llm, index, selected_text, intention, reader_position) -> str

`respond_detailed(...)` is the same dispatch but returns
`{answer, chapters, entity}` — the chapters touched (for the per-intention eval's
spoiler check) and the recalled entity. It also accepts `reader_position=None`,
which disables the position bound (the unbounded baseline the eval contrasts
against).

The four intentions differ only in (a) whether they retrieve and how, and (b)
the prompt framing. The anti-spoiler guarantee is identical across all of them:
every retrieval is bounded by `reader_position` (the same `_in_bounds` filter
the eval validates), and the prompts forbid drawing on out-of-bounds knowledge.

  | Intention    | Retrieval                          | Spoiler risk |
  |--------------|------------------------------------|--------------|
  | paraphrase   | none — operates on the span itself | low          |
  | define       | small bounded embedding context    | low          |
  | contextualize| bounded embedding retrieval        | medium       |
  | recall       | bounded exhaustive mention gather  | high         |
"""

from __future__ import annotations

import json
import re

from . import config
from .index import EmbeddingIndex
from .llm_client import LLMClient
from .retrieval import (
    extract_entity,
    format_context,
    recall_retrieve,
    retrieve_embedding,
    retrieve_lexical,
)

INTENTIONS = ("define", "paraphrase", "contextualize", "recall")

# Shared framing, mirroring qa.build_system_prompt: tell the model what it IS
# (a bounded companion) rather than handing it prohibitions (Zhou et al. 2023).
_PREAMBLE = (
    'You are a reading companion for "{title}" by {author}. You are reading '
    "alongside the reader and only know what they have read so far — the "
    "passages provided are their knowledge up to this point. Never use outside "
    "knowledge of this document's content, and never reference or hint at details "
    "past the provided passages. General world knowledge (vocabulary, history, "
    "customs, literary conventions) is fine; document-specific content is not."
)


def _preamble(title: str | None = None, author: str | None = None) -> str:
    return _PREAMBLE.format(
        title=(title or config.BOOK_TITLE).strip(),
        author=(author or config.BOOK_AUTHOR).strip() or "Unknown author",
    )


def _parse_define(raw: str) -> tuple[str, list[dict]]:
    """The define model's JSON output -> (overall meaning, clean [{word, definition}]).

    Tolerant of ```fences``` and stray prose; returns ("", []) on failure so `_define`
    can always emit valid JSON downstream.
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[\w]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        obj = {}
    meaning = str(obj.get("meaning", "")).strip()
    out = []
    for d in obj.get("definitions", []) or []:
        if isinstance(d, dict):
            word, definition = str(d.get("word", "")).strip(), str(d.get("definition", "")).strip()
            if word and definition:
                out.append({"word": word, "definition": definition})
    return meaning, out


def _define(
    llm: LLMClient,
    index: EmbeddingIndex,
    span: str,
    pos: int | None,
    title: str | None = None,
    author: str | None = None,
):
    # Find where the term actually appears (lexical substring) so the model sees the
    # real usage — embedding kNN on a single word often misses the literal occurrence.
    # Fall back to embedding similarity when there is no in-bounds literal match.
    ctx = retrieve_lexical(index.chunks, span, pos, max_results=3) or retrieve_embedding(
        index, span, pos, top_k=4
    )
    system = (
        _preamble(title, author) + "\n\n"
        "The reader selected some text and wants help understanding it. Produce TWO things:\n"
        "1. MEANING — a brief plain-language explanation of what the selected text means "
        "as a whole, in its context (1–2 sentences). Use the passages below for context; "
        "never reference anything past them.\n"
        "2. DEFINITIONS — the individual words in the selection a typical reader would most "
        "likely need defined (medium-to-difficult, archaic, technical, or uncommon; skip "
        "common, everyday words), each with a SHORT definition of its meaning AS USED here.\n\n"
        'Return ONLY a JSON object, no prose, no markdown fences:\n'
        '{"meaning": "<overall meaning of the selection>", '
        '"definitions": [{"word": "<the word as it appears in the text>", "definition": "<short definition as used here>"}]}\n'
        'If nothing is hard enough to need defining, use an empty list. Never refuse.'
    )
    user = (
        f'SELECTED TEXT:\n"""{span}"""\n\n'
        f"SURROUNDING CONTEXT (the reader's passages so far):\n{format_context(ctx)}"
        if ctx
        else f'SELECTED TEXT:\n"""{span}"""\n\n(No surrounding passages retrieved.)'
    )
    # Re-serialize clean JSON so downstream (validator, frontend) can plain-parse it.
    # Generous budget: the combined meaning + word list is larger, and reasoning models
    # (cheap ones via OpenRouter in dev) need headroom before emitting the JSON.
    meaning, definitions = _parse_define(llm.complete(system, user, max_tokens=2000))
    return json.dumps({"meaning": meaning, "definitions": definitions}), ctx, None


def _paraphrase(
    llm: LLMClient,
    span: str,
    title: str | None = None,
    author: str | None = None,
):
    system = (
        _preamble(title, author) + "\n\n"
        "The reader selected a passage and wants it restated in simpler, clearer "
        "English. Paraphrase ONLY the selected passage — keep the same meaning and "
        "tense, do not add information, do not explain what happens next, do not "
        "foreshadow. Just say the same thing in plainer words."
    )
    user = f'SELECTED PASSAGE:\n"""{span}"""\n\nParaphrase it in plain English.'
    # No retrieval: paraphrase grounds on the selected span itself (D15).
    return llm.complete(system, user), [], None


def _contextualize(
    llm: LLMClient,
    index: EmbeddingIndex,
    span: str,
    pos: int | None,
    title: str | None = None,
    author: str | None = None,
):
    ctx = retrieve_embedding(index, span, pos, top_k=config.TOP_K)
    system = (
        _preamble(title, author) + "\n\n"
        "The reader selected a passage and wants historical, cultural, or thematic "
        "context for it. Lead with general world knowledge (the period, customs, "
        "references). For anything specific to this document's story, people, or "
        "claims, use ONLY the passages below — do not reference how a theme or situation "
        "develops later. Keep it short and concrete."
    )
    user = (
        f'SELECTED PASSAGE:\n"""{span}"""\n\n'
        f"PASSAGES THE READER HAS READ:\n{format_context(ctx)}"
        if ctx
        else f'SELECTED PASSAGE:\n"""{span}"""\n\n(No in-bounds passages retrieved.)'
    )
    return llm.complete(system, user), ctx, None


def _recall(
    llm: LLMClient,
    index: EmbeddingIndex,
    span: str,
    pos: int | None,
    title: str | None = None,
    author: str | None = None,
):
    # The reader highlighted a name/subject; recover its earlier mentions.
    # A short selection IS the entity — use it verbatim. extract_entity is for
    # pulling a name out of a longer phrase, and (being question-oriented) it can
    # return a whole chatty sentence, which then matches nothing. So only invoke
    # it for multi-word spans, and only trust its result if that string actually
    # occurs in the text; otherwise fall back to the raw selection.
    entity = span.strip()
    if len(entity.split()) > 4:
        extracted = extract_entity(llm, span)
        if extracted and any(extracted.lower() in c.text.lower() for c in index.chunks):
            entity = extracted
    hits = recall_retrieve(index.chunks, entity, reader_pos=pos)
    if not hits:
        doc_title = (title or config.BOOK_TITLE).strip()
        upto = "the whole document" if pos is None else f"{doc_title} position {pos}"
        msg = f'Nothing about "{entity}" has come up yet in what you\'ve read (up to {upto}).'
        return msg, [], entity
    system = (
        _preamble(title, author) + "\n\n"
        'The reader wants to remember what they have already encountered about a '
        'subject earlier in the document ("who was this again?"). Using ONLY the '
        "passages below — every earlier mention within what they have read — "
        "summarise what has been shown about it, in chronological order (earliest "
        "to latest). Cite chunk_ids. Do not add anything beyond these passages and "
        "do not speculate about what comes next."
    )
    user = (
        f'SUBJECT TO RECALL: "{entity}"\n\n'
        f"EARLIER MENTIONS (chronological, within what the reader has read):\n"
        f"{format_context(hits)}\n\n"
        f'Summarise what has been shown about "{entity}" so far.'
    )
    return llm.complete(system, user), hits, entity


def _dispatch(
    llm: LLMClient,
    index: EmbeddingIndex,
    span: str,
    intention: str,
    pos: int | None,
    title: str | None = None,
    author: str | None = None,
):
    """Return (answer, chunks, entity) for a non-empty span. Raises on unknown.

    `chunks` is the list[Chunk] used as grounding (empty for paraphrase, which
    operates on the span itself). Callers that only need chapter indices derive
    them with `[c.chapter_index for c in chunks]`.
    """
    if intention == "paraphrase":
        return _paraphrase(llm, span, title, author)
    if intention == "define":
        return _define(llm, index, span, pos, title, author)
    if intention == "contextualize":
        return _contextualize(llm, index, span, pos, title, author)
    if intention == "recall":
        return _recall(llm, index, span, pos, title, author)
    raise ValueError(f"Unknown intention {intention!r}; expected one of {INTENTIONS}")


def respond(
    llm: LLMClient,
    index: EmbeddingIndex,
    selected_text: str,
    intention: str,
    reader_position: int | None = config.READER_POSITION,
    title: str | None = None,
    author: str | None = None,
) -> str:
    """Route a (selection, intention, position) triple to a bounded response."""
    intention = (intention or "").lower().strip()
    span = (selected_text or "").strip()
    if not span:
        return "Select some text first, then choose what you'd like."
    return _dispatch(llm, index, span, intention, reader_position, title, author)[0]


def respond_detailed(
    llm: LLMClient,
    index: EmbeddingIndex,
    selected_text: str,
    intention: str,
    reader_position: int | None = config.READER_POSITION,
    title: str | None = None,
    author: str | None = None,
) -> dict:
    """Same dispatch as respond(), but expose {answer, chapters, entity}.

    `reader_position=None` disables the position bound (the unbounded baseline
    the per-intention eval contrasts against). `chapters` is the set of chapter
    indices the answer was allowed to draw on — the eval checks none exceed the
    bound, and that the unbounded arm reaches past it.
    """
    intention = (intention or "").lower().strip()
    span = (selected_text or "").strip()
    if not span:
        return {"answer": "Select some text first, then choose what you'd like.",
                "chapters": [], "entity": None}
    answer, chunks, entity = _dispatch(llm, index, span, intention, reader_position, title, author)
    return {"answer": answer, "chapters": [c.chapter_index for c in chunks], "entity": entity}


def respond_with_evidence(
    llm: LLMClient,
    index: EmbeddingIndex,
    selected_text: str,
    intention: str,
    reader_position: int | None = config.READER_POSITION,
    title: str | None = None,
    author: str | None = None,
) -> dict:
    """Same dispatch as respond(), but also return the retrieved grounding chunks.

    This is what the validator (LLM 3) needs: the judge must ground its verdict on
    *exactly the same* position-bounded passages the generator saw (D15), so we
    surface them here instead of re-retrieving (no drift, no double cost). Returns
    {answer, chunks, chapters, entity}; `chunks` is a list[Chunk] (empty for
    paraphrase, which grounds on the selected span itself).
    """
    intention = (intention or "").lower().strip()
    span = (selected_text or "").strip()
    if not span:
        return {"answer": "Select some text first, then choose what you'd like.",
                "chunks": [], "chapters": [], "entity": None}
    answer, chunks, entity = _dispatch(
        llm, index, span, intention, reader_position, title, author
    )
    return {"answer": answer, "chunks": chunks,
            "chapters": [c.chapter_index for c in chunks], "entity": entity}
