import logging
from config import AI_ENABLED, ANTHROPIC_KEY, CLAUDE_VERSION, SCORE_CUTOFF

log = logging.getLogger(__name__)

INSTRUCTIONS = """You answer questions strictly from the provided context.

Rules:
- Read all context carefully before answering.
- Give a direct answer in 1-2 sentences.
- If the answer is not in the context, respond: "This information is not available in the provided documents."
- Do not guess or add outside knowledge."""


def _rate_confidence(hits):
    if not hits:
        return "low"
    avg = sum(h["distance"] for h in hits) / len(hits)
    if avg < SCORE_CUTOFF * 0.5:
        return "high"
    elif avg < SCORE_CUTOFF:
        return "medium"
    return "low"


def _get_sources(hits):
    seen, sources = set(), []
    for h in hits:
        if h["source"] not in seen:
            seen.add(h["source"])
            sources.append(h["source"])
    return sources


def _build_prompt(question, hits):
    sections = [f"[Doc {i}: {h['source']}]\n{h['text']}" for i, h in enumerate(hits, 1)]
    return "Context:\n" + "\n\n".join(sections) + f"\n\nQuestion: {question}"


def _ask_claude(question, hits):
    try:
        import anthropic
        ai = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        resp = ai.messages.create(
            model=CLAUDE_VERSION,
            max_tokens=150,
            system=INSTRUCTIONS,
            messages=[{"role": "user", "content": _build_prompt(question, hits)}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("Answer:"):
            raw = raw.split("Answer:", 1)[1].strip()
        if "Sources:" in raw:
            raw = raw.split("Sources:")[0].strip()
        return raw
    except Exception as err:
        log.error("Claude error: %s", err)
        return _extract_answer(question, hits)


def _extract_answer(question, hits):
    combined = " ".join(h["text"] for h in hits)
    clean    = " ".join(combined.replace("\n", " ").replace("#", "").split())
    sentences = [s.strip() for s in clean.split(".") if len(s.strip()) > 20]
    keywords  = {w.lower() for w in question.split() if len(w) > 2}

    ranked = sorted(
        [(len(keywords & set(s.lower().split())), s) for s in sentences],
        reverse=True
    )
    best = [s for _, s in ranked[:2] if _ > 0]

    if best:
        result = ". ".join(best)
        return result if result.endswith(".") else result + "."
    return "This information is not available in the provided documents."


def generate_answer(question, hits):
    if not hits:
        return {"answer": "This information is not available in the provided documents.", "citations": [], "confidence": "low"}

    answer = _ask_claude(question, hits) if AI_ENABLED else _extract_answer(question, hits)
    return {"answer": answer, "citations": _get_sources(hits), "confidence": _rate_confidence(hits)}


def generate_answer_stream(question, hits):
    if not hits:
        yield "This information is not available in the provided documents."
        return

    if AI_ENABLED:
        try:
            import anthropic
            ai = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            with ai.messages.stream(
                model=CLAUDE_VERSION,
                max_tokens=150,
                system=INSTRUCTIONS,
                messages=[{"role": "user", "content": _build_prompt(question, hits)}],
            ) as stream:
                for token in stream.text_stream:
                    yield token
        except Exception as err:
            log.error("Claude stream error: %s", err)
            yield _extract_answer(question, hits)
    else:
        words = _extract_answer(question, hits).split()
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
