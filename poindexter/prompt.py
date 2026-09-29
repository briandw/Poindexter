"""The prompts: one for answering from context, one closed-book prompt for screening."""

from __future__ import annotations

from poindexter.contract import MAX_ANSWER_CHARS, Answer, Rejection, Unit, _extract_object

SYSTEM = f"""\
You answer a question using only the context units the user provides. Each unit is \
one line that starts with its id in square brackets, like [u1].

Respond with exactly one JSON object and nothing else:
{{"answer": "<string or null>", "cites": ["<unit id>"], "abstain": false}}

Rules:
1. Answer only from the context units. Do not use outside knowledge.
2. Cite the id of every unit you relied on, and only ids that appear in the context.
3. If the context does not contain the answer, abstain: \
{{"answer": null, "cites": [], "abstain": true}}
4. The answer is the shortest span that answers the question: a few words, not a \
sentence (a name, number, date, or short phrase), at most {MAX_ANSWER_CHARS} characters.
5. Your response is the JSON object alone, with nothing before or after it and no \
explanation, including when you abstain."""

CLOSED_BOOK_SYSTEM = f"""\
You answer a question from your own knowledge. No context is provided.

Respond with exactly one JSON object and nothing else:
{{"answer": "<string or null>", "cites": [], "abstain": false}}

Rules:
1. Answer from what you know.
2. cites is always the empty list [].
3. If you do not know the answer, abstain: {{"answer": null, "cites": [], "abstain": true}}
4. The answer is the shortest span that answers the question: a few words, not a \
sentence (a name, number, date, or short phrase), at most {MAX_ANSWER_CHARS} characters.
5. Your response is the JSON object alone, with nothing before or after it and no \
explanation, including when you abstain."""

NO_CONTEXT = "(no context units)"


def build_prompt(units: list[Unit], question: str) -> tuple[str, str]:
    lines = [f"[{u.id}] {u.text}" for u in units] or [NO_CONTEXT]
    return SYSTEM, "Context:\n" + "\n".join(lines) + f"\n\nQuestion: {question}"


def build_closed_book_prompt(question: str) -> tuple[str, str]:
    return CLOSED_BOOK_SYSTEM, f"Question: {question}"


def retry_prompt(user: str, rejection: Rejection) -> str:
    return (
        f"{user}\n\nYour previous response was rejected: {rejection.code}: "
        f"{rejection.message}. Respond with only the JSON object."
    )


def validate_closed_book(raw: str) -> Answer | Rejection:
    """Validator for the closed-book prompt: the same schema, but cites must be []."""
    obj = _extract_object(raw)
    if isinstance(obj, Rejection):
        return obj
    if set(obj) != {"answer", "cites", "abstain"}:
        got = sorted(obj)
        return Rejection("BAD_KEYS", f"keys must be exactly answer, cites, abstain; got {got}")
    answer, cites, abstain = obj["answer"], obj["cites"], obj["abstain"]
    if not isinstance(abstain, bool):
        return Rejection("BAD_ABSTAIN", "abstain must be true or false")
    if cites != []:
        return Rejection("BAD_CITES", "cites must be the empty list []")
    if answer is not None and not isinstance(answer, str):
        return Rejection("BAD_ANSWER", "answer must be a string or null")
    if abstain:
        if answer is not None:
            return Rejection("BAD_ABSTAIN", "abstain is true, so answer must be null")
        return Answer(text=None, cites=[], abstain=True)
    if answer is None or not answer.strip():
        return Rejection("BAD_ANSWER", "abstain is false, so answer must be a non-empty string")
    if len(answer) > MAX_ANSWER_CHARS:
        return Rejection("BAD_ANSWER", f"answer is over {MAX_ANSWER_CHARS} characters")
    return Answer(text=answer, cites=[], abstain=False)
