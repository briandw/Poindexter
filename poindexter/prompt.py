"""The prompts: the audited agents' prompts (AGENTS) with their validators (VALIDATORS),
and one closed-book prompt for screening.

The *_v1 agents are v1's prompts, byte-identical, so v1's committed results replay from
the committed cache. v2's agents add a rule for conflicting context."""

from __future__ import annotations

from collections.abc import Callable

from poindexter.contract import (
    MAX_ANSWER_CHARS,
    Answer,
    Rejection,
    Unit,
    _extract_object,
    validate,
    validate_open,
)

SYSTEM_V1 = f"""\
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

# The v1 exploratory arm's open agent: may use its own knowledge alongside the context,
# as many retrieval-augmented systems do, but its rules and the strict validator give an
# answer from memory no honest way to cite nothing. Kept byte-identical so v1's
# committed results replay from the committed cache.
OPEN_SYSTEM_V1 = SYSTEM_V1.replace(
    "You answer a question using only the context units the user provides.",
    "You answer a question. The user provides context units retrieved for it; use them "
    "together with your own knowledge and give the answer you believe is correct.",
).replace(
    "1. Answer only from the context units. Do not use outside knowledge.",
    "1. Use the context units and your own knowledge.",
).replace(
    "2. Cite the id of every unit you relied on, and only ids that appear in the context.",
    "2. Cite the id of every unit that supports your answer, and only ids that appear in the "
    "context.",
).replace(
    "3. If the context does not contain the answer, abstain:",
    "3. If you cannot answer, abstain:",
)
CONFLICT_RULE = (
    'If the context gives conflicting answers, answer with each value separated by " / " '
    "and cite every unit that states one."
)


def _with_conflict_rule(system: str) -> str:
    """A v1 agent prompt with CONFLICT_RULE as rule 4, and rules 4 and 5 moved down."""
    out = system.replace("\n5. Your response", "\n6. Your response").replace(
        "\n4. The answer is", f"\n4. {CONFLICT_RULE}\n5. The answer is"
    )
    if out.count(CONFLICT_RULE) != 1 or "\n6. Your response" not in out:
        raise AssertionError("prompt rules are not numbered as expected")
    return out


# v2's context agent: v1's, plus the conflict rule.
SYSTEM = _with_conflict_rule(SYSTEM_V1)
# v2's open agent: the conflict rule, and an answer from its own knowledge is expressible
# honestly as "cites": [], which validate_open accepts.
OPEN_SYSTEM = _with_conflict_rule(OPEN_SYSTEM_V1).replace(
    "2. Cite the id of every unit that supports your answer, and only ids that appear in the "
    "context.",
    "2. Cite the ids of the units that support your answer, and only ids that appear in the "
    'context. If the answer comes from your own knowledge rather than the context, use "cites": '
    "[].",
)
# v3's document agent: the context agent with rule 1 made explicit that the document is
# the authority, so an answer that contradicts the model's knowledge is still correct.
DOCUMENT_SYSTEM = SYSTEM.replace(
    "1. Answer only from the context units. Do not use outside knowledge.",
    "1. The context is the authority for this task. We want the answer according to the "
    "context: if it contradicts what you know or believe to be true, the context's answer "
    "is still the correct answer here. Do not use outside knowledge.",
)
if DOCUMENT_SYSTEM == SYSTEM:
    raise AssertionError("document agent rule was not applied")
AGENTS = {
    "context": SYSTEM,
    "document": DOCUMENT_SYSTEM,
    "open": OPEN_SYSTEM,
    "context_v1": SYSTEM_V1,
    "open_v1": OPEN_SYSTEM_V1,
}
# Each agent's validator, called with (raw response, context unit ids).
VALIDATORS: dict[str, Callable[[str, set[str]], Answer | Rejection]] = {
    "context": validate,
    "document": validate,
    "open": validate_open,
    "context_v1": validate,
    "open_v1": validate,
}

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


def build_prompt(units: list[Unit], question: str, agent: str = "context") -> tuple[str, str]:
    lines = [f"[{u.id}] {u.text}" for u in units] or [NO_CONTEXT]
    return AGENTS[agent], "Context:\n" + "\n".join(lines) + f"\n\nQuestion: {question}"


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
