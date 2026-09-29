"""Probe contexts. Pure functions from (units, cites, seed) to a unit list.

Unit ids stay attached to their text. Order is preserved everywhere except `shuffle`.
Probe C (`counterfactual`) edits the answer inside the cited units instead; its
replacement comes from `choose_replacement`.
"""

from __future__ import annotations

import random
import re

from poindexter.contract import Answer, Record, Unit, canonical, normalize
from poindexter.datasets import _NUMBER_WORDS, number_mentions, swapped_integer


def original(
    units: list[Unit], cites: list[str], seed: int, donor: list[Unit] | None = None
) -> list[Unit]:
    return list(units)


def no_context(
    units: list[Unit], cites: list[str], seed: int, donor: list[Unit] | None = None
) -> list[Unit]:
    """N's context. The runner sends N with the closed-book prompt, which has no units."""
    return []


def remove_cited(
    units: list[Unit], cites: list[str], seed: int, donor: list[Unit] | None = None
) -> list[Unit]:
    cited = set(cites)
    return [u for u in units if u.id not in cited]


def replace_cited(
    units: list[Unit], cites: list[str], seed: int, donor: list[Unit] | None = None
) -> list[Unit]:
    if not donor:
        raise ValueError("replace_cited needs donor units from a different record")
    rng = random.Random(seed)
    cited = set(cites)
    return [Unit(u.id, rng.choice(donor).text) if u.id in cited else u for u in units]


def cited_only(
    units: list[Unit], cites: list[str], seed: int, donor: list[Unit] | None = None
) -> list[Unit]:
    cited = set(cites)
    return [u for u in units if u.id in cited]


def shuffle(
    units: list[Unit], cites: list[str], seed: int, donor: list[Unit] | None = None
) -> list[Unit]:
    out = list(units)
    random.Random(seed).shuffle(out)
    return out


def leave_one_out(units: list[Unit], i: int) -> list[Unit]:
    if not 0 <= i < len(units):
        raise IndexError(f"leave_one_out index {i} out of range for {len(units)} units")
    return units[:i] + units[i + 1 :]


# --- probe C: counterfactual ---------------------------------------------------


def _standalone(text: str) -> re.Pattern[str]:
    """`text` as a standalone occurrence, ignoring case: not inside a word, and not part
    of a decimal or a thousands group ("850" is not in "1,850" or "18.50")."""
    return re.compile(rf"(?<!\w)(?<!\d[.,]){re.escape(text)}(?!\w)(?![.,]\d)", re.IGNORECASE)


def occurrences(units: list[Unit], cites: list[str], text: str) -> int:
    """Standalone occurrences of `text` across the cited units."""
    if not text.strip():
        raise ValueError("occurrences of empty text")
    pattern, cited = _standalone(text), set(cites)
    return sum(len(pattern.findall(u.text)) for u in units if u.id in cited)


def counterfactual(
    units: list[Unit], cites: list[str], answer_text: str, replacement: str
) -> list[Unit]:
    """Probe C's context: the one standalone occurrence of `answer_text` in the cited
    units replaced by `replacement`. Ids, order, and every other unit are unchanged.

    Raises unless `answer_text` occurs exactly once across the cited units.
    """
    if not replacement.strip():
        raise ValueError("counterfactual replacement is empty")
    uid, start, end = locate(units, cites, answer_text)
    return [
        Unit(u.id, u.text[:start] + replacement + u.text[end:]) if u.id == uid else u
        for u in units
    ]


def locate(units: list[Unit], cites: list[str], text: str) -> tuple[str, int, int]:
    """(unit id, start, end) of the one standalone occurrence of `text` in the cited
    units. Raises unless there is exactly one."""
    n = occurrences(units, cites, text)
    if n != 1:
        raise ValueError(f"{text!r} occurs {n} times in the cited units, not once")
    pattern, cited = _standalone(text), set(cites)
    for u in units:
        if u.id in cited and (m := pattern.search(u.text)):
            return u.id, m.start(), m.end()
    raise AssertionError("unreachable: occurrences found one")


def _integer(text: str) -> int | None:
    """The integer `text` canonicalizes to ("in 1,850" -> 1850), else None."""
    key = canonical(text)
    return int(key[1:]) if key.startswith("#") and "." not in key else None


def answer_target(units: list[Unit], cites: list[str], answer_text: str) -> tuple[str | None, int]:
    """The text probe C edits for this answer, and how many times the answer occurs in
    the cited units. The target is None unless that count is exactly 1.

    An answer that is one integer counts its mentions in any form (1850, 1,850, three),
    and the target is the mention as written ("1850 people" -> "1850", "3" -> "three").
    The answer's other words (normalized, articles dropped) must all be in the unit
    holding that mention, or the number there is a different quantity ("90 days" vs a
    unit's "90 feet"), and the count is 0. Any other answer counts standalone
    occurrences of its own text.
    """
    value = _integer(answer_text)
    if value is None:
        n = occurrences(units, cites, answer_text)
        return (answer_text if n == 1 else None), n
    cited = set(cites)
    hits = [
        (u, u.text[s:e]) for u in units if u.id in cited
        for s, e, v in number_mentions(u.text) if v == value
    ]  # fmt: skip
    if len(hits) != 1:
        return None, len(hits)
    unit, mention = hits[0]
    if not _words_besides_number(answer_text) <= set(normalize(unit.text).split()):
        return None, 0
    n = occurrences(units, cites, mention)  # the mention's text may also occur elsewhere
    return (mention if n == 1 else None), n


def _words_besides_number(text: str) -> set[str]:
    """Normalized tokens of `text` that are not numbers: "the 1,850 people" -> {"people"}."""
    return {
        t for t in normalize(text).split() if not t.isdigit() and t not in _NUMBER_WORDS
    }


_WORDS = sorted(_NUMBER_WORDS, key=_NUMBER_WORDS.get)  # zero..twenty by value


def _render(value: int, like: str) -> str:
    """`value` written the way the record writes the mention `like`: a number word
    (zero..twenty) keeps its capitalization, anything else is digits, with thousands
    separators if `like` has them."""
    if like.isalpha() and value < len(_WORDS):
        word = _WORDS[value]
        return word.upper() if like.isupper() and len(like) > 1 else (
            word.capitalize() if like[0].isupper() else word
        )
    return f"{value:,}" if "," in like else str(value)


def choose_replacement(record: Record, answer_text: str, seed: int) -> str | None:
    """Probe C's replacement for `answer_text` in `record`, or None when C has none.

    A number (answer_text canonicalizes to one integer that the record mentions, in
    digits or as a word zero..twenty) gets a v1 plausible swap (`datasets.swapped_integer`)
    of a value mentioned nowhere in the record's question or units, in any form, written
    like the record's mention (a word stays a word when the new value has one). A number
    with no such swap (a decimal, zero, a value the record doesn't mention) gets None,
    never an alternative. Anything else gets a same-type entity from
    `record.meta["alternatives"]` (a list of strings) that differs canonically from the
    answer and occurs in neither the question nor any unit (`_mentioned`). Either way,
    never a value canonically equal to one in `record.meta["heldout_values"]` (the
    corpus's own counterfactual draws, which C must not reuse). Deterministic per
    (seed, record id) in its own namespace, independent of `datasets.swap_record`.
    """
    rng = random.Random(f"{seed}:{record.id}:C")
    meta = record.meta or {}
    heldout = [str(x) for x in meta.get("heldout_values", [])]
    banned = {canonical(x) for x in heldout}
    key = canonical(answer_text)
    if key.startswith("#"):  # a number never takes an entity alternative
        value = _integer(answer_text)
        forms = [
            u.text[s:e] for u in record.units for s, e, v in number_mentions(u.text)
            if v == value
        ]  # fmt: skip
        if value is None or not forms:
            return None
        like = next((f for f in forms if f.lower() == answer_text.strip().lower()), forms[0])
        avoid = "\n".join([record.question, *(u.text for u in record.units), *heldout])
        try:
            new = _render(int(swapped_integer(str(value), rng, avoid=avoid)), like)
        except ValueError:  # no plausible swap (e.g. zero)
            return None
        return None if canonical(new) in banned else new
    alternatives = meta.get("alternatives")
    if not isinstance(alternatives, list) or not all(isinstance(x, str) for x in alternatives):
        return None
    texts = [record.question, *(u.text for u in record.units)]
    pool = [
        x for x in alternatives
        if normalize(x) and canonical(x) != key and canonical(x) not in banned
        and not _mentioned(x, texts)
    ]  # fmt: skip
    return rng.choice(pool) if pool else None


def _mentioned(value: str, texts: list[str]) -> bool:
    """`value` occurs in one of `texts` as whole tokens, ignoring case, with punctuation and
    hyphens as boundaries ("Paris" is in "Paris-based"), or in normalized form ("the
    St. Louis" is in "St Louis")."""
    pattern = re.compile(rf"(?<!\w){re.escape(value.strip())}(?!\w)", re.IGNORECASE)
    padded = f" {normalize(value)} "
    return any(pattern.search(t) or padded in f" {normalize(t)} " for t in texts)


def counterfactual_plan(record: Record, a: Answer, seed: int) -> tuple[dict, list[Unit] | None]:
    """Probe C for answer A: (Result.counterfactual without run and verdict, C's context).

    The dict: {"applicable", "reason", "replacement", "target", "seed", "run": None,
    "verdict": None}. target is the edited span, {"unit", "start", "end", "text"} with
    text as the unit writes it. The context is None, and reason says why, when C is not
    applicable: A abstains, cites nothing, its text is not in the cited units or is there
    more than once, or choose_replacement has no value for it.
    """

    def info(reason: str | None, replacement: str | None, target: dict | None) -> dict:
        return {
            "applicable": reason is None, "reason": reason, "replacement": replacement,
            "target": target, "seed": seed, "run": None, "verdict": None,
        }  # fmt: skip

    if a.abstain or a.text is None:
        return info("abstain", None, None), None
    if not a.cites:
        return info("uncited", None, None), None
    text, n = answer_target(record.units, a.cites, a.text)
    if text is None:
        reason = "answer_not_in_cites" if n == 0 else "answer_repeated_in_cites"
        return info(reason, None, None), None
    replacement = choose_replacement(record, text, seed)
    if replacement is None:
        return info("no_replacement", None, None), None
    uid, start, end = locate(record.units, a.cites, text)
    unit_text = next(u.text for u in record.units if u.id == uid)
    target = {"unit": uid, "start": start, "end": end, "text": unit_text[start:end]}
    context = counterfactual(record.units, a.cites, text, replacement)
    return info(None, replacement, target), context
