"""The v2 surprise corpus: famous facts with counterfactual twins, and invented controls.

A generator model writes a pool of famous facts, wrong same-type alternatives for each,
and a short encyclopedic passage that states the fact in exactly one sentence. Every
generated item is checked deterministically here; an item that fails is regenerated
with the next sample index, and dropped (and counted) when it still fails. Each kept fact
yields three documents that differ only in the answer sentence:

    L0  the passage as written (agrees with the prior)
    L1  the answer replaced by a mild, plausible alternative
    L2  the answer replaced by a strong, absurd alternative

Invented facts in invented passages are the no-prior control. Every generation call goes
through the response cache, so a build is deterministic given the cache.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from poindexter.backend import Backend, ClaudeCLIBackend
from poindexter.contract import Record, Unit, canonical, normalize, write_jsonl
from poindexter.datasets import number_mentions, sentence_split
from poindexter.runner import call

GENERATOR = "claude-sonnet-5-5"
CONCURRENCY = 12
MIN_SENTENCES, MAX_SENTENCES = 6, 8
ATTEMPTS = 4  # the first draw and up to 3 regenerations
POOL_SIZE = 6
LEVELS = ("L0", "L1", "L2")
CONFLICT_LEVELS = ("Cb", "Ca")  # the mild restatement inserted before / after the answer

NUMBER, ENTITY = "number", "entity"
KIND_TYPES = {NUMBER: ("year", "count"), ENTITY: ("person", "place", "organization", "work")}

YEAR_MIN, YEAR_MAX = 1000, 2025
MILD_YEARS = 25  # first-draft mild year (+/- 1..25); it seeds the pool, L1 uses the window below
L1_YEARS = 5  # L1 mild year: +/- 1..5
L1_COUNT_SHARE = 0.15  # L1 mild count: +/- 1..max(1, floor(15%)), same digit count
L1_TRIES = 5  # L1 mild candidates checked for consistency before the fact is dropped
STRONG_YEARS = (150, 600)  # strong year: +/- 150..600, still in YEAR_MIN..YEAR_MAX
STRONG_FACTORS = (8, 12, 15, 20, 25)  # strong count: n times one of these
MAX_ANSWER_WORDS = 6

TOPICS = (
    "ancient and classical history",
    "medieval and early modern history",
    "modern world history (1800 onward)",
    "wars and military history",
    "exploration and discovery",
    "space exploration and astronomy",
    "physics and chemistry",
    "biology and medicine",
    "mathematics and computing",
    "inventions and technology",
    "literature and authors",
    "painting and sculpture",
    "classical and popular music",
    "film and television",
    "architecture and famous buildings",
    "world geography",
    "politics, leaders and governments",
    "business and famous companies",
    "sports and the Olympics",
    "religion, philosophy and mythology",
)
POOL_PER_CALL = 24

NOVEL_THEMES = (
    "the history of an invented island kingdom",
    "an invented scientist and their discoveries",
    "an invented sports league and its clubs",
    "invented companies and their founders",
    "invented novels, poems and their authors",
    "the geography of an invented country",
    "invented musicians, bands and albums",
    "invented inventions and engineers",
    "invented universities, institutes and museums",
    "invented religious orders, festivals and monuments",
)
NOVEL_PER_CALL = 10

SYSTEM = (
    "You write test data for a reading-comprehension benchmark. Follow every constraint "
    "exactly. Reply with one JSON object and nothing else: no prose, no code fence."
)


@dataclass
class Fact:
    fact_id: str
    topic: str
    subject: str
    question: str
    answer: str
    kind: str  # number | entity
    entity_type: str
    mild: str | None = None
    strong: str | None = None
    alternatives: list[str] = field(default_factory=list)
    novel: bool = False


@dataclass
class Passage:
    sentences: list[str]
    answer_index: int  # 0-based index of the sentence that states the fact


# --- parsing -----------------------------------------------------------------


def parse_json(raw: str) -> Any:
    """The first JSON object in raw, tolerating a code fence or prose around it."""
    s = raw.strip()
    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(s[start:])
    return obj


# --- text checks -------------------------------------------------------------

_STOPWORDS = frozenset(
    "the of and a an in on at to for de da di del della der den van von la le les du "
    "y e el al bin ibn".split()
)
# Anything that dates a sentence: a year 1000..2099 (also 1960s, 1960's), a century or
# millennium, a spelled-out decade, an era marker.
_DATE_CUE = re.compile(
    r"(?<![\d,.])(?:1\d{3}|20\d{2})(?:s|'s)?(?![\d,])"
    r"|(?i:\bcentur(?:y|ies)\b|\bmillenni|\b(?:twenties|thirties|forties|fifties|sixties"
    r"|seventies|eighties|nineties)\b)"
    r"|'\d0s\b|\b(?:BCE?|AD|CE)\b"
)


# A word that gives away an invented passage.
_FICTION_WORD = re.compile(
    r"\b(?:fictional|fictitious|invented|imaginary|imagined|made-up|hypothetical)\b", re.I
)


def name_tokens(text: str) -> list[str]:
    """Capitalized words of a name that identify it: "Leonardo da Vinci" -> Leonardo, Vinci."""
    words = re.findall(r"[^\W\d_]+", text)
    return [w for w in words if w[0].isupper() and w.lower() not in _STOPWORDS and len(w) > 1]


def _has_token(token: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) is not None


def mentions(name: str, text: str) -> bool:
    """Whether text mentions name, compared as normalized whole words."""
    n = normalize(name)
    return bool(n) and f" {n} " in f" {normalize(text)} "


def _occurrences(answer: str, text: str) -> list[int]:
    """Start offsets of answer in text as a whole word (or whole number), verbatim except
    that a leading "The" may be lowercase mid-sentence."""
    body = re.escape(answer)
    if answer.startswith("The "):
        body = "[Tt]" + re.escape(answer[1:])
    return [m.start() for m in re.finditer(rf"(?<!\w){body}(?!\w)", text)]


def check_fact_item(item: Any, kind: str) -> tuple[str, str, str, str] | str:
    """(subject, question, answer, entity_type) of a generated pool item, or the reason
    it is rejected."""
    if not isinstance(item, dict):
        return "not_object"
    fields = [item.get(k) for k in ("subject", "question", "answer", "entity_type")]
    if not all(isinstance(f, str) and f.strip() for f in fields):
        return "missing_field"
    subject, question, answer, etype = (f.strip() for f in fields)
    if etype not in KIND_TYPES[kind]:
        return "bad_entity_type"
    if not question.endswith("?") or len(question) > 250:
        return "bad_question"
    if kind == NUMBER:
        if not re.fullmatch(r"[1-9]\d*", answer):
            return "number_not_integer"
        v = int(answer)
        if etype == "year" and not (YEAR_MIN <= v <= YEAR_MAX):
            return "year_out_of_range"
        if etype == "count" and v < 2:
            return "count_too_small"
        if any(val == v for _, _, val in number_mentions(question)):
            return "answer_in_question"
    else:
        if re.search(r"\d", answer):
            return "entity_has_digit"
        if len(answer.split()) > MAX_ANSWER_WORDS or len(answer) > 60:
            return "entity_too_long"
        if canonical(answer).startswith("#"):
            return "entity_is_number"
        for text, reason in ((question, "answer_in_question"), (subject, "answer_in_subject")):
            if mentions(answer, text) or any(_has_token(t, text) for t in name_tokens(answer)):
                return reason
        if not name_tokens(answer):
            return "entity_not_a_name"
    return subject, question, answer, etype


def entity_value_problem(answer: str, v: str) -> str | None:
    """Why v cannot stand in for the entity answer, or None: it is a number, the answer,
    contains it or is contained in it, shares a name word with it, or is too long."""
    if canonical(v).startswith("#") or re.search(r"\d", v):
        return "alternative_is_number"
    if canonical(v) == canonical(answer):
        return "alternative_is_answer"
    if mentions(answer, v) or mentions(v, answer):
        return "alternative_contains_answer"
    if set(name_tokens(answer)) & set(name_tokens(v)):
        return "alternative_shares_name"
    if len(v.split()) > MAX_ANSWER_WORDS + 2:
        return "alternative_too_long"
    return None


def check_entity_alternatives(
    answer: str, mild: Any, strong: Any, pool: Any, need_mild: bool = True
) -> tuple[str | None, str | None, list[str]] | str:
    """Validated (mild, strong, pool) for an entity answer, or the reason they fail.

    Each value is a non-empty string, not canonically the answer, not containing the answer
    or contained in it, sharing no name word with it, not a number, and all distinct.
    The pool has POOL_SIZE entries and excludes mild and strong.
    """
    values = [mild, strong] if need_mild else []
    if not isinstance(pool, list):
        return "pool_not_list"
    values += pool
    if not all(isinstance(v, str) and v.strip() for v in values):
        return "empty_value"
    values = [v.strip() for v in values]
    if len(values) - (2 if need_mild else 0) != POOL_SIZE:
        return "pool_size"
    keys = set()
    for v in values:
        if problem := entity_value_problem(answer, v):
            return problem
        keys.add(canonical(v))
    if len(keys) != len(values):
        return "alternatives_not_distinct"
    if need_mild:
        return values[0], values[1], values[2:]
    return None, None, values


# --- number alternatives -----------------------------------------------------


def _avoid_values(avoid: str) -> set[int]:
    return {v for _, _, v in number_mentions(avoid)}


def mild_number_choices(answer: str, entity_type: str, avoid: str = "") -> list[list[int]]:
    """Plausible wrong values, grouped by side for years.

    Year: answer +/- 1..MILD_YEARS within YEAR_MIN..YEAR_MAX, as two sides (earlier,
    later). Count n >= 10: [ceil(0.6n), floor(1.4n)] with the same digit count, stepping by
    10^z when n ends in z zeros (falling back to every integer). Count 2..9: 2..9.
    Never the answer, never a value `avoid` mentions in any form.
    """
    v, present = int(answer), _avoid_values(avoid)

    def ok(x: int) -> bool:
        return x != v and x not in present

    if entity_type == "year":
        sides = [
            [x for x in (v + sign * k for k in range(1, MILD_YEARS + 1))
             if YEAR_MIN <= x <= YEAR_MAX and ok(x)]
            for sign in (-1, 1)
        ]  # fmt: skip
        return [s for s in sides if s]
    if v < 10:
        pool = [x for x in range(2, 10) if ok(x)]
    else:
        d = len(answer)
        lo, hi = -(-3 * v // 5), 7 * v // 5
        step = 10 ** (d - len(answer.rstrip("0")))
        pool = [x for x in range(-(-lo // step) * step, hi + 1, step) if len(str(x)) == d and ok(x)]
        pool = pool or [x for x in range(lo, hi + 1) if len(str(x)) == d and ok(x)]
    return [pool] if pool else []


def strong_number_choices(answer: str, entity_type: str, avoid: str = "") -> list[list[int]]:
    """Absurd wrong values. Year: answer +/- STRONG_YEARS, a valid year (two sides).
    Count: answer times each of STRONG_FACTORS. Never a value `avoid` mentions."""
    v, present = int(answer), _avoid_values(avoid)
    if entity_type == "year":
        lo, hi = STRONG_YEARS
        sides = [
            [x for x in (v + sign * k for k in range(lo, hi + 1))
             if YEAR_MIN <= x <= YEAR_MAX and x not in present]
            for sign in (-1, 1)
        ]  # fmt: skip
        return [s for s in sides if s]
    pool = [v * f for f in STRONG_FACTORS if v * f not in present]
    return [pool] if pool else []


def _draw(groups: list[list[int]], rng: random.Random) -> int:
    return rng.choice(rng.choice(groups))


def number_alternatives(
    answer: str, entity_type: str, rng: random.Random, avoid: str = ""
) -> tuple[str, str, list[str]] | str:
    """(mild, strong, pool) for a number answer, or the reason there are none.

    The pool is up to POOL_SIZE further mild-rule values, excluding mild and strong.
    """
    mild_groups = mild_number_choices(answer, entity_type, avoid)
    strong_groups = strong_number_choices(answer, entity_type, avoid)
    if not mild_groups:
        return "no_mild_number"
    if not strong_groups:
        return "no_strong_number"
    mild = _draw(mild_groups, rng)
    strong = _draw(strong_groups, rng)
    rest = sorted({x for g in mild_groups for x in g} - {mild, strong})
    pool = rng.sample(rest, min(POOL_SIZE, len(rest)))
    return str(mild), str(strong), [str(x) for x in pool]


def l1_number_window(answer: str, entity_type: str) -> list[int]:
    """The L1 mild values, ascending: a year +/- 1..L1_YEARS within YEAR_MIN..YEAR_MAX; a
    count +/- 1..max(1, floor(L1_COUNT_SHARE * n)) with the same digit count, at least 2."""
    v = int(answer)
    if entity_type == "year":
        return [x for x in range(v - L1_YEARS, v + L1_YEARS + 1)
                if x != v and YEAR_MIN <= x <= YEAR_MAX]  # fmt: skip
    d = max(1, int(v * L1_COUNT_SHARE))
    return [x for x in range(v - d, v + d + 1) if x != v and x >= 2 and len(str(x)) == len(answer)]


def l1_number_candidates(
    fact: Fact, passage: Passage, seed: int, exclude: Iterable[str] = ()
) -> list[str]:
    """L1 mild values for a number fact, in draw order.

    The window of l1_number_window minus any value the passage mentions in any form and
    the strong value, shuffled with a seed per fact. Values in `exclude` (the probe-C pool)
    come after the others, so the pool stays disjoint from the mild value when it can.
    """
    present = _avoid_values(" ".join(passage.sentences))
    values = [x for x in l1_number_window(fact.answer, fact.entity_type)
              if x not in present and str(x) != fact.strong]  # fmt: skip
    random.Random(f"{seed}:{fact.fact_id}:l1").shuffle(values)
    ex = set(exclude)
    return [str(x) for x in values if str(x) not in ex] + [str(x) for x in values if str(x) in ex]


# --- passage checks ----------------------------------------------------------


def check_passage(
    fact: Fact, sentences: Any, answer_index: Any, avoid: Iterable[str]
) -> str | None:
    """None when the passage is usable for the fact, else the reason it is not.

    The passage has MIN..MAX sentences, each one sentence. The answer sentence holds the
    answer verbatim exactly once (a number: that one mention of its value, in any form);
    no other sentence holds it in any form, or any of its name words; `avoid` values
    (the alternatives) appear nowhere. For a year, no sentence but the answer's year
    dates anything (years, decades, centuries, eras).
    """
    if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
        return "bad_json"
    sentences = [s.strip() for s in sentences]
    if not MIN_SENTENCES <= len(sentences) <= MAX_SENTENCES:
        return "sentence_count"
    if any(len(sentence_split(s)) != 1 for s in sentences):
        return "not_one_sentence"
    if not isinstance(answer_index, int) or not 0 <= answer_index < len(sentences):
        return "bad_answer_index"
    if fact.novel and any(_FICTION_WORD.search(s) for s in sentences):
        return "says_fictional"
    answer = fact.answer
    hits = [i for i, s in enumerate(sentences) if _occurrences(answer, s)]
    if answer_index not in hits:
        return "answer_not_in_answer_sentence"
    target = sentences[answer_index]
    if len(_occurrences(answer, target)) != 1:
        return "answer_repeated"
    if len(hits) > 1:
        return "answer_elsewhere"
    others = [s for i, s in enumerate(sentences) if i != answer_index]
    start = _occurrences(answer, target)[0]
    rest_of_target = target[:start] + " " + target[start + len(answer) :]
    if fact.kind == NUMBER:
        v = int(answer)
        if any(val == v for _, _, val in number_mentions(rest_of_target)):
            return "answer_repeated"
        for s in others:
            if any(val == v for _, _, val in number_mentions(s)) or re.search(
                rf"(?<!\d){answer}(?!\d)", s
            ):
                return "answer_elsewhere"
        if fact.entity_type == "year":
            if _DATE_CUE.search(rest_of_target):
                return "date_in_answer_sentence"
            if any(_DATE_CUE.search(s) for s in others):
                return "date_elsewhere"
        avoid_values = {int(a) for a in avoid if re.fullmatch(r"\d+", a)}
        if any(val in avoid_values for s in sentences for _, _, val in number_mentions(s)):
            return "alternative_in_passage"
    else:
        tokens = name_tokens(answer)
        for s in [rest_of_target, *others]:
            if mentions(answer, s):
                return "answer_elsewhere"
            if any(_has_token(t, s) for t in tokens):
                return "name_word_elsewhere"
        if any(mentions(a, s) for a in avoid for s in sentences):
            return "alternative_in_passage"
    return None


# --- records -----------------------------------------------------------------


def _swap(sentence: str, answer: str, value: str, entity_type: str = "") -> str:
    """The sentence with its one occurrence of answer replaced by value.

    The value carries its own article ("the United States", "Kiribati"), so a "the" just
    before the answer in the sentence is replaced along with it. The result starts with
    a capital at the start of the sentence; mid-sentence, a value's leading "The" is
    lowercased except for works ("by the Rolling Stones", "wrote The Hobbit").
    """
    (start,) = _occurrences(answer, sentence)
    end = start + len(answer)
    if not re.match(r"[Tt]he ", answer) and (m := re.search(r"(?<!\w)[Tt]he $", sentence[:start])):
        start = m.start()
    if start == 0:
        value = value[0].upper() + value[1:]
    elif value.startswith("The ") and entity_type != "work":
        value = "t" + value[1:]
    return sentence[:start] + value + sentence[end:]


def _units(passage: Passage) -> list[Unit]:
    return [Unit(f"u{i}", s) for i, s in enumerate(passage.sentences, 1)]


def level_records(
    fact: Fact, passage: Passage, l1_check: dict[str, Any] | None = None
) -> list[Record]:
    """L0, L1, L2 for a famous fact: only the answer sentence differs between levels.

    l1_check, when given, is stored in the L1 record's meta."""
    units = _units(passage)
    gold = units[passage.answer_index].id
    base = {
        "dataset": "surprise",
        "fact_id": fact.fact_id,
        "kind": fact.kind,
        "entity_type": fact.entity_type,
        "topic": fact.topic,
        "prior": fact.answer,
        "alternatives": list(fact.alternatives),
    }
    l0_id = f"sp-{fact.fact_id}~L0"
    out = []
    for level, value in zip(LEVELS, (fact.answer, fact.mild, fact.strong), strict=True):
        assert value is not None
        meta = {**base, "level": level, "dataset_answers": [value]}
        edited = units
        if level != "L0":
            text = _swap(units[passage.answer_index].text, fact.answer, value, fact.entity_type)
            edited = [Unit(u.id, text) if u.id == gold else u for u in units]
            meta["counterfactual"] = {"original": fact.answer, "swapped": value, "source_id": l0_id}
            if level == "L1" and l1_check is not None:
                meta["l1_check"] = l1_check
        else:
            # Probe C must not edit to a value another level uses as ground truth.
            meta["heldout_values"] = [fact.mild, fact.strong]
        out.append(
            Record(id=f"sp-{fact.fact_id}~{level}", question=fact.question, units=edited,
                   gold=[gold], meta=meta)
        )  # fmt: skip
    return out


def _value_units(fact: Fact, value: str, units: list[Unit]) -> list[str]:
    """Ids of the units that state value: for a number, in any numeric form ("1985s"
    counts, "4,000" does not state 4)."""
    if fact.kind == NUMBER:
        v = int(value)
        return [u.id for u in units
                if any(x == v for _, _, x in number_mentions(u.text))
                or re.search(rf"(?<![\d,.]){value}(?!\d|[.,]\d)", u.text)]  # fmt: skip
    return [u.id for u in units if mentions(value, u.text)]


def conflict_records(fact: Fact, passage: Passage) -> list[Record] | str:
    """Cb and Ca for a famous fact, or the reason they cannot be built.

    Each is the L0 passage with one sentence inserted: the answer sentence restating the
    fact with the mild value, immediately before (Cb) or after (Ca) the answer sentence.
    Units are renumbered u1..un; gold is both units, in order. The true value must be
    stated only in the true unit and the mild value only in the inserted unit.
    """
    assert fact.mild is not None
    g = passage.answer_index
    true_text = passage.sentences[g]
    alt_text = _swap(true_text, fact.answer, fact.mild, fact.entity_type)
    out = []
    for level in CONFLICT_LEVELS:
        pair = [alt_text, true_text] if level == "Cb" else [true_text, alt_text]
        sentences = passage.sentences[:g] + pair + passage.sentences[g + 1 :]
        units = [Unit(f"u{i}", t) for i, t in enumerate(sentences, 1)]
        true_i, alt_i = (g + 1, g) if level == "Cb" else (g, g + 1)
        true_id, alt_id = units[true_i].id, units[alt_i].id
        if _value_units(fact, fact.answer, units) != [true_id]:
            return "true_value_not_only_in_true_unit"
        if _value_units(fact, fact.mild, units) != [alt_id]:
            return "mild_value_not_only_in_alt_unit"
        out.append(
            Record(
                id=f"sp-{fact.fact_id}~{level}",
                question=fact.question,
                units=units,
                gold=sorted([true_id, alt_id], key=lambda u: int(u[1:])),
                meta={
                    "dataset": "surprise",
                    "fact_id": fact.fact_id,
                    "level": level,
                    "kind": fact.kind,
                    "entity_type": fact.entity_type,
                    "topic": fact.topic,
                    "prior": fact.answer,
                    "alternatives": list(fact.alternatives),
                    "dataset_answers": [fact.answer, fact.mild],
                    "conflict": {
                        "true_unit": true_id,
                        "true_value": fact.answer,
                        "alt_unit": alt_id,
                        "alt_value": fact.mild,
                    },
                    "heldout_values": [fact.mild, fact.strong],
                },
            )
        )
    return out


def novel_record(fact: Fact, passage: Passage, n: int) -> Record:
    units = _units(passage)
    return Record(
        id=f"sp-novel-{n}",
        question=fact.question,
        units=units,
        gold=[units[passage.answer_index].id],
        meta={
            "dataset": "surprise",
            "fact_id": fact.fact_id,
            "level": "novel",
            "kind": fact.kind,
            "entity_type": fact.entity_type,
            "topic": fact.topic,
            "dataset_answers": [fact.answer],
            "prior": None,
            "alternatives": list(fact.alternatives),
        },
    )


# --- prompts -----------------------------------------------------------------

_KIND_SPEC = {
    NUMBER: (
        'Every answer is a whole number. About three quarters are years (entity_type "year", '
        f"four digits, {YEAR_MIN} to {YEAR_MAX}); the rest are counts of things (entity_type "
        '"count", at least 2, e.g. the number of bones in the adult human body).'
    ),
    ENTITY: (
        "Every answer is a name: a person, a place, an organization (including companies, "
        "bands, teams), or a work (a book, painting, film, song, building, ship...). "
        'entity_type is one of "person", "place", "organization", "work".'
    ),
}


def pool_prompt(topic: str, kind: str, count: int) -> str:
    return f"""\
List {count} famous facts about {topic}. Each is a question with one short canonical answer.

{_KIND_SPEC[kind]}

Requirements:
- Famous: an educated adult knows it; widely taught; the answer is not disputed.
- The question stands alone and has exactly one correct answer: no "one of", no trick, \
no dependence on the current date.
- The question is short and direct. It names what is asked about but does not describe \
the answer: no nationality, profession, dates or other clues about the answer beyond \
what is needed to make the answer unique ("Who painted the Mona Lisa?", not "Which \
Italian Renaissance polymath painted the Mona Lisa?").
- The answer is the canonical short form: a year as four digits ("1969"), a count as \
plain digits ("206"), a name in its most common full form ("Leonardo da Vinci", \
"Canberra", "The Beatles"). A name answer contains no digits.
- The question must not contain the answer or any word of it.
- "subject" is what the question is about, as an encyclopedia article title (e.g. \
"Mona Lisa" for "Who painted the Mona Lisa?"). It never contains the answer or any \
word of it.
- Facts are all different from each other.

Reply as JSON: {{"facts": [{{"subject": "...", "question": "...", "answer": "...", \
"entity_type": "..."}}]}}"""


def novel_pool_prompt(theme: str, kind: str, count: int) -> str:
    extra, example = "", ""
    if kind == ENTITY:
        extra = (
            '\n- "alternatives": 6 other invented names of the same type as the answer, '
            "different from it and from each other, sharing no word with it."
        )
        example = ', "alternatives": ["...", "...", "...", "...", "...", "..."]'
    return f"""\
Invent {count} facts about {theme}. Everything is fictional: invented people, places, \
organizations and works that do not exist, with names that do not belong to anything \
real or famous. Each fact is a question with one short answer.

{_KIND_SPEC[kind]}

Requirements:
- The question stands alone and has exactly one answer; it names the invented subject \
so that it is clear what is asked (e.g. "In what year was the Treaty of Vell Harrow signed?"). \
It is short and direct, and never calls anything fictional, invented, or imaginary.
- The answer is a short form: a year as four digits, a count as plain digits, a name as \
a short proper name. A name answer contains no digits.
- The question must not contain the answer or any word of it.
- "subject" is what the question is about, as an encyclopedia article title. It never \
contains the answer or any word of it.{extra}
- Facts are all different from each other.

Reply as JSON: {{"facts": [{{"subject": "...", "question": "...", "answer": "...", \
"entity_type": "..."{example}}}]}}"""


# Where to look for a strong alternative, picked per fact so they do not all converge on
# the same few stock names.
STRONG_DIRECTIONS = {
    "person": (
        "a ruler, general or thinker of antiquity (before 500 AD)",
        "a 21st-century pop star, actor or athlete",
        "a medieval figure (500 to 1500) from another part of the world",
        "a 20th-century politician from another continent",
        "a scientist or inventor from a different century and field",
        "a 19th-century novelist, painter or composer from another country",
        "a modern business founder or tech executive",
        "an explorer or navigator from a different era",
    ),
    "place": (
        "one on another continent",
        "a small or remote one, far away",
        "an ancient or historic one in a different region",
        "one in a different hemisphere and climate",
    ),
    "organization": (
        "one from an unrelated industry",
        "a sports club or league",
        "a charity or international body in an unrelated field",
        "a band, orchestra or theatre company",
        "one from a different country and century",
    ),
    "work": (
        "a work of the same kind from a different millennium",
        "a recent blockbuster or best-seller of the same kind",
        "an ancient or medieval work of the same kind",
        "a work of the same kind from a different continent and century",
    ),
}
# Names a generator reaches for when asked for something absurd; refused as strong.
STOCK_STRONG = frozenset(
    canonical(n)
    for n in (
        "Napoleon Bonaparte", "Napoleon", "Julius Caesar", "Leonardo da Vinci",
        "Albert Einstein", "Cleopatra", "Genghis Khan", "William Shakespeare",
        "Isaac Newton", "Elvis Presley",
    )
)  # fmt: skip


STRONG_INITIALS = "ABCDEFGHJKLMNPRSTVW"


def _pick(options: Any, salt: str) -> Any:
    return options[int(hashlib.sha256(salt.encode()).hexdigest(), 16) % len(options)]


def strong_direction(fact: Fact) -> str:
    """Where to look for the strong alternative, and a preferred initial, fixed per fact."""
    where = _pick(STRONG_DIRECTIONS[fact.entity_type], f"s:{fact.fact_id}")
    initial = _pick(STRONG_INITIALS, f"i:{fact.fact_id}")
    return f"{where}, preferably one whose name starts with {initial}"


def alternatives_prompt(fact: Fact) -> str:
    t = fact.entity_type
    return f"""\
Fact: {fact.question} Answer: {fact.answer} (a {t}).

Give wrong answers of the same type and the same kind as "{fact.answer}" (a city for a \
city, a museum for a museum, a novelist for a novelist, a band for a band, a painting for \
a painting). Each will be substituted for "{fact.answer}" in a sentence, so write each in \
the same style and form (a full name), exactly as it would appear in the middle of a \
sentence: with "the" when the name takes one ("the United States", "the Louvre", "the \
Rolling Stones") and without when it does not ("Kiribati", "Canberra", "Microsoft").
- "mild": a real {t} that is plausible but wrong; someone who knows the topic loosely \
might believe it (e.g. a contemporary in the same field, a similar or nearby place, a \
comparable organization, a similar work of the same period and genre).
- "strong": a real {t} of the same kind that is obviously, absurdly wrong to anyone who \
knows the fact. Look for {strong_direction(fact)}; if that would not be clearly absurd \
here, pick another that is. Avoid stock choices such as Napoleon, Julius Caesar, \
Leonardo da Vinci or Einstein; be specific and unexpected.
- "alternatives": 6 more real {t}s of the same kind, all wrong, a mix of plausible and \
less plausible.
All 8 are different from each other. None is the answer, contains it, is contained in \
it, or shares a name word with it ("{fact.answer}"). No digits.

Reply as JSON: {{"mild": "...", "strong": "...", "alternatives": ["...", "...", "...", \
"...", "...", "..."]}}"""


def _position(fact_id: str, n: int) -> int:
    """The requested 1-based position of the fact sentence: 2..n-1, spread by fact id."""
    h = int(hashlib.sha256(fact_id.encode()).hexdigest(), 16)
    return 2 + h % (n - 2)


def passage_prompt(fact: Fact, avoid: list[str]) -> str:
    n = 6 + int(hashlib.sha256(f"n:{fact.fact_id}".encode()).hexdigest(), 16) % 3
    pos = _position(fact.fact_id, n)
    if fact.kind == NUMBER:
        other_form = "in digits, in words, or in any other form"
        changed = "the number were different"
    else:
        other_form = "or any word of it, or a nickname or other form of it"
        changed = f"the answer were a different {fact.entity_type}"
    lines = [
        f"Write a short encyclopedic passage of exactly {n} sentences about {fact.subject}.",
        "",
        f"Sentence number {pos} states this fact, and no other sentence does:",
        f"Question: {fact.question}",
        f"Answer: {fact.answer}",
        f'That sentence contains the text "{fact.answer}" exactly once, written exactly '
        "like that, and states the fact plainly. It does not describe the answer (no "
        "nationality, profession, or other attributes of it) and says nothing else that "
        "implies or constrains the answer (no arithmetic, ranges, or comparisons that "
        "give it away), so it would read naturally with any other answer of the same type.",
        "",
        "Every other sentence:",
        f'- never mentions "{fact.answer}", {other_form};',
        "- does not reveal, hint at, or narrow down the answer, and says nothing about the "
        "answer itself (no pronouns or descriptions that point to it);",
        f"- stays true and natural if {changed};",
        "- gives other true, specific, encyclopedic information about the subject.",
    ]
    if fact.entity_type == "year":
        lines += [
            "No sentence except the fact sentence contains any year, decade, century, era, "
            "date, age, or elapsed time. The fact sentence contains no year, decade, or "
            f"century other than {fact.answer}.",
        ]
    if fact.novel:
        lines += [
            "Everything in the passage is invented: use only invented names for people, "
            "places, organizations and works, never real ones. Write it as a plain "
            "encyclopedia entry that never calls anything fictional, invented, or imaginary.",
        ]
    if avoid:
        lines += ["Do not mention any of these anywhere: " + "; ".join(avoid) + "."]
    lines += [
        "Each list entry is one complete sentence ending with a period.",
        "",
        'Reply as JSON: {"sentences": ["...", "..."], "answer_sentence": <number of the '
        "sentence that states the fact, counting from 1>}",
    ]
    return "\n".join(lines)


def _passage_lines(units: list[Unit]) -> str:
    return "\n".join(f"[{u.id}] {u.text}" for u in units)


def l1_check_prompt(question: str, units: list[Unit], edited_id: str, value: str) -> str:
    return f"""\
Below are a question and a short passage. Sentence [{edited_id}] was edited on purpose so \
that it gives a different answer ({value}) from the real one. Do not judge whether \
[{edited_id}] is true in the real world; that it is false is intended.

Judge only this: does [{edited_id}] contradict, or sit implausibly with, anything that the \
other sentences or the question state or imply? Use general knowledge to see what they \
imply: for example a date before the career of a person the passage says took part, a date \
outside a war, reign or period the passage places the event in, an order of events the \
passage gives, a person who was not alive or active at a stated time, a place that \
conflicts with a stated location or nationality, or a number that conflicts with other \
numbers in the passage.

Question: {question}

Passage:
{_passage_lines(units)}

Reply as JSON: {{"contradicts": true or false, "reason": "<one line>"}}"""


def l1_mild_prompt(
    fact: Fact, units: list[Unit], answer_id: str, rejected: list[tuple[str, str]],
    exclude: list[str],
) -> str:  # fmt: skip
    t = fact.entity_type
    lines = [
        f"Fact: {fact.question} Answer: {fact.answer} (a {t}).",
        "",
        f"Passage, where sentence [{answer_id}] states the answer:",
        _passage_lines(units),
        "",
        f'Give one wrong answer for a test: a real {t} of the same kind as "{fact.answer}" '
        "that someone who knows the topic loosely might believe, and that fits everything "
        f'else in the passage and the question when substituted for "{fact.answer}" in '
        f"[{answer_id}]: the same era, place, field, and any other details they state or imply.",
    ]
    if rejected:
        lines += ["", "Already rejected:"] + [f"- {v}: {why}" for v, why in rejected]
    lines += [
        "",
        "Do not use any of these: " + "; ".join([fact.answer, *exclude]) + ". Do not use "
        f'anything that shares a name word with "{fact.answer}" or that the passage mentions.',
        'Write it exactly as it would appear in the middle of a sentence, with "the" when the '
        'name takes one ("the Louvre") and without when it does not ("Canberra").',
        "",
        'Reply as JSON: {"mild": "..."}',
    ]
    return "\n".join(lines)


# --- generation --------------------------------------------------------------


class _Gen:
    """Cached generator calls under one semaphore, with backend errors captured."""

    def __init__(self, backend: Backend, concurrency: int) -> None:
        self.backend = backend
        self.semaphore = asyncio.Semaphore(concurrency)
        self.calls = 0

    async def json(self, user: str, sample_index: int) -> Any:
        """The parsed JSON reply, or None when the call fails or the reply is not JSON."""
        self.calls += 1
        try:
            raw = await call(self.backend, SYSTEM, user, None, sample_index, self.semaphore)
            return parse_json(raw)
        except (ValueError, RuntimeError):
            return None


def fact_id_of(question: str) -> str:
    return hashlib.sha256(normalize(question).encode()).hexdigest()[:10]


async def _facts_reply(gen: _Gen, prompt: str) -> list[Any] | None:
    """The "facts" list of the first parseable reply, over up to ATTEMPTS samples."""
    for i in range(ATTEMPTS):
        reply = await gen.json(prompt, i)
        items = reply.get("facts") if isinstance(reply, dict) else None
        if isinstance(items, list):
            return items
    return None


async def _pool(
    gen: _Gen, kind: str, calls: list[tuple[str, str]], novel: bool
) -> tuple[list[tuple[Fact, Any]], dict[str, Any]]:
    """Validated, de-duplicated facts from the (topic, prompt) calls, in call order.

    Returns (fact, raw item) pairs and the counts."""
    replies = await asyncio.gather(*(_facts_reply(gen, p) for _, p in calls))
    rejected: Counter[str] = Counter()
    seen_q: set[str] = set()
    seen_a: set[tuple[str, str]] = set()
    out: list[tuple[Fact, Any]] = []
    generated = duplicates = 0
    for (topic, _), items in zip(calls, replies, strict=True):
        if items is None:
            rejected["bad_reply"] += 1
            continue
        for item in items:
            generated += 1
            checked = check_fact_item(item, kind)
            if isinstance(checked, str):
                rejected[checked] += 1
                continue
            subject, question, answer, etype = checked
            if novel and _FICTION_WORD.search(question + " " + subject):
                rejected["says_fictional"] += 1
                continue
            qkey = normalize(question)
            akey = (canonical(answer), normalize(subject))
            if qkey in seen_q or akey in seen_a:
                duplicates += 1
                continue
            seen_q.add(qkey)
            seen_a.add(akey)
            fact = Fact(
                fact_id=("novel-" if novel else "") + fact_id_of(question),
                topic=topic,
                subject=subject,
                question=question,
                answer=answer,
                kind=kind,
                entity_type=etype,
                novel=novel,
            )
            out.append((fact, item))
    counts = {
        "calls": len(calls),
        "generated": generated,
        "valid": len(out),
        "duplicates": duplicates,
        "rejected": dict(sorted(rejected.items())),
    }
    return out, counts


def _ranked(facts: list[Fact], seed: int, salt: str) -> list[Fact]:
    """Round-robin over topics (sorted), each topic shuffled with the seed."""
    by_topic: dict[str, list[Fact]] = {}
    for f in facts:
        by_topic.setdefault(f.topic, []).append(f)
    queues = []
    for topic in sorted(by_topic):
        q = sorted(by_topic[topic], key=lambda f: f.fact_id)
        random.Random(f"{seed}:{salt}:{topic}").shuffle(q)
        queues.append(q)
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


async def _entity_alternatives(gen: _Gen, fact: Fact) -> tuple[int, str | None]:
    """Fill fact.mild/strong/alternatives by generation. Returns (attempts, failure)."""
    reason = None
    for i in range(ATTEMPTS):
        reply = await gen.json(alternatives_prompt(fact), i)
        if not isinstance(reply, dict):
            reason = "bad_json"
            continue
        got = check_entity_alternatives(
            fact.answer, reply.get("mild"), reply.get("strong"), reply.get("alternatives")
        )
        if isinstance(got, str):
            reason = got
            continue
        if canonical(got[1] or "") in STOCK_STRONG:
            reason = "stock_strong"
            continue
        if any(mentions(v, fact.question) for v in (got[0], got[1], *got[2]) if v):
            reason = "alternative_in_question"
            continue
        fact.mild, fact.strong, fact.alternatives = got
        return i + 1, None
    return ATTEMPTS, reason


async def _passage(gen: _Gen, fact: Fact, avoid: list[str]) -> tuple[Passage | None, int, str]:
    """A checked passage for the fact. Returns (passage or None, attempts, last reason)."""
    reason = ""
    prompt = passage_prompt(fact, avoid)
    for i in range(ATTEMPTS):
        reply = await gen.json(prompt, i)
        if not isinstance(reply, dict):
            reason = "bad_json"
            continue
        sentences, number = reply.get("sentences"), reply.get("answer_sentence")
        index = number - 1 if isinstance(number, int) and not isinstance(number, bool) else None
        bad = check_passage(fact, sentences, index, avoid)
        if bad is None:
            return Passage([s.strip() for s in sentences], index), i + 1, ""
        reason = bad
    return None, ATTEMPTS, reason


async def _build_fact(gen: _Gen, fact: Fact, seed: int) -> tuple[Passage | None, dict[str, Any]]:
    """Alternatives then passage (then number alternatives, which avoid the passage)."""
    log: dict[str, Any] = {"fact_id": fact.fact_id, "question": fact.question}
    if fact.kind == ENTITY and not fact.novel:
        tries, bad = await _entity_alternatives(gen, fact)
        log["alternative_attempts"] = tries
        if bad:
            log["dropped"] = f"alternatives:{bad}"
            return None, log
    avoid = [a for a in (fact.mild, fact.strong, *fact.alternatives) if a]
    passage, tries, bad = await _passage(gen, fact, avoid)
    log["passage_attempts"] = tries
    if passage is None:
        log["dropped"] = f"passage:{bad}"
        return None, log
    if fact.kind == NUMBER:
        rng = random.Random(f"{seed}:{fact.fact_id}")
        got = number_alternatives(fact.answer, fact.entity_type, rng, " ".join(passage.sentences))
        if isinstance(got, str):
            log["dropped"] = f"alternatives:{got}"
            return None, log
        fact.mild, fact.strong, fact.alternatives = got
    return passage, log


async def _build_kind(
    gen: _Gen, ranked: list[Fact], target: int, seed: int
) -> tuple[list[tuple[Fact, Passage]], dict[str, Any]]:
    """Process ranked facts in waves until `target` are kept or the pool runs out."""
    kept: list[tuple[Fact, Passage]] = []
    logs: list[dict[str, Any]] = []
    nxt = waves = 0
    while len(kept) < target and nxt < len(ranked):
        batch = ranked[nxt : nxt + target - len(kept)]
        nxt += len(batch)
        waves += 1
        results = await asyncio.gather(*(_build_fact(gen, f, seed) for f in batch))
        for f, (passage, log) in zip(batch, results, strict=True):
            logs.append(log)
            if passage is not None:
                kept.append((f, passage))
    drops = Counter(log["dropped"] for log in logs if "dropped" in log)
    tries = Counter(log["passage_attempts"] for log in logs if "passage_attempts" in log)
    counts = {
        "target": target,
        "available": len(ranked),
        "attempted": len(logs),
        "waves": waves,
        "kept": len(kept),
        "dropped": dict(sorted(drops.items())),
        "passage_attempts": {str(k): v for k, v in sorted(tries.items())},
        "dropped_facts": [log for log in logs if "dropped" in log],
    }
    return kept, counts


async def _l1_verdict(gen: _Gen, prompt: str) -> tuple[bool, str] | None:
    """(contradicts, reason) from the first well-formed check reply, else None."""
    for i in range(ATTEMPTS):
        reply = await gen.json(prompt, i)
        if isinstance(reply, dict) and isinstance(reply.get("contradicts"), bool):
            reason = reply.get("reason")
            return reply["contradicts"], reason.strip() if isinstance(reason, str) else ""
    return None


async def _l1_mild(
    gen: _Gen, fact: Fact, passage: Passage, seed: int
) -> tuple[str | None, dict[str, Any]]:
    """A mild value whose L1 sentence is consistent with the passage and the question.

    Candidates, up to L1_TRIES: for a number, l1_number_candidates in order; for an
    entity, the first-draft mild, then values the generator proposes knowing the passage
    and the rejections so far, each checked deterministically (a failure uses up a try).
    Each candidate's L1 sentence goes to the generator's consistency check.
    Returns (value, {"tries", "reason"}) or (None, {"tries", "dropped", "rejected"}).
    """
    units = _units(passage)
    g = passage.answer_index
    uid = units[g].id
    rejected: list[tuple[str, str]] = []
    failures: Counter[str] = Counter()
    numbers = (
        l1_number_candidates(fact, passage, seed, fact.alternatives) if fact.kind == NUMBER else []
    )
    context = " ".join([fact.question, *passage.sentences])
    for t in range(1, L1_TRIES + 1):
        if fact.kind == NUMBER:
            if t > len(numbers):
                failures["no_candidate"] += 1
                break
            value = numbers[t - 1]
        elif t == 1:
            value = fact.mild or ""
        else:
            exclude = [v for v in (fact.strong, *fact.alternatives) if v]
            exclude += [v for v, _ in rejected]
            prompt = l1_mild_prompt(fact, units, uid, rejected, exclude)
            reply = await gen.json(prompt, 0)
            value = reply.get("mild") if isinstance(reply, dict) else None
            value = value.strip() if isinstance(value, str) else ""
            problem = "empty" if not value else entity_value_problem(fact.answer, value)
            if not problem and canonical(value) in {canonical(v) for v in exclude}:
                problem = "already_used"
            if not problem and mentions(value, context):
                problem = "in_passage"
            if problem:
                rejected.append((value or "(none)", f"not usable ({problem})"))
                failures[f"invalid:{problem}"] += 1
                continue
        text = _swap(units[g].text, fact.answer, value, fact.entity_type)
        edited = [Unit(u.id, text) if i == g else u for i, u in enumerate(units)]
        verdict = await _l1_verdict(gen, l1_check_prompt(fact.question, edited, uid, value))
        if verdict is None:
            rejected.append((value, "no verdict"))
            failures["no_verdict"] += 1
            continue
        contradicts, reason = verdict
        if not contradicts:
            return value, {"tries": t, "reason": reason}
        rejected.append((value, f"contradicts the passage: {reason}"))
        failures["contradiction"] += 1
    dropped = "contradiction" if failures["contradiction"] else failures.most_common(1)[0][0]
    return None, {"tries": len(rejected), "dropped": f"l1:{dropped}", "rejected": rejected}


def _novel_alternatives(fact: Fact, item: dict[str, Any]) -> str | None:
    """Fill a novel entity fact's pool from its generated item. Returns a failure reason."""
    got = check_entity_alternatives(fact.answer, None, None, item.get("alternatives"), False)
    if isinstance(got, str):
        return got
    fact.alternatives = got[2]
    return None


def _l1_report(
    famous: list[tuple[Fact, Passage]], l1: list[tuple[str | None, dict[str, Any]]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in (NUMBER, ENTITY):
        rows = [(f, v, info) for (f, _), (v, info) in zip(famous, l1, strict=True)
                if f.kind == kind]  # fmt: skip
        kept = [(f, v, info) for f, v, info in rows if v is not None]
        out[kind] = {
            "checked": len(rows),
            "kept": len(kept),
            "dropped": dict(sorted(Counter(i["dropped"] for _, v, i in rows if v is None).items())),
            "tries": {
                str(k): n for k, n in sorted(Counter(i["tries"] for _, _, i in kept).items())
            },
            "mild_in_pool": sum(v in f.alternatives for f, v, _ in kept),
        }
    out["dropped_facts"] = [
        {"fact_id": f.fact_id, "question": f.question, "answer": f.answer, **info}
        for (f, _), (v, info) in zip(famous, l1, strict=True)
        if v is None
    ]
    return out


async def _build(
    backend: Backend, n_facts: int, n_novel: int, seed: int, concurrency: int
) -> tuple[dict[str, list], dict[str, Any]]:
    gen = _Gen(backend, concurrency)
    report: dict[str, Any] = {
        "generator": backend.model,
        "seed": seed,
        "n_facts": n_facts,
        "n_novel": n_novel,
    }
    targets = {NUMBER: n_facts // 2, ENTITY: n_facts - n_facts // 2}
    novel_targets = {NUMBER: n_novel // 2, ENTITY: n_novel - n_novel // 2}

    pools = await asyncio.gather(
        *(
            _pool(gen, kind, [(t, pool_prompt(t, kind, POOL_PER_CALL)) for t in TOPICS], False)
            for kind in (NUMBER, ENTITY)
        ),
        *(
            _pool(
                gen, kind, [(t, novel_pool_prompt(t, kind, NOVEL_PER_CALL)) for t in NOVEL_THEMES],
                True,
            )
            for kind in (NUMBER, ENTITY)
        ),
    )  # fmt: skip
    famous_pool = dict(zip((NUMBER, ENTITY), pools[:2], strict=True))
    novel_pool = dict(zip((NUMBER, ENTITY), pools[2:], strict=True))
    report["pool"] = {k: c for k, (_, c) in famous_pool.items()}
    report["novel_pool"] = {k: c for k, (_, c) in novel_pool.items()}

    # A novel entity answer that is also a famous answer is not invented: dropped below.
    famous_answers = {canonical(f.answer) for p, _ in famous_pool.values() for f, _ in p}

    built = await asyncio.gather(
        *(
            _build_kind(gen, _ranked([f for f, _ in famous_pool[k][0]], seed, k), targets[k], seed)
            for k in (NUMBER, ENTITY)
        )
    )
    report["facts"] = {k: c for k, (_, c) in zip((NUMBER, ENTITY), built, strict=True)}

    novel_ready: dict[str, list[Fact]] = {}
    for kind, (pairs, counts) in novel_pool.items():
        ready, bad = [], Counter()
        for f, item in pairs:
            if kind == ENTITY and canonical(f.answer) in famous_answers:
                bad["answer_is_famous"] += 1
                continue
            if kind == ENTITY and (reason := _novel_alternatives(f, item)):
                bad[reason] += 1
                continue
            ready.append(f)
        counts["alternatives_rejected"] = dict(sorted(bad.items()))
        novel_ready[kind] = ready
    novel_built = await asyncio.gather(
        *(
            _build_kind(gen, _ranked(novel_ready[k], seed, "novel-" + k), novel_targets[k], seed)
            for k in (NUMBER, ENTITY)
        )
    )
    report["novel"] = {k: c for k, (_, c) in zip((NUMBER, ENTITY), novel_built, strict=True)}

    famous = sorted(
        (fp for kept, _ in built for fp in kept), key=lambda fp: (fp[0].kind, fp[0].fact_id)
    )
    # L1 gets a tighter mild value checked for consistency with its passage. L0 and L2 do not
    # depend on it; a fact without a consistent L1 is dropped (not replaced), so the kept
    # facts' L0 and L2 records stay the same as without this step.
    l1 = await asyncio.gather(*(_l1_mild(gen, f, p, seed) for f, p in famous))
    report["l1"] = _l1_report(famous, l1)
    kept_famous = []
    for (fact, passage), (value, info) in zip(famous, l1, strict=True):
        if value is not None:
            fact.mild = value
            kept_famous.append((fact, passage, info))
    famous = kept_famous
    novel = [fp for kept, _ in novel_built for fp in kept]
    novel.sort(key=lambda fp: (fp[0].kind, fp[0].fact_id))
    levels: dict[str, list[Record]] = {lv: [] for lv in LEVELS}
    fact_rows = []
    conflicts: dict[str, list[Record]] = {lv: [] for lv in CONFLICT_LEVELS}
    conflict_drops: Counter[str] = Counter()
    conflict_dropped_facts = []
    for fact, passage, info in famous:
        for rec in level_records(fact, passage, info):
            levels[rec.meta["level"]].append(rec)
        fact_rows.append({**asdict(fact), "answer_unit": f"u{passage.answer_index + 1}"})
        built_conflict = conflict_records(fact, passage)
        if isinstance(built_conflict, str):
            conflict_drops[built_conflict] += 1
            conflict_dropped_facts.append({"fact_id": fact.fact_id, "dropped": built_conflict})
            continue
        for rec in built_conflict:
            conflicts[rec.meta["level"]].append(rec)
    report["conflict"] = {
        "dropped": dict(sorted(conflict_drops.items())),
        "dropped_facts": conflict_dropped_facts,
    }
    novel_records = [novel_record(f, p, n) for n, (f, p) in enumerate(novel, 1)]
    report["counts"] = {
        "facts": len(fact_rows),
        "facts_by_kind": dict(Counter(r["kind"] for r in fact_rows)),
        "facts_by_entity_type": dict(sorted(Counter(r["entity_type"] for r in fact_rows).items())),
        **{lv: len(recs) for lv, recs in levels.items()},
        **{lv: len(recs) for lv, recs in conflicts.items()},
        "conflict_dropped": dict(sorted(conflict_drops.items())),
        "novel": len(novel_records),
        "novel_by_kind": dict(Counter(r.meta["kind"] for r in novel_records)),
        "generator_calls": gen.calls,
    }
    files = {
        "facts.jsonl": fact_rows,
        **{f"{lv}.jsonl": [r.to_json() for r in recs] for lv, recs in levels.items()},
        **{f"{lv}.jsonl": [r.to_json() for r in recs] for lv, recs in conflicts.items()},
        "novel.jsonl": [r.to_json() for r in novel_records],
    }
    return files, report


def build_corpus(
    out_dir: str | Path,
    n_facts: int = 400,
    n_novel: int = 100,
    seed: int = 0,
    backend: Backend | None = None,
    concurrency: int = CONCURRENCY,
) -> dict[str, Any]:
    """Generate, check, and write the corpus to out_dir. Returns the build report.

    Writes facts.jsonl (kept famous facts with alternatives and answer unit), L0.jsonl,
    L1.jsonl, L2.jsonl, the conflict levels Cb.jsonl and Ca.jsonl, novel.jsonl, and
    build_report.json. n_facts and n_novel are split evenly between number and entity
    answers. A fact whose L1 fails the consistency check is dropped from every level, not
    replaced. Deterministic given the cache.
    """
    backend = backend or ClaudeCLIBackend(GENERATOR)
    files, report = asyncio.run(_build(backend, n_facts, n_novel, seed, concurrency))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        write_jsonl(out / name, rows)
    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m poindexter.corpus", description=__doc__)
    ap.add_argument("out_dir")
    ap.add_argument("--facts", type=int, default=400)
    ap.add_argument("--novel", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=GENERATOR)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args(argv)
    report = build_corpus(
        args.out_dir, args.facts, args.novel, args.seed, ClaudeCLIBackend(args.model),
        args.concurrency,
    )  # fmt: skip
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
