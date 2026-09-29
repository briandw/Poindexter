import json
import random
import re

import pytest

from poindexter import corpus as C
from poindexter.contract import read_records

FILLER = [
    "The subject has been described in many reference works.",
    "Scholars continue to study its background and influence.",
    "Its reputation grew steadily among specialists and the public.",
    "Several museums and libraries hold material related to it.",
    "Critics have discussed its importance at length.",
    "Its legacy remains a common topic in classrooms.",
    "Visitors and readers often ask about its origins.",
]


def fact(answer, kind="entity", entity_type="person", **kw):
    return C.Fact(
        fact_id=kw.pop("fact_id", "abc"),
        topic="t",
        subject=kw.pop("subject", "Mona Lisa"),
        question=kw.pop("question", "Who painted the Mona Lisa?"),
        answer=answer,
        kind=kind,
        entity_type=entity_type,
        **kw,
    )


def passage_with(answer_sentence, at=2, n=6, others=None):
    s = list(others or FILLER[: n - 1])
    s.insert(at, answer_sentence)
    return s


# --- verification rules -------------------------------------------------------


def test_good_entity_passage_passes():
    f = fact("Leonardo da Vinci")
    s = passage_with("The portrait was painted by Leonardo da Vinci in Florence.")
    assert C.check_passage(f, s, 2, ["Raphael"]) is None


def test_good_year_passage_passes():
    f = fact("1969", "number", "year", question="In what year did Apollo 11 land on the Moon?")
    s = passage_with("Apollo 11 landed on the Moon on 20 July 1969.")
    assert C.check_passage(f, s, 2, []) is None


@pytest.mark.parametrize(
    "sentences, index, reason",
    [
        (FILLER[:5], 0, "sentence_count"),
        (FILLER[:6] + ["The painter was Leonardo da Vinci."] + FILLER[:2], 6, "sentence_count"),
        (passage_with("It was painted by Leonardo da Vinci."), 9, "bad_answer_index"),
        (passage_with("It was painted by Leonardo da Vinci."), 1, "answer_not_in_answer_sentence"),
        (passage_with("Leonardo da Vinci and Leonardo da Vinci."), 2, "answer_repeated"),
        (
            passage_with(
                "It was painted by Leonardo da Vinci.",
                others=["Leonardo da Vinci lived."] + FILLER[:4],
            ),
            2,
            "answer_elsewhere",
        ),
        (
            passage_with(
                "It was painted by Leonardo da Vinci.",
                others=["Leonardo kept notebooks."] + FILLER[:4],
            ),
            2,
            "name_word_elsewhere",
        ),
        (
            passage_with("Leonardo da Vinci painted it, and Vinci is his town."),
            2,
            "name_word_elsewhere",
        ),
        (
            passage_with(
                "It was painted by Leonardo da Vinci.", others=["Raphael admired it."] + FILLER[:4]
            ),
            2,
            "alternative_in_passage",
        ),
        (passage_with("It was painted by Leonardo da Vinci. It is small."), 2, "not_one_sentence"),
    ],
)
def test_entity_passage_rejections(sentences, index, reason):
    assert C.check_passage(fact("Leonardo da Vinci"), sentences, index, ["Raphael"]) == reason


def year_fact():
    return fact("1969", "number", "year", question="In what year did Apollo 11 land?")


@pytest.mark.parametrize(
    "answer_sentence, other, reason",
    [
        ("It landed in 1969.", "The crew trained in 1966.", "date_elsewhere"),
        ("It landed in 1969.", "It was a feat of the twentieth century.", "date_elsewhere"),
        ("It landed in 1969.", "Its era, the 1960s, was busy.", "date_elsewhere"),
        ("It landed in 1969.", "The rocket weighed 1969 tonnes.", "answer_elsewhere"),
        ("It landed in 1969, not in 1968.", "The crew trained hard.", "date_in_answer_sentence"),
        ("It landed in 1969 (1969 exactly).", "The crew trained hard.", "answer_repeated"),
        ("It landed in 1,969 days.", "The crew trained hard.", "answer_not_in_answer_sentence"),
    ],
)
def test_year_passage_rejections(answer_sentence, other, reason):
    s = passage_with(answer_sentence, others=[other] + FILLER[:4])
    assert C.check_passage(year_fact(), s, 2, []) == reason


def test_novel_passage_must_not_say_fictional():
    f = fact("Tovren Halsk", question="Who first won the Kessary Prize?", novel=True)
    ok = passage_with("The first winner was Tovren Halsk.")
    assert C.check_passage(f, ok, 2, []) is None
    bad = passage_with(
        "The first winner was Tovren Halsk.",
        others=["The prize is awarded in a fictional town."] + FILLER[:4],
    )
    assert C.check_passage(f, bad, 2, []) == "says_fictional"


def test_strong_direction_is_stable_and_typed():
    f = fact("Leonardo da Vinci", fact_id="x1")
    assert C.strong_direction(f) == C.strong_direction(f)
    assert C.strong_direction(f).split(", preferably")[0] in C.STRONG_DIRECTIONS["person"]
    assert C.strong_direction(f) in C.alternatives_prompt(f)


def test_count_in_any_numeric_form_is_rejected_elsewhere():
    f = fact("12", "number", "count", question="How many signs are in the zodiac?")
    ok = passage_with("The zodiac has 12 signs.", others=["It dates back 3000 years."] + FILLER[:4])
    assert C.check_passage(f, ok, 2, []) is None
    for other in ["Each of the twelve signs has a symbol.", "There are 12 houses too."]:
        s = passage_with("The zodiac has 12 signs.", others=[other] + FILLER[:4])
        assert C.check_passage(f, s, 2, []) == "answer_elsewhere"
    s = passage_with("The zodiac has 12 signs.", others=["It lists 15 stars."] + FILLER[:4])
    assert C.check_passage(f, s, 2, ["15"]) == "alternative_in_passage"


def test_leading_the_may_be_lowercase():
    f = fact("The Beatles", entity_type="organization", question="Which band recorded Abbey Road?")
    s = passage_with("The album was recorded by the Beatles in London.")
    assert C.check_passage(f, s, 2, []) is None
    rec = C.level_records(
        C.Fact(**{**f.__dict__, "mild": "The Rolling Stones", "strong": "Daft Punk"}),
        C.Passage(s, 2),
    )
    assert "by the Rolling Stones in" in rec[1].units[2].text
    assert "by Daft Punk in" in rec[2].units[2].text


@pytest.mark.parametrize(
    "sentence, answer, value, etype, expected",
    [
        ("The Soviet Union launched it.", "Soviet Union", "the United States", "place",
         "The United States launched it."),
        ("The Soviet Union launched it.", "Soviet Union", "Kiribati", "place",
         "Kiribati launched it."),
        ("It hangs in the Louvre in Paris.", "Louvre", "Te Papa Tongarewa", "place",
         "It hangs in Te Papa Tongarewa in Paris."),
        ("It hangs in the Louvre in Paris.", "Louvre", "the Prado", "place",
         "It hangs in the Prado in Paris."),
        ("Canberra is the capital.", "Canberra", "the Hague", "place", "The Hague is the capital."),
        ("It was written by Tolkien.", "Tolkien", "Lewis", "person", "It was written by Lewis."),
        ("He wrote Ulysses.", "Ulysses", "The Hobbit", "work", "He wrote The Hobbit."),
        ("Then came the Beatles.", "The Beatles", "Queen", "organization", "Then came Queen."),
        ("It was the other one.", "other", "The Band", "organization", "It was the Band one."),
    ],
)  # fmt: skip
def test_swap_articles(sentence, answer, value, etype, expected):
    assert C._swap(sentence, answer, value, etype) == expected


# --- fact items and alternatives ----------------------------------------------


@pytest.mark.parametrize(
    "item, kind, expected",
    [
        ({"subject": "Mona Lisa", "question": "Who painted the Mona Lisa?",
          "answer": "Leonardo da Vinci", "entity_type": "person"}, "entity", None),
        ({"subject": "Apollo 11", "question": "When did Apollo 11 land?", "answer": "1969",
          "entity_type": "year"}, "number", None),
        ({"subject": "X", "question": "Who painted it?", "answer": "Leo",
          "entity_type": "year"}, "entity", "bad_entity_type"),
        ({"subject": "X", "question": "When?", "answer": "1969s",
          "entity_type": "year"}, "number", "number_not_integer"),
        ({"subject": "X", "question": "When?", "answer": "0776",
          "entity_type": "year"}, "number", "number_not_integer"),
        ({"subject": "X", "question": "When?", "answer": "776",
          "entity_type": "year"}, "number", "year_out_of_range"),
        ({"subject": "X", "question": "How many after 206 bones?", "answer": "206",
          "entity_type": "count"}, "number", "answer_in_question"),
        ({"subject": "Paris", "question": "Which Paris landmark is tallest?",
          "answer": "Eiffel Tower", "entity_type": "work"}, "entity", None),
        ({"subject": "X", "question": "Who founded the Ford Motor Company?",
          "answer": "Henry Ford", "entity_type": "person"}, "entity", "answer_in_question"),
        ({"subject": "Presidency of George Washington", "question": "Who was first president?",
          "answer": "George Washington", "entity_type": "person"}, "entity", "answer_in_subject"),
        ({"subject": "X", "question": "Which novel?", "answer": "Catch-22",
          "entity_type": "work"}, "entity", "entity_has_digit"),
        ({"subject": "X", "question": "Who?", "answer": "",
          "entity_type": "person"}, "entity", "missing_field"),
    ],
)  # fmt: skip
def test_check_fact_item(item, kind, expected):
    got = C.check_fact_item(item, kind)
    if expected is None:
        assert got == (item["subject"], item["question"], item["answer"], item["entity_type"])
    else:
        assert got == expected


POOL = ["Titian", "Rembrandt", "Caravaggio", "Vermeer", "Botticelli", "Giotto"]


def test_entity_alternatives_valid():
    got = C.check_entity_alternatives("Leonardo da Vinci", "Raphael", "Andy Warhol", POOL)
    assert got == ("Raphael", "Andy Warhol", POOL)


@pytest.mark.parametrize(
    "mild, strong, pool, reason",
    [
        ("leonardo da vinci", "Andy Warhol", POOL, "alternative_is_answer"),
        ("Raphael", "Leonardo da Vinci Jr", POOL, "alternative_contains_answer"),
        ("Raphael", "Paolo Vinci", POOL, "alternative_shares_name"),
        ("Raphael", "Andy Warhol", POOL[:5], "pool_size"),
        ("Raphael", "Raphael", POOL, "alternatives_not_distinct"),
        ("Raphael", "Andy Warhol", POOL[:5] + ["Raphael"], "alternatives_not_distinct"),
        ("Raphael", "1990", POOL, "alternative_is_number"),
        ("Raphael", "", POOL, "empty_value"),
        (None, "Andy Warhol", POOL, "empty_value"),
        ("Raphael", "Andy Warhol", "Titian", "pool_not_list"),
    ],
)
def test_entity_alternatives_rejected(mild, strong, pool, reason):
    assert C.check_entity_alternatives("Leonardo da Vinci", mild, strong, pool) == reason


def test_year_alternatives():
    for year in ["1969", "1066", "2020", "1850"]:
        for s in range(20):
            got = C.number_alternatives(year, "year", random.Random(s))
            assert not isinstance(got, str)
            mild, strong, pool = got
            v = int(year)
            assert 1 <= abs(int(mild) - v) <= C.MILD_YEARS
            assert abs(int(strong) - v) >= 150
            assert C.YEAR_MIN <= int(mild) <= C.YEAR_MAX and C.YEAR_MIN <= int(strong) <= C.YEAR_MAX
            assert len(pool) == C.POOL_SIZE and len(set(pool)) == C.POOL_SIZE
            assert not {year, mild, strong} & set(pool)


def test_count_alternatives():
    mild, strong, pool = C.number_alternatives("206", "count", random.Random(0))
    assert 124 <= int(mild) <= 288 and mild != "206" and len(mild) == 3
    assert int(strong) // 206 in C.STRONG_FACTORS
    mild, _, _ = C.number_alternatives("4", "count", random.Random(0), avoid="five and 6 and 7")
    assert mild in {"2", "3", "8", "9"}
    mild, _, _ = C.number_alternatives("8000", "count", random.Random(1))
    assert int(mild) % 1000 == 0


@pytest.mark.parametrize(
    "answer, etype, expected",
    [
        ("1969", "year", [1964, 1965, 1966, 1967, 1968, 1970, 1971, 1972, 1973, 1974]),
        ("2023", "year", [2018, 2019, 2020, 2021, 2022, 2024, 2025]),
        ("4", "count", [3, 5]),
        ("2", "count", [3]),
        ("12", "count", [11, 13]),
        ("20", "count", [17, 18, 19, 21, 22, 23]),
        ("100", "count", list(range(101, 116))),
        ("206", "count", [x for x in range(176, 237) if x != 206]),
    ],
)
def test_l1_number_window(answer, etype, expected):
    assert C.l1_number_window(answer, etype) == expected


def test_l1_number_candidates_avoid_passage_and_put_pool_last():
    f = fact("1969", "number", "year", fact_id="n1", question="When did it land?", strong="1500")
    p = C.Passage(
        passage_with("It landed in 1969.", others=["It weighed 1971 kg."] + FILLER[:4]), 2
    )
    pool = ["1964", "1965", "1966", "1967", "1968", "1970"]
    got = C.l1_number_candidates(f, p, seed=0, exclude=pool)
    assert sorted(got) == ["1964", "1965", "1966", "1967", "1968", "1970", "1972", "1973", "1974"]
    assert set(got[:3]) == {"1972", "1973", "1974"}
    assert got == C.l1_number_candidates(f, p, seed=0, exclude=pool)


def test_entity_value_problem():
    assert C.entity_value_problem("Leonardo da Vinci", "Giorgione") is None
    assert C.entity_value_problem("Leonardo da Vinci", "Paolo Vinci") == "alternative_shares_name"
    assert C.entity_value_problem("Leonardo da Vinci", "the leonardo da vinci") == (
        "alternative_is_answer"
    )


# --- levels ---------------------------------------------------------------------


def test_level_records():
    f = fact(
        "1969", "number", "year", fact_id="f1", question="In what year did Apollo 11 land?",
        mild="1972", strong="1512", alternatives=["1960", "1961", "1962", "1963", "1964", "1965"],
    )  # fmt: skip
    s = passage_with("Apollo 11 landed on 20 July 1969.", at=3)
    recs = C.level_records(f, C.Passage(s, 3))
    assert [r.id for r in recs] == ["sp-f1~L0", "sp-f1~L1", "sp-f1~L2"]
    for r, value in zip(recs, ["1969", "1972", "1512"], strict=True):
        assert r.gold == ["u4"]
        assert r.question == f.question
        assert r.meta["dataset"] == "surprise" and r.meta["fact_id"] == "f1"
        assert r.meta["dataset_answers"] == [value]
        assert r.meta["prior"] == "1969"
        assert r.meta["alternatives"] == f.alternatives
        assert r.units[3].text == f"Apollo 11 landed on 20 July {value}."
        others = [u for i, u in enumerate(r.units) if i != 3]
        assert others == [u for i, u in enumerate(recs[0].units) if i != 3]
    assert "counterfactual" not in recs[0].meta
    assert recs[0].meta["heldout_values"] == ["1972", "1512"]
    assert "heldout_values" not in recs[1].meta and "heldout_values" not in recs[2].meta
    assert recs[1].meta["counterfactual"] == {
        "original": "1969", "swapped": "1972", "source_id": "sp-f1~L0"
    }  # fmt: skip
    assert recs[2].meta["level"] == "L2" and recs[2].meta["counterfactual"]["swapped"] == "1512"


def conflict_fact():
    return fact(
        "1985", "number", "year", fact_id="la", question="In what year was Live Aid held?",
        mild="1984", strong="1612", alternatives=["1980", "1981", "1982", "1983", "1986", "1987"],
    )  # fmt: skip


def test_conflict_records():
    f = conflict_fact()
    s = passage_with("The Live Aid concert was held in 1985.", at=2)
    cb, ca = C.conflict_records(f, C.Passage(s, 2))
    assert [cb.id, ca.id] == ["sp-la~Cb", "sp-la~Ca"]
    for r in (cb, ca):
        assert [u.id for u in r.units] == [f"u{i}" for i in range(1, 8)]
        assert r.meta["dataset_answers"] == ["1985", "1984"]
        assert r.meta["prior"] == "1985" and r.meta["alternatives"] == f.alternatives
        assert [u.text for u in r.units[:2]] == s[:2] and [u.text for u in r.units[4:]] == s[3:]
    assert cb.meta["level"] == "Cb" and ca.meta["level"] == "Ca"
    assert cb.units[2].text == "The Live Aid concert was held in 1984."
    assert cb.units[3].text == "The Live Aid concert was held in 1985."
    assert cb.meta["conflict"] == {
        "true_unit": "u4", "true_value": "1985", "alt_unit": "u3", "alt_value": "1984"
    }  # fmt: skip
    assert ca.units[2].text.endswith("1985.") and ca.units[3].text.endswith("1984.")
    assert ca.meta["conflict"]["true_unit"] == "u3" and ca.meta["conflict"]["alt_unit"] == "u4"
    assert cb.gold == ["u3", "u4"] == ca.gold
    assert cb.meta["heldout_values"] == ["1984", "1612"] == ca.meta["heldout_values"]


def test_conflict_records_check_values():
    f = conflict_fact()
    other = ["Tickets were first printed in 1984."] + FILLER[:4]
    s = passage_with("The Live Aid concert was held in 1985.", at=2, others=other)
    assert C.conflict_records(f, C.Passage(s, 2)) == "mild_value_not_only_in_alt_unit"
    z = fact("12", "number", "count", question="How many signs?", mild="13", strong="180")
    other = ["Each of the twelve signs has a symbol."] + FILLER[:4]
    s = passage_with("The zodiac has 12 signs.", at=2, others=other)
    assert C.conflict_records(z, C.Passage(s, 2)) == "true_value_not_only_in_true_unit"
    other = ["It spans 13,000 years and the 1300s."] + FILLER[:4]
    s = passage_with("The zodiac has 12 signs.", at=2, others=other)
    assert not isinstance(C.conflict_records(z, C.Passage(s, 2)), str)
    e = fact("Leonardo da Vinci", mild="Giorgione", strong="Andy Warhol", alternatives=POOL)
    s = passage_with(
        "It was painted by Leonardo da Vinci.", others=["Giorgione saw it."] + FILLER[:4]
    )
    assert C.conflict_records(e, C.Passage(s, 2)) == "mild_value_not_only_in_alt_unit"
    s = passage_with("It was painted by Leonardo da Vinci.")
    cb, _ = C.conflict_records(e, C.Passage(s, 2))
    assert cb.units[2].text == "It was painted by Giorgione."


# --- end to end with a scripted generator --------------------------------------

FAMOUS = {
    "number": [
        ("Apollo 11", "In what year did Apollo 11 land on the Moon?", "1969", "year"),
        ("Human skeleton", "How many bones are in the adult human body?", "206", "count"),
        ("Berlin Wall", "In what year did the Berlin Wall fall?", "1989", "year"),  # leaks
        ("Apollo 11", "When did Apollo 11 land on the Moon?", "1969", "year"),  # duplicate
        ("X", "Year?", "1969s", "year"),  # invalid
    ],
    "entity": [
        ("Mona Lisa", "Who painted the Mona Lisa?", "Leonardo da Vinci", "person"),
        ("Australia", "What is the capital city of Australia?", "Canberra", "place"),
        ("Hamlet", "Who wrote the play Hamlet?", "William Shakespeare", "person"),
    ],
}
ALTS = {
    # first sample invalid (mild is the answer), second a stock strong, third valid
    "Leonardo da Vinci": [
        {"mild": "Leonardo da Vinci", "strong": "Andy Warhol", "alternatives": POOL},
        {"mild": "Raphael", "strong": "Napoleon Bonaparte", "alternatives": POOL},
        {"mild": "Raphael", "strong": "Andy Warhol", "alternatives": POOL},
    ],
    "Canberra": [
        {"mild": "Sydney", "strong": "Reykjavik",
         "alternatives": ["Melbourne", "Perth", "Adelaide", "Brisbane", "Hobart", "Darwin"]},
    ],
    # always invalid
    "William Shakespeare": [
        {"mild": "Christopher Marlowe", "strong": "Stephen King", "alternatives": POOL[:3]},
    ],
}  # fmt: skip
NOVEL = {
    "number": [
        ("Treaty of Vell Harrow", "In what year was the Treaty of Vell Harrow signed?", "1742",
         "year"),
        ("Orvane Lighthouse", "How many lamps does the Orvane Lighthouse hold?", "14", "count"),
    ],
    "entity": [
        ("Kessary Prize", "Who first won the Kessary Prize?", "Tovren Halsk", "person",
         ["Imra Dune", "Castel Wode", "Pell Arrow", "Juna Ferris", "Odo Brack", "Lisel Varn"]),
        ("Brumhold Institute", "Which city hosts the Brumhold Institute?", "Quellmarch", "place",
         ["Arnsby", "Tolvik", "Merrowgate", "Pashkent", "Villory", "Dunmere"]),
    ],
}  # fmt: skip


class FakeGenerator:
    model = "fake-generator"

    def __init__(self):
        self.calls = 0

    async def complete(self, system, user, temperature, sample_index):
        self.calls += 1
        if user.startswith("List "):
            kind = "number" if "whole number" in user else "entity"
            if f"about {C.TOPICS[1]}." in user and sample_index == 0:
                return '{"facts": [{"subject" "broken'  # unparseable: retried
            if f"about {C.TOPICS[0]}." not in user:
                return json.dumps({"facts": []})
            facts = [dict(zip(["subject", "question", "answer", "entity_type"], f, strict=True))
                     for f in FAMOUS[kind]]  # fmt: skip
            return "```json\n" + json.dumps({"facts": facts}) + "\n```"
        if user.startswith("Invent "):
            kind = "number" if "whole number" in user else "entity"
            if f"about {C.NOVEL_THEMES[0]}." not in user:
                return json.dumps({"facts": []})
            keys = ["subject", "question", "answer", "entity_type", "alternatives"]
            return json.dumps({"facts": [dict(zip(keys, f, strict=False)) for f in NOVEL[kind]]})
        if user.startswith("Fact: ") and "Give wrong answers" in user:
            answer = re.search(r"Answer: (.*) \(a ", user).group(1)
            replies = ALTS[answer]
            return json.dumps(replies[min(sample_index, len(replies) - 1)])
        if user.startswith("Write a short encyclopedic passage"):
            n = int(re.search(r"exactly (\d+) sentences", user).group(1))
            pos = int(re.search(r"Sentence number (\d+)", user).group(1))
            answer = re.search(r"^Answer: (.*)$", user, re.M).group(1)
            others = FILLER[: n - 1]
            leak = answer == "1989" or (answer == "Canberra" and sample_index == 0)
            if leak:
                others = [f"It is often linked with {answer}."] + others[1:]
            sentences = others[: pos - 1] + [f"The key fact is {answer}."] + others[pos - 1 :]
            return json.dumps({"sentences": sentences, "answer_sentence": pos})
        if user.startswith("Below are a question and a short passage"):
            value = re.search(r"answer \((.*?)\) from the real one", user).group(1)
            # Every L1 draw for 1969 contradicts (the fact is dropped); Raphael contradicts.
            bad = value == "Raphael" or (value.isdigit() and abs(int(value) - 1969) <= 5)
            return json.dumps({"contradicts": bad, "reason": "scripted"})
        if user.startswith("Fact: ") and "Give one wrong answer for a test" in user:
            assert "- Raphael: contradicts the passage: scripted" in user
            if "- Titian: not usable (already_used)" not in user:
                return json.dumps({"mild": "Titian"})  # in the pool: rejected, uses a try
            return json.dumps({"mild": "Giorgione"})
        raise AssertionError(f"unexpected prompt: {user[:80]}")


def test_build_corpus_end_to_end(tmp_path):
    tmp_path = tmp_path / "out"
    gen = FakeGenerator()
    report = C.build_corpus(tmp_path, n_facts=6, n_novel=4, seed=0, backend=gen, concurrency=4)

    assert report["pool"]["number"]["generated"] == 5
    assert report["pool"]["number"]["valid"] == 3
    assert report["pool"]["number"]["duplicates"] == 1
    assert report["pool"]["number"]["rejected"] == {"number_not_integer": 1}
    assert report["pool"]["entity"]["rejected"] == {}

    num, ent = report["facts"]["number"], report["facts"]["entity"]
    assert num["kept"] == 2 and num["dropped"] == {"passage:answer_elsewhere": 1}
    assert num["passage_attempts"] == {"1": 2, "4": 1}
    assert ent["kept"] == 2 and ent["dropped"] == {"alternatives:pool_size": 1}
    assert ent["passage_attempts"] == {"1": 1, "2": 1}
    dropped = {d["question"]: d for d in ent["dropped_facts"]}
    assert dropped["Who wrote the play Hamlet?"]["alternative_attempts"] == C.ATTEMPTS
    assert report["counts"]["generator_calls"] == gen.calls

    l1 = report["l1"]
    assert l1["number"] == {
        "checked": 2, "kept": 1, "dropped": {"l1:contradiction": 1}, "tries": {"1": 1},
        "mild_in_pool": 0,
    }  # fmt: skip
    assert l1["entity"] == {
        "checked": 2, "kept": 2, "dropped": {}, "tries": {"1": 1, "3": 1}, "mild_in_pool": 0
    }  # fmt: skip
    (gone,) = l1["dropped_facts"]
    assert gone["answer"] == "1969" and gone["tries"] == C.L1_TRIES
    drawn = [int(v) for v, _ in gone["rejected"]]
    assert len(set(drawn)) == C.L1_TRIES and all(1 <= abs(v - 1969) <= 5 for v in drawn)

    counts = report["counts"]
    assert counts["facts"] == 3 and counts["L0"] == counts["L1"] == counts["L2"] == 3
    assert counts["Cb"] == counts["Ca"] == 3 and counts["conflict_dropped"] == {}
    for lv in C.CONFLICT_LEVELS:
        for r in read_records(tmp_path / f"{lv}.jsonl"):
            assert r.meta["dataset_answers"][1] == next(
                x.meta["dataset_answers"][0]
                for x in levels_l1(tmp_path)
                if x.meta["fact_id"] == r.meta["fact_id"]
            )
    assert counts["novel"] == 4

    facts = [json.loads(line) for line in (tmp_path / "facts.jsonl").read_text().splitlines()]
    by_answer = {f["answer"]: f for f in facts}
    assert by_answer["Leonardo da Vinci"]["mild"] == "Giorgione"
    assert by_answer["Canberra"]["mild"] == "Sydney"
    assert by_answer["Canberra"]["alternatives"][0] == "Melbourne"
    assert 206 - 30 <= int(by_answer["206"]["mild"]) <= 206 + 30
    assert by_answer["206"]["mild"] not in by_answer["206"]["alternatives"]
    assert abs(int(by_answer["206"]["strong"]) - 206) > 1000

    levels = {lv: read_records(tmp_path / f"{lv}.jsonl") for lv in C.LEVELS}
    for r0, r1, r2 in zip(*levels.values(), strict=True):
        fid = r0.meta["fact_id"]
        assert (r0.id, r1.id, r2.id) == (f"sp-{fid}~L0", f"sp-{fid}~L1", f"sp-{fid}~L2")
        f = next(x for x in facts if x["fact_id"] == fid)
        g = int(r0.gold[0][1:]) - 1
        assert r0.gold[0] == f["answer_unit"]
        assert r1.units[g].text == r0.units[g].text.replace(f["answer"], f["mild"])
        assert r2.units[g].text == r0.units[g].text.replace(f["answer"], f["strong"])
        assert r1.meta["dataset_answers"] == [f["mild"]]
        assert r2.meta["counterfactual"]["source_id"] == r0.id
        assert "l1_check" not in r0.meta and "l1_check" not in r2.meta
        assert r1.meta["l1_check"]["reason"] == "scripted"
    tries = {r.meta["prior"]: r.meta["l1_check"]["tries"] for r in levels["L1"]}
    assert tries == {"206": 1, "Leonardo da Vinci": 3, "Canberra": 1}

    novel = read_records(tmp_path / "novel.jsonl")
    assert [r.id for r in novel] == [f"sp-novel-{i}" for i in range(1, 5)]
    for r in novel:
        assert r.meta["level"] == "novel" and r.meta["prior"] is None
        assert r.meta["dataset_answers"][0] in r.units[int(r.gold[0][1:]) - 1].text
        assert len(r.meta["alternatives"]) == C.POOL_SIZE
    halsk = next(r for r in novel if r.meta["dataset_answers"] == ["Tovren Halsk"])
    assert halsk.meta["alternatives"] == NOVEL["entity"][0][4]

    written = json.loads((tmp_path / "build_report.json").read_text())
    assert written["counts"] == counts

    # A rebuild is served from the cache and writes the same files.
    before = {p.name: p.read_text() for p in tmp_path.iterdir()}
    again = FakeGenerator()
    C.build_corpus(tmp_path, n_facts=6, n_novel=4, seed=0, backend=again, concurrency=4)
    assert again.calls == 0
    assert {p.name: p.read_text() for p in tmp_path.iterdir()} == before


def levels_l1(path):
    return read_records(path / "L1.jsonl")


def test_parse_json_tolerates_fence_and_prose():
    assert C.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert C.parse_json('Here: {"a": [1, 2]} done') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        C.parse_json("no json")
