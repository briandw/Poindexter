import pytest

from poindexter import probes
from poindexter.contract import Unit

UNITS = [Unit(f"u{i}", f"text {i}") for i in range(1, 6)]
CITES = ["u4", "u2"]
DONOR = [Unit("d1", "donor one"), Unit("d2", "donor two")]


def ids(units):
    return [u.id for u in units]


def test_original_and_no_context():
    assert probes.original(UNITS, CITES, 0) == UNITS
    assert probes.no_context(UNITS, CITES, 0) == []


def test_remove_cited_preserves_order():
    assert ids(probes.remove_cited(UNITS, CITES, 0)) == ["u1", "u3", "u5"]


def test_cited_only_in_original_order():
    out = probes.cited_only(UNITS, CITES, 0)
    assert ids(out) == ["u2", "u4"]
    assert out == [UNITS[1], UNITS[3]]


def test_replace_cited_uses_donor_text_and_keeps_ids():
    out = probes.replace_cited(UNITS, CITES, 7, DONOR)
    assert ids(out) == ids(UNITS)
    donor_texts = {u.text for u in DONOR}
    for before, after in zip(UNITS, out, strict=True):
        if before.id in CITES:
            assert after.text in donor_texts
        else:
            assert after == before
    assert out == probes.replace_cited(UNITS, CITES, 7, DONOR)


@pytest.mark.parametrize("donor", [None, []])
def test_replace_cited_needs_donor(donor):
    with pytest.raises(ValueError):
        probes.replace_cited(UNITS, CITES, 0, donor)


def test_shuffle_is_a_seeded_permutation():
    out = probes.shuffle(UNITS, CITES, 3)
    assert sorted(out, key=lambda u: u.id) == UNITS
    assert out == probes.shuffle(UNITS, CITES, 3)
    assert any(probes.shuffle(UNITS, CITES, s) != UNITS for s in range(5))
    assert ids(UNITS) == ["u1", "u2", "u3", "u4", "u5"]  # input untouched


def test_leave_one_out():
    for i in range(len(UNITS)):
        out = probes.leave_one_out(UNITS, i)
        assert out == UNITS[:i] + UNITS[i + 1 :]
    with pytest.raises(IndexError):
        probes.leave_one_out(UNITS, 5)
