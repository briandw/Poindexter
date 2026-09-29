"""Probe contexts. Pure functions from (units, cites, seed) to a unit list.

Unit ids stay attached to their text. Order is preserved everywhere except `shuffle`.
"""

from __future__ import annotations

import random

from poindexter.contract import Unit


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
