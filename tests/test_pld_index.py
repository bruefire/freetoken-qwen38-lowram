from __future__ import annotations

import pytest
from freetoken.pld_index import PLDIndex


def _reference_draft(
    tokens: list[int], min_ngram: int, max_ngram: int, max_tokens: int
) -> list[int]:
    """Brute-force longest-suffix most-recent-occurrence lookup."""
    length = len(tokens)
    for n in range(min(max_ngram, length - 1), min_ngram - 1, -1):
        suffix = tokens[length - n :]
        for start in range(length - n - 1, -1, -1):
            if tokens[start : start + n] == suffix:
                return tokens[start + n : start + n + max_tokens]
    return []


def test_pld_index_validates_ngram_range():
    with pytest.raises(ValueError):
        PLDIndex(0, 4)
    with pytest.raises(ValueError):
        PLDIndex(5, 4)


def test_pld_index_finds_most_recent_continuation():
    index = PLDIndex(2, 4)
    index.extend([1, 2, 3, 9, 1, 2, 3, 8, 7, 1, 2, 3])
    # Longest suffix [1, 2, 3] most recently continued with 8, not 9.
    assert index.draft(2) == [8, 7]


def test_pld_index_never_matches_the_query_suffix_itself():
    index = PLDIndex(2, 4)
    index.extend([5, 6, 7])
    assert index.draft(2) == []


def test_pld_index_returns_partial_tail_continuation():
    index = PLDIndex(2, 3)
    index.extend([1, 2, 9, 1, 2])
    assert index.draft(3) == [9, 1, 2]
    index2 = PLDIndex(2, 3)
    index2.extend([3, 4, 3, 4])
    # The matched occurrence ends two tokens before the tail, so only two
    # continuation tokens exist.
    assert index2.draft(3) == [3, 4]


def test_pld_index_zero_budget_and_incremental_extend():
    index = PLDIndex(2, 4)
    index.extend([1, 2, 3, 1, 2])
    assert index.draft(0) == []
    assert index.draft(2) == [3, 1]
    index.extend([3])
    assert index.draft(2) == [1, 2]


@pytest.mark.parametrize("min_ngram,max_ngram", [(2, 4), (3, 6), (6, 12)])
def test_pld_index_matches_brute_force_reference(min_ngram, max_ngram):
    # Deterministic pseudo-random sequence over a small alphabet so repeats
    # occur; verify every prefix against the reference scan.
    tokens: list[int] = []
    state = 12345
    for _ in range(160):
        state = (state * 1103515245 + 12345) % (2**31)
        tokens.append(state % 7)
    index = PLDIndex(min_ngram, max_ngram)
    for i, token in enumerate(tokens):
        index._append(token)
        got = index.draft(3)
        want = _reference_draft(tokens[: i + 1], min_ngram, max_ngram, 3)
        assert got == want, f"prefix length {i + 1}"
