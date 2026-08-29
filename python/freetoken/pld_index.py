"""Prompt-lookup (n-gram) draft index for speculative decoding.

Maintains a host-side index over the request's prompt plus emitted tokens and
proposes the continuation of the most recent earlier occurrence of the current
suffix. Drafting is a dictionary lookup, so it costs no model forward; the
proposals are verified by the existing speculative target path.

The module is deliberately torch-free so the offline policy-replay tooling can
simulate exactly the semantics the engine executes.
"""

from __future__ import annotations

from collections.abc import Iterable


class PLDIndex:
    """Suffix n-gram index over one request's token history.

    For every n in ``[min_ngram, max_ngram]`` the index maps each n-gram to
    the end position of its most recent occurrence strictly before the current
    end of the sequence. ``draft`` returns the continuation of the longest
    matching current suffix.

    Memory grows with one dictionary entry per (position, n); at the current
    8K MTP context limit this is a few hundred thousand small entries. Cap the
    n-gram range before raising that limit.
    """

    def __init__(self, min_ngram: int = 6, max_ngram: int = 12):
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError("PLD needs 1 <= min_ngram <= max_ngram")
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.tokens: list[int] = []
        self._last_end: dict[int, dict[tuple[int, ...], int]] = {
            n: {} for n in range(min_ngram, max_ngram + 1)
        }

    def __len__(self) -> int:
        return len(self.tokens)

    def extend(self, tokens: Iterable[int]) -> None:
        for token in tokens:
            self._append(int(token))

    def _append(self, token: int) -> None:
        # Register the grams ending at the current length before the new token
        # lands, so a later lookup of the then-current suffix never finds
        # itself, only strictly earlier occurrences.
        end = len(self.tokens)
        for n in range(self.min_ngram, self.max_ngram + 1):
            if n > end:
                break
            self._last_end[n][tuple(self.tokens[end - n : end])] = end
        self.tokens.append(token)

    def draft(self, max_tokens: int) -> list[int]:
        """Continuation of the longest matched suffix, up to `max_tokens`."""
        if max_tokens < 1:
            return []
        length = len(self.tokens)
        for n in range(min(self.max_ngram, length - 1), self.min_ngram - 1, -1):
            suffix = tuple(self.tokens[length - n : length])
            end = self._last_end[n].get(suffix)
            if end is not None:
                return self.tokens[end : end + max_tokens]
        return []


__all__ = ["PLDIndex"]
