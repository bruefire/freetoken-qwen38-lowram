"""Qwen4 QSA KV pool: paged GQA K/V plus per-token raw index keys."""

from __future__ import annotations

from .bsa_pool import BSAKVCache


class QSAKVCache(BSAKVCache):
    """Correctness-first QSA storage using the common GQA index-slab layout.

    The index slab stores one raw 128-wide BF16 key per token and QSA layer.
    Logical 4-token blocks are compressed on demand by the attention backend.

    TODO(qsa-cache): replace this with SGLang's one compressed key per complete
    4-token block plus a four-slot ring for the incomplete block. That reduces
    the index slab by nearly 4x without changing QSA selection semantics.
    """


__all__ = ["QSAKVCache"]
