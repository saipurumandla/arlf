import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

type Vector = npt.NDArray[np.float32]

_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> Vector: ...


class HashingEmbedder:
    """Bag of words hashed into a fixed width unit vector."""

    def __init__(self, dim: int = _DIM) -> None:
        self._dim = dim

    def embed(self, text: str) -> Vector:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest, "big") % self._dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        return vec / norm


def cosine(a: Vector, b: Vector) -> float:
    return float(np.dot(a, b))
