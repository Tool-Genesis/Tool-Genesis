# src/memories/vector_db_block.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import hashlib
import math
import os

from src.core.memories.base import MemoryBlock
from src.core.memories.records import ContextRecord, MemoryRecord


class BaseEmbedding:
    """Minimal embedding interface."""
    def get_output_dim(self) -> int:
        raise NotImplementedError

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class HashEmbedding(BaseEmbedding):
    """
    Zero-dependency bag-of-words hashing embedding (fast & reproducible).
    - Tokenizes by simple lowercasing & split on non-alnum.
    - Hashes each token -> bucket index in fixed dim.
    - Signed hashing to reduce collisions' bias.
    """
    def __init__(self, dim: int = 512, seed: int = 0):
        assert dim > 0
        self.dim = dim
        self.seed = seed

    def get_output_dim(self) -> int:
        return self.dim

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # simple, fast tokenizer
        out, cur = [], []
        for ch in text.lower():
            if ch.isalnum():
                cur.append(ch)
            else:
                if cur:
                    out.append("".join(cur))
                    cur = []
        if cur:
            out.append("".join(cur))
        return out

    def _bucket(self, token: str) -> int:
        h = hashlib.blake2b((token + str(self.seed)).encode("utf-8"), digest_size=8).digest()
        # map to [0, dim-1]
        return int.from_bytes(h, "little") % self.dim

    def _sign(self, token: str) -> int:
        h = hashlib.md5((token + "sign").encode("utf-8")).digest()
        return 1 if (h[0] & 1) == 1 else -1

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        for tok in tokens:
            i = self._bucket(tok)
            s = self._sign(tok)
            vec[i] += float(s)
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class OpenAIEmbedding(BaseEmbedding):
    """
    Optional OpenAI embedding backend. Requires:
        pip install openai
        export OPENAI_API_KEY=...
    """
    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            import openai  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAIEmbedding requires `openai` package. `pip install openai`"
            ) from e
        self._openai = openai
        self.model = model
        self._dim = 1536  # text-embedding-3-small output dim

    def get_output_dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        # OpenAI python SDK v1.x
        from openai import OpenAI  # lazy import inside to avoid hard dep when unused
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


# ===================== Vector storage =====================

@dataclass
class VectorRecord:
    id: str
    vector: List[float]
    payload: Dict[str, Any]


@dataclass
class VectorDBQuery:
    query_vector: List[float]
    top_k: int = 3


@dataclass
class QueryResult:
    record: VectorRecord
    similarity: float


class BaseVectorStorage:
    def add(self, records: List[VectorRecord]) -> None:
        raise NotImplementedError

    def query(self, q: VectorDBQuery) -> List[QueryResult]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class InMemoryStorage(BaseVectorStorage):
    """Simple in-memory vector store with cosine similarity."""
    def __init__(self, vector_dim: int):
        self._dim = vector_dim
        self._store: List[VectorRecord] = []

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # safe cosine (assuming normalized inputs is fine but we compute dot anyway)
        dot = sum(x * y for x, y in zip(a, b))
        # if vectors come normalized, denom ~= 1; otherwise compute:
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def add(self, records: List[VectorRecord]) -> None:
        for r in records:
            if len(r.vector) != self._dim:
                raise ValueError(f"vector dim {len(r.vector)} != expected {self._dim}")
            self._store.append(r)

    def query(self, q: VectorDBQuery) -> List[QueryResult]:
        res = []
        for r in self._store:
            sim = self._cosine(q.query_vector, r.vector)
            res.append(QueryResult(record=r, similarity=sim))
        res.sort(key=lambda x: x.similarity, reverse=True)
        return res[: max(0, q.top_k)]

    def clear(self) -> None:
        self._store.clear()


# ===================== VectorDBBlock (drop-in) =====================

class VectorDBBlock(MemoryBlock):
    """
    Lightweight vector-memory block with pluggable embedding and storage backends.

    Defaults:
      - Embedding: HashEmbedding(dim=512) — no external dependencies
      - Storage: InMemoryStorage — simple local vector store

    Optional:
      - Set `UseOpenAIEmbedding=True` to use OpenAI embeddings (requires `openai`).
    """

    def __init__(
        self,
        storage: Optional[BaseVectorStorage] = None,
        embedding: Optional[BaseEmbedding] = None,
        UseOpenAIEmbedding: bool = False,
    ) -> None:
        if embedding is None:
            if UseOpenAIEmbedding:
                embedding = OpenAIEmbedding()  # requires `openai`
            else:
                embedding = HashEmbedding(dim=512)
        self.embedding = embedding
        self.vector_dim = self.embedding.get_output_dim()
        self.storage = storage or InMemoryStorage(vector_dim=self.vector_dim)

    def retrieve(self, keyword: str, limit: int = 3) -> List[ContextRecord]:
        """
        Embed the query `keyword`, search the vector storage, and return
        up to `limit` matched context records ordered by similarity.
        """
        query_vector = self.embedding.embed(keyword)
        results = self.storage.query(VectorDBQuery(query_vector=query_vector, top_k=limit))
        out: List[ContextRecord] = []
        for r in results:
            payload = r.record.payload or {}
            try:
                mr = MemoryRecord.from_dict(payload)
            except Exception:
                # Fallback: payload may already be a MemoryRecord-like dict
                mr = MemoryRecord(
                    message=payload.get("message"),
                    role_at_backend=payload.get("role_at_backend"),
                    timestamp=payload.get("timestamp"),
                    agent_id=payload.get("agent_id"),
                )
            ts = payload.get("timestamp", None)
            out.append(ContextRecord(memory_record=mr, score=r.similarity, timestamp=ts))
        return out

    def write_records(self, records: List[MemoryRecord]) -> None:
        """
        Embed and persist a batch of `MemoryRecord` items into the vector storage.
        """
        v_records: List[VectorRecord] = []
        for rec in records:
            content = getattr(rec.message, "content", None)
            if content is None:
                # allow dict-like message
                content = str(rec.message)
            vec = self.embedding.embed(str(content))
            v_records.append(
                VectorRecord(
                    vector=vec,
                    payload=rec.to_dict(),
                    id=str(getattr(rec, "uuid", id(rec))),
                )
            )
        self.storage.add(v_records)

    def clear(self) -> None:
        """Remove all stored vectors and payloads."""
        self.storage.clear()