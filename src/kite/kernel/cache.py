"""Content-addressed response cache (SQLite) and the wrapper that consults it."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Literal

from kite.kernel.base import Kernel
from kite.kernel.types import KernelAnswer, KernelRequest, KernelResponse, compact_json

Granularity = Literal["question", "request"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    created_at REAL NOT NULL
)
"""


def cache_key(*parts: str) -> str:
    """sha256 over length-prefixed parts, so no two part lists can collide by concatenation."""
    digest = hashlib.sha256()
    for part in parts:
        data = part.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class CacheStore:
    """One SQLite file. The key hashes the exact bytes that would be sent to the model."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)
        self._db.commit()

    def get(self, key: str) -> dict | None:
        row = self._db.execute("SELECT backend, model, answer_json, latency_ms FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return {"backend": row[0], "model": row[1], "answer": json.loads(row[2]), "latency_ms": row[3]}

    def put(self, key: str, *, backend: str, model: str, request_json: str, answer: dict, latency_ms: float) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, backend, model, request_json, json.dumps(answer, ensure_ascii=False), latency_ms, time.time()),
        )
        self._db.commit()

    def stats(self) -> dict[str, int]:
        rows = self._db.execute("SELECT backend || ':' || model, COUNT(*) FROM responses GROUP BY 1").fetchall()
        return dict(rows)

    def export_jsonl_gz(self, path: str | Path) -> int:
        cursor = self._db.execute("SELECT key, backend, model, request_json, answer_json, latency_ms, created_at FROM responses")
        count = 0
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in cursor:
                handle.write(json.dumps(list(row), ensure_ascii=False) + "\n")
                count += 1
        return count

    def import_jsonl_gz(self, path: str | Path) -> int:
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                self._db.execute("INSERT OR IGNORE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)", json.loads(line))
                count += 1
        self._db.commit()
        return count

    def close(self) -> None:
        self._db.close()


class CachedKernel:
    """Serve answers from the cache; send only what is missing to the inner kernel.

    granularity="question": one entry per (state, question). Valid only for backends that evaluate
    each question in isolation (Jev). granularity="request": one entry per whole request (LLM
    adapters, where the model sees every question at once).
    """

    def __init__(
        self,
        inner: Kernel,
        store: CacheStore,
        *,
        granularity: Granularity,
        options: str = "",
        salt: str = "",
        on_lookup=None,
    ) -> None:
        self._inner = inner
        self._store = store
        self._granularity = granularity
        self._options = options
        self._salt = salt
        self._on_lookup = on_lookup
        self.name = inner.name
        self.model_id = inner.model_id
        self.hits = 0
        self.misses = 0

    def _key(self, state_json: str, question_json: str) -> str:
        return cache_key(self._inner.name, self._inner.model_id, self._options, state_json, question_json, self._salt)

    def _count(self, hits: int, misses: int) -> None:
        self.hits += hits
        self.misses += misses
        if self._on_lookup is not None:
            self._on_lookup(hits, misses)

    async def evaluate(self, request: KernelRequest) -> KernelResponse:
        state_json = compact_json(request.state)
        if self._granularity == "request":
            return await self._evaluate_whole(request, state_json)
        return await self._evaluate_per_question(request, state_json)

    async def _evaluate_whole(self, request: KernelRequest, state_json: str) -> KernelResponse:
        questions_json = compact_json({name: q.to_api() for name, q in request.questions.items()})
        key = self._key(state_json, questions_json)
        hit = self._store.get(key)
        if hit is not None:
            self._count(len(request.questions), 0)
            answers = {name: KernelAnswer.model_validate(raw) for name, raw in hit["answer"].items()}
            return KernelResponse(answers=answers, model=hit["model"], backend=hit["backend"], cached=dict.fromkeys(answers, True))
        self._count(0, len(request.questions))
        response = await self._inner.evaluate(request)
        self._store.put(
            key,
            backend=response.backend,
            model=response.model,
            request_json=state_json + "\n" + questions_json,
            answer={name: answer.model_dump() for name, answer in response.answers.items()},
            latency_ms=response.latency_ms,
        )
        return response

    async def _evaluate_per_question(self, request: KernelRequest, state_json: str) -> KernelResponse:
        keys = {name: self._key(state_json, compact_json(q.to_api())) for name, q in request.questions.items()}
        found = {name: self._store.get(key) for name, key in keys.items()}
        missing = {name: q for name, q in request.questions.items() if found[name] is None}
        self._count(len(request.questions) - len(missing), len(missing))

        fresh: KernelResponse | None = None
        if missing:
            fresh = await self._inner.evaluate(KernelRequest(state=request.state, questions=missing))
            for name, answer in fresh.answers.items():
                self._store.put(
                    keys[name],
                    backend=fresh.backend,
                    model=fresh.model,
                    request_json=state_json + "\n" + compact_json(missing[name].to_api()),
                    answer=answer.model_dump(),
                    latency_ms=fresh.latency_ms,
                )

        answers: dict[str, KernelAnswer] = {}
        cached: dict[str, bool] = {}
        for name in request.questions:
            hit = found[name]
            if hit is not None:
                answers[name] = KernelAnswer.model_validate(hit["answer"])
                cached[name] = True
            else:
                assert fresh is not None
                answers[name] = fresh.answers[name]
                cached[name] = False

        if fresh is not None:
            return fresh.model_copy(update={"answers": answers, "cached": cached})
        first = next(iter(found.values()))
        assert first is not None
        return KernelResponse(answers=answers, model=first["model"], backend=first["backend"], cached=cached)
