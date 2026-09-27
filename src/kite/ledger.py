"""The prospective forecast ledger (research plan v5, E2): seal a forecast now, reveal it after the results.

A forecast is a JSON record. Sealing publishes only SHA-256(salt + canonical JSON) with a UTC time; the record and its
salt stay private until the study's results are public, when both are revealed and anyone can recompute the digest.
The salt stops anyone from confirming a guessed forecast against the digest before the reveal.

Layout of a ledger directory:
    LEDGER.jsonl        public: one line per sealed forecast (id, digest, time, source)
    sealed/<id>.json    private until the reveal: record, salt, digest
    revealed/<id>.json  public after the reveal: the same file
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple


class Sealed(NamedTuple):
    digest: str
    salt: str


class Entry(NamedTuple):
    forecast_id: str
    digest: str
    sealed_at: str
    source: str


def canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest_of(record: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(salt.encode("ascii") + canonical(record)).hexdigest()


def seal(record: dict[str, Any]) -> Sealed:
    salt = secrets.token_hex(16)
    return Sealed(digest_of(record, salt), salt)


def verify(record: dict[str, Any], salt: str, digest: str) -> bool:
    return secrets.compare_digest(digest_of(record, salt), digest)


class Ledger:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.public = self.root / "LEDGER.jsonl"

    def ids(self) -> set[str]:
        if not self.public.exists():
            return set()
        return {json.loads(line)["forecast_id"] for line in self.public.read_text(encoding="utf-8").splitlines() if line}

    def seal(self, record: dict[str, Any], source: str) -> Entry:
        forecast_id = record["forecast_id"]
        if forecast_id in self.ids():
            raise ValueError(f"{forecast_id} is already sealed; a sealed forecast is never replaced")
        sealed = seal(record)
        entry = Entry(forecast_id, sealed.digest, datetime.now(UTC).isoformat(timespec="seconds"), source)
        (self.root / "sealed").mkdir(parents=True, exist_ok=True)
        private = {"record": record, "salt": sealed.salt, "digest": sealed.digest, "sealed_at": entry.sealed_at}
        (self.root / "sealed" / f"{forecast_id}.json").write_text(json.dumps(private, indent=2, ensure_ascii=False), encoding="utf-8")
        with self.public.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry._asdict(), ensure_ascii=False) + "\n")
        return entry

    def reveal(self, forecast_id: str) -> dict[str, Any]:
        private = json.loads((self.root / "sealed" / f"{forecast_id}.json").read_text(encoding="utf-8"))
        if not verify(private["record"], private["salt"], private["digest"]):
            raise ValueError(f"{forecast_id}: the sealed file no longer matches its digest")
        (self.root / "revealed").mkdir(parents=True, exist_ok=True)
        (self.root / "revealed" / f"{forecast_id}.json").write_text(json.dumps(private, indent=2, ensure_ascii=False), encoding="utf-8")
        return private
