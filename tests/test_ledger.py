import json

import pytest

from kite.ledger import Ledger, canonical, digest_of, seal, verify

RECORD = {
    "forecast_id": "tess-demo-001",
    "study": {"source": "TESS", "title": "demo"},
    "conditions": [{"id": "control", "mean": 0.41}, {"id": "treatment", "mean": 0.47}],
    "models": {"kernel": "jev-1.13.0"},
}


def test_canonical_json_ignores_key_order():
    shuffled = json.loads(json.dumps(RECORD))
    shuffled = {key: shuffled[key] for key in reversed(list(shuffled))}
    assert canonical(RECORD) == canonical(shuffled)


def test_seal_and_verify_round_trip():
    sealed = seal(RECORD)
    assert len(sealed.salt) == 32 and len(sealed.digest) == 64
    assert verify(RECORD, sealed.salt, sealed.digest)


def test_any_change_breaks_the_seal():
    sealed = seal(RECORD)
    changed = json.loads(json.dumps(RECORD))
    changed["conditions"][1]["mean"] = 0.48
    assert not verify(changed, sealed.salt, sealed.digest)
    assert not verify(RECORD, "0" * 32, sealed.digest)


def test_salts_differ_so_equal_records_do_not_share_a_digest():
    assert seal(RECORD).digest != seal(RECORD).digest
    assert digest_of(RECORD, "ab" * 16) == digest_of(RECORD, "ab" * 16)


def test_ledger_appends_and_reveals(tmp_path):
    ledger = Ledger(tmp_path)
    entry = ledger.seal(RECORD, source="TESS embargo list")
    assert (tmp_path / "LEDGER.jsonl").read_text(encoding="utf-8").count("\n") == 1
    public = json.loads((tmp_path / "LEDGER.jsonl").read_text(encoding="utf-8"))
    assert public == {"forecast_id": "tess-demo-001", "digest": entry.digest, "sealed_at": entry.sealed_at, "source": "TESS embargo list"}
    assert "salt" not in public
    with pytest.raises(ValueError):
        ledger.seal(RECORD, source="again")
    revealed = ledger.reveal("tess-demo-001")
    assert verify(revealed["record"], revealed["salt"], revealed["digest"])
    assert (tmp_path / "revealed" / "tess-demo-001.json").exists()
