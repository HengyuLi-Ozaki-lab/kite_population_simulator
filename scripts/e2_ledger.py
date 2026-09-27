"""E2: seal, reveal and verify prospective forecasts (research plan v5 §4; protocol in forecasts/README.md).

uv run python scripts/e2_ledger.py seal   <record.json> --source "<where the study came from>"
uv run python scripts/e2_ledger.py reveal <forecast_id>      # only after the study's results are public
uv run python scripts/e2_ledger.py verify <forecast_id>      # recompute a revealed digest against LEDGER.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kite.ledger import Ledger, verify

ROOT = Path("forecasts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("record", type=Path)
    seal.add_argument("--source", required=True)
    sub.add_parser("reveal").add_argument("forecast_id")
    sub.add_parser("verify").add_argument("forecast_id")
    args = parser.parse_args()
    ledger = Ledger(ROOT)

    if args.command == "seal":
        entry = ledger.seal(json.loads(args.record.read_text(encoding="utf-8")), source=args.source)
        print(f"sealed {entry.forecast_id} at {entry.sealed_at}\n  digest {entry.digest}")
    elif args.command == "reveal":
        ledger.reveal(args.forecast_id)
        print(f"revealed {args.forecast_id} -> {ROOT / 'revealed' / (args.forecast_id + '.json')}")
    else:
        line = next(json.loads(x) for x in ledger.public.read_text(encoding="utf-8").splitlines() if json.loads(x)["forecast_id"] == args.forecast_id)
        revealed = json.loads((ROOT / "revealed" / f"{args.forecast_id}.json").read_text(encoding="utf-8"))
        ok = revealed["digest"] == line["digest"] and verify(revealed["record"], revealed["salt"], line["digest"])
        print(f"{args.forecast_id}: {'matches' if ok else 'DOES NOT MATCH'} the ledger line sealed at {line['sealed_at']}")


if __name__ == "__main__":
    main()
