"""A3c: reconcile the two blind codings into the labels the publication-status test uses.

Follows docs/data/tess-publication-coding.md:

    agreement   - agreement between coder A and coder B, before anything is reconciled, and the list of
                  disagreements for the third agent (data/socsci210/tess/coding/disagreements.json)
    finalize    - agreed codes plus the third agent's decisions (adjudication.json), OpenAlex citation
                  counts for accepted publications, written to docs/data/tess_publication_status.csv

Usage:
    uv run python scripts/a3c_reconcile_labels.py agreement
    uv run python scripts/a3c_reconcile_labels.py finalize
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

CODING = Path("data/socsci210/tess/coding")
METADATA = Path("data/socsci210/tess/metadata.json")
LABELS = Path("docs/data/tess_publication_status.csv")
CUTOFF = 2025


def load(coder: str) -> dict[str, dict]:
    out = {}
    for path in sorted(CODING.glob(f"{coder}[0-9].json")):
        for row in json.loads(path.read_text()):
            out[row["study_id"]] = row
    return out


def year_of(row: dict) -> int | None:
    year = row.get("earliest_public_year") or (row.get("citation") or {}).get("year")
    return int(year) if year else None


def public(row: dict) -> bool:
    published = row.get("publication") == "yes" and (year_of(row) or 0) < CUTOFF
    return bool(row.get("tess_page_reports_findings")) or published


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def agreement() -> None:
    meta = json.loads(METADATA.read_text())
    a, b = load("A"), load("B")
    missing = {"A": sorted(set(meta) - set(a)), "B": sorted(set(meta) - set(b))}
    ids = sorted(set(a) & set(b))
    stats = {}
    for name, f in (
        ("results_public", public),
        ("tess_page_reports_findings", lambda r: bool(r.get("tess_page_reports_findings"))),
        ("publication_yes", lambda r: r.get("publication") == "yes"),
    ):
        x, y = [f(a[i]) for i in ids], [f(b[i]) for i in ids]
        stats[name] = {"percent": sum(p == q for p, q in zip(x, y, strict=True)) / len(ids), "kappa": kappa(x, y), "A_true": sum(x), "B_true": sum(y)}
    disagreements = []
    for i in ids:
        reasons = []
        if bool(a[i].get("tess_page_reports_findings")) != bool(b[i].get("tess_page_reports_findings")):
            reasons.append("tess_page_reports_findings")
        if (a[i].get("publication") == "yes") != (b[i].get("publication") == "yes"):
            reasons.append("publication")
        elif a[i].get("publication") == "yes" and (year_of(a[i]) or 0) < CUTOFF and (year_of(b[i]) or 0) >= CUTOFF:
            reasons.append("publication year")
        if a[i].get("publication") == b[i].get("publication") == "unclear":
            reasons.append("both unclear")
        if reasons:
            disagreements.append(
                {
                    "study_id": i,
                    "reasons": reasons,
                    "metadata": {k: meta[i].get(k) for k in ("osf_title", "tess_slug", "index_pis", "field_period")},
                    "coder_A": a[i],
                    "coder_B": b[i],
                }
            )
    (CODING / "disagreements.json").write_text(json.dumps(disagreements, indent=1, ensure_ascii=False))
    (CODING / "agreement.json").write_text(json.dumps({"studies": len(ids), "missing": missing, "agreement": stats}, indent=1))
    print(f"{len(ids)} studies coded by both; missing {missing}")
    for name, s in stats.items():
        print(f"  {name:28} agreement {s['percent']:.1%}  kappa {s['kappa']:.2f}  (A true {s['A_true']}, B true {s['B_true']})")
    print(f"{len(disagreements)} disagreements written to {CODING / 'disagreements.json'}")


def fetch(url: str) -> dict:
    """GET with one request per second, waiting out OpenAlex's rate limit (HTTP 429) instead of giving up."""
    for attempt in range(4):
        time.sleep(1 + 5 * attempt)
        try:
            return json.loads(urllib.request.urlopen(url, timeout=30).read())
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 3:
                raise
    raise RuntimeError("unreachable")


def openalex(citation: dict) -> int | None:
    norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())  # noqa: E731
    try:
        if citation.get("doi"):
            doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", citation["doi"]).strip()
            return fetch(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}")["cited_by_count"]
        words = re.sub(r"[^A-Za-z0-9 ]", " ", citation["title"])  # commas and colons break the filter syntax
        results = fetch("https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"title.search:{words}", "per-page": 5}))["results"]
        exact = [r for r in results if norm(r.get("title")) == norm(citation["title"])]
        return exact[0]["cited_by_count"] if exact else None
    except Exception as error:  # noqa: BLE001 - a missing count is recorded as missing
        print(f"  OpenAlex lookup failed for {citation.get('title', '')[:60]}: {error}")
        return None


def finalize() -> None:
    meta = json.loads(METADATA.read_text())
    a, b = load("A"), load("B")
    decided = (
        {row["study_id"]: row for row in json.loads((CODING / "adjudication.json").read_text())} if (CODING / "adjudication.json").exists() else {}
    )
    rows = []
    for i in sorted(meta):
        m, x, y = meta[i], a.get(i, {}), b.get(i, {})
        if i in decided:
            final, source = decided[i], "adjudicated"
        else:
            final, source = (x if x.get("publication") == "yes" else y if y.get("publication") == "yes" else x), "agreed"
        unclear = x.get("publication") == y.get("publication") == "unclear"
        publication = "yes" if final.get("publication") == "yes" else "no"
        pub_year = year_of(final) if publication == "yes" else None
        published = publication == "yes" and (pub_year or 0) < CUTOFF
        page = bool(final.get("tess_page_reports_findings"))
        citation = final.get("citation") or {} if publication == "yes" else {}
        start = re.search(r"(\d{4})", m.get("field_period") or "")
        rows.append({
            "study_id": i, "split": m["split"], "reliable_contrasts": m["reliable_contrasts"], "osf_title": m["osf_title"],
            "tess_url": f"https://www.tessexperiments.org/study/{m['tess_slug']}" if m.get("tess_slug") else "",
            "field_start_year": int(start.group(1)) if start else None,
            "tess_page_reports_findings": page, "publication": publication, "pub_year": pub_year,
            "pub_authors": citation.get("authors", ""), "pub_title": citation.get("title", ""), "pub_venue": citation.get("venue", ""),
            "pub_url": citation.get("url", ""), "pub_doi": citation.get("doi") or "",
            "citations": openalex(citation) if published and citation.get("title") else None,
            "results_public": page or published, "published": published, "unclear": unclear,
            "coder_a_public": public(x) if x else None, "coder_b_public": public(y) if y else None, "reconciliation": source,
        })  # fmt: skip
    frame = pd.DataFrame(rows)
    frame.to_csv(LABELS, index=False)
    for split, g in frame.groupby("split"):
        counts = {k: int(g[k].sum()) for k in ("results_public", "published", "unclear")}
        print(f"{split}: {len(g)} studies; {counts}")
    print(f"wrote {LABELS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["agreement", "finalize"])
    args = parser.parse_args()
    agreement() if args.step == "agreement" else finalize()


if __name__ == "__main__":
    main()
