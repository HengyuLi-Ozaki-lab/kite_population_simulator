"""A3c: public metadata for the studies in the publication-status test, for the coders to start from.

For each study (the 91 seen and 25 unseen studies with a reliable human contrast) it collects:
- the OSF node's title and creation date (the study id is the TESS project's OSF id);
- the matching entry of the TESS past-studies index (by title), and from that study's TESS page its PIs,
  field period, which sections the page has, and the text of those sections - verified by the page's
  own link back to the OSF id;
- for studies the index does not match, the OSF wiki, which carries the PIs and field period.

Only extracted fields are written, to data/socsci210/tess/metadata.json (not committed); pages are not
stored. One request per second.

Usage:
    uv run python scripts/a3c_tess_metadata.py
"""

from __future__ import annotations

import difflib
import html
import json
import re
import time
import urllib.request
from pathlib import Path

from a3c_seen_runs import SEEN, scope

TEST = Path("data/socsci210/prepared/test")
OUT = Path("data/socsci210/tess/metadata.json")
INDEX = "https://www.tessexperiments.org/paststudies"
AGENT = "jev-kite-research/0.1 (metadata lookup, one request per second)"
_last = [0.0]


def get(url: str) -> str:
    for attempt in range(3):
        wait = 1.0 - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENT}), timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001 - retried, then raised
            if attempt == 2:
                raise
            print(f"  retry {url}: {error}")
            time.sleep(3)
    raise RuntimeError("unreachable")


def text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def norm(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", html.unescape(title).lower().replace("’", "'")).strip()


def parse_index(page: str) -> list[dict]:
    entries, year = [], None
    for token in re.finditer(r'<a name="(\d{4})"></a>|<div class = "study.*?</div>', page, re.S):
        if token.group(1):
            year = int(token.group(1))
            continue
        block = token.group(0)
        link = re.search(r'href="\./study/([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if link:
            pis = [text(p) for p in re.findall(r'<ul class="name[^"]*">(.*?)</ul>', block, re.S)]
            entries.append({"slug": link.group(1), "title": text(link.group(2)), "pis": pis, "index_year": year})
    return entries


def parse_study(page: str) -> dict:
    sections = {}
    parts = re.split(r'<div class="subheading[^"]*">([^<]+)</div>', page)
    for name, body in zip(parts[1::2], parts[2::2], strict=True):
        sections[text(name)] = text(re.split(r"<footer|</main>", body)[0])
    field = re.search(r"<strong>Field period: </strong>([^<]+)</p>", page)
    size = re.search(r"<strong>Sample size: </strong>([^<]+)</p>", page)
    title = re.search(r"<h2[^>]*>(.*?)</h2>", page, re.S)
    head = page[: page.find('class="subheading')] if 'class="subheading' in page else page
    people = [text(p) for p in re.findall(r'<p class = "font-bold mb-2">(.*?)</p>', head, re.S)]
    return {
        "tess_title": text(title.group(1)) if title else None,
        "osf_links": sorted(set(re.findall(r"osf\.io/([a-z0-9]{5})", page))),
        "field_period": field.group(1).strip() if field else None,
        "sample_size": size.group(1).strip() if size else None,
        "people": [p for p in people if p],
        "sections": sections,
    }


def main() -> None:
    seen_cells, seen = scope(SEEN)
    test_cells, test = scope(TEST)
    studies = {s: {"split": "seen", "reliable_contrasts": n} for s, n in seen.items()}
    studies |= {s: {"split": "unseen", "reliable_contrasts": n} for s, n in test.items()}
    print(f"{len(seen)} seen + {len(test)} unseen studies")

    index = parse_index(get(INDEX))
    print(f"index: {len(index)} past studies")
    by_title = {norm(e["title"]): e for e in index}

    for study, row in sorted(studies.items()):
        node = json.loads(get(f"https://api.osf.io/v2/nodes/{study}/"))["data"]["attributes"]
        row |= {"osf_title": node["title"], "osf_created": node["date_created"][:10], "osf_tags": node.get("tags", [])}
        best = difflib.get_close_matches(norm(node["title"]), list(by_title), n=1, cutoff=0.85)
        if best:
            entry = by_title[best[0]]
            page = parse_study(get(f"https://www.tessexperiments.org/study/{entry['slug']}"))
            ratio = difflib.SequenceMatcher(None, norm(node["title"]), best[0]).ratio()
            row |= {"tess_slug": entry["slug"], "index_year": entry["index_year"], "index_pis": entry["pis"], "title_match": round(ratio, 3), **page}
            row["tess_verified"] = study in page["osf_links"]
        else:
            row["tess_slug"] = None
        if not row.get("tess_verified"):  # the OSF wiki carries PIs and field period when no TESS page can be tied to the id
            wikis = json.loads(get(f"https://api.osf.io/v2/nodes/{study}/wikis/"))["data"]
            row["osf_wiki"] = text(get(f"https://api.osf.io/v2/wikis/{wikis[0]['id']}/content/")) if wikis else None
        print(f"  {study} {row['split']:6} {'TESS ' + row['tess_slug'] if row.get('tess_slug') else 'no index match':28} {row['osf_title'][:60]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(studies, indent=1, ensure_ascii=False))
    matched = sum(bool(r.get("tess_slug")) for r in studies.values())
    verified = sum(bool(r.get("tess_verified")) for r in studies.values())
    summary = sum("Summary of Results" in r.get("sections", {}) for r in studies.values())
    print(f"wrote {OUT}: {matched} matched to a TESS page ({verified} verified by the OSF link), {summary} pages with a results summary")


if __name__ == "__main__":
    main()
