"""Check that our OpinionQA human distributions equal the ones the benchmark's authors compute.

Usage:
    git clone https://github.com/tatsu-lab/opinions_qa data/opinionqa/repo
    uv run python scripts/check_opinionqa_parity.py data/opinionqa data/opinionqa/repo --wave 26

We import the authors' `helpers.extract_human_opinions` from the clone (their repo has no license, so
we do not copy it) and compare every (question, attribute, group) distribution with ours.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kite.eval.opinionqa import OVERALL, human_distributions, load_wave, wave_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="unpacked OpinionQA bundle (contains human_resp/)")
    parser.add_argument("repo", type=Path, help="clone of tatsu-lab/opinions_qa")
    parser.add_argument("--wave", type=int, default=26)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    import helpers as authors  # noqa: PLC0415

    questions, groups, responses = load_wave(args.root, args.wave)
    ours = human_distributions(questions, groups, responses, args.wave)

    folder = wave_dir(args.root, args.wave)
    info = pd.read_csv(folder / "info.csv").set_index("key")
    metadata = pd.read_csv(folder / "metadata.csv")
    metadata["options"] = metadata["options"].map(ast.literal_eval)
    stub = pd.DataFrame(
        {
            "qkey": [question.key for question in questions],
            "ordinal_refs": [question.options for question in questions],
            "refusal_refs": [ast.literal_eval(info.loc[question.key, "references"])[len(question.options) :] for question in questions],
            "ordinal": [question.ordinal for question in questions],
        }
    )

    compared, worst = 0, 0.0
    for attribute in [OVERALL, *groups]:
        theirs = authors.extract_human_opinions(responses, stub, metadata, demographic=attribute, wave=args.wave)
        for row in theirs.itertuples(index=False):
            mine = ours[(row.qkey, row.attribute, str(row.group))]
            worst = max(worst, float(np.abs(mine - np.asarray(row.D_H, dtype=float)).max()))
            compared += 1

    print(f"wave {args.wave}: compared {compared} distributions (ours has {len(ours)}); largest absolute difference {worst:.2e}")
    return 0 if compared == len(ours) and worst < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
