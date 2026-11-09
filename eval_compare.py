#!/usr/bin/env python3
"""Compare two eval runs case by case.

Input: two JSONL files, one object per line, each with an "id" and a boolean "pass".
Both files must cover exactly the same ids. That is what makes the comparison paired.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from math import comb
from pathlib import Path


def load(path: str) -> dict[str, bool]:
    rows: dict[str, bool] = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f"{path}:{number}: duplicate id {row['id']!r}")
        rows[row["id"]] = bool(row["pass"])
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. Only the cases where the runs disagree count."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def paired_interval(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Agresti-Min interval for the difference in matched proportions: add 0.5 to each cell."""
    bt, ct, nt = b + 0.5, c + 0.5, n + 2
    diff = (bt - ct) / nt
    se = math.sqrt(max(bt + ct - (bt - ct) ** 2 / nt, 0)) / nt
    return max(-1.0, diff - z * se), min(1.0, diff + z * se)


def required_cases(
    effect: float, discordant: float, alpha_z: float = 1.96, power_z: float = 0.8416
) -> int:
    """Cases needed to detect `effect` (a difference in pass rate) with a paired test.

    `discordant` is the share of cases where the two runs disagree. Connor (1987).
    """
    if not 0 < abs(effect) <= discordant <= 1:
        raise ValueError("need 0 < |effect| <= discordant <= 1")
    root = alpha_z * math.sqrt(discordant) + power_z * math.sqrt(discordant - effect**2)
    return math.ceil((root / effect) ** 2)


def compare(a: dict[str, bool], b: dict[str, bool], alpha: float = 0.05) -> dict:
    if a.keys() != b.keys():
        missing = sorted(a.keys() ^ b.keys())[:5]
        raise ValueError(f"runs cover different cases, e.g. {missing}; a paired test needs both")
    if not a:
        raise ValueError("no cases to compare")
    gained = sorted(k for k in a if b[k] and not a[k])
    lost = sorted(k for k in a if a[k] and not b[k])
    n = len(a)
    p = mcnemar_exact(len(gained), len(lost))
    low, high = paired_interval(len(gained), len(lost), n)
    diff = (len(gained) - len(lost)) / n
    if p < alpha:
        verdict = "b_better" if diff > 0 else "b_worse"
    else:
        verdict = "no_detectable_difference"
    return {
        "n": n,
        "pass_a": sum(a.values()) / n,
        "pass_b": sum(b.values()) / n,
        "diff": diff,
        "ci_low": low,
        "ci_high": high,
        "p": p,
        "gained": gained,
        "lost": lost,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 only when the candidate is significantly worse",
    )
    args = parser.parse_args(argv)
    r = compare(load(args.baseline), load(args.candidate), args.alpha)
    print(f"cases           {r['n']}")
    print(f"baseline        {r['pass_a']:.1%}")
    print(f"candidate       {r['pass_b']:.1%}")
    print(f"difference      {r['diff']:+.1%}  (95% CI {r['ci_low']:+.1%} to {r['ci_high']:+.1%})")
    print(f"flips           +{len(r['gained'])} gained, -{len(r['lost'])} lost")
    print(f"McNemar exact p {r['p']:.4f}")
    print(f"verdict         {r['verdict']}")
    for label, ids in (("lost", r["lost"]), ("gained", r["gained"])):
        if ids:
            print(f"{label}: {', '.join(ids[:20])}{' ...' if len(ids) > 20 else ''}")
    return 1 if args.fail_on_regression and r["verdict"] == "b_worse" else 0


if __name__ == "__main__":
    sys.exit(main())
