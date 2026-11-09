#!/usr/bin/env python3
"""The three simulations behind the blog post. Results are written to results/.

Case model: each eval case has its own pass probability drawn from a Beta distribution with a
given mean. Low concentration means cases are mostly always-pass or always-fail; high concentration
means every case is equally flaky. Two runs over the same cases are therefore correlated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_compare import mcnemar_exact, required_cases

SIZES = [20, 50, 100, 200, 500, 1000]
MEAN = 0.8


def cases(rng, sims, n, conc):
    return rng.beta(MEAN * conc, (1 - MEAN) * conc, (sims, n))


def mcnemar_many(b, c):
    pairs, inverse = np.unique(np.stack([b, c], 1), axis=0, return_inverse=True)
    table = np.array([mcnemar_exact(int(x), int(y)) for x, y in pairs])
    return table[inverse.ravel()]


def unpaired_z(a_pass, b_pass, n):
    pooled = (a_pass + b_pass) / (2 * n)
    se = np.sqrt(2 * pooled * (1 - pooled) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(b_pass - a_pass) / n / se
    return np.nan_to_num(z) > 1.96


def run(rng, p, q):
    a = rng.random(p.shape) < p
    b = rng.random(p.shape) < q
    return a, b


def identical_prompts(rng, sims, conc=2.0):
    rows = []
    for n in SIZES:
        p = cases(rng, sims, n, conc)
        a, b = run(rng, p, p)
        diff = b.mean(1) - a.mean(1)
        sig = mcnemar_many((b & ~a).sum(1), (a & ~b).sum(1)) < 0.05
        rows.append(
            {
                "n": n,
                "looks_3pt_better": float((diff >= 0.03).mean()),
                "looks_5pt_different": float((np.abs(diff) >= 0.05).mean()),
                "mcnemar_false_positive": float(sig.mean()),
            }
        )
    return rows


def real_improvement(rng, sims, effect=0.03):
    rows = []
    for conc in (0.5, 2.0, 20.0):
        for n in SIZES:
            p = cases(rng, sims, n, conc)
            q = p + (effect / (1 - MEAN)) * (1 - p)
            a, b = run(rng, p, q)
            gained, lost = (b & ~a).sum(1), (a & ~b).sum(1)
            rows.append(
                {
                    "concentration": conc,
                    "n": n,
                    "power_mcnemar": float(
                        ((mcnemar_many(gained, lost) < 0.05) & (gained > lost)).mean()
                    ),
                    "power_unpaired": float(
                        (unpaired_z(a.sum(1), b.sum(1), n) & (b.sum(1) > a.sum(1))).mean()
                    ),
                    "discordant_rate": float(((gained + lost) / n).mean()),
                }
            )
    return rows


def winners_curse(rng, sims, candidates=20, conc=2.0):
    rows = []
    for n in (50, 100, 200, 500):
        p = cases(rng, sims, n, conc)
        base = rng.random(p.shape) < p
        best_gain, naive_sig, corrected_sig = [], 0, 0
        for s in range(sims):
            runs = rng.random((candidates, n)) < p[s]
            score = runs.mean(1)
            winner = int(score.argmax())
            best_gain.append(score[winner] - base[s].mean())
            pv = mcnemar_exact(
                int((runs[winner] & ~base[s]).sum()), int((base[s] & ~runs[winner]).sum())
            )
            naive_sig += pv < 0.05 and score[winner] > base[s].mean()
            corrected_sig += pv < 0.05 / candidates and score[winner] > base[s].mean()
        # the holdout gain is zero by construction: every candidate has the baseline's true rate
        rows.append(
            {
                "n": n,
                "candidates": candidates,
                "apparent_gain_points": float(np.mean(best_gain) * 100),
                "true_gain_points": 0.0,
                "uncorrected_significant": naive_sig / sims,
                "bonferroni_significant": corrected_sig / sims,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small run for CI")
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    sims = 500 if args.quick else 20000
    out = {
        "seed": args.seed,
        "sims": sims,
        "identical_prompts": identical_prompts(rng, sims),
        "real_improvement_3pt": real_improvement(rng, sims),
        "winners_curse": winners_curse(rng, min(sims, 3000)),
        "required_cases": [
            {"effect_points": e, "discordant": d, "cases": required_cases(e / 100, d)}
            for e in (1, 2, 3, 5, 10)
            for d in (0.1, 0.2, 0.4)
            if e / 100 <= d
        ],
    }
    if args.quick:
        print("quick run ok:", {k: len(v) for k, v in out.items() if isinstance(v, list)})
        return
    Path("results").mkdir(exist_ok=True)
    Path("results/simulation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
