# eval-compare

Compare two LLM eval runs case by case. Exact McNemar test, a paired confidence interval, sample-size planning and a CI gate, in one standard-library Python file.

Companion to [Your Prompt Didn't Get Better. Your Eval Set Is Just Small.](https://cloudandsre.com/blog/comparing-llm-evals-paired-tests/)

## Use it

Each run is a JSONL file with one object per case. Both files must cover the same ids.

```json
{"id": "case-001", "pass": true}
```

```bash
python eval_compare.py baseline.jsonl candidate.jsonl
```

```text
cases           200
baseline        80.0%
candidate       83.5%
difference      +3.5%  (95% CI -2.3% to +9.3%)
flips           +21 gained, -14 lost
McNemar exact p 0.3105
verdict         no_detectable_difference
```

Add `--fail-on-regression` in CI. It exits 1 only when the candidate is significantly worse, because at realistic eval sizes a candidate can rarely prove that it's better.

## How many cases do you need?

```python
from eval_compare import required_cases

required_cases(effect=0.03, discordant=0.20)  # 1742
```

`discordant` is the share of cases where the two runs disagree. For small effects, `n ≈ 7.85 × D / d²`.

## The simulations

`simulate.py` (needs numpy, seeded) produces [`results/simulation.json`](results/simulation.json):

- Identical prompts look at least 3 points better in 31.9% of runs on 50 cases.
- A real 3-point gain is detected 6.1% of the time on 100 cases and 51.3% on 1,000.
- The best of 20 variants identical to the baseline shows +5.9 points on 100 cases.

Case outcomes follow a Beta model with a mean of 80%. It assumes binary grading, independent cases and a perfect grader.

## Development

```bash
pip install -r requirements-ci.txt
pytest -q && ruff check .
```

## License

Apache-2.0
