"""Paired comparison of each federated condition against the baselines.

Reads ``data/benchmark_results.json`` (written by ``python -m app.benchmark``) and,
for every condition, prints the mean per-seed accuracy difference against the
local-only and centralized baselines, its standard error and the paired t
statistic. Baselines and conditions share seeds (hence splits), so pairing is valid.

With n seeds the two-sided 5 % critical value is the Student-t quantile with n-1
degrees of freedom (2.776 for n = 5). This is an exploratory summary of 5 runs on
303 rows, not evidence of a population effect.

    python scripts/paired_stats.py [path/to/benchmark_results.json]
"""
import json
import math
import pathlib
import statistics
import sys

# two-sided 5 % Student-t critical values by degrees of freedom
_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
           8: 2.306, 9: 2.262, 10: 2.228}


def _paired(diffs):
    n = len(diffs)
    mean = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(n) if n > 1 else float("nan")
    t = mean / se if se and not math.isnan(se) else float("nan")
    return mean, se, t


def main(path: pathlib.Path) -> int:
    data = json.loads(path.read_text())
    local = [r["accuracy"] for r in data["local_only_baseline"]["per_seed"]]
    central = [r["accuracy"] for r in data["centralized_baseline"]["per_seed"]]
    n = len(local)
    crit = _T_CRIT.get(n - 1)
    print(f"seeds={data['metadata']['seeds']}  git={data['metadata']['git_commit'][:10]} "
          f"dirty={data['metadata'].get('git_dirty')}")
    print(f"two-sided 5% critical |t| for df={n - 1}: {crit}")
    print(f"{'condition':<28}{'vs':<9}{'mean diff':>10}{'SE':>8}{'t':>7}  significant")
    for cond in data["ablation_full"]:
        accs = [r["final_global_accuracy"] for r in cond["seed_runs"]]
        for name, base in (("local", local), ("central", central)):
            mean, se, t = _paired([a - b for a, b in zip(accs, base)])
            sig = "yes" if crit and abs(t) > crit else "no"
            print(f"{cond['label']:<28}{name:<9}{mean:>10.4f}{se:>8.4f}{t:>7.2f}  {sig}")
    return 0


if __name__ == "__main__":
    default = pathlib.Path(__file__).resolve().parent.parent / "data" / "benchmark_results.json"
    sys.exit(main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default))
