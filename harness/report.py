"""Report generation from a completed run's results.

Consumes the results.json structure produced by main.py and emits:
  - a per-case CSV (easy to load into a spreadsheet)
  - a human-readable Markdown summary with a leaderboard
"""

import csv
import os
from tabulate import tabulate


def _fmt_size(num_bytes: int) -> str:
    mb = num_bytes / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def write_cases_csv(results: dict, path: str) -> None:
    """Flat per-case CSV: one row per (model, test, case)."""
    fields = [
        "model", "test", "case", "passed", "reason",
        "completion_tokens", "tokens_per_s", "wall_time_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model in results["models"]:
            for test in model["tests"]:
                for case in test["cases"]:
                    stats = case.get("stats", {})
                    writer.writerow({
                        "model": model["model_name"],
                        "test": test["test_id"],
                        "case": case["case_id"],
                        "passed": case["passed"],
                        "reason": case["reason"],
                        "completion_tokens": stats.get("completion_tokens", 0),
                        "tokens_per_s": stats.get("tokens_per_s", 0),
                        "wall_time_s": stats.get("wall_time_s", 0),
                    })


def _model_summary_rows(results: dict):
    """Build leaderboard rows sorted by all-pass then size then speed."""
    rows = []
    for model in results["models"]:
        total = 0
        passed = 0
        per_test = {}
        tps_values = []
        for test in model["tests"]:
            t_pass = sum(1 for c in test["cases"] if c["passed"])
            t_total = len(test["cases"])
            per_test[test["test_id"]] = f"{t_pass}/{t_total}"
            total += t_total
            passed += t_pass
            for c in test["cases"]:
                tps = c.get("stats", {}).get("tokens_per_s", 0)
                if tps:
                    tps_values.append(tps)

        avg_tps = round(sum(tps_values) / len(tps_values), 1) if tps_values else 0
        all_pass = (passed == total and total > 0)
        rows.append({
            "model": model["model_name"],
            "size_bytes": model.get("size_bytes", 0),
            "all_pass": all_pass,
            "score": f"{passed}/{total}",
            "passed": passed,
            "total": total,
            "per_test": per_test,
            "avg_tps": avg_tps,
            "load_time_s": round(model.get("load_time_s", 0), 2),
        })

    # Best = passes everything, then smallest, then fastest.
    rows.sort(key=lambda r: (not r["all_pass"], r["size_bytes"], -r["avg_tps"]))
    return rows


def write_markdown(results: dict, path: str) -> str:
    rows = _model_summary_rows(results)
    test_ids = results.get("test_ids", [])

    header = ["Rank", "Model", "Size", "All Pass", "Score"]
    header += [t for t in test_ids]
    header += ["Avg tok/s", "Load (s)"]

    table = []
    for i, r in enumerate(rows, 1):
        line = [
            i,
            r["model"],
            _fmt_size(r["size_bytes"]),
            "YES" if r["all_pass"] else "no",
            r["score"],
        ]
        line += [r["per_test"].get(t, "-") for t in test_ids]
        line += [r["avg_tps"], r["load_time_s"]]
        table.append(line)

    md = []
    md.append(f"# CTF Small-Model Benchmark Report\n")
    md.append(f"- Run ID: `{results.get('run_id', 'n/a')}`")
    md.append(f"- Flag: `{results.get('flag', '')}`")
    md.append(f"- Models tested: {len(rows)}")
    md.append(f"- Tests: {', '.join(test_ids)}\n")

    # Best model callout.
    best = next((r for r in rows if r["all_pass"]), None)
    if best:
        md.append(f"## Best model (passes all, smallest)\n")
        md.append(f"**{best['model']}** — {_fmt_size(best['size_bytes'])}, "
                  f"{best['avg_tps']} tok/s avg, score {best['score']}\n")
    else:
        md.append("## Best model\n\n_No model passed all tests._\n")

    md.append("## Leaderboard\n")
    md.append(tabulate(table, headers=header, tablefmt="github"))
    md.append("")

    content = "\n".join(md)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def generate(results: dict, results_dir: str) -> None:
    csv_path = os.path.join(results_dir, "cases.csv")
    md_path = os.path.join(results_dir, "report.md")
    write_cases_csv(results, csv_path)
    content = write_markdown(results, md_path)
    print("\n" + content)
    print(f"\nArtifacts written to: {results_dir}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
