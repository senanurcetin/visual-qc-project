"""Write the real-data CNN results into README.md from docs/data/neu-cls-dl/summary.json.

    python analysis/update_docs_from_dl.py

Only the block between the DL-RESULTS markers is rewritten. A smoke-test summary (synthetic data)
is refused, so a pipeline check can never be published as a result. tests/test_docs_consistency.py
fails if the block in README.md drifts from the summary file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUMMARY = ROOT / "docs" / "data" / "neu-cls-dl" / "summary.json"
BASELINE = ROOT / "docs" / "data" / "neu-cls-case-study" / "benchmark-comparison.json"
START, END = "<!-- DL-RESULTS:START -->", "<!-- DL-RESULTS:END -->"


class SmokeSummary(ValueError):
    pass


def render_block(summary: dict, rf: dict) -> str:
    """Markdown for the results block (no markers)."""
    if summary.get("smoke_test"):
        raise SmokeSummary("summary.json comes from --smoke-test (synthetic data); refusing to publish it")
    t, cal = summary["test"], summary["calibration"]
    lo, hi = t["accuracy_ci95"]
    inside = lo <= rf["accuracy"] <= hi
    verdict = (
        f"The Random Forest's accuracy ({rf['accuracy']:.3f}) lies inside the CNN's 95% interval, so on {summary['split']['test']} "
        "holdout samples the two are not clearly separated."
        if inside
        else f"The Random Forest's accuracy ({rf['accuracy']:.3f}) lies outside the CNN's 95% interval "
        f"({'below' if rf['accuracy'] < lo else 'above'} it)."
    )
    rows = [
        "| Model | Accuracy | Macro F1 | ECE before → after calibration |",
        "|-------|----------|----------|--------------------------------|",
        f"| Random Forest (HOG + Gabor + grid) | {rf['accuracy']:.3f} | {rf['macro_f1']:.3f} | n/a |",
        f"| {summary['architecture']}{'' if summary['pretrained'] else ' (no pretraining)'} | "
        f"{t['accuracy']:.3f} ({lo:.3f}–{hi:.3f}) | {t['macro_f1']:.3f} ({t['macro_f1_ci95'][0]:.3f}–{t['macro_f1_ci95'][1]:.3f}) | "
        f"{cal['ece_before']:.3f} → {cal['ece_after']:.3f} |",
    ]
    notes = [
        f"Same {summary['split']['test']}-row holdout for both models; brackets are bootstrap 95% intervals. "
        f"{summary['epochs']} epochs on {summary['device']} in {summary['train_seconds']:.0f} s, seed {summary['seed']}, "
        f"calibration temperature {cal['temperature']:.3f}.",
    ]
    cv = summary.get("cross_validation")
    if cv:
        notes.append(f"{cv['folds']}-fold cross-validation accuracy: {cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f}.")
    notes.append(verdict)
    return "\n".join([*rows, "", *notes])


def apply(readme: str, block: str) -> str:
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise ValueError(f"README.md must contain exactly one {START} … {END} pair")
    head, rest = readme.split(START)
    _, tail = rest.split(END)
    return f"{head}{START}\n{block}\n{END}{tail}"


def main() -> int:
    if not SUMMARY.exists():
        print(f"{SUMMARY.relative_to(ROOT)} not found: run analysis/run_dl_case_study.py first", file=sys.stderr)
        return 1
    rf = next(m for m in json.loads(BASELINE.read_text(encoding="utf-8")) if m["model"] == "random_forest")
    try:
        block = render_block(json.loads(SUMMARY.read_text(encoding="utf-8")), rf)
    except SmokeSummary as exc:
        print(exc, file=sys.stderr)
        return 2
    README.write_text(apply(README.read_text(encoding="utf-8"), block), encoding="utf-8")
    print("README.md updated. Review the diff, then commit README.md and docs/data/neu-cls-dl/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
