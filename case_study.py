from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, abort, jsonify, render_template_string

ARTIFACT_DIR = Path(__file__).resolve().parent / "docs" / "data" / "neu-cls-case-study"

CASE_STUDY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual QC Case Study</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #07111f;
            --panel: #0f1c2d;
            --panel-soft: #13253b;
            --border: #214166;
            --text: #e6efff;
            --muted: #8da4c4;
            --accent: #6ee7b7;
            --accent-2: #60a5fa;
            --danger: #fca5a5;
            --warning: #fde68a;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: 'Inter', sans-serif;
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.18), transparent 32%),
                radial-gradient(circle at top right, rgba(110, 231, 183, 0.16), transparent 28%),
                var(--bg);
            color: var(--text);
        }
        a { color: var(--accent); text-decoration: none; }
        .page {
            max-width: 1180px;
            margin: 0 auto;
            padding: 32px 20px 64px;
        }
        .hero, .panel {
            background: rgba(15, 28, 45, 0.92);
            border: 1px solid var(--border);
            border-radius: 20px;
            backdrop-filter: blur(14px);
        }
        .hero {
            padding: 28px;
            display: grid;
            gap: 22px;
            margin-bottom: 22px;
        }
        .eyebrow {
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        h1, h2, h3, p { margin: 0; }
        h1 { font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.05; }
        .lede {
            max-width: 760px;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
        }
        .hero-grid, .grid {
            display: grid;
            gap: 16px;
        }
        .hero-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
        .metric {
            padding: 18px;
            background: rgba(19, 37, 59, 0.85);
            border: 1px solid rgba(96, 165, 250, 0.14);
            border-radius: 16px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
        }
        .metric-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.7rem;
            font-weight: 500;
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 8px;
            line-height: 1.45;
        }
        .grid {
            grid-template-columns: 1.15fr 0.85fr;
            align-items: start;
        }
        .panel {
            padding: 22px;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 14px;
        }
        .section-copy, .bullet-list li {
            color: var(--muted);
            line-height: 1.62;
        }
        .bullet-list {
            margin: 0;
            padding-left: 18px;
            display: grid;
            gap: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }
        th, td {
            padding: 10px 0;
            border-bottom: 1px solid rgba(141, 164, 196, 0.18);
            text-align: left;
        }
        th {
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        td strong {
            color: var(--text);
        }
        .mono {
            font-family: 'IBM Plex Mono', monospace;
        }
        .taxonomy {
            display: grid;
            gap: 12px;
        }
        .taxonomy-item {
            padding: 14px;
            border-radius: 14px;
            border: 1px solid rgba(141, 164, 196, 0.14);
            background: rgba(19, 37, 59, 0.6);
        }
        .taxonomy-item h3 {
            font-size: 0.98rem;
            margin-bottom: 6px;
        }
        .taxonomy-item p {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .footer-note {
            margin-top: 22px;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.65;
        }
        .danger { color: var(--danger); }
        .warning { color: var(--warning); }
        @media (max-width: 920px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <main class="page">
        <section class="hero">
            <div>
                <div class="eyebrow">Support Case Study</div>
                <h1>{{ summary.project }}</h1>
            </div>
            <p class="lede">{{ summary.operational_takeaway }}</p>
            <div class="hero-grid">
                <div class="metric">
                    <div class="metric-label">Final Model</div>
                    <div class="metric-value">{{ summary.final_model.macro_f1 }}</div>
                    <div class="metric-note">Macro-F1 from a deterministic 80/20 holdout on {{ summary.dataset.total_images }} NEU-CLS images.</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{{ summary.final_model.accuracy }}</div>
                    <div class="metric-note">Six-class steel surface defect triage with a free-tier friendly classical CV pipeline.</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Review Queue Lift</div>
                    <div class="metric-value">{{ summary.review_queue.yield_lift_vs_random }}x</div>
                    <div class="metric-note">{{ summary.review_queue.selected_budget_label }} capture more routing errors than random spot-checking.</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Holdout Coverage</div>
                    <div class="metric-value">{{ summary.dataset.evaluation_split.holdout_samples }}</div>
                    <div class="metric-note">Balanced validation set sampled with a fixed seed for reproducibility.</div>
                </div>
            </div>
        </section>

        <section class="grid">
            <div class="panel">
                <h2 class="section-title">What this proves</h2>
                <ul class="bullet-list">
                    <li>This repo is no longer only a simulated dashboard. It now includes a measurable computer-vision benchmark tied to a real steel surface defect dataset.</li>
                    <li>The workflow story remains practical: low-confidence predictions can be routed into a human review queue instead of pretending the classifier is fully autonomous.</li>
                    <li>The pipeline stays zero-cost friendly and reproducible: classical HOG features, scikit-learn models, JSON artifacts, and a Flask surface that can run on free infrastructure.</li>
                </ul>
                <p class="footer-note">Dataset: <a href="{{ summary.dataset.source_url }}">{{ summary.dataset.name }}</a> under {{ summary.dataset.license }}. The raw dataset is not committed to Git; it is downloaded on demand by the analysis script.</p>
            </div>
            <div class="panel">
                <h2 class="section-title">Benchmark Comparison</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Macro-F1</th>
                            <th>Train Time</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for row in benchmarks %}
                        <tr>
                            <td><strong>{{ row.model }}</strong></td>
                            <td class="mono">{{ row.accuracy }}</td>
                            <td class="mono">{{ row.macro_f1 }}</td>
                            <td class="mono">{{ row.training_seconds }}s</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </section>

        <section class="grid" style="margin-top: 18px;">
            <div class="panel">
                <h2 class="section-title">Review Queue Design</h2>
                <p class="section-copy" style="margin-bottom: 14px;">The classifier exposes a useful operator story: route the least-confident predictions to a human reviewer, rather than reviewing defects at random.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Review Budget</th>
                            <th>Errors Captured</th>
                            <th>Capture Rate</th>
                            <th>Yield Lift</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for row in review_queue.review_budgets %}
                        <tr>
                            <td class="mono">{{ (row.review_fraction * 100) | int }}%</td>
                            <td class="mono">{{ row.captured_errors }} / {{ row.reviewed_samples }}</td>
                            <td class="mono">{{ row.error_capture_rate }}</td>
                            <td class="mono">{{ row.yield_lift_vs_random }}x</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
            <div class="panel">
                <h2 class="section-title">Confusion Hotspots</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Actual</th>
                            <th>Predicted</th>
                            <th>Count</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for row in confusion.hotspots %}
                        <tr>
                            <td>{{ row.actual }}</td>
                            <td>{{ row.predicted }}</td>
                            <td class="mono">{{ row.count }}</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </section>

        <section class="grid" style="margin-top: 18px;">
            <div class="panel">
                <h2 class="section-title">Defect Taxonomy</h2>
                <div class="taxonomy">
                {% for item in taxonomy %}
                    <div class="taxonomy-item">
                        <h3>{{ item.class_name }}</h3>
                        <p>{{ item.description }}</p>
                    </div>
                {% endfor %}
                </div>
            </div>
            <div class="panel">
                <h2 class="section-title">Per-Class Metrics</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Class</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for row in class_metrics %}
                        <tr>
                            <td>{{ row.class_name }}</td>
                            <td class="mono">{{ row.precision }}</td>
                            <td class="mono">{{ row.recall }}</td>
                            <td class="mono">{{ row.f1 }}</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
                <p class="footer-note">Selection note: {{ model_selection.selection_reason }}</p>
            </div>
        </section>

        <section class="panel" style="margin-top: 18px;">
            <h2 class="section-title">Limitations and next step</h2>
            <ul class="bullet-list">
            {% for item in summary.limitations %}
                <li>{{ item }}</li>
            {% endfor %}
            </ul>
            <p class="footer-note"><span class="warning">Next step:</span> {{ model_selection.next_step }}</p>
        </section>
    </main>
</body>
</html>
"""

case_study_bp = Blueprint("case_study", __name__)


def _load_json(filename: str):
    path = ARTIFACT_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_case_study_artifacts() -> dict:
    summary = _load_json("summary.json")
    if summary is None:
        raise FileNotFoundError("Case-study artifacts are missing. Run python analysis/run_neu_case_study.py.")
    return {
        "summary": summary,
        "benchmarks": _load_json("benchmark-comparison.json") or [],
        "taxonomy": _load_json("defect-taxonomy.json") or [],
        "confusion": _load_json("confusion-matrix.json") or {"hotspots": [], "labels": [], "matrix": []},
        "review_queue": _load_json("review-queue.json") or {"review_budgets": [], "misclassified_examples": []},
        "class_metrics": _load_json("class-metrics.json") or [],
        "model_selection": _load_json("model-selection.json") or {},
        "dataset_profile": _load_json("dataset-profile.json") or {},
    }


@case_study_bp.route("/api/case-study")
def case_study_api():
    try:
        return jsonify(load_case_study_artifacts())
    except FileNotFoundError as exc:
        abort(503, description=str(exc))


@case_study_bp.route("/case-study")
def case_study_page():
    try:
        artifacts = load_case_study_artifacts()
    except FileNotFoundError as exc:
        abort(503, description=str(exc))
    return render_template_string(CASE_STUDY_TEMPLATE, **artifacts)
