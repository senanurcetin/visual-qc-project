# Visual QC Project

Visual QC Project is a Flask-based manufacturing quality-control simulator focused on inspection workflows, KPI tracking, and report generation. It combines computer vision, event traceability, and operational metrics in a single lightweight web application.

Demo: [Portfolio project entry](https://senanur-cetin.vercel.app/#projects)

Portfolio role: `support case study`

## Why this project exists

Inspection systems are rarely useful on model output alone. Quality teams still need per-unit traceability, pass/fail history, yield context, and exports that can be reviewed outside the application. Visual QC Project exists to show that inspection logic can be packaged as an operational workflow instead of a toy CV demo.

## Case-study frame

### Problem

Inspection workflows often separate defect detection from reporting and review, which makes quality oversight slower and less traceable.

### Business context

In a manufacturing setting, the useful output is not only a classification result. It is a reviewable record of what happened, when it happened, and how it affects quality KPIs.

### Data or signal source

The project simulates a vision feed, per-unit pass/fail outcomes, OEE updates, and timestamped production logs stored in SQLite.

### Workflow and logic approach

The app combines an OpenCV-driven visual surface with stateful event handling, historian-style logging, and an Excel export path for offline QA review.

### Evaluation and key metrics

- **Inspection outcomes:** OK / NOK per unit
- **Traceability:** timestamped SQLite production logs
- **Operational metrics:** OEE, total output, yield, and fail counts
- **Reporting proof:** exportable Excel report for review and handoff

### Operational outcome

The result is a compact industrial QC concept where inspection, storage, KPI context, and reporting stay in one operator-facing flow.

## What it does

- Simulates a production line inspection workflow.
- Streams a synthetic vision feed for operator monitoring.
- Tracks KPIs such as OEE, pass/fail counts, and production status.
- Stores production events in SQLite.
- Exports structured Excel reports for review and offline analysis.

## Stack

- Python
- Flask
- OpenCV
- Pandas
- SQLite
- XlsxWriter

## Architecture snapshot

- **Application shell:** single-file Flask app serving the operator-facing HMI
- **Vision layer:** synthetic OpenCV inspection feed with defect simulation
- **Persistence layer:** SQLite historian for timestamped production logs
- **Reporting layer:** Excel export for offline QA review and handoff
- **Deployment shape:** lightweight local or free-tier Flask demo

## What this proves

- You can package industrial computer vision into a traceable QA workflow.
- You understand how quality stakeholders consume outputs, not only how vision logic runs.
- You can connect inspection events to KPI and reporting surfaces that are legible beyond ML audiences.

## Local setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

The app runs on `http://localhost:8080`.

## Quality checks

```bash
python -m py_compile main.py
python -c "import main; print(main.app.name)"
```

These are the same smoke checks enforced in GitHub Actions.

## Limitations

- This repo is strongest as an industrial workflow and reporting case study, not as a benchmark-heavy defect-classification project.
- The current signal layer is simulated rather than tied to a labeled production image dataset.

## Portfolio note

This repository is a compact industrial computer-vision support case study. It is meant to show inspection logic, reporting design, and quality-system thinking rather than claim production-grade deployment.

## License

MIT
