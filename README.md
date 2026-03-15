# Visual QC Project

Visual QC Project is a Flask-based manufacturing quality control simulator focused on inspection workflows, KPI tracking, and report generation. It combines computer vision, process simulation, and operational metrics in a single lightweight web application.

Demo: [Portfolio project entry](https://senanur-cetin.vercel.app/#projects)

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

## Local setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

The app runs on `http://localhost:8080`.

## Portfolio note

This repository is a compact manufacturing simulation project built to show industrial workflow thinking, inspection logic, and reporting design.

## License

MIT
