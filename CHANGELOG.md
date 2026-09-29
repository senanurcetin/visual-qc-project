# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions are not tagged yet.

## [Unreleased]

### Added
- `run_dl_case_study.py --preflight`: checks GPU, disk, dataset layout / reachability, pretrained weights and output paths before a long run; exits 1 when the run cannot work.
- Abuse limits on the public write endpoints: per-visitor and total row quotas with retention pruning for review decisions (429 / 503), and per-client rate limits on `/api/review` and `/api/classify` (`Retry-After`). Client identity ignores spoofable forwarding headers.
- Operator review queue in the HMI: confirm or correct the predicted defect of rejected units; decisions persist per visitor (Postgres via `DATABASE_URL`, SQLite fallback) with versioned migrations.
- `/api/admin/corrections.csv` (quality-engineer export, `REVIEW_ADMIN_TOKEN` bearer token) and `/api/review/export.csv`.
- `/spc`: p-chart of the reject rate per batch with 3-sigma control limits.
- `POST /api/classify`: ONNX Runtime serving of the exported CNN (503 with a reason when unavailable).
- `analysis/run_dl_case_study.py`: ResNet18 / EfficientNet-B0 on the Random Forest's exact holdout, temperature scaling, ECE, bootstrap intervals, K-fold CV, Grad-CAM, ONNX export and a synthetic `--smoke-test`.
- `FrameSource` abstraction (simulator or looped video via `FRAME_SOURCE`) and `analysis/bench_frames.py` latency report.
- Dockerfile, `docker-compose.yml` (app + Postgres), `.env.example`.
- Tooling: ruff, mypy, coverage floor, pre-commit, Dependabot; CI runs a Postgres service and a CPU-torch smoke job.
- `docs/roadmap.md`; `tests/test_docs_consistency.py` keeps README/hiring-summary numbers in sync with `docs/data/`.

### Changed
- `main.py` split into `hmi_page`, `camera`, `report`, `line_session` modules (behaviour unchanged).
- `line_sim.snapshot`/`history` cache OK counts in blocks: a 7-day-old running line went from ~330 ms to ~0.4 ms per poll.

### Fixed
- Deploy failure caused by a `pyproject.toml` without a `[project]` table (tool config now lives in `ruff.toml` and `.coveragerc`).
