# Visual QC — QA acceptance (2026-10-14)

> **Release outcome — 14 October 2026:** PR #46 merged to `main` as `e431433`. CircleCI Python/PostgreSQL validation **#10 passed**; the HMI subsequently passed **6/6 actual Vercel production Chromium E2E tests**. Production screenshot PR #47 and README documentation correction PR #48 also merged; the screenshot commit `676dbc3` was deployed READY. The *proposed PR*, *must run CI* and *merge only after* instructions below describe the **earlier release gate**, not an outstanding blocker for PR #46. The five local test skips (four database-backed, one CNN training-dependent) were not counted as passing; PostgreSQL integration was tested separately in CI.

---

## Proposed PR
- PR #46: fix deterministic Python test discovery and release OpenCV temporary video handles
- Branch: `fix/test-discovery-imports-20261014`
- Main/runtime HMI behavior is unchanged. Do not assume the draft branch is in production.

## Locally witnessed gates
- Python 3.11 with isolated `opencv-python-headless==4.14.0.94`: `python -m unittest discover -s tests -v` returned exit 0; 168 tests run, five skips and no errors or failures.
- Four of the skips are PostgreSQL-specific integration tests because no disposable PG service was available locally; one skip is contingent on a committed CNN training run.
- Separate six real Chromium HMI tests all passed on local Flask, including simulated line controls, ESTOP/reset, responsive mobile HUD, report export and RAG.
- `ruff check .` passed.

## Required CircleCI acceptance
- CircleCI `python-postgres-quality` has Python 3.12 and disposable PostgreSQL 16 sidecar; it must run on this PR's final tip without `[skip ci]`.
- Required: locked dependency install, `ruff check .`, `mypy`, syntax compile, chart generation, database ready check, `coverage run -m unittest discover -s tests -v`, and app import.
- The four PostgreSQL tests are accepted only when actual disposable DB-backed assertions pass. Do not connect to any production datastore or reuse saved credentials.
- Merge only after a successful build result and a final production smoke plan.

## Ownership and deployment
- No model rebuilds, source-data modification or production changes are authorized by this QA file.
- Review the final PR diff, then merge with exact expected head SHA. After merge, verify hosting deployment and browser smoke on the effective production URL.
