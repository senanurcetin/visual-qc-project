---
description: Read-only source-backed acceptance for Vision Defect QA agent
---

# Vision Defect QA — custom agent acceptance

Read-only source acceptance. Invocation must be explicit: GitHub Copilot CLI `--agent=vision-defect-qa` or a supported IDE agent selector. The external test harness, not the model response, must record the requested agent ID and CLI exit code. Do not mark NOT RUN just because no IDE selector was used. If the actual loaded profile is not independently shown by runtime evidence, mark profile identity UNVERIFIED separately from source-review results.

Read these checked-in files (if missing, record that fact):
- `README.md`
- `classify.py`
- `camera.py`
- `review.py`
- `tests/test_classify.py`
- `tests/test_camera.py`
- `tests/test_review.py`
- `tests/test_rag.py`

Using ONLY files you actually inspect, answer:
1. Explain classification image decode, preprocessing, logit/probability and calibration validation; identify an edge case involving invalid data or channel shape.
2. Describe simulated/video frame behavior and the difference between local test mocks and a tested live camera.
3. Trace how reviewer corrections are recorded/exported and list one privacy or CSV/format risk grounded in code.
4. Explain why NEU-CLS model scores and offline RAG tests do not establish robustness to novel camera conditions or authoritative production answers.

For each numbered check, give PASS, FAIL, or UNVERIFIED and cite concrete path names plus the relevant function, condition or test. A passing source review is NOT proof that code was executed. Label all unrun tests, live-device claims, runtime deployments and CI UNVERIFIED. Do not edit files, execute shell commands, run tests, contact external APIs, install dependencies, touch credentials, or trigger CI/GitHub Actions/CircleCI/Vercel. No live cameras, model downloads, industrial safety claims, or new benchmark claims.
