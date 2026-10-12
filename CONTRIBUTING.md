# Contributing to Visual QC — Steel Surface Quality Analytics

Contributions are welcome for reproducibility, documentation, tests and well-scoped technical improvements. Open an issue describing the change before proposing a major feature or architecture rewrite.

## Local checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m unittest discover -s tests -v
```

The full test suite can require a test PostgreSQL service and optional Python packages. CNN and RAG evaluations have additional dependencies; consult the repository workflows and documentation before running resource-intensive experiments.

## Pull request requirements

- Explain the problem, the exact code or data paths changed and how the result was validated.
- Add tests or reproducible evidence for changes affecting model behavior, metrics or user-facing features.
- Document dataset provenance, license constraints, split methodology and evaluation limitations.
- Include no secrets, personally identifiable data, proprietary plant data, or bulky downloaded training datasets.
- Keep metrics honest: label CV, holdout and demo/synthetic results separately.

## Responsible disclosure

Please use the private process in [SECURITY.md](SECURITY.md) for vulnerabilities rather than a public issue.
