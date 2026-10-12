# Security policy

## Reporting a vulnerability

Please disclose privately by emailing [senanur.cetin.work@gmail.com](mailto:senanur.cetin.work@gmail.com) with the subject `[visual-qc-project security]`. Include the affected version or commit, impact, and minimal safe reproduction steps. Do **not** disclose credentials, personal data, exploits against third-party systems, or sensitive logs in public GitHub issues.

This is a portfolio/research project with no guaranteed response time or supported security-release schedule.

## Use and deployment boundaries

The image classification and RAG components are research demonstrations, not factory release gates. Treat uploaded data and retrieved knowledge as untrusted input, do not add real production or personnel data to the public demo, and independently verify operator guidance.

Use environment variables or a dedicated secret manager for credentials. Do not commit .env files, data containing personal information, or private model/data artifacts.
