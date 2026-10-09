# Security policy

<!-- STATUS: research -->

This repository is an experimental prototype. It does not publish a supported
version range or a security-response SLA. The previous version table was
boilerplate and did not describe this project.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
[repository security advisories](https://github.com/sgm-audio/stem-midi-pro/security/advisories/new)
if private reporting is enabled. If it is unavailable, contact the repository
maintainers through a private channel before opening a public issue. Do not
include credentials, private keys, or private audio in a report.

Maintainers should confirm the private-reporting setting, supported-version
policy, and response expectations. No response-time commitment is currently
published.

## Dependency range findings (reviewed 2026-10-07)

The manifests are unpinned and this environment did not resolve or install the
runtime dependency set. `python-multipart>=0.0.18` addresses CVE-2024-53981,
but that floor is now below later advisories: CVE-2026-42561 is patched in
0.0.27, CVE-2026-53539 in 0.0.30, and CVE-2026-53540 in 0.0.31. The `/process`
route parses attacker-controlled multipart uploads. Its 50 MiB file-byte
check runs after FastAPI has parsed the multipart form, so it is not a total
pre-parser request-body limit; enforce an upstream body cap before deployment.
See the [python-multipart advisory for CVE-2026-42561](https://github.com/advisories/GHSA-pp6c-gr5w-3c5g),
[CVE-2026-53539](https://github.com/advisories/GHSA-5rvq-cxj2-64vf), and
[CVE-2026-53540](https://github.com/advisories/GHSA-v9pg-7xvm-68hf).

The broad `fastapi>=0.100.0` range also permits old Starlette versions. Relevant
multipart parsing advisories include Starlette CVE-2024-47874 (fixed in
0.40.0) and CVE-2025-54121 (fixed in 0.47.2). A separate Range-header DoS is
fixed in 0.49.1, but its reported impact is limited to `FileResponse` or
`StaticFiles`; the current API uses `StreamingResponse` and does not expose
those file-serving paths. See the [multipart advisory](https://github.com/advisories/GHSA-f96h-pmfr-66vw),
[upload rollover advisory](https://github.com/advisories/GHSA-2c2j-9gv5-cj73),
and [Range-header advisory](https://github.com/advisories/GHSA-7f5h-v6xp-fcq8).

These are range-level exposure findings, not evidence that a vulnerable
version is installed. Before deployment, maintainers should review and approve
compatible dependency floors (including `python-multipart>=0.0.31` as a
candidate), resolve a lockfile, and run a current dependency audit. This floor
change is proposed, not applied, pending compatibility validation.
