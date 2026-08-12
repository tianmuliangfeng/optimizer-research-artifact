# Anonymous submission release

This repository snapshot is intended for double-blind peer review. It contains
the source release and the compact `core-results/` evidence package. The
restricted internal `full-archive` is not part of this distribution.

## Identity boundary

- Publish this snapshot from a fresh Git history.
- Use anonymous commit metadata during review.
- Do not expose the private upstream GitHub origin in the manuscript, README,
  issues, releases, or repository metadata.
- The only paper-facing repository address is
  `{{ANONYMOUS_REPOSITORY_URL}}`. Replace this placeholder only after an
  anonymous mirror has been created and audited.
- Keep all original W&B projects private or inaccessible during review. The
  compact evidence package pseudonymizes W&B identities and container hosts.

## Local checks

From the repository root:

```bash
python -B -m unittest discover -s tests
python -B core-results/tools/validate_core_results_package.py core-results
python -B tools/validate_anonymous_submission.py .
```

The repository-level submission manifest and `SHA256SUMS` are generated after
the source and compact evidence have passed these checks.

## Licensing

Newly authored code is under the MIT License. Third-party attributions and
preserved notices are listed in `THIRD_PARTY_NOTICES.md` and
`third_party/licenses/`.

After acceptance, replace anonymous authorship and the anonymous URL only in a
separately audited camera-ready/public release.
