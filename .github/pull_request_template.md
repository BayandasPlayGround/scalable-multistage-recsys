## Production Impact

- [ ] This PR is safe to deploy after merge to `main`.
- [ ] I understand that `main` is the production branch.
- [ ] I have called out any user-facing, model-quality, data, infrastructure, or security impact below.

## Review Notes

Describe what changed and what the reviewer should focus on.

## Validation

- [ ] `pytest -p no:cacheprovider`
- [ ] `ruff check .`
- [ ] Docker build and `/health` plus `/ready` smoke check
- [ ] Security and dependency checks
- [ ] Azure manifest validation, if infrastructure changed

## Deployment Notes

Describe any required environment variables, bundle activation steps, rollback notes, or monitoring checks.
