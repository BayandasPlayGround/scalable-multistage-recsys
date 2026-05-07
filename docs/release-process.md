# Release Process

[Back to docs hub](README.md) | [Back to main README](../README.md)

`main` is the production branch. No code should reach `main` without a pull request, one human approval, and passing CI.

## Normal Change Flow

1. Create a feature branch from the latest `main`.
2. Push the branch and open a pull request into `main`.
3. Complete the PR template, including production impact and validation notes.
4. Wait for all required CI jobs to pass.
5. Get one non-author CODEOWNER approval.
6. Merge the PR into `main`.
7. Verify the production deployment, `/health`, `/ready`, active bundle state, and monitoring logs.

## Required CI Checks

Configure these GitHub branch-protection required status checks for `main`:

- `Lint / Ruff`
- `Tests / Pytest`
- `Security / Source and dependencies`
- `Container / Build and smoke`
- `Infra / Azure manifests`

The CI workflow runs on pull requests to `main`, pushes to `main`, and manual dispatch. Pull requests are the required path; push runs are the post-merge confirmation for production.

## GitHub Branch Protection

Enable a branch ruleset or branch protection rule for `main` with these settings:

- Require a pull request before merging.
- Require at least `1` approval.
- Require review from Code Owners.
- Dismiss stale approvals when new commits are pushed.
- Require conversation resolution before merging.
- Require branches to be up to date before merging.
- Require the CI checks listed above.
- Block force pushes.
- Block branch deletion.
- Do not allow bypassing the above settings, including administrators.

The default CODEOWNER is `@BayandasPlayGround`. If that account is also the PR author, add at least one collaborator or team because authors cannot approve their own required review.

## Emergency Changes

Use the same PR path for emergencies. If production is down and an administrator bypass is ever used outside this policy, open a follow-up PR that documents the incident, the diff, the validation performed, and any missing tests.

## Rollback

For application code, revert the production PR with a new PR into `main`. For model serving issues, activate the previous known-good bundle and confirm `/ready`, `/models/active`, recommendation responses, and monitoring summaries before closing the incident.
