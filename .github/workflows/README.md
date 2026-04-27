# CI/CD Workflow

> Single GitHub Actions workflow that mirrors the local `dvc repro` + `pytest` + `docker compose build` story to a hosted CI environment.

## Files

| Path | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Main CI/CD pipeline — runs on push, PR, and manual dispatch |

## Job graph

```
                    ┌── lint ──┐
                    │          ├──── docker-build ──┐
                    ├── test ──┤                    ├── ci-summary
                    │          ├── dvc-validate ────┤
                    │          │                    │
                    └──────────┴── dvc-repro ───────┘
                                  (gated on secret)
```

| Job | Purpose | Runtime | Always runs? |
|---|---|---|---|
| `lint` | ruff style + import check | ~30s | ✅ |
| `test` | pytest 42 tests | ~3 min | ✅ |
| `dvc-validate` | `dvc dag` + `dvc status` | ~1 min | ✅ |
| `dvc-repro` | Full pipeline reproduce | ~5–10 min | Only if `ENABLE_DVC_REPRO=true` |
| `docker-build` | Build all images | ~5 min | ✅ |
| `ci-summary` | Fan-in success gate | ~5s | ✅ |

Total wall-clock time: ~5–8 minutes (parallel) without `dvc-repro`, ~10–15 minutes with it.

## Mapping to assignment rubric

| Rubric item | How this CI demonstrates it |
|---|---|
| Source Control & CI [2] | `.github/workflows/ci.yml` triggered on push/PR |
| "DVC DAG representing CI pipeline" [2] | `dvc-validate` job + `dvc-repro` job (when enabled) |
| Software Engineering [5] — tests | `test` job runs pytest on every push |
| Software Engineering [5] — lint | `lint` job runs ruff |
| Reproducibility | `dvc-repro` confirms `dvc.lock` matches reality |

## Where to put it

Drop the file at `.github/workflows/ci.yml` (relative to repo root). Both GitHub and DagsHub will pick it up automatically — DagsHub supports the GitHub Actions YAML schema natively.

## Required setup

### Always (any CI provider)
Nothing. The default jobs (`lint`, `test`, `dvc-validate`, `docker-build`) run with no configuration.

### Optional — enable full pipeline reproduce in CI
1. Generate a DagsHub access token: profile → Settings → Tokens → Generate new token.
2. In your repo on GitHub (or DagsHub):
   - Settings → Secrets and variables → Actions → New repository secret
     - Name: `DAGSHUB_USERNAME` → your DagsHub login
     - Name: `DAGSHUB_TOKEN` → the token from step 1
   - Settings → Secrets and variables → Actions → Variables → New repository variable
     - Name: `ENABLE_DVC_REPRO` → value `true`
3. Re-run the workflow. The `dvc-repro` job will now run.

> **Why a Variable, not just secrets?** GitHub Actions doesn't let `if:` clauses reference secrets directly (security policy — secret presence/absence shouldn't be visible in workflow logic). Using a repo Variable as the gate is the standard workaround.

## Local testing before pushing

You can dry-run most of the CI locally:

```powershell
# Lint
ruff check src api dags frontend tests --select E,F,W,I --ignore E501

# Tests
pytest -v

# DVC validation
python -m dvc dag
python -m dvc status

# Docker validation
docker compose -f docker-compose.yml config
docker compose -f docker-compose.yml -f docker-compose.mlflow-serve.yml config
docker compose -f docker-compose.yml build --parallel
```

If all four blocks pass locally, CI will pass.

## Common failures and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `ruff check` fails on import order | Imports unsorted | `ruff check --fix src api ...` to auto-fix |
| `pytest` collection error on `airflow/logs/scheduler/latest` | Windows symlink in repo (shouldn't be in CI Linux) | Confirm `pytest.ini` has `testpaths = tests` |
| `dvc dag` complains about missing dep | `dvc.yaml` references a file that's neither tracked nor on disk | Add the file to repo OR adjust `deps` |
| `docker compose build` fails on missing env var | `.env` not committed (correct) but compose tries to interpolate | Set the var in workflow `env:` section or use `${VAR:-default}` syntax in `docker-compose.yml` |
| `dvc-repro` fails on `dvc pull` 401 | `DAGSHUB_TOKEN` secret not set or expired | Regenerate token, update secret |
| `dvc-repro` job is skipped | `ENABLE_DVC_REPRO` variable not set to `true` | Set the repo variable |

## Future extensions (NOT required for assignment)

If you wanted to demonstrate full CD beyond CI:

1. **Auto-tag and release** — On every push to `main`, calculate next semver tag and create a release.
2. **Push images to a registry** — After successful build, `docker compose push` to Docker Hub or DagsHub registry.
3. **Auto-deploy** — On tag, SSH to a deployment host and `docker compose pull && up -d`.
4. **Performance regression gate** — In `dvc-repro`, fail if F1 drops below the previous run.
5. **Slack notification on model registry promotion** — Webhook step gated on `models/test_metrics.json` change.

For an academic project, the current `ci.yml` is the right scope — demonstrating CI patterns without aspirational deploy targets that don't actually exist.

## Viva talking points

If asked about CI:

> "The CI pipeline mirrors my local development workflow. Every push triggers ruff for code quality, pytest for the 42-test suite, `dvc dag` to validate the pipeline definition, and `docker compose build` to verify all six images build correctly. The `dvc-repro` job is gated behind a feature flag and runs the full data pipeline against DagsHub-stored data — that's the true reproducibility test, asserting that anyone with read access can recreate the exact model from a clean checkout."

If asked why no CD:

> "The assignment is a single-host Docker Compose deployment, so CD would mean SSHing to a target host and running `docker compose up`. I've kept the workflow scoped to CI because adding deploy steps without an actual deploy target would be aspirational. The `docker-build` job validates that any environment running `docker compose build` will succeed, which is the precondition for whatever CD path is added later."
