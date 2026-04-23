# Tests

Red-green TDD for the DLBS morphometry benchmark. New scripts or features
get a failing test first, then the minimum code to pass, then a refactor
if warranted.

## Layout

- `test_shell.sh` — plain-bash tests for shell scripts (no `bats`
  dependency; self-contained). Each test function returns 0 for PASS,
  non-zero for FAIL.
- `test_python.py` — pytest tests for the Python feature extractors +
  ridge baseline.
- `fixtures/` — tiny BIDS-like inputs for smoke tests; synthetic or
  checked-in subset of sub-1003.

## Run

```bash
# shell tests
bash experiments/dlbs_morphometry_benchmark/tests/test_shell.sh

# python tests
pytest experiments/dlbs_morphometry_benchmark/tests/test_python.py
```

## Policy

- Every new script in `scripts/` must land with at least one test that
  would fail without the implementation.
- Bugs found in production get a regression test BEFORE the fix.
- CI (GitHub Actions) will run these on every PR once set up.
