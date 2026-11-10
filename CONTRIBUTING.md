# Contributing

1. Open an issue first: what is wrong or missing, in about three lines.
2. Branch from `main` as `<issue-number>-<slug>`, for example `12-kidney-mesh`.
3. One concern per pull request. Say what changed, how it behaves now and how
   you tested it. Attach a render when a model changes.
4. Sign and sign off your commits (`git commit -S -s`). `main` accepts signed
   commits only.
5. Before you push, run what CI runs (Lint, Unit tests, Type check, Blender
   self-check, Secret scan, Data guard, Licence check):

   ```bash
   uv venv .venv -p 3.12 && uv pip install --group test -e '.[mesh,qa]'   # once
   ruff check scripts blender tests && ruff format --check scripts blender tests
   shellcheck scripts/*.sh
   .venv/bin/pytest
   .venv/bin/python scripts/mesh_check.py
   .venv/bin/python scripts/bake_check.py
   npx pyright@1.1.414
   python3 scripts/data_guard.py
   python3 scripts/licence_check.py
   gitleaks detect --source . --config .gitleaks.toml --redact
   scripts/self-check.sh   # needs Blender 5.2+; skips without it, fails in CI
   ```

   Put tests in `tests/test_*.py`; they can import modules from `scripts/`.
   Scripts that import `bpy` are not type checked: list them under
   `[tool.pyright] exclude` in `pyproject.toml` (a test enforces this).

## Data

- Use only CC BY or CC0 sources, and add each new one to [NOTICE](NOTICE) with
  its licence, record URL and credit line.
- Never use non-commercial datasets or models, the SMPL family, or a scan of a
  real person.
- Never commit CT volumes, segmentations or meshes. Keep them in `data/`, which
  git ignores, and pin each one by SHA-256.
- Assets derived from CC BY-SA sources go in `assets/cc-by-sa/` only.
