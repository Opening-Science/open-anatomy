# Contributing

1. Open an issue first: what is wrong or missing, in about three lines.
2. Branch from `main` as `<issue-number>-<slug>`, for example `12-kidney-mesh`.
3. One concern per pull request. Say what changed, how it behaves now and how
   you tested it. Attach a render when a model changes.
4. Sign and sign off your commits (`git commit -S -s`). `main` accepts signed
   commits only.
5. Before you push:

   ```bash
   ruff check scripts
   ruff format --check scripts
   scripts/self-check.sh   # needs Blender 5.2+; skips without it
   ```

## Data

- Use only CC BY or CC0 sources, and add each new one to [NOTICE](NOTICE) with
  its licence, record URL and credit line.
- Never use non-commercial datasets or models, the SMPL family, or a scan of a
  real person.
- Never commit CT volumes, segmentations or meshes. Keep them in `data/`, which
  git ignores, and pin each one by SHA-256.
- Assets derived from CC BY-SA sources go in `assets/cc-by-sa/` only.
