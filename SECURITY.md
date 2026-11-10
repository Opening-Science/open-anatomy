# Security Policy

## Supported versions

open-anatomy has no releases yet. Security fixes are made on `main` only.

## Reporting a vulnerability

Do not report suspected vulnerabilities in public issues, pull requests or
discussions. Report them privately through GitHub:
[**Report a vulnerability**](https://github.com/Opening-Science/open-anatomy/security/advisories/new)
(Security tab, then Advisories). If that is not available to you, email
[peter.dike@unibe.ch](mailto:peter.dike@unibe.ch).

Say what the issue is, the affected commit or path, its impact and how to
reproduce it. Do not attach scans, credentials or tokens; describe them instead.

## Scope

In scope: the scripts and CI workflows in this repository, and the released
models and manifests.

Out of scope: TotalSegmentator, Blender, ANNY and the source datasets (report
those to their own projects), and copies built or hosted by others.

Also report privately if you find any of these in the repository or a release:

- **A real person's scan, or anything derived from one.** Only the open
  datasets named in [NOTICE](NOTICE) are used. A scan of a real person outside
  them, or a mesh, segmentation or model built from such a scan, must never be
  committed or released.
- **Credentials.** No passwords, API keys or tokens belong in the repository,
  including in scripts, CI files and logs.
- **Data in git.** CT volumes, segmentations and meshes stay in the gitignored
  `data/` folder, each pinned by SHA-256. A data file in git, or a release file
  that does not match its pinned SHA-256, is a problem to report.
- **An input under the wrong licence.** Data inputs must be CC BY or CC0, and
  tools Apache-2.0.

## What to expect

The maintainer will acknowledge a report within five business days and give a
status update after a first look. Please keep the report confidential until a
fix is out or a disclosure is agreed. The project may publish a GitHub security
advisory once users can protect themselves.
