# GPS Injection Detection — Project Overview

## Read this first

This is the orientation document for the **GPS Injection Detection** project.
Any Claude Code session working on this project should read this file first,
then read the phase document corresponding to the current work phase.

## What is this project?

A pure-Python pipeline for detecting energetic electron injections in
LANL-GPS particle data. The pipeline operates on the publicly-available
LANL-GPS ASCII data product (v1.10) and produces a catalog of detected
injection events with derived quantities (onset MLT/UTC/L, MLT span,
penetration depth, dispersionless vs. dispersed classification per platform).

The scientific motivation comes from Rogers et al. (2024), "Energetic
Electron Injections Inside of GEO Observed by an Operational Constellation"
(submitted to *Earth and Space Science*; LA-UR-24-29218), which demonstrated
that GPS data can identify and characterize substorm injections inside GEO
but did so manually for two case studies. This project generalizes that
manual workflow into an unsupervised detection algorithm.

## Project lead

Anthony J. Rogers (`arogers@SpaceScience.org`). All design decisions trace
back to discussions with him; do not invent new design directions
unilaterally. If something seems underspecified, flag it and ask.

## Existing infrastructure

A standalone GPS data reader exists at `/home/buck/GPS_injection/gps_reader.py`
(see prior session handoff: `GPS_HANDOFF.md`). It exposes:

- `read_gps_ascii(path, ...)` → `(DataFrame, metadata dict)`
- `list_variables(meta)` → prints variable table
- `get_variable_info(meta, name)` → metadata for one variable

This reader is **complete and tested**. Do not modify it without explicit
direction. The detection pipeline builds on top of it.

A larger Python library called `inj_trace` exists at
`/home/buck/op_code/inj_trace/` that wraps LANLGeoMag and SHIELDS-PTM for
particle tracing. **The detection pipeline does not depend on `inj_trace`.**
Integration with `inj_trace` is a possible future direction (for using
particle tracing to refine boundary estimates on detected events) but is
explicitly out of scope for the initial pipeline.

## Test data

- `/home/buck/data/gps/ns56_030406_v1.10.ascii` — 7-day file, NS56,
  2003-04-06 onward. All rows valid, ~12% NaN rate. Useful for reader-level
  testing.
- `/home/buck/data/gps/ns79_230122_v1.10.ascii` — 1-day file, NS79,
  2023-01-22. All rows have `dropped_data=1`. Useful for testing the
  filtering logic.

For algorithm validation, additional GPS files covering the two paper events
will be needed (2016-09-27 and 2022-04-10). These can be downloaded from
NOAA NCEI: <https://www.ncei.noaa.gov/products/GPS-energetic-particles>.
The events involve specific spacecraft listed in the paper (see Phase 2 doc).

## Development phases

The pipeline is being built in four phases. Each phase has its own handoff
document. **Work one phase at a time.** Do not pre-implement work from a
later phase.

| Phase | Document | Status |
|-------|----------|--------|
| 1. Multi-file GPS loading | `HANDOFF_PHASE1_MULTIFILE.md` | Complete |
| 2. Stage 1: per-platform inflection detection | `HANDOFF_PHASE2_STAGE1.md` | Complete |
| 3. Stage 2: multi-platform coincidence | `HANDOFF_PHASE3_STAGE2.md` | Complete |
| 4. Stages 3+4: onset, characterization, catalog | `HANDOFF_PHASE4_CATALOG.md` | Complete |

When a phase is complete and verified, mark it complete in this table and
move to the next phase in a fresh Claude Code session if practical.

## Cross-cutting design decisions (apply to all phases)

These were settled during the design discussion and apply throughout. Do not
revisit without explicit direction.

1. **Pure numpy/pandas where reasonable.** Avoid `scipy.signal` unless there
   is a specific feature it provides that numpy can't replicate in a few
   lines. `scipy.optimize`, `scipy.stats`, etc. are fine where genuinely
   useful. `networkx` is fine for graph operations if needed (Stage 2 uses
   union-find, but networkx would also be acceptable).
2. **Standalone module(s) at `/home/buck/GPS_injection/`.** The detection
   pipeline is parallel to `gps_reader.py`, not part of `inj_trace`.
   Suggested module name: `gps_injection_detect.py` or a small package
   `gps_injection_detect/` if it grows beyond a single file.
3. **Long-format DataFrames as the primary intermediate representation.**
   Stage 1 outputs one row per (platform, time, channel) inflection
   candidate. Stage 2 operates on this. Per-channel data is preserved
   throughout, not collapsed early.
4. **Provenance dict on every event record.** See `HANDOFF_PHASE4_CATALOG.md`
   for the schema. Includes algorithm version, parameters, the
   loaded → with-data → with-candidates → in-cluster funnel of platforms,
   and source file paths.
5. **Reproduce both paper events as the validation gate.** Before declaring
   any phase complete, the partial pipeline through that phase must be
   verified to produce the expected output for the 2016-09-27 and
   2022-04-10 events. Specific expected outputs are listed in each phase's
   doc.
6. **No machine learning in the initial pipeline.** A future phase may
   layer an ML detector on top of (or alongside) the threshold-based
   detector. Not now.
7. **Two test events is a known constraint.** Do not synthesize fake events
   or "augment" the test set. If thresholds need tuning beyond what the two
   events support, surface that as a question rather than guessing.

## Files Claude Code may create or modify

- `/home/buck/GPS_injection/gps_injection_detect.py` (or package) — primary
  deliverable
- `/home/buck/GPS_injection/tests/test_gps_injection_detect.py` (or similar) —
  unit tests, integration tests against the paper events
- `/home/buck/GPS_injection/gps_reader.py` — only for the multi-file extension in
  Phase 1, and only with explicit direction

## Files Claude Code must not modify without explicit direction

- `/home/buck/op_code/inj_trace/` — entire package (lives in op_code; not part of this project)
- The contents of `/home/buck/data/gps/`

## Style and tooling notes

Anthony's preferences (from project user preferences):

- Python 3.x
- Concise explanations; he asks for more detail when needed
- Local installs preferred (`pip3 install --user ...`) over global
- He uses Claude Code regularly for other projects; assume baseline fluency
  with the standard CC workflow

## When in doubt

Stop and ask. The phase docs are detailed but not exhaustive. If a question
isn't answered by the relevant phase doc plus this overview, surface it
rather than picking a direction. Anthony would much rather answer a
clarification question than untangle a wrong-direction implementation.
