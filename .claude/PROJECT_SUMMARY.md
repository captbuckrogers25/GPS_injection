# GPS Injection Detection — Project Summary

This document summarizes the design discussion that produced the GPS
injection detection pipeline. It serves two purposes:

1. **Record of contributions.** A reviewable account of what was decided
   by the human (Anthony Rogers) versus the AI assistant (Claude), in the
   event that questions about authorship or AI involvement arise.
2. **Handoff context for future development.** Particularly for a future
   Claude session investigating a machine-learning-based detection
   approach, which Anthony has indicated interest in pursuing after the
   threshold-based pipeline is operational.

The discussion took place in a single conversation on 2026-04-26.

---

## 1. Project context

Anthony Rogers is the lead author of *"Energetic Electron Injections
Inside of GEO Observed by an Operational Constellation"* (Rogers, Morley,
Gattiker; submitted to *Earth and Space Science*, LA-UR-24-29218). The
paper demonstrates that LANL-GPS particle data can identify substorm
electron injections inside geosynchronous orbit, presenting two manually
analyzed case studies (2016-09-27 and 2022-04-10).

Anthony is preparing a proposal to expand this work, with two goals
he articulated up front:

- (a) Identify electron injections similar to those in the paper using
      GPS data and other publicly available data.
- (b) Categorize the injections found, using primarily GPS particle data.

The discussion focused first on (a), with (b) shaped by what (a) makes
feasible.

---

## 2. Discussion arc

### 2.1 Initial brainstorm (Anthony's prompt, Claude's response)

Claude proposed a range of approaches for (a) including ancillary trigger
sources (SuperMAG SML, Kyoto AL/AE, GOES dipolarizations, Pi2
pulsations), GPS-internal detection methods (sliding-window inflection
detection, multi-platform coincidence, energy-dispersion fingerprinting,
ML/anomaly detection), and confirmation data (THEMIS, GOES SEISS, POES,
Van Allen Probes). For (b), Claude proposed categorization axes including
spatial extent, energy characteristics, temporal characteristics,
geophysical context, and a tiered classification scheme.

### 2.2 Anthony's scoping decisions

**On the broader plan:**
- Focus on GPS-internal detection, not ancillary triggers, for the
  initial pipeline.
- Set machine learning aside ("put a pin in") for later consideration.
- Avoid energy-dispersion fingerprinting as a *detection* filter, because
  dispersionless events are scientifically critical (they bound the
  injection region) and any dispersion-based filter would
  systematically discard them. Dispersion classification is acceptable
  *post-detection* as a descriptive label.

This was the key scientific constraint that shaped the algorithm: the
detector must be sensitive to flux rises that look injection-like
without requiring an energy-time signature on any single platform.

### 2.3 Algorithm design

Claude proposed a four-stage algorithm:

- **Stage 1 (per platform):** Identify inflection candidates per energy
  channel using the d²J/dt² → dJ/dt sequence from Figure 1 of the paper.
  Filter on L ≤ 6, SNR threshold over a pre-rise baseline, and require
  multi-channel coincidence at the merge step.
- **Stage 2 (across platforms):** Cluster candidates that are
  kinematically consistent — same MLT and simultaneous within
  measurement uncertainty (the dispersionless case), or eastward in MLT
  with delay consistent with gradient-curvature drift kinematics. Use
  union-find for cluster extraction.
- **Stage 3:** Identify the onset platform(s) (earliest time in the
  cluster, or all tied within tolerance for the dispersionless onset
  case).
- **Stage 4:** Compute derived quantities (MLT span, penetration depth,
  per-platform dispersion classification) and emit catalog records with
  full provenance.

The key design move that preserves dispersionless events is in Stage 2:
the time-tolerance window is centered on zero, not on a positive
expected drift delay. Dispersionless candidates fall out as the
zero-delay, zero-MLT-separation special case of multi-platform
coincidence, rather than requiring special handling.

### 2.4 Anthony's design refinements

In the iteration that followed:

- **Multi-channel requirement (≥3 channels).** Anthony added this not
  for technical reasons but because reviewer culture demands it: "it's
  the first question asked by every reviewer." Including it up front
  forecloses that line of questioning.
- **L-detrending deferred.** Anthony agreed it was probably not worth
  the implementation cost up front. Multi-platform coincidence should
  handle most L-gradient false positives.
- **Linear vs. log-flux derivatives — both implemented as a flag.**
  Anthony asked for both as a tunable parameter, since he expects to
  experiment with which works better in practice.
- **Avoid `scipy.signal` if possible.** Anthony has a general preference
  for staying within numpy when feasible. Claude proposed a pure-numpy
  local-maximum helper.
- **N (SNR threshold) easily tunable.** Anthony expects to tune this
  during exploration.
- **Per-channel results preserved.** Both Claude and Anthony agreed
  that retaining per-channel inflection records (rather than collapsing
  to per-platform candidates immediately) preserves a useful data
  product for future event types.
- **Union-find for Stage 2.** Anthony's preference was simple and
  correct over more elaborate alternatives.
- **Provenance dict.** Both agreed this was important. Claude proposed
  the schema, Anthony confirmed.
- **Two test events is a hard constraint.** Anthony confirmed there is
  no larger curated test set; building one is part of the proposal
  scope, not a precondition. Claude proposed an honest validation
  strategy: tune so both paper events detect with comfortable margin,
  apply to the full archive, characterize the false-positive rate from
  the other direction (consistency with substorm context in ancillary
  data), and use the curated catalog as ground truth for any future
  ML work.

### 2.5 Implementation handoff structure

Anthony asked for handoff documents for Claude Code, deferring to
Claude on the question of one-doc-vs-many. Claude recommended multiple
documents (one overview + one per phase), reasoning that:

- Context window discipline is better with concentrated docs.
- Phase isolation matches the verification cadence (paper events as a
  validation gate at each phase).
- Recovery from problems is cleaner if the doc for one phase doesn't
  also describe later phases.

The four phases are:

1. Multi-file GPS loading (extension to existing reader)
2. Stage 1: per-platform inflection detection
3. Stage 2: multi-platform coincidence
4. Stages 3 & 4: characterization and catalog

Each phase doc specifies goal, deliverables, detailed design, tests
(including the paper events as the gate), and explicit non-goals.

---

## 3. Authorship summary

For review purposes:

**Anthony Rogers contributed:**
- All scientific framing and project goals.
- The constraint to preserve dispersionless events (which fundamentally
  shaped the detector design).
- The decision to deduce dispersion *post-detection* rather than use it
  as a filter.
- The reviewer-defensibility argument for the multi-channel coincidence
  requirement.
- The decision to defer L-detrending, ML, and ancillary-data integration
  to future phases.
- The constraint that the test set is the two paper events; no
  fabrication.
- All threshold defaults will be tuned by Anthony based on his domain
  knowledge.
- The decision to use the existing `gps_reader.py` as the data-access
  foundation.
- All scientific judgment about what constitutes a valid detection.

**Claude (the AI assistant) contributed:**
- Initial broad survey of detection approaches and categorization axes
  (Anthony chose the subset to pursue).
- The four-stage decomposition of the algorithm.
- The kinematic-consistency formulation for Stage 2 (with the
  zero-centered tolerance window that preserves dispersionless events
  — this was the technically novel piece, but it followed directly
  from Anthony's constraint).
- Concrete numpy-based implementation sketches (local-maxima helper,
  union-find).
- The provenance dict schema.
- The validation strategy given only two ground-truth events.
- The phased-handoff document structure.
- Drafting of the handoff documents themselves.

The arc was: Anthony set scientific goals and constraints; Claude
proposed concrete technical structures that satisfied those constraints;
Anthony refined and approved or modified each choice. No design
decision was finalized without Anthony's explicit assent, and no novel
scientific claim was made by the AI.

---

## 4. For a future ML-focused session

If revisiting the machine-learning approach Anthony deferred:

### Context to load
- This document.
- The paper PDF (`GPS_inj_AJR_2024_v1.pdf`) for scientific context.
- The completed threshold-based pipeline (`gps_injection_detect.py`)
  and its catalog output, which is the labeled training data that
  didn't exist at the start of the project.

### Why ML was deferred initially
At the start of this project, only two labeled events existed (the two
case studies in the paper). That is not a usable training set. The
threshold-based pipeline was the path to generate a larger labeled
catalog, *which then becomes the training data for ML*.

By the time a future session revisits ML, the threshold-based catalog
should provide hundreds (ideally thousands) of detected events with
manual review status, plus a likely larger pool of *rejected*
candidates from threshold tuning. Both classes are necessary.

### Likely ML directions worth considering
Discussed but not pursued in this conversation:
- Autoencoder or 1D CNN trained on the stack-plot representation
  (Figure 4 / Figure 8 of the paper) to flag injection-shaped patterns.
- Anomaly detection on the per-platform flux time series with
  cross-platform consistency as a confidence score.
- A learned-threshold replacement for the `snr_threshold` filter,
  conditioned on L, MLT, and recent flux statistics.

### Constraints the ML approach must respect
These come from Anthony, not Claude, and were the underlying
constraints on the threshold-based approach. They apply equally to ML:

- **Dispersionless events must be detectable.** Any ML approach trained
  on energy-dispersion features as the *primary* signal will fail this
  test. Train on multi-platform coincidence patterns, with dispersion
  as one of many descriptive features.
- **Reviewer defensibility.** Black-box detectors will face skepticism.
  An interpretability story (e.g., showing which time/platform
  contributions drove a positive detection) is essential.
- **The two paper events must remain detectable.** They are the
  scientific bedrock.

### What changes in the proposal context
By the time ML is on the table, the catalog itself is a contribution
worth publishing — independent of the ML work. ML is a refinement on
top of an established threshold pipeline, not a replacement for it.
The proposal text should reflect that staging.

---

## 5. Files produced by this discussion

In the planning directory:

- `HANDOFF_OVERVIEW.md` — top-level orientation for any Claude Code
  session
- `HANDOFF_PHASE1_MULTIFILE.md` — multi-file GPS loading
- `HANDOFF_PHASE2_STAGE1.md` — per-platform inflection detection
- `HANDOFF_PHASE3_STAGE2.md` — multi-platform coincidence
- `HANDOFF_PHASE4_CATALOG.md` — characterization and catalog output
- `PROJECT_SUMMARY.md` — this document

No code was written during this discussion; that begins with the first
Claude Code session against `HANDOFF_PHASE1_MULTIFILE.md`.

---

*End of project summary.*
