# Phase 3 — Stage 2: Multi-Platform Coincidence

**Read `HANDOFF_OVERVIEW.md` first.** Phases 1 and 2 must be complete and
verified against the paper events before starting this phase.

## Goal

Take the per-platform merged candidates from Phase 2 and identify clusters
of candidates across platforms that are kinematically consistent with a
single injection event. The output of this phase is a list of clusters,
each containing 2 or more platforms (preferably 3 or more), with
candidates that satisfy the gradient-curvature drift kinematics.

## Scientific basis

Eastward gradient-curvature drift of injected electrons should produce
later onset times at platforms further east in MLT. The expected delay
scales with energy (roughly inversely) and with MLT separation. Two
candidates are kinematically consistent if:

- They are at the same MLT (within tolerance) AND simultaneous within
  measurement uncertainty (≈240 s) — this is the **dispersionless** case,
- OR they are at different MLTs with the *eastward* candidate occurring
  later in time, with delay consistent with drift kinematics for at least
  one of the energy channels detected.

The Δt-tolerance window must include zero so that dispersionless events
are preserved. This is critical: dispersionless events bound the
injection region and are scientifically the most valuable (per Anthony's
explicit guidance).

## Deliverables

Add to `gps_injection_detect.py`:

```python
def find_coincidences(merged_candidates, simul_tolerance_s=240,
                      min_platforms=3, drift_model='simple'):
    """Cluster per-platform merged candidates across platforms.

    Returns a list of clusters, where each cluster is a DataFrame
    (subset of input rows) representing a single candidate injection
    event observed across multiple platforms.
    """
```

Plus a small UnionFind helper (private, in the same module).

## Detailed design — `find_coincidences`

### Input

A DataFrame from concatenating per-platform `merge_channels` outputs
(use `pd.concat([...])` over a dict of platforms). Required columns are
those produced by `merge_channels` plus `platform`. Index can be the
default integer index.

### UnionFind helper

```python
class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self):
        from collections import defaultdict
        d = defaultdict(list)
        for i in range(len(self.parent)):
            d[self.find(i)].append(i)
        return list(d.values())
```

### Pairwise consistency check

For two candidates `A` (at platform P_A, time t_A, MLT φ_A) and `B` (at
platform P_B, time t_B, MLT φ_B):

1. **Different platforms.** Skip if `P_A == P_B`.
2. **MLT direction.** Compute `Δφ = (φ_B - φ_A) mod 24` mapped into the
   range `[-12, +12)`. Eastward drift is positive Δφ. If the candidate
   relationship is to be tested both directions, you can swap A and B as
   needed; the canonical form is "later UTC platform should be at later
   MLT (or the same MLT)."

   Concretely: order so that `t_B ≥ t_A`, then require `Δφ ≥ -tolerance_mlt`
   where `tolerance_mlt` is small (e.g., 0.5 hours) to allow same-MLT
   simultaneous detections (dispersionless case).
3. **Time tolerance.**
   - **Lower bound:** `t_B - t_A ≥ -simul_tolerance_s`. (i.e., `t_B`
     can be slightly before `t_A` within measurement uncertainty.)
   - **Upper bound:** `t_B - t_A ≤ Δt_max(Δφ)`, where
     `Δt_max` is computed from a drift model (see below).
4. **Drift model.** For `drift_model='simple'`, use the
   gradient-curvature drift period for an electron at L ≈ 5 in a dipole
   field. For 120 keV electrons (the lowest energy in the analysis),
   the drift period is ~1.5 hours; for higher energies it's faster.
   Use the *slowest* expected drift period as the upper bound (most
   permissive), since we don't want to reject valid events. Concretely:
   
   ```python
   # Drift period in seconds for the slowest energy considered (120 keV at L=5)
   T_drift_max = 1.5 * 3600  # ≈ 5400 s for 120 keV at L=5
   # Maximum delay for a given MLT separation:
   # If MLT separation is Δφ hours, the time to drift that far is
   # (Δφ / 24) * T_drift_max
   delta_t_max = (delta_phi_hours / 24.0) * T_drift_max
   ```

   This is deliberately a loose bound — we are filtering against random
   coincidences, not doing precise kinematics. A factor-of-2 margin on
   `T_drift_max` is fine; favor inclusivity.
5. **MLT span sanity.** Reject pairs with |Δφ| > 12 hours. (The candidate
   in question would be on the dayside, where injection-driven drifts
   shouldn't deposit fresh electrons in any reasonable timeframe;
   typically this means the pair is unrelated.)

If all checks pass, the pair is kinematically consistent. Add the edge
to the union-find structure.

### Cluster extraction

After all pairs are checked, extract connected components from the
union-find. A cluster is *valid* if:

- It contains candidates from at least `min_platforms` *unique*
  platforms (default 3, but allow tuning to 2 for sparse events).

Note: a single platform can contribute multiple merged candidates to a
cluster (e.g., if a slow-rising injection produces multiple SNR-passing
inflections at the same platform). Count *unique platforms*, not unique
rows, when applying `min_platforms`.

Return a list of clusters, where each cluster is a DataFrame containing
the subset of rows belonging to that cluster, sorted by `time_onset`.

### Edge cases

- **MLT wraparound.** Handle the 0/24 hour boundary correctly. The
  modular arithmetic in step 2 above takes care of this if implemented
  carefully. Add a unit test specifically for this.
- **Same-MLT, same-time, different platforms.** This is the dispersionless
  case. It must pass the consistency check. (The check should already
  pass: Δφ ≈ 0, Δt ≈ 0, both within tolerances.) Add a unit test.
- **Empty input.** If `merged_candidates` is empty, return `[]`.
- **No clusters.** If no pairs are consistent, return `[]`.

## Tests

Create `/home/buck/GPS_injection/tests/test_stage2.py`.

1. **UnionFind sanity.** Standard tests for find/union/groups.
2. **Synthetic dispersionless cluster.** Three platforms at MLT 23.5,
   00.0, 00.5, all with `time_onset` within 240 s of each other. Assert:
   one cluster with 3 platforms.
3. **Synthetic dispersed cluster.** Three platforms at MLT 0, 4, 8, with
   `time_onset` increasing at a rate consistent with drift kinematics.
   Assert: one cluster with 3 platforms.
4. **Synthetic uncorrelated.** Three platforms at random MLTs with
   onsets hours apart. Assert: zero clusters.
5. **MLT wraparound.** Two platforms straddling 23.5 / 00.5 with
   simultaneous onsets. Assert: they cluster.
6. **Westward (wrong direction).** Two platforms where the later-time
   candidate is *westward* of the earlier. Assert: they do not cluster.
7. **Mixed dispersionless + dispersed.** Five platforms in a single
   event: two simultaneous near midnight (dispersionless) plus three
   delayed eastward at increasing MLT (dispersed). Assert: all five end
   up in one cluster.
8. **Paper Event 1 (2016-09-27).** Run Phases 1+2 to produce merged
   candidates for all 8 platforms reported in the paper (ns54, ns62,
   ns55, ns61, ns68, ns60, ns72, ns73). Run `find_coincidences`. Assert:
   exactly one cluster (or, if ns73 is borderline, the dominant cluster)
   contains all of these platforms within a single event near 14:48 UTC.
9. **Paper Event 2 (2022-04-10).** Same for the 8 platforms reported
   for the second event (ns65, ns53, ns68, ns63, ns56, ns71, ns70, ns57).
   Assert: cluster contains all of these.

Tests 8 and 9 are the validation gate.

## Non-goals for this phase

- **No event characterization (onset selection, MLT span, dispersion
  classification, penetration depth).** That's Phase 4.
- **No catalog output to file.** That's Phase 4.
- **No drift model beyond the simple constant-period bound.** A
  T89-or-better drift calculation would be more accurate but isn't
  needed; the simple bound is generous enough that real events pass
  comfortably. If `drift_model='simple'` proves insufficient on the
  paper events, surface that to Anthony rather than building a more
  complex model unilaterally.
- **No magnetopause-shadowing or SPE filtering.** Those are post-detection
  contextual filters, applied to the catalog as a whole — not part of
  the coincidence logic.

## Done criterion

- `find_coincidences` exists and is documented.
- All seven synthetic tests pass.
- Both paper events produce single clusters containing the expected
  platform sets.
- Anthony has reviewed and approved.

When done, update the status in `HANDOFF_OVERVIEW.md` and move to
`HANDOFF_PHASE4_CATALOG.md`.

---

## Implementation notes (added 2026-04-28)

Phase 3 is complete. All 11 tests pass (`tests/test_stage2.py`).

### Event 1 platform count

With default parameters (`snr_threshold=1.5`, `l_max=6.0`), `find_coincidences`
recovers **5 of 8** expected platforms for Event 1 (ns54, ns60, ns61, ns68,
ns73). The three absent platforms have physical/data causes that were
investigated and confirmed:

| Platform | Reason absent |
|----------|---------------|
| ns62 | At L up to 22 during the 14:00–16:00 UTC window; most samples filtered by `l_max=6.0` |
| ns72 | Candidates are 1–1.5 h before the main cluster; at 14:48 UTC it is at MLT 15–17, near the kinematic range limit |
| ns55 | No merged candidate at `snr_threshold=1.5`; appears at snr=1.0 at MLT ~3–4 near the event |

Anthony confirmed (2026-04-28) that a cluster of ≥ 3 platforms satisfying
the kinematic criteria is sufficient to identify an event. The validation
test asserts ≥ 3 platforms from the expected set, consistent with the
`min_platforms=3` default.

Event 2 (2022-04-10) recovers all 8 expected platforms cleanly.

### Drift model note

The `drift_model='simple'` bound (T_drift_max = 1.5 h for 120 keV at L=5)
proved comfortably inclusive for both events. No need to revisit unless
future events show false negatives.
