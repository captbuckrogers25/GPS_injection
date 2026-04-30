# Phase 4 — Stages 3 & 4: Onset, Characterization, and Catalog

**Read `HANDOFF_OVERVIEW.md` first.** Phases 1, 2, and 3 must be complete
and verified against the paper events before starting this phase.

## Goal

Take the clusters produced by Phase 3 and generate a structured catalog
of detected injection events. Each catalog entry includes onset
identification, derived quantities (MLT span, penetration depth,
dispersionless/dispersed classification of platforms), and a complete
provenance dict. Also wire up a top-level `detect_injections` convenience
function that runs the full pipeline end-to-end.

## Deliverables

Add to `gps_injection_detect.py`:

```python
def characterize_event(cluster, parameters):
    """Compute derived quantities for a single cluster.

    Returns an event record (dict).
    """

def detect_injections(gps_data_dict, **kwargs):
    """End-to-end pipeline: per-platform Stage 1, merge, multi-platform
    Stage 2, characterize. Returns a list of event records.
    """

def write_catalog(events, path, format='json'):
    """Write events to disk. Supports JSON (one file) or JSONL
    (one event per line) for ease of incremental appending.
    """
```

## Detailed design — `characterize_event`

### Inputs

- `cluster`: DataFrame of merged candidates from a single Phase 3 cluster.
- `parameters`: dict of pipeline parameters (passed through to the
  provenance dict — see below).

### Per-platform dispersion classification

For each platform in the cluster, examine `channel_times` (the dict
produced by `merge_channels` in Phase 2). A platform is classified as:

- **Dispersionless** if the spread of channel onset times (max − min
  across channels) is ≤ `dispersion_tolerance_s` (default 240 s, i.e.,
  one cadence step).
- **Dispersed** if the spread exceeds the tolerance AND there is a
  monotonic ordering of onset time vs. 1/energy (later onset for lower
  energy). Use `np.corrcoef` between onset time and 1/energy to detect
  the ordering: `r > 0.7` is a reasonable starting threshold.
- **Ambiguous** if neither — flag this in the output but don't crash.

If a single platform has multiple merged candidates in the cluster
(which should be rare), use the earliest one for classification.

### Onset platform(s)

Find the row(s) with the earliest `time_onset` in the cluster. If
multiple platforms are tied within `dispersion_tolerance_s`, all of them
are recorded as onset platforms (the dispersionless-onset case).

### Derived quantities

Per the design discussion, compute:

| Field | Definition |
|-------|-----------|
| `event_id` | `f"{onset_utc.isoformat()}_{onset_platform}"` (use the first onset platform if tied) |
| `onset_utc` | Earliest `time_onset` |
| `onset_mlt` | MLT at the onset platform at `onset_utc` |
| `onset_l` | L at the onset platform at `onset_utc` |
| `onset_platforms` | List of platform IDs sharing the earliest onset (single-element list if not tied) |
| `cluster_platforms` | List of all unique platforms in the cluster, sorted by MLT |
| `mlt_span` | Span of MLT covered by the cluster (handle wraparound) |
| `penetration_l` | Minimum L observed across **dispersionless** platforms (NaN if none) |
| `dispersionless_platforms` | List of platform IDs classified as dispersionless |
| `dispersed_platforms` | List of platform IDs classified as dispersed |
| `ambiguous_platforms` | List of platform IDs that are neither |
| `max_rise_amplitude` | Max `max_snr * baseline_mad` across the cluster (or simply `max(max_snr)` if amplitude isn't needed in absolute units; check with Anthony) |
| `n_platforms` | `len(cluster_platforms)` |
| `provenance` | See below |

### Provenance dict

```python
provenance = {
    'algorithm_version': '0.1.0',  # Module-level constant
    'detection_timestamp': pd.Timestamp.utcnow().isoformat(),
    'parameters': {
        # All thresholds from this run
        'scale': ...,
        'snr_threshold': ...,
        'l_max': ...,
        'channels_used': [...],
        'energy_max_mev': ...,
        'n_channels_required': ...,
        'simul_tolerance_s': ...,
        'min_platforms': ...,
        'drift_model': ...,
        'dispersion_tolerance_s': ...,
    },
    'platforms_loaded': [...],          # All keys in the input gps_data_dict
    'platforms_with_data': [...],       # Platforms with rows passing initial filters
    'platforms_with_candidates': [...], # Platforms with ≥1 Stage-1 merged candidate
    'platforms_in_cluster': [...],      # Final cluster membership (== cluster_platforms)
    'data_files': [...],                # Source file paths if available (None if loaded
                                        # from in-memory DataFrames)
    'gps_data_version': 'v1.10',        # Pulled from metadata if available
}
```

The full provenance is verbose, but storage isn't a concern for a catalog
of hundreds-to-thousands of events. If catalog size becomes annoying
later, separate "detection runs" and "events" tables can normalize this.

## Detailed design — `detect_injections`

A convenience wrapper that runs the full pipeline:

```python
def detect_injections(gps_data_dict, *, scale='log', snr_threshold=3.0,
                      l_max=6.0, channels=None,
                      n_channels_required=3, simul_tolerance_s=240,
                      min_platforms=3, drift_model='simple',
                      dispersion_tolerance_s=240,
                      data_files=None):
    """Run the full detection pipeline.

    Parameters
    ----------
    gps_data_dict : dict[str, pandas.DataFrame]
        From `read_gps_multi`.
    data_files : list[str] or None
        Optional list of source file paths, recorded in the
        provenance dict for each event. If None, the provenance dict
        will record `data_files=None`.
    ... (other parameters as defined above)

    Returns
    -------
    events : list[dict]
        List of event records, sorted by `onset_utc`.
    """
```

Internally:

1. For each platform, run `find_inflections` then `merge_channels`.
   Collect the per-platform `merged_candidates` into one big DataFrame
   with `pd.concat`.
2. Track which platforms had data, which had candidates, etc., for the
   provenance dict.
3. Run `find_coincidences`.
4. For each cluster, run `characterize_event`.
5. Sort by `onset_utc` and return.

The pipeline assembly is the only place where parameter defaults are
chosen for the user; downstream functions should expose them too for
direct use, but `detect_injections` provides a one-call convenience.

## Detailed design — `write_catalog`

A thin wrapper around JSON / JSONL serialization. Not a database, not a
schema — just persistence.

```python
def write_catalog(events, path, format='json'):
    """Serialize events to disk.

    'json' writes a single JSON document (a list).
    'jsonl' writes one event per line (easier for streaming/appending).
    """
```

Handle datetime serialization by converting `pd.Timestamp` to ISO format
strings. NumPy types (`np.int64`, `np.float64`) need conversion to
native Python types — pandas' `to_json` orient='records' handles this,
or use `default=str` with `json.dumps`.

## Tests

Create `/home/buck/GPS_injection/tests/test_stage4.py` (or rename to
`test_full_pipeline.py`).

1. **`characterize_event` synthetic.** Hand-construct a small cluster
   DataFrame and verify all derived quantities.
2. **Dispersion classification.** Synthetic clusters with each of:
   all dispersionless, all dispersed, mixed, ambiguous. Verify
   classification.
3. **MLT span wraparound.** Cluster spanning 22 → 02 MLT. Verify
   `mlt_span` ≈ 4 hours, not 20.
4. **Provenance completeness.** Verify all expected keys are present in
   the provenance dict for a synthetic event.
5. **`write_catalog` round-trip.** Generate events, write JSON/JSONL,
   read back, verify equivalence (modulo Timestamp → str conversion).
6. **Paper Event 1 end-to-end.** Run `detect_injections` on the full set
   of GPS files for 2016-09-27. Assert: at least one event with
   onset near 14:48 UTC, onset platform ns54, ≥7 platforms in cluster,
   `dispersionless_platforms` includes ns54, ns62, ns55,
   `dispersed_platforms` includes ns61, ns68, ns60, ns72, ns73.
7. **Paper Event 2 end-to-end.** Same for 2022-04-10. Assert: event near
   04:30 UTC, ns65 and ns57 both in `onset_platforms` (tied
   dispersionless onset), ≥7 platforms in cluster.

Tests 6 and 7 are the final validation gate for the entire pipeline.

## Non-goals for this phase

- **No automated false-positive filtering using ancillary data** (SYM-H,
  Hp30, GOES dipolarization). That's a downstream catalog-curation step
  that Anthony will run manually for a first pass. Could be a future
  Phase 5.
- **No catalog database.** JSON/JSONL files on disk are sufficient.
- **No web visualization, dashboard, or GUI.** Plain text output.
- **No integration with `inj_trace`.** Particle tracing for boundary
  refinement is a future phase if Anthony wants it.
- **No SuperMAG / OMNI download or correlation.** A bridge to those
  ancillary data sources may be added later but isn't part of this
  pipeline.

## Done criterion

- `characterize_event`, `detect_injections`, and `write_catalog` exist
  and are documented.
- All synthetic tests pass.
- Both paper events are detected end-to-end with provenance dicts that
  match what's expected.
- Anthony has reviewed the catalog output for both paper events and
  confirmed it captures the science accurately.

When done, update the status in `HANDOFF_OVERVIEW.md`. The pipeline is
ready for first scientific use; threshold tuning and false-positive
characterization on the larger archive is the next activity, but is
exploratory rather than implementation.
