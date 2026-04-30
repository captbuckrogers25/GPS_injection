# Phase 2 — Stage 1: Per-Platform Inflection Detection

**Read `HANDOFF_OVERVIEW.md` first.** Phase 1 (multi-file loading) must be
complete and merged before starting this phase.

## Goal

Implement the per-platform inflection-detection logic. For a single GPS
platform's DataFrame, identify all (time, energy channel) pairs where the
electron flux shows an injection-like inflection signature, and merge
across channels to produce per-platform candidate events that satisfy a
multi-channel coincidence requirement.

## Scientific basis

Replicates and extends the Figure 1 method from Rogers et al. (2024). For a
flux time series `J(t)`, an injection onset is identified as a local
maximum in `d²J/dt²` immediately preceding a local maximum in `dJ/dt`.
Anthony's experience from manual analysis: the multi-channel coincidence
requirement (≥3 channels firing within one cadence step) is the single
most important filter for reviewer-defensibility.

## Deliverables

A new module: `/home/buck/GPS_injection/gps_injection_detect.py` (or package).
First-pass public functions for this phase:

```python
def find_inflections(df, scale='log', snr_threshold=3.0, l_max=6.0,
                     channels=None, baseline_window_min=30,
                     forward_window_samples=5):
    """Per-channel inflection detection for one GPS platform.

    Returns a long-format DataFrame with one row per detected
    (channel, time) inflection candidate.
    """

def merge_channels(inflections_long, n_channels_required=3,
                   time_tolerance_s=240):
    """Group per-channel inflections into per-platform per-time candidates.

    Returns a DataFrame with one row per merged candidate, including
    a list of channels that fired and the per-channel timing for use in
    later dispersionless-vs-dispersed classification.
    """
```

## Detailed design — `find_inflections`

### Inputs
- `df`: DataFrame for one platform from `read_gps_ascii` (with DatetimeIndex)
- `scale`: `'log'` or `'linear'` — how to scale flux before differentiating
- `snr_threshold`: rise amplitude must exceed N × MAD of pre-rise baseline
- `l_max`: only consider samples where L ≤ this value
- `channels`: list of channel numbers (1-indexed) to process. Default `None`
  means "all channels with energy < 1 MeV." (See "Channel selection" below.)
- `baseline_window_min`: minutes before the candidate inflection to use for
  noise estimation
- `forward_window_samples`: how many samples after a `d²J/dt²` peak to look
  for the matching `dJ/dt` peak. Default 5 samples ≈ 20 min.

### Steps

1. **Filter rows.** Drop rows where `dropped_data == 1`. Drop rows where
   `L_LGM_T89IGRF > l_max` (or fall back to `L_shell` if `L_LGM_T89IGRF`
   is mostly NaN — emit a warning if falling back).

2. **Channel selection.** If `channels=None`, examine the columns
   `electron_diff_flux_energy1` through `electron_diff_flux_energy15`.
   Take the mean energy across the file (excluding NaN), and select
   channels where mean energy < 1.0 MeV. Record the selected channel
   indices in the output (so downstream can know what was actually used).
   Channels 1-6 will likely be the result based on the variable table in
   `GPS_HANDOFF.md`, but verify rather than hardcode.

3. **For each channel:** extract the flux time series `electron_diff_fluxN`
   for channel N. Mask non-positive and NaN values. Apply scaling:
   - `log`: `signal = log10(flux)` after masking
   - `linear`: `signal = flux`
   Use `np.gradient` to compute `dJdt` and `d²Jdt²` with respect to time.
   Time should be in seconds (use `df.index.astype('int64') / 1e9` or
   convert via timedelta — be explicit about units).

4. **Find local maxima** in `d²J/dt²` using a pure-numpy strict-local-max
   helper (one is sketched below; do not use `scipy.signal.find_peaks`).
   For each `d²J/dt²` peak at index `i`, search indices `i+1` through
   `i+forward_window_samples` for a `dJ/dt` peak. If found at index `j`,
   that pair `(i, j)` is a candidate.

5. **SNR check.** For each candidate, compute the rise amplitude
   = flux(at peak `j`) − median(flux over `baseline_window_min` minutes
   before index `i`). Compute MAD of flux over the same baseline window.
   Require `rise_amplitude > snr_threshold * MAD`.

   Note: the meaningful range of `snr_threshold` differs significantly
   between `scale='log'` and `scale='linear'` because the noise
   characteristics of the derivative differ. Document this in the
   docstring. Default 3.0 is a starting guess to be tuned.

6. **Record candidate.** For each surviving candidate, append a row to
   the output DataFrame with fields:

   | Column | Description |
   |--------|-------------|
   | `platform` | Spacecraft ID (e.g. `'ns54'`) — pulled from a parameter or DataFrame attribute, see below |
   | `channel` | Channel number (1-indexed) |
   | `energy_mev` | Mean energy of this channel over the file |
   | `time_d2jdt2_peak` | Timestamp of the `d²J/dt²` peak |
   | `time_djdt_peak` | Timestamp of the `dJ/dt` peak |
   | `time_onset` | The earlier of the two; this is the canonical onset time |
   | `l_shell` | L at `time_onset` |
   | `mlt` | `local_time` at `time_onset` |
   | `flux_pre` | Median flux in the baseline window |
   | `flux_peak` | Flux at `time_djdt_peak` |
   | `rise_amplitude` | `flux_peak - flux_pre` |
   | `baseline_mad` | MAD of flux in the baseline window |
   | `snr` | `rise_amplitude / baseline_mad` |
   | `scale_used` | `'log'` or `'linear'` |

   The `platform` field needs to come from somewhere. Two options:
   (a) accept a `platform` argument to `find_inflections`, or
   (b) attach the spacecraft ID as a DataFrame attribute in
   `read_gps_multi` (e.g. `df.attrs['platform'] = 'ns54'`) and pull it
   from there. Option (b) is cleaner; if you go that route, update
   `read_gps_multi` to set `df.attrs['platform']` and document the
   dependency in this module's docstring.

### Local-maxima helper (pure numpy)

```python
def _local_maxima(x, min_separation=1):
    """Return indices where x[i] > x[i-1] and x[i] > x[i+1].

    NaN-safe: NaN values are treated as not-a-maximum.
    """
    x = np.asarray(x, dtype=float)
    is_max = np.zeros(len(x), dtype=bool)
    # Strict interior maxima; NaN comparisons are False, which is correct here
    is_max[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    if min_separation > 1:
        idx = np.where(is_max)[0]
        kept = []
        last = -min_separation
        for i in idx:
            if i - last >= min_separation:
                kept.append(i)
                last = i
        is_max[:] = False
        is_max[kept] = True
    return np.where(is_max)[0]
```

### NaN handling

`np.gradient` propagates NaN aggressively (one NaN can poison neighboring
samples). Two reasonable approaches:

1. **Pre-fill with interpolation.** Use `pd.Series.interpolate()` on the
   flux time series before differentiating. Limit the interpolation to
   short gaps (e.g., `limit=2`) to avoid bridging real data gaps.
2. **Mask-then-skip.** Compute gradient anyway, accept NaN propagation,
   and rely on the fact that a candidate near a gap will fail the SNR
   check because the baseline window has too few points.

Start with approach 1 (limited interpolation) since it's more graceful.
If this proves problematic, switch to approach 2 and document the change.

## Detailed design — `merge_channels`

Group rows in the long-format inflection table by `(platform, time)`
within `time_tolerance_s` seconds. Within each group, count unique
channels. If `n_channels >= n_channels_required`, emit a merged candidate.

Output columns:
- `platform`
- `time_onset` — the earliest `time_onset` in the group (or median, see below)
- `l_shell`, `mlt` — at the merged onset time
- `n_channels_fired` — count of unique channels in the group
- `channels_fired` — list of channel numbers
- `channel_times` — dict mapping channel number → onset time (preserves
  per-channel timing for later dispersion analysis)
- `max_snr` — max SNR across channels in the group
- `min_snr` — min SNR across channels in the group

For the merged onset time: use the earliest channel's `time_onset`.
Rationale: subsequent analysis (Stage 2) uses this for cross-platform
coincidence; using the earliest is conservative and matches the paper's
visual identification of the leading inflection.

## Tests

Create `/home/buck/GPS_injection/tests/test_stage1.py`.

Required tests (the paper events anchor the validation; the rest are
unit-level sanity checks):

1. **Local-maxima helper sanity.** Synthetic input with known peaks.
2. **Synthetic injection.** Construct a flux time series with a known
   step-up at a known time; verify `find_inflections` recovers it.
3. **Synthetic non-injection.** Flat noisy flux; verify `find_inflections`
   returns an empty DataFrame.
4. **Synthetic dispersionless event.** Step-up at the same time across
   multiple channels; verify `merge_channels` returns one merged
   candidate with `n_channels_fired` equal to the synthetic count.
5. **Synthetic dispersed event.** Step-up with 1/E delay across channels
   spanning more than `time_tolerance_s`; verify behavior matches the
   tolerance setting.
6. **Paper Event 1 (2016-09-27).** Load NS54 data for that day. Run
   `find_inflections` (default parameters) and `merge_channels` (default
   `n_channels_required=3`). Assert: at least one merged candidate is
   detected near 14:48 UTC (within ±5 min tolerance) at L ≈ 5.25 and
   MLT ≈ 23.83. (Use `pytest.approx` with appropriate tolerance.)
7. **Paper Event 2 (2022-04-10).** Load NS65 data for that day. Run the
   same. Assert: at least one merged candidate near 04:30 UTC, at the
   appropriate L and MLT. (Refer to Figure 8 / Section 5.2 of the paper
   for exact expected values; a quick re-read at this point is worth it.)
8. **Both scaling modes.** Run Event 1 with `scale='log'` and
   `scale='linear'`. Assert: both detect the event. (Different
   `snr_threshold` values may be needed; document the values used.)

Tests 6 and 7 are the validation gate. If they fail, debug before
proceeding to Phase 3.

## Non-goals for this phase

- **No multi-platform logic.** That's Stage 2 / Phase 3.
- **No event characterization beyond what's in the merged candidate
  table.** Dispersionless-vs-dispersed classification at the
  *cluster* level is Phase 4. Note that `merge_channels` already
  preserves per-channel timing in `channel_times`, which is what
  Phase 4 will use; you do not need to compute the classification here.
- **No catalog file output.** That's Phase 4.
- **No L-detrending false-positive filtering.** Anthony has decided to
  defer this. Multi-platform coincidence (Phase 3) is expected to handle
  most L-gradient false positives.

## Done criterion

- `find_inflections` and `merge_channels` exist and are documented.
- All synthetic tests pass.
- Both paper events (2016-09-27 NS54, 2022-04-10 NS65) are detected by
  the per-platform stage with reasonable parameter values.
- Anthony has reviewed and approved the threshold values used to make
  the paper events detect.

When done, update the status in `HANDOFF_OVERVIEW.md` and move to
`HANDOFF_PHASE3_STAGE2.md`.
