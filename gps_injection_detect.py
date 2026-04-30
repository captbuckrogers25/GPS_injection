"""
gps_injection_detect.py
========================
Per-platform inflection detection for LANL GPS particle data.

Detects energetic electron injection events by identifying inflection
signatures in differential flux time series: a local maximum in d²J/dt²
immediately preceding a local maximum in dJ/dt, subject to a multi-channel
SNR and coincidence requirement.

Depends on gps_reader.read_gps_multi, which sets df.attrs['platform'] for
each spacecraft DataFrame. If using read_gps_ascii directly, pass
platform= to find_inflections.

Phases 2–4 of the injection detection pipeline:

  Stage 1 (Phase 2): per-platform inflection detection
    find_inflections   — per-channel inflection candidates
    merge_channels     — group coincident multi-channel candidates

  Stage 2 (Phase 3): multi-platform coincidence clustering
    find_coincidences  — cluster candidates across platforms using
                         gradient-curvature drift kinematics

  Stages 3+4 (Phase 4): characterization, catalog
    characterize_event  — derive onset/MLT/L/dispersion quantities
    detect_injections   — end-to-end convenience wrapper
    write_catalog       — serialize event list to JSON / JSONL
"""

import warnings

import numpy as np
import pandas as pd


ALGORITHM_VERSION = '0.1.0'


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _local_maxima(x, min_separation=1):
    """Return indices where x[i] > x[i-1] and x[i] > x[i+1].

    NaN-safe: NaN values are treated as not-a-maximum.
    """
    x = np.asarray(x, dtype=float)
    is_max = np.zeros(len(x), dtype=bool)
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


def _select_channels(df, channels):
    """Return list of channel numbers (1-indexed) to process.

    If channels is None, selects all channels with mean energy < 1.0 MeV.
    """
    if channels is not None:
        return list(channels)
    selected = []
    for i in range(1, 16):
        col = f'electron_diff_flux_energy{i}'
        if col not in df.columns:
            break
        mean_e = df[col].mean(skipna=True)
        if not np.isnan(mean_e) and mean_e < 1.0:
            selected.append(i)
    return selected


def _empty_inflections():
    """Empty DataFrame with the canonical inflection schema."""
    cols = [
        'platform', 'channel', 'energy_mev',
        'time_d2jdt2_peak', 'time_djdt_peak', 'time_onset',
        'l_shell', 'mlt',
        'flux_pre', 'flux_peak', 'rise_amplitude', 'baseline_mad',
        'snr', 'scale_used',
    ]
    return pd.DataFrame(columns=cols)


def _empty_merged():
    """Empty DataFrame with the canonical merged-candidate schema."""
    cols = [
        'platform', 'time_onset', 'l_shell', 'mlt',
        'n_channels_fired', 'channels_fired', 'channel_times',
        'channel_energies', 'max_snr', 'min_snr',
    ]
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Stage 3+4 helpers
# ---------------------------------------------------------------------------

def _mlt_span(mlts):
    """Minimum circular arc (hours) spanning a set of MLT values.

    Computes 24 − largest_gap so that a cluster straddling midnight counts
    as a small span rather than a near-24-hour span.
    """
    mlts = np.asarray(mlts, dtype=float)
    mlts = mlts[~np.isnan(mlts)] % 24.0
    if len(mlts) < 2:
        return 0.0
    angles = np.sort(mlts)
    gaps = np.diff(angles)
    wrap_gap = 24.0 - angles[-1] + angles[0]
    return float(24.0 - np.append(gaps, wrap_gap).max())


def _classify_platform_dispersion(channel_times, channel_energies,
                                   dispersion_tolerance_s):
    """Return 'dispersionless', 'dispersed', or 'ambiguous' for one platform.

    channel_times  : dict {channel_int: pd.Timestamp}
    channel_energies: dict {channel_int: energy_mev} (may be empty)
    dispersion_tolerance_s : float
    """
    if len(channel_times) < 2:
        return 'dispersionless'

    channels = sorted(channel_times.keys())
    t_vals = np.array([
        channel_times[ch].value / 1e9
        if isinstance(channel_times[ch], pd.Timestamp)
        else float(channel_times[ch])
        for ch in channels
    ])
    spread = t_vals.max() - t_vals.min()

    if spread <= dispersion_tolerance_s:
        return 'dispersionless'

    # Try dispersed: later onset for lower energy → positive corr(t, 1/E)
    if channel_energies:
        e_vals = np.array([channel_energies.get(ch, np.nan) for ch in channels])
        valid = ~np.isnan(e_vals) & (e_vals > 0)
        if valid.sum() >= 2:
            t_sub = t_vals[valid]
            inv_e = 1.0 / e_vals[valid]
            if np.std(t_sub) > 0 and np.std(inv_e) > 0:
                r = np.corrcoef(t_sub, inv_e)[0, 1]
                if r > 0.7:
                    return 'dispersed'

    return 'ambiguous'


# ---------------------------------------------------------------------------
# Stage 2 helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_inflections(df, scale='log', snr_threshold=1.5, l_max=6.0,
                     channels=None, baseline_window_min=30,
                     forward_window_samples=5, platform=None):
    """Per-channel inflection detection for one GPS platform.

    For each energy channel, identifies (time, channel) pairs where the
    electron flux shows an injection-like inflection: a local maximum in
    d²J/dt² immediately preceding a local maximum in dJ/dt, with rise
    amplitude exceeding snr_threshold × MAD of a pre-event baseline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from read_gps_ascii / read_gps_multi with DatetimeIndex.
        read_gps_multi sets df.attrs['platform'] automatically; if using
        read_gps_ascii directly, pass platform= instead.
    scale : {'log', 'linear'}
        Transform applied before differentiating. 'log' uses log10(flux);
        'linear' uses raw flux. Note: meaningful snr_threshold values
        differ between modes because noise characteristics of the derivative
        differ (log-space MAD is dimensionless; linear-space MAD is in flux
        units). Default 1.5 is set to recover events where pre-event flux
        enhancements inflate the baseline MAD; later stages handle false positives.
    snr_threshold : float
        Rise amplitude in signal space must exceed snr_threshold × MAD of
        the pre-event baseline.
    l_max : float
        Discard samples where L > l_max. Uses L_LGM_T89IGRF; falls back
        to L_shell if L_LGM_T89IGRF is mostly NaN (warns on fallback).
    channels : list of int or None
        1-indexed channel numbers to process. None = auto-select all
        channels with mean energy < 1.0 MeV.
    baseline_window_min : float
        Minutes of data before the candidate d²J/dt² peak to use for
        noise estimation (median and MAD).
    forward_window_samples : int
        Number of samples after the d²J/dt² peak to search for a matching
        dJ/dt peak. 5 samples ≈ 20 min at 4-min cadence.
    platform : str or None
        Spacecraft ID override. If None, reads from df.attrs['platform'].

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per surviving (channel, time)
        candidate. Empty DataFrame (with correct columns) if none survive.

        Columns
        -------
        platform, channel, energy_mev,
        time_d2jdt2_peak, time_djdt_peak, time_onset,
        l_shell, mlt,
        flux_pre, flux_peak, rise_amplitude, baseline_mad, snr, scale_used
    """
    if platform is None:
        platform = df.attrs.get('platform', 'unknown')

    # --- Step 1: row filtering ---
    l_col = 'L_LGM_T89IGRF'
    if l_col not in df.columns or df[l_col].isna().mean() > 0.5:
        if 'L_shell' in df.columns:
            warnings.warn(
                f"find_inflections ({platform}): L_LGM_T89IGRF is mostly NaN; "
                "falling back to L_shell.",
                stacklevel=2,
            )
            l_col = 'L_shell'

    keep = pd.Series(True, index=df.index)
    if 'dropped_data' in df.columns:
        keep &= (df['dropped_data'] != 1)
    if l_col in df.columns:
        keep &= (df[l_col].fillna(np.inf) <= l_max)

    df_filt = df.loc[keep].sort_index()

    if len(df_filt) < 10:
        return _empty_inflections()

    # Time axis in seconds (float, for gradient)
    t_sec = df_filt.index.astype('int64').to_numpy() / 1e9

    # Estimate sample interval for baseline window sizing
    dt_median = np.median(np.diff(t_sec)) if len(t_sec) > 1 else 240.0
    baseline_samples = max(3, int(baseline_window_min * 60 / dt_median))

    # --- Step 2: channel selection ---
    ch_list = _select_channels(df_filt, channels)
    if not ch_list:
        return _empty_inflections()

    records = []

    # --- Step 3–6: per-channel detection ---
    for ch in ch_list:
        flux_col = f'electron_diff_flux{ch}'
        energy_col = f'electron_diff_flux_energy{ch}'

        if flux_col not in df_filt.columns:
            continue

        raw_flux = df_filt[flux_col].to_numpy(dtype=float)
        energy_mev = (df_filt[energy_col].mean(skipna=True)
                      if energy_col in df_filt.columns else np.nan)

        # Apply scaling
        if scale == 'log':
            with np.errstate(divide='ignore', invalid='ignore'):
                sig = np.where(raw_flux > 0, np.log10(raw_flux), np.nan)
        else:
            sig = raw_flux.copy()

        # Interpolate short gaps (limit=2 samples) before differentiating
        sig_interp = (pd.Series(sig)
                      .interpolate(method='linear', limit=2)
                      .to_numpy(dtype=float))

        if np.all(np.isnan(sig_interp)):
            continue

        dJdt = np.gradient(sig_interp, t_sec)
        d2Jdt2 = np.gradient(dJdt, t_sec)

        d2_peaks = _local_maxima(d2Jdt2)

        for i in d2_peaks:
            # Search forward for a dJ/dt peak.
            # Include dJdt[i] as left context so that dJdt[i+1] can be
            # detected as a local max when it's immediately followed by
            # a decline (common for sharp injections at 4-min cadence).
            j_end = min(i + forward_window_samples + 1, len(sig_interp))
            ext_dj = dJdt[i:j_end]   # includes dJdt[i] as left neighbor
            if len(ext_dj) < 2:
                continue

            # Find interior maxima; offset ≥ 1 means position is after i.
            # Only accept peaks where dJdt > 0 (flux is actually rising).
            # Negative-dJdt local maxima are artefacts of the declining
            # pre-event phase and would fail the SNR check, but filtering
            # them here avoids shadowing the correct positive-dJdt peak.
            peaks_in_ext = _local_maxima(ext_dj)
            peaks_after_i = peaks_in_ext[peaks_in_ext >= 1]
            peaks_after_i = peaks_after_i[ext_dj[peaks_after_i] > 0]

            if len(peaks_after_i) == 0:
                # Boundary fallback: if dJdt is still rising at the last
                # available sample (data ends before the peak is visible),
                # accept the last sample as j. The SNR check still applies.
                if len(ext_dj) >= 2 and ext_dj[-1] >= ext_dj[-2] and ext_dj[-1] > 0:
                    j = i + len(ext_dj) - 1
                else:
                    continue
            else:
                j = i + peaks_after_i[0]

            # One-step lookahead: with 4-min cadence, centered-difference
            # gradients can place the dJdt peak one sample before the actual
            # signal peak.  If the signal is still rising at j+1 and j+1 is
            # within the forward window, advance j by one.
            fwd_limit = min(i + forward_window_samples + 1, len(sig_interp))
            if (j + 1 < fwd_limit
                    and not np.isnan(sig_interp[j + 1])
                    and sig_interp[j + 1] > sig_interp[j]):
                j = j + 1

            # Baseline window: [i - baseline_samples, i) in sig space
            b_start = max(0, i - baseline_samples)
            baseline_sig = sig_interp[b_start:i]
            baseline_sig = baseline_sig[~np.isnan(baseline_sig)]
            if len(baseline_sig) < 3:
                continue

            sig_pre = np.median(baseline_sig)
            sig_peak = sig_interp[j]
            if np.isnan(sig_peak):
                continue

            rise_amplitude = sig_peak - sig_pre
            if rise_amplitude <= 0:
                continue

            baseline_mad = np.median(np.abs(baseline_sig - sig_pre))
            if baseline_mad == 0 or np.isnan(baseline_mad):
                continue

            snr = rise_amplitude / baseline_mad
            if snr < snr_threshold:
                continue

            # Convert back to flux units for reporting
            if scale == 'log':
                flux_pre_val = 10.0 ** sig_pre
                flux_peak_val = raw_flux[j] if not np.isnan(raw_flux[j]) else 10.0 ** sig_peak
            else:
                flux_pre_val = sig_pre
                flux_peak_val = sig_peak

            time_d2 = df_filt.index[i]
            time_dj = df_filt.index[j]
            time_onset = time_d2  # d²J/dt² peak is always ≤ dJ/dt peak

            l_at_onset = df_filt[l_col].iloc[i]
            mlt_at_onset = (df_filt['local_time'].iloc[i]
                            if 'local_time' in df_filt.columns else np.nan)

            records.append({
                'platform': platform,
                'channel': ch,
                'energy_mev': energy_mev,
                'time_d2jdt2_peak': time_d2,
                'time_djdt_peak': time_dj,
                'time_onset': time_onset,
                'l_shell': l_at_onset,
                'mlt': mlt_at_onset,
                'flux_pre': flux_pre_val,
                'flux_peak': flux_peak_val,
                'rise_amplitude': rise_amplitude,
                'baseline_mad': baseline_mad,
                'snr': snr,
                'scale_used': scale,
            })

    if not records:
        return _empty_inflections()

    return pd.DataFrame(records)


def merge_channels(inflections_long, n_channels_required=3,
                   time_tolerance_s=480):
    """Group per-channel inflections into per-platform per-time candidates.

    For each platform, scans the per-channel inflection table and groups
    detections that fall within time_tolerance_s of each other. Groups
    with ≥ n_channels_required unique channels become merged candidates.

    Grouping is greedy, left-to-right in time: each group is seeded by
    the earliest unassigned inflection and absorbs all detections within
    time_tolerance_s of that seed.

    Parameters
    ----------
    inflections_long : pd.DataFrame
        Output of find_inflections (or pd.concat of multiple calls).
        Must have columns: platform, channel, time_onset, l_shell, mlt,
        snr.
    n_channels_required : int
        Minimum unique channels to form a merged candidate. Default 3.
    time_tolerance_s : float
        Coincidence window in seconds. Default 480 (two 4-min cadence steps),
        which accommodates slight energy dispersion across channels while
        still providing a tight multi-platform coincidence window in Stage 2.

    Returns
    -------
    pd.DataFrame
        One row per merged candidate. Empty if none pass the filter.

        Columns
        -------
        platform, time_onset, l_shell, mlt,
        n_channels_fired, channels_fired, channel_times,
        max_snr, min_snr
    """
    if inflections_long is None or inflections_long.empty:
        return _empty_merged()

    results = []

    for platform, grp in inflections_long.groupby('platform', sort=False):
        grp = grp.sort_values('time_onset').reset_index(drop=True)

        # Convert time_onset to float seconds for arithmetic
        t_sec = np.array([t.value / 1e9 for t in grp['time_onset']])

        used = np.zeros(len(grp), dtype=bool)

        for idx in range(len(grp)):
            if used[idx]:
                continue

            t0 = t_sec[idx]
            in_window = (t_sec >= t0) & (t_sec <= t0 + time_tolerance_s)
            window_rows = grp.iloc[np.where(in_window)[0]]

            # Always mark window as processed to avoid re-visiting
            used[in_window] = True

            n_unique = window_rows['channel'].nunique()
            if n_unique < n_channels_required:
                continue

            onset_time = window_rows['time_onset'].min()

            # L and MLT from the row with the earliest time_onset
            onset_row = window_rows.loc[
                window_rows['time_onset'] == onset_time
            ].iloc[0]

            # channel_times / channel_energies: one entry per channel (first wins)
            ch_times = {}
            ch_energies = {}
            has_energy = 'energy_mev' in window_rows.columns
            for _, row in window_rows.iterrows():
                ch = row['channel']
                if ch not in ch_times:
                    ch_times[ch] = row['time_onset']
                    if has_energy:
                        ch_energies[ch] = row['energy_mev']

            results.append({
                'platform': platform,
                'time_onset': onset_time,
                'l_shell': onset_row['l_shell'],
                'mlt': onset_row['mlt'],
                'n_channels_fired': n_unique,
                'channels_fired': sorted(window_rows['channel'].unique().tolist()),
                'channel_times': ch_times,
                'channel_energies': ch_energies,
                'max_snr': window_rows['snr'].max(),
                'min_snr': window_rows['snr'].min(),
            })

    if not results:
        return _empty_merged()

    return pd.DataFrame(results)


def find_coincidences(merged_candidates, simul_tolerance_s=240,
                      min_platforms=3, drift_model='simple'):
    """Cluster per-platform merged candidates across platforms.

    Returns a list of clusters, where each cluster is a DataFrame
    (subset of input rows) representing a single candidate injection
    event observed across multiple platforms.

    Parameters
    ----------
    merged_candidates : pd.DataFrame
        Concatenated output of merge_channels across all platforms.
        Required columns: platform, time_onset, mlt.
    simul_tolerance_s : float
        Time window (seconds) used as both the measurement-uncertainty
        allowance for simultaneous detections and the additive buffer on
        the drift upper-bound. Default 240 (one 4-min cadence step).
    min_platforms : int
        Minimum unique platforms required to form a valid cluster.
        Default 3; use 2 for sparse events.
    drift_model : str
        Only 'simple' is implemented. Uses the gradient-curvature drift
        period for 250 keV electrons at L=5 (T_drift ≈ 0.5 h) as the
        slowest (most permissive) upper bound.

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per valid cluster, sorted by time_onset.
        Empty list if no clusters pass the min_platforms threshold.
    """
    if merged_candidates is None or merged_candidates.empty:
        return []

    df = merged_candidates.reset_index(drop=True)
    n = len(df)

    t_sec = np.array([t.value / 1e9 for t in df['time_onset']])
    mlt = df['mlt'].to_numpy(dtype=float)
    platforms = df['platform'].to_numpy()

    # Drift period for slowest (lowest) energy channel: ~120 keV at L=5
    T_drift_max = 0.5 * 3600  # 1800 s; most permissive bound
    tolerance_mlt = 0.5       # hours; allows same-MLT simultaneous detections

    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if platforms[i] == platforms[j]:
                continue

            # Canonical ordering: A is the earlier platform
            if t_sec[i] <= t_sec[j]:
                t_a, mlt_a = t_sec[i], mlt[i]
                t_b, mlt_b = t_sec[j], mlt[j]
            else:
                t_a, mlt_a = t_sec[j], mlt[j]
                t_b, mlt_b = t_sec[i], mlt[i]

            delta_t = t_b - t_a  # ≥ 0

            # MLT separation: positive means B is eastward of A
            delta_phi = (mlt_b - mlt_a) % 24.0
            if delta_phi >= 12.0:
                delta_phi -= 24.0  # map to [-12, +12)

            # Sanity: reject if MLT span > 12 h (likely unrelated)
            if abs(delta_phi) > 12.0:
                continue

            # Direction check: later platform must be eastward (or within tolerance)
            if delta_phi < -tolerance_mlt:
                continue

            # Time upper bound: drift time for this MLT separation + measurement slack
            delta_t_max = (max(0.0, delta_phi) / 24.0) * T_drift_max + simul_tolerance_s
            if delta_t > delta_t_max:
                continue

            uf.union(i, j)

    clusters = []
    for group in uf.groups():
        subset = df.iloc[group]
        if subset['platform'].nunique() >= min_platforms:
            clusters.append(
                subset.sort_values('time_onset').reset_index(drop=True)
            )

    return clusters


def characterize_event(cluster, parameters):
    """Compute derived quantities for a single cluster.

    Parameters
    ----------
    cluster : pd.DataFrame
        One cluster from find_coincidences. Columns: platform, time_onset,
        l_shell, mlt, channel_times, channel_energies (optional), max_snr.
    parameters : dict
        Pipeline parameters and provenance tracking fields. Expected keys
        mirror the detect_injections signature plus platforms_loaded,
        platforms_with_data, platforms_with_candidates, data_files, and
        gps_data_version. Missing keys default to None / empty list.

    Returns
    -------
    dict
        Event record with all derived quantities and a provenance sub-dict.
    """
    dispersion_tolerance_s = parameters.get('dispersion_tolerance_s', 240)
    has_ch_energies = 'channel_energies' in cluster.columns

    dispersionless = []
    dispersed = []
    ambiguous = []
    platform_l = {}

    for platform, rows in cluster.groupby('platform'):
        row = rows.sort_values('time_onset').iloc[0]
        ch_times = row['channel_times'] if isinstance(row['channel_times'], dict) else {}
        ch_energies = (row['channel_energies']
                       if has_ch_energies and isinstance(row['channel_energies'], dict)
                       else {})

        cls = _classify_platform_dispersion(ch_times, ch_energies, dispersion_tolerance_s)
        if cls == 'dispersionless':
            dispersionless.append(platform)
            l_val = row['l_shell']
            platform_l[platform] = float(l_val) if pd.notna(l_val) else np.nan
        elif cls == 'dispersed':
            dispersed.append(platform)
        else:
            ambiguous.append(platform)

    # Onset platforms: all platforms within dispersion_tolerance_s of the minimum onset
    t_min = cluster['time_onset'].min()
    tol = pd.Timedelta(seconds=dispersion_tolerance_s)
    onset_platforms = sorted(
        cluster.loc[(cluster['time_onset'] - t_min) <= tol, 'platform']
        .unique().tolist()
    )
    onset_platform = onset_platforms[0]

    onset_row = (cluster[cluster['platform'] == onset_platform]
                 .sort_values('time_onset').iloc[0])
    onset_mlt = float(onset_row['mlt']) if pd.notna(onset_row['mlt']) else np.nan
    onset_l = float(onset_row['l_shell']) if pd.notna(onset_row['l_shell']) else np.nan

    # cluster_platforms sorted by MLT
    plat_mlt = cluster.groupby('platform')['mlt'].first()
    cluster_platforms = sorted(
        plat_mlt.index.tolist(),
        key=lambda p: float(plat_mlt[p]) % 24.0 if pd.notna(plat_mlt[p]) else 999.0,
    )

    # MLT span (wraparound-aware)
    mlts = plat_mlt.dropna().to_numpy(dtype=float)
    mlt_span = _mlt_span(mlts)

    # penetration_l: minimum L among dispersionless platforms
    l_vals = [v for p, v in platform_l.items() if not np.isnan(v)]
    penetration_l = float(min(l_vals)) if l_vals else np.nan

    max_rise_amplitude = float(cluster['max_snr'].max())
    n_platforms = len(cluster_platforms)

    onset_utc = t_min
    event_id = f"{onset_utc.isoformat()}_{onset_platform}"

    provenance = {
        'algorithm_version': ALGORITHM_VERSION,
        'detection_timestamp': pd.Timestamp.now('UTC').isoformat(),
        'parameters': {
            'scale':                 parameters.get('scale'),
            'snr_threshold':         parameters.get('snr_threshold'),
            'l_max':                 parameters.get('l_max'),
            'channels_used':         parameters.get('channels_used'),
            'energy_max_mev':        parameters.get('energy_max_mev'),
            'n_channels_required':   parameters.get('n_channels_required'),
            'simul_tolerance_s':     parameters.get('simul_tolerance_s'),
            'min_platforms':         parameters.get('min_platforms'),
            'drift_model':           parameters.get('drift_model'),
            'dispersion_tolerance_s': parameters.get('dispersion_tolerance_s'),
        },
        'platforms_loaded':           parameters.get('platforms_loaded', []),
        'platforms_with_data':        parameters.get('platforms_with_data', []),
        'platforms_with_candidates':  parameters.get('platforms_with_candidates', []),
        'platforms_in_cluster':       cluster_platforms,
        'data_files':                 parameters.get('data_files', None),
        'gps_data_version':           parameters.get('gps_data_version', 'v1.10'),
    }

    return {
        'event_id':                event_id,
        'onset_utc':               onset_utc,
        'onset_mlt':               onset_mlt,
        'onset_l':                 onset_l,
        'onset_platforms':         onset_platforms,
        'cluster_platforms':       cluster_platforms,
        'mlt_span':                mlt_span,
        'penetration_l':           penetration_l,
        'dispersionless_platforms': sorted(dispersionless),
        'dispersed_platforms':      sorted(dispersed),
        'ambiguous_platforms':      sorted(ambiguous),
        'max_rise_amplitude':      max_rise_amplitude,
        'n_platforms':             n_platforms,
        'provenance':              provenance,
    }


def detect_injections(gps_data_dict, *, scale='log', snr_threshold=3.0,
                      l_max=6.0, channels=None,
                      n_channels_required=3, simul_tolerance_s=240,
                      min_platforms=3, drift_model='simple',
                      dispersion_tolerance_s=240,
                      data_files=None):
    """Run the full detection pipeline end-to-end.

    Parameters
    ----------
    gps_data_dict : dict[str, pd.DataFrame]
        Per-platform DataFrames from read_gps_multi.
    scale : {'log', 'linear'}
        Passed to find_inflections.
    snr_threshold : float
        Passed to find_inflections.
    l_max : float
        Passed to find_inflections.
    channels : list of int or None
        Passed to find_inflections.
    n_channels_required : int
        Passed to merge_channels.
    simul_tolerance_s : float
        Passed to find_coincidences.
    min_platforms : int
        Passed to find_coincidences.
    drift_model : str
        Passed to find_coincidences.
    dispersion_tolerance_s : float
        Passed to characterize_event for dispersion classification and
        onset-platform tie-breaking.
    data_files : list[str] or None
        Source file paths recorded in each event's provenance dict.

    Returns
    -------
    list[dict]
        Event records sorted by onset_utc.
    """
    platforms_loaded = list(gps_data_dict.keys())
    platforms_with_data = []
    platforms_with_candidates = []
    merged_list = []

    for platform, df in gps_data_dict.items():
        # Has valid (non-dropped) rows?
        if len(df) > 0:
            if 'dropped_data' in df.columns:
                has_data = bool((df['dropped_data'] != 1).any())
            else:
                has_data = True
        else:
            has_data = False
        if has_data:
            platforms_with_data.append(platform)

        infl = find_inflections(df, scale=scale, snr_threshold=snr_threshold,
                                l_max=l_max, channels=channels)
        mc = merge_channels(infl, n_channels_required=n_channels_required)
        if not mc.empty:
            platforms_with_candidates.append(platform)
            merged_list.append(mc)

    if not merged_list:
        return []

    all_merged = pd.concat(merged_list, ignore_index=True)
    clusters = find_coincidences(all_merged,
                                 simul_tolerance_s=simul_tolerance_s,
                                 min_platforms=min_platforms,
                                 drift_model=drift_model)

    params = {
        'scale':                  scale,
        'snr_threshold':          snr_threshold,
        'l_max':                  l_max,
        'channels_used':          channels,
        'energy_max_mev':         None,
        'n_channels_required':    n_channels_required,
        'simul_tolerance_s':      simul_tolerance_s,
        'min_platforms':          min_platforms,
        'drift_model':            drift_model,
        'dispersion_tolerance_s': dispersion_tolerance_s,
        'platforms_loaded':       platforms_loaded,
        'platforms_with_data':    platforms_with_data,
        'platforms_with_candidates': platforms_with_candidates,
        'data_files':             data_files,
        'gps_data_version':       'v1.10',
    }

    events = [characterize_event(c, params) for c in clusters]
    events.sort(key=lambda e: e['onset_utc'])
    return events


def write_catalog(events, path, format='json'):
    """Serialize events to disk.

    Parameters
    ----------
    events : list[dict]
        Output of detect_injections (or a list of characterize_event results).
    path : str or Path
        Destination file path.
    format : {'json', 'jsonl'}
        'json'  — single JSON document (a list).
        'jsonl' — one JSON object per line; easier for incremental appending.
    """
    import json

    def _serialize(v):
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        if isinstance(v, dict):
            return {k: _serialize(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_serialize(item) for item in v]
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return None if np.isnan(v) else float(v)
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    records = [{k: _serialize(v) for k, v in event.items()} for event in events]

    path = str(path)
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(records, f, indent=2)
    elif format == 'jsonl':
        with open(path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')
    else:
        raise ValueError(f"Unsupported format: {format!r}. Use 'json' or 'jsonl'.")
