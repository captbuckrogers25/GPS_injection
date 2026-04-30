#!/usr/bin/env python3
"""
plot_injections.py
==================
Plot stacked electron flux time series for GPS injection events.

For each selected event in a catalog JSON, produces a multi-panel PDF
showing electron differential flux vs. time for each spacecraft in the
injection cluster, sorted by magnetic local time at onset. Format follows
Figures 4 and 8 in Rogers et al. (2024).

Usage
-----
    python3 plot_injections.py CATALOG.json
    python3 plot_injections.py CATALOG.json --events 0 2 5
    python3 plot_injections.py CATALOG.json --output-dir plots/ --max-platforms 8
    python3 plot_injections.py CATALOG.json --before 2 --after 4 --e-max 1.6
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gps_reader import read_gps_multi


_FLUX_UNITS = r'cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$'

# Matplotlib style choices for a publication figure
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'lines.linewidth': 0.9,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(s):
    t = pd.Timestamp(s)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def _channel_energies(df):
    """Return {channel_int: energy_MeV} for all channels present in df."""
    out = {}
    for i in range(1, 16):
        col = f'electron_diff_flux_energy{i}'
        if col in df.columns:
            e = float(df[col].median(skipna=True))
            if not np.isnan(e):
                out[i] = e
    return out


def _select_channels(ch_energies, e_max_mev, channels=None):
    """Return sorted list of channel ints to plot."""
    if channels is not None:
        return sorted(set(channels) & set(ch_energies.keys()))
    return sorted(ch for ch, e in ch_energies.items() if e <= e_max_mev)


def _load_event_data(event, platforms_to_load=None):
    """Load GPS data files for selected platforms in an event.

    Reads only files for the requested platforms to avoid loading the
    full dataset. Returns dict {platform: DataFrame}.
    """
    if platforms_to_load is None:
        platforms_to_load = set(event['cluster_platforms'])
    else:
        platforms_to_load = set(platforms_to_load)

    files = [
        Path(f) for f in event['provenance']['data_files']
        if Path(f).parent.name in platforms_to_load
    ]
    if not files:
        return {}
    data, _ = read_gps_multi(files)
    return data


def _l_col(df):
    """Return the L-shell column name to use for this DataFrame."""
    if 'L_LGM_T89IGRF' in df.columns and df['L_LGM_T89IGRF'].notna().any():
        return 'L_LGM_T89IGRF'
    if 'L_shell' in df.columns:
        return 'L_shell'
    return None


def _panel_label(platform, df, onset):
    """Build the per-panel annotation string: 'NS55  L=4.58  MLT=3.5'.

    L and MLT are sampled at the nearest valid point within ±30 min of onset
    where L is in the expected injection range (1–7).  Falls back to blank
    if no such sample is found (spacecraft outside injection region at onset).
    """
    parts = [platform.upper()]
    lcol = _l_col(df)
    if lcol and 'local_time' in df.columns and len(df) > 0:
        window = df.loc[
            onset - pd.Timedelta(minutes=30) : onset + pd.Timedelta(minutes=30)
        ]
        # prefer samples where L is in injection range
        valid = window[lcol].between(1.0, 7.0) if len(window) > 0 else pd.Series(dtype=bool)
        if valid.any():
            row = window.loc[valid].iloc[0]
        elif len(window) > 0:
            idx = window.index.searchsorted(onset)
            row = window.iloc[min(idx, len(window) - 1)]
        else:
            row = None

        if row is not None:
            l_val = row[lcol]
            mlt_val = row['local_time']
            if not np.isnan(l_val) and 1.0 <= l_val <= 7.0:
                parts.append(f'L={l_val:.2f}')
            if not np.isnan(mlt_val):
                parts.append(f'MLT={mlt_val:.1f}')
    return '  '.join(parts)


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------

def plot_event(event, gps_data, output_path,
               before_h=2.0, after_h=4.0,
               max_platforms=None, e_max_mev=1.0, channels=None):
    """Create and save a stacked flux time-series figure for one event.

    Parameters
    ----------
    event : dict
        Event record from the catalog (output of detect_injections).
    gps_data : dict[str, pd.DataFrame]
        GPS data for the platforms in event['cluster_platforms'].
    output_path : Path
        Destination PDF file path.
    before_h : float
        Hours before onset shown in each panel.
    after_h : float
        Hours after onset shown in each panel.
    max_platforms : int or None
        Maximum number of spacecraft panels. None = all.
    e_max_mev : float
        Maximum channel energy (MeV) to include. Ignored when channels is set.
    channels : list[int] or None
        Explicit 1-indexed channel list. None = auto-select by e_max_mev.
    """
    onset = _parse_ts(event['onset_utc'])
    t_start = onset - pd.Timedelta(hours=before_h)
    t_end = onset + pd.Timedelta(hours=after_h)

    # Platforms are already MLT-sorted in cluster_platforms (from characterize_event)
    platforms = [p for p in event['cluster_platforms'] if p in gps_data]
    if max_platforms is not None:
        platforms = platforms[:max_platforms]
    if not platforms:
        print(f'  No data loaded for event {event["event_id"]}, skipping.')
        return

    # Determine energy channels from the first platform that has data in window
    ch_energies = {}
    for p in platforms:
        df = gps_data[p]
        window = df.loc[t_start:t_end]
        if len(window) > 0:
            ch_energies = _channel_energies(window)
            break
    if not ch_energies:
        print(f'  No energy channel info found for event {event["event_id"]}, skipping.')
        return

    ch_list = _select_channels(ch_energies, e_max_mev, channels)
    if not ch_list:
        print(f'  No channels within E <= {e_max_mev} MeV, skipping.')
        return

    # Color mapping: plasma colormap over energy range (low E = dark purple, high E = yellow)
    e_arr = np.array([ch_energies[ch] for ch in ch_list])
    norm = mcolors.LogNorm(vmin=e_arr.min(), vmax=e_arr.max())
    cmap = matplotlib.colormaps['plasma']
    ch_colors = {ch: cmap(norm(ch_energies[ch])) for ch in ch_list}

    # Dispersion classification labels for panel annotation
    disp_label = {}
    for p in platforms:
        if p in event.get('dispersionless_platforms', []):
            disp_label[p] = 'DL'
        elif p in event.get('dispersed_platforms', []):
            disp_label[p] = 'D'
        else:
            disp_label[p] = ''

    # ---- Figure layout ----
    n = len(platforms)
    panel_h = 1.35   # inches per panel
    cbar_w = 0.55    # inches for colorbar column
    fig_w = 7.5
    fig_h = panel_h * n + 0.9

    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), sharex=True,
                             squeeze=False, constrained_layout=True)
    axes = axes.flatten()

    # ---- Title ----
    disp_counts = (
        f'{len(event["dispersionless_platforms"])} DL / '
        f'{len(event["dispersed_platforms"])} D / '
        f'{len(event["ambiguous_platforms"])} (?)'
    )
    fig.suptitle(
        f'{onset.strftime("%Y-%m-%d %H:%M UTC")}   '
        f'MLT={event["onset_mlt"]:.1f}  L={event["onset_l"]:.2f}  '
        f'n={event["n_platforms"]}  [{disp_counts}]',
        fontsize=9,
    )

    # ---- Per-platform panels ----
    for ax_idx, platform in enumerate(platforms):
        ax = axes[ax_idx]
        df = gps_data[platform]

        # Drop flagged data
        if 'dropped_data' in df.columns:
            df = df.loc[df['dropped_data'] != 1]

        window = df.loc[t_start:t_end]

        if len(window) == 0:
            ax.text(0.5, 0.5, f'{platform.upper()}  (no data in window)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=8)
            ax.set_ylabel(_FLUX_UNITS, fontsize=6)
            continue

        # Plot each channel
        for ch in ch_list:
            fcol = f'electron_diff_flux{ch}'
            if fcol not in window.columns:
                continue
            flux = window[fcol].astype(float).copy()
            flux[flux <= 0] = np.nan
            if flux.notna().any():
                ax.semilogy(window.index, flux, color=ch_colors[ch],
                           linewidth=0.9, rasterized=True)

        # Onset vertical line
        ax.axvline(onset, color='k', linestyle='--', linewidth=0.9, zorder=5)

        # Panel label (upper left)
        label = _panel_label(platform, window, onset)
        dl = disp_label.get(platform, '')
        if dl:
            label += f'  [{dl}]'
        ax.text(0.005, 0.97, label, transform=ax.transAxes,
                ha='left', va='top', fontsize=7.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.75, edgecolor='none'))

        ax.set_xlim(t_start, t_end)
        ax.set_ylabel(_FLUX_UNITS, fontsize=6, labelpad=1)
        ax.yaxis.set_minor_locator(plt.NullLocator())

        if ax_idx < n - 1:
            ax.tick_params(axis='x', which='both', labelbottom=False)

    # ---- Bottom axis: time labels ----
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator())
    axes[-1].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[15, 30, 45]))
    axes[-1].set_xlabel(f'Time (UTC)  {onset.strftime("%Y-%m-%d")}', fontsize=8)
    fig.autofmt_xdate(rotation=0, ha='center')

    # ---- Colorbar: energy axis ----
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), aspect=40 * n / 8, pad=0.01,
                        shrink=0.92, fraction=0.04)
    cbar.set_label('Energy (MeV)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks(e_arr)
    cbar.set_ticklabels([f'{e:.3g}' for e in e_arr])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot stacked electron flux time series for GPS injection events.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('catalog', metavar='CATALOG.json',
                        help='Injection catalog JSON file')
    parser.add_argument('--events', nargs='*', type=int, default=None,
                        metavar='IDX',
                        help='0-based event indices to plot. Default: all events.')
    parser.add_argument('--output-dir', default='./plots',
                        metavar='DIR',
                        help='Directory for output PDFs')
    parser.add_argument('--before', type=float, default=2.0,
                        metavar='HOURS',
                        help='Hours before onset to show in each panel')
    parser.add_argument('--after', type=float, default=4.0,
                        metavar='HOURS',
                        help='Hours after onset to show in each panel')
    parser.add_argument('--max-platforms', type=int, default=None,
                        metavar='N',
                        help='Maximum number of spacecraft panels per figure')
    parser.add_argument('--e-max', type=float, default=1.0,
                        metavar='MEV',
                        help='Maximum channel energy (MeV) to plot')
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.is_file():
        sys.exit(f'Catalog not found: {catalog_path}')

    with open(catalog_path) as f:
        events = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = (args.events if args.events is not None
               else list(range(len(events))))

    print(f'Catalog  : {catalog_path.name}  ({len(events)} events)')
    print(f'Plotting : {len(indices)} event(s)')
    print(f'Output   : {output_dir}')
    print(f'Window   : -{args.before}h / +{args.after}h around onset')
    print(f'E range  : ≤ {args.e_max} MeV')
    if args.max_platforms:
        print(f'Platforms: up to {args.max_platforms} per figure')

    for i in indices:
        if i < 0 or i >= len(events):
            print(f'  Warning: index {i} out of range (0–{len(events)-1}), skipping.')
            continue

        ev = events[i]
        onset_tag = (ev['onset_utc']
                     .replace(':', '')
                     .replace('+00:00', '')
                     .replace('T', 'T')[:15])
        out_path = output_dir / f'injection_{i:04d}_{onset_tag}.pdf'

        print(f'\nEvent {i:3d}: {ev["onset_utc"]}  n={ev["n_platforms"]}  '
              f'MLT={ev["onset_mlt"]:.1f}  L={ev["onset_l"]:.2f}')

        platforms_needed = ev['cluster_platforms']
        if args.max_platforms:
            platforms_needed = platforms_needed[:args.max_platforms]

        print(f'  Loading data for {len(platforms_needed)} platform(s)...')
        gps_data = _load_event_data(ev, platforms_to_load=platforms_needed)
        print(f'  Loaded {len(gps_data)} platform(s).')

        plot_event(
            ev, gps_data, out_path,
            before_h=args.before,
            after_h=args.after,
            max_platforms=args.max_platforms,
            e_max_mev=args.e_max,
        )

    print(f'\nDone. PDFs in {output_dir}/')


if __name__ == '__main__':
    main()
