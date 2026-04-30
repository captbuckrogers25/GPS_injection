"""
gps_reader.py
=============
Reader for LANL GPS particle data ASCII files (ns??_YYMMDD_v*.ascii).

The file format consists of a JSON metadata header (lines prefixed with '#',
delimited by '#{' and '#}') followed by whitespace-separated numeric data.
Column names, units, fill values, and descriptions are extracted from the
header automatically.

Usage
-----
    from gps_reader import read_gps_ascii, list_variables

    df, meta = read_gps_ascii('/path/to/ns79_230122_v1.10.ascii')

    # df has a DatetimeIndex and one column per ELEMENT_NAME in the header.
    # Fill values are replaced by NaN by default.
    # meta is a dict of variable-name -> info dict (units, description, etc.)

    # Print a summary of available variables
    list_variables(meta)

    # Access electron differential flux channels
    e_flux = df[[c for c in df.columns if c.startswith('electron_diff_flux')]]
"""

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sentinel values commonly used in space-physics ASCII products.
# These are checked in addition to the per-variable FILL_VALUE in the header.
# ---------------------------------------------------------------------------
_EXTRA_SENTINELS = (-999.0, 999.0, -9.99e2, 9.99e2, -1.0e26, 1.0e26)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_header(path: Path):
    """
    Read the '#{...#}' JSON block and return (meta_dict, n_header_lines).

    Each header line is prefixed with '#'.  The opening delimiter is '#{' and
    the closing delimiter is '#}'.  Stripping the leading '#' from every line
    yields valid JSON.
    """
    header_lines = []
    n_header = 0
    with open(path, 'r') as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.rstrip('\n')
            if stripped.startswith('#'):
                header_lines.append(stripped[1:])   # drop leading '#'
                n_header = lineno
                if stripped.rstrip() == '#}':
                    break
            else:
                # First non-comment line after header — should not happen, but
                # guard against malformed files.
                break

    json_text = '\n'.join(header_lines)
    meta = json.loads(json_text)
    return meta, n_header


def _variable_defs(meta: dict):
    """
    Extract and return variable definition dicts sorted by START_COLUMN.

    Variable defs are top-level keys whose value is a dict containing a
    'START_COLUMN' key.  Top-level scalar fields (Copyright, contacts, etc.)
    are ignored.
    """
    defs = []
    for key, val in meta.items():
        if isinstance(val, dict) and 'START_COLUMN' in val:
            defs.append(val)
    defs.sort(key=lambda v: v['START_COLUMN'])
    return defs


def _build_column_list(var_defs: list):
    """
    Return a list of (col_name, fill_value, units) tuples in column order.

    Multi-element variables (DIMENSION > 1) contribute one entry per element
    using the names in ELEMENT_NAMES.
    """
    columns = []
    for vd in var_defs:
        fill = vd.get('FILL_VALUE', np.nan)
        units = vd.get('UNITS', '')
        elem_names = vd.get('ELEMENT_NAMES', [vd['NAME']])
        for name in elem_names:
            columns.append((name, fill, units))
    return columns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_gps_ascii(
    path,
    replace_fill: bool = True,
    extra_sentinels=_EXTRA_SENTINELS,
    parse_time: bool = True,
) -> tuple:
    """
    Read a LANL GPS particle data ASCII file into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the .ascii file.
    replace_fill : bool, default True
        Replace per-variable fill values declared in the header with NaN.
    extra_sentinels : sequence of float, optional
        Additional sentinel values to replace with NaN regardless of variable.
        Defaults to ``(-999, 999, -9.99e2, 9.99e2, -1e26, 1e26)``, which are
        common space-physics fill values that appear in these files but may not
        be declared in every variable's FILL_VALUE field.
        Pass an empty sequence to skip extra sentinel replacement.
    parse_time : bool, default True
        Derive a UTC DatetimeIndex from the ``year`` and ``decimal_day``
        columns and set it as the DataFrame index.  The ``year`` and
        ``decimal_day`` columns are kept in the DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        One row per time sample.  Column names match ``ELEMENT_NAMES`` from
        the header.  Index is a DatetimeIndex in UTC (when ``parse_time=True``)
        or a default RangeIndex.
    meta : dict
        Parsed header information.  Variable definitions are accessible by
        their ``NAME`` key.  Each variable dict contains:
        ``NAME``, ``DESCRIPTION``, ``UNITS``, ``DIMENSION``,
        ``FILL_VALUE``, ``ELEMENT_NAMES``, ``ELEMENT_LABELS``,
        ``START_COLUMN``.

    Examples
    --------
    >>> df, meta = read_gps_ascii('ns79_230122_v1.10.ascii')
    >>> df['L_shell'].dropna().plot()

    >>> # Electron differential flux channels
    >>> e_flux = df[[c for c in df.columns if c.startswith('electron_diff_flux')
    ...              and not c.startswith('electron_diff_flux_energy')]]
    """
    path = Path(path)

    # --- Parse header ---
    meta, n_header = _parse_header(path)
    var_defs = _variable_defs(meta)
    col_info = _build_column_list(var_defs)
    col_names = [c[0] for c in col_info]

    # --- Read data ---
    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        names=col_names,
        skiprows=n_header,
        dtype=float,
        low_memory=False,
    )

    # Force integer columns to remain numeric (year, svn_number, etc.)
    # Pandas may infer them as float due to fill values; that is fine here.

    # --- Replace fill values ---
    if replace_fill:
        for vd in var_defs:
            declared_fill = vd.get('FILL_VALUE')
            if declared_fill is None:
                continue
            elem_names = vd.get('ELEMENT_NAMES', [vd['NAME']])
            for col in elem_names:
                if col not in df.columns:
                    continue
                mask = np.isclose(df[col].to_numpy(dtype=float, na_value=np.nan),
                                  declared_fill, rtol=1e-6, atol=0,
                                  equal_nan=False)
                df.loc[mask, col] = np.nan

    if extra_sentinels:
        sentinels = list(extra_sentinels)
        for col in df.columns:
            arr = df[col].to_numpy(dtype=float, na_value=np.nan)
            mask = np.zeros(len(arr), dtype=bool)
            for sv in sentinels:
                mask |= np.isclose(arr, sv, rtol=1e-6, atol=0, equal_nan=False)
            df.loc[mask, col] = np.nan

    # --- Build DatetimeIndex ---
    if parse_time and 'year' in df.columns and 'decimal_day' in df.columns:
        years = df['year'].fillna(0).astype(int).astype(str)
        year_starts = pd.to_datetime(years + '-01-01', format='%Y-%m-%d',
                                     errors='coerce', utc=True)
        day_offsets = pd.to_timedelta(
            df['decimal_day'].fillna(np.nan) - 1.0, unit='D'
        )
        dt_index = year_starts + day_offsets
        df.index = dt_index
        df.index.name = 'time_utc'

    return df, meta


def list_variables(meta: dict, show_units: bool = True) -> None:
    """
    Print a human-readable summary of the variables defined in a GPS file header.

    Parameters
    ----------
    meta : dict
        The ``meta`` dict returned by :func:`read_gps_ascii`.
    show_units : bool, default True
        Include units and start-column in the output.
    """
    var_defs = _variable_defs(meta)
    print(f"{'NAME':<35} {'DIM':<6} {'START_COL':<11} {'UNITS':<25} DESCRIPTION")
    print('-' * 110)
    for vd in var_defs:
        name = vd.get('NAME', '')
        dim = vd.get('DIMENSION', [1])
        start = vd.get('START_COLUMN', '')
        units = vd.get('UNITS', '') if show_units else ''
        desc = vd.get('DESCRIPTION', '')
        # Truncate description for readability
        if len(desc) > 50:
            desc = desc[:47] + '...'
        dim_str = str(dim[0]) if isinstance(dim, list) else str(dim)
        print(f"{name:<35} {dim_str:<6} {str(start):<11} {units:<25} {desc}")


def get_variable_info(meta: dict, name: str) -> dict:
    """
    Return the metadata dict for a single variable by its NAME.

    Parameters
    ----------
    meta : dict
        The ``meta`` dict returned by :func:`read_gps_ascii`.
    name : str
        Variable name as it appears in the header (e.g. ``'L_shell'``).

    Returns
    -------
    dict or None
        The variable info dict, or None if not found.
    """
    val = meta.get(name)
    if isinstance(val, dict) and 'START_COLUMN' in val:
        return val
    return None


def read_gps_multi(paths, **kwargs):
    """Load multiple GPS ASCII files into a dict keyed by spacecraft ID.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Paths to GPS ASCII files. May span multiple spacecraft and/or
        multiple days. Each spacecraft's files are concatenated in time order.
    **kwargs
        Forwarded to :func:`read_gps_ascii` (e.g. ``replace_fill``,
        ``parse_time``).

    Returns
    -------
    data : dict[str, pandas.DataFrame]
        Keys are spacecraft IDs (e.g. ``'ns54'``, ``'ns56'``). Values are
        time-sorted DataFrames with the same columns produced by
        :func:`read_gps_ascii`. Per-file metadata is not preserved in this
        return value (see ``metadata`` for that).
    metadata : dict[str, dict]
        Keys are spacecraft IDs. Values are the metadata dict from the
        *first* file loaded for that spacecraft.

    Examples
    --------
    >>> data, meta = read_gps_multi([
    ...     'ns56_030406_v1.10.ascii',
    ...     'ns79_230122_v1.10.ascii',
    ... ])
    >>> ns56_df = data['ns56']
    >>> ns56_df['L_shell'].dropna().plot()
    """
    # Group paths by spacecraft ID parsed from filename basename.
    # Pattern: ns<digits>_YYMMDD_v*.ascii
    _sc_re = re.compile(r'^(ns\d+)_', re.IGNORECASE)

    per_sc_frames = {}   # sc_id -> list of DataFrames
    per_sc_meta = {}     # sc_id -> metadata dict (first file wins)

    for raw_path in paths:
        path = Path(raw_path)
        m = _sc_re.match(path.name)
        if m is None:
            warnings.warn(
                f"read_gps_multi: cannot parse spacecraft ID from filename "
                f"'{path.name}' — skipping.",
                stacklevel=2,
            )
            continue

        sc_id = m.group(1).lower()

        try:
            df, meta = read_gps_ascii(path, **kwargs)
        except Exception as exc:
            warnings.warn(
                f"read_gps_multi: failed to load '{path}' ({exc}) — skipping.",
                stacklevel=2,
            )
            continue

        if sc_id not in per_sc_frames:
            per_sc_frames[sc_id] = []
            per_sc_meta[sc_id] = meta

        per_sc_frames[sc_id].append(df)

    data = {}
    for sc_id, frames in per_sc_frames.items():
        if len(frames) == 1:
            data[sc_id] = frames[0]
        else:
            combined = pd.concat(frames)
            combined.sort_index(inplace=True)

            # Detect duplicate timestamps and warn; keep first occurrence.
            n_dup = combined.index.duplicated(keep='first').sum()
            if n_dup:
                warnings.warn(
                    f"read_gps_multi: {n_dup} duplicate timestamp(s) found for "
                    f"spacecraft '{sc_id}' after concatenation — keeping first "
                    f"occurrence. Check for overlapping input files.",
                    stacklevel=2,
                )
                combined = combined[~combined.index.duplicated(keep='first')]

            data[sc_id] = combined

        data[sc_id].attrs['platform'] = sc_id

    return data, per_sc_meta
