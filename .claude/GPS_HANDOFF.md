# GPS Data / inj_trace Integration — Handoff Document

## Context

Anthony Rogers is working on `inj_trace`, a Python library for simulating and
visualizing energetic electron injections in the inner magnetosphere.  It wraps
**LANLGeoMag** (`lgmpy`) and **SHIELDS-PTM** for field modeling and particle
tracing, respectively.  See `PLAN.md` and `README.md` for full details.

A separate GPS particle data reader was written in a prior session.  The goal
is to eventually integrate that GPS data into the `inj_trace` workflow (exact
integration design TBD).

---

## GPS Reader: `gps_reader.py`

**Location**: `/home/buck/op_code/gps_reader.py`  
**Status**: Complete and tested.  Standalone module — NOT yet part of the
`inj_trace` package.

### What it does

Reads LANL GPS particle data ASCII files (e.g. `ns56_030406_v1.10.ascii`) into
a pandas DataFrame, parsing all metadata from the embedded JSON header.

### Public API

```python
from gps_reader import read_gps_ascii, list_variables, get_variable_info

# Main reader — returns (DataFrame, metadata dict)
df, meta = read_gps_ascii(
    path,
    replace_fill=True,       # Replace declared fill values with NaN
    extra_sentinels=(...),   # Also replace ±999, ±1e26 (common space-physics fills)
    parse_time=True,         # Build UTC DatetimeIndex from year + decimal_day
)

# Print table of all 46 variables (name, dim, units, description)
list_variables(meta)

# Get full metadata dict for one variable
info = get_variable_info(meta, 'electron_diff_flux')
# → dict with DESCRIPTION, UNITS, DIMENSION, FILL_VALUE, ELEMENT_NAMES, START_COLUMN
```

### File format

- Header: lines 1–514, `#{...#}` block, each line prefixed with `#`.
  Strip `#` from each line → valid JSON.
- Data: line 515+, whitespace-delimited, 269 columns, no column header row.
- Column names come from `ELEMENT_NAMES` in each variable's metadata block.
- Variables are ordered by `START_COLUMN` (0-indexed).

### Key variables in the files

| Variable | Columns | Units | Notes |
|---|---|---|---|
| `decimal_day` | 0 | days | Day of year (fractional) |
| `Geographic_Latitude/Longitude` | 1–2 | degrees | Often -999 fill when not computed |
| `Rad_Re` | 3 | R_E | Orbital radius; GPS ≈ 4.17 Re |
| `rate_electron_measured1–11` | 4–14 | Hz | Raw electron count rates |
| `rate_proton_measured1–5` | 15–19 | Hz | Raw proton count rates |
| `dropped_data` | 25 | flag | **1 = ignore all data this row** |
| `L_shell` | 29 | R_E | T89/IGRF L-shell |
| `L_LGM_T89IGRF` | 33 | R_E | Same as above but LGM-computed |
| `local_time` | 35 | hours | Magnetic local time |
| `b_satellite` | 37 | gauss | Field magnitude at spacecraft |
| `electron_diff_flux1–15` | 241–255 | cm⁻²s⁻¹sr⁻¹MeV⁻¹ | Differential electron flux |
| `electron_diff_flux_energy1–15` | 226–240 | MeV | Energy bins: 0.12–10 MeV |
| `electron_density_fit` | 59 | cm⁻³ | Fitted electron density |
| `efitpars1–9` | 256–264 | various | 9-param electron fit parameters |

### Fill value handling

The header declares `FILL_VALUE: -1.0` for all variables, but the data uses
multiple sentinels in practice:
- `-1.0` — declared fill; caught by per-variable replacement
- `-999` / `999` — used for position, L-shell when computation fails; caught by `extra_sentinels`
- `-1e26` / `1e26` — used for some computed quantities; caught by `extra_sentinels`
- Large unique negative values (`-9.8e30` etc.) in `bfield_ratio` — these are
  **numerical artifacts** (B_sat/B_eq near equatorial crossing), NOT fill values.
  Do not blanket-replace; filter downstream as needed.

### Test data files

Both in `/home/buck/data/gps/`:

| File | Spacecraft | Date | Notes |
|---|---|---|---|
| `ns79_230122_v1.10.ascii` | NS79 | 2023-01-22 | `dropped_data=1` for all 2520 rows — entire day flagged invalid |
| `ns56_030406_v1.10.ascii` | NS56 | 2003-04-06 to 2003-04-12 | All 2520 rows good, 11.9% NaN rate, full flux data |

---

## What Was NOT Done Yet

The GPS reader is a standalone module.  The intended next step is to integrate
it into the `inj_trace` package, but the design is TBD.  Possibilities to
discuss with Anthony:

1. **Add `inj_trace/io/gps.py`** — move/import `gps_reader.py` into the
   package as `inj_trace.io.read_gps_ascii`, expose from `inj_trace.__init__`.

2. **Decide what GPS data feeds into inj_trace** — likely candidates:
   - `electron_diff_flux` channels → observational flux for comparison against
     PTM simulation output (validation/context)
   - `L_shell` / `L_LGM_T89IGRF` → for plotting alongside `TrajectoryData.compute_lstar()`
   - `local_time` → for equatorial plane plots

3. **Multi-file loading** — GPS data comes one file per spacecraft per day.
   A helper to load and concatenate multiple files into one DataFrame would be
   useful (e.g., `read_gps_multi(paths)` or `read_gps_dir(directory, pattern)`).

4. **Visualization integration** — plot GPS flux observations overlaid on PTM
   equatorial or L-shell plots.  Could be a new `inj_trace/visualization/obs.py`
   module.

---

## inj_trace Package State

Fully implemented at `/home/buck/op_code/inj_trace/`.  All modules exist:

```
inj_trace/
├── config.py            ← path config for lgmpy / SHIELDS-PTM
├── fields/              ← FieldGrid, PTMFieldWriter, model wrappers (T89/TS04/OP77)
├── runner/              ← PTMRunConfig, PTMExecutor
├── postprocess/         ← TrajectoryData, FluxMapResult
├── visualization/       ← equatorial, lshell, timeseries, trajectories3d, animation
└── cli/                 ← inj-config, inj-make-fields, inj-run, inj-plot
```

External dependencies (local installs, not pip):
- `lgmpy` at `~/LANLGeoMag/Python/` — configure with `inj-config set`
- `ptm` / `ptm_python` at `~/SHIELDS-PTM/` — configure with `inj-config set`

See `PLAN.md` for full module-by-module implementation details and verification
steps.
