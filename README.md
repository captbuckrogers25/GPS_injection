# GPS Injection Detection

Automated detection and characterization of energetic electron injection events in LANL-GPS particle data (v1.10 ASCII product). Generalizes the manual workflow from Rogers et al. (2024) (LA-UR-24-29218) into an unsupervised, threshold-based pipeline.

## Overview

Substorm electron injections are identified by detecting rapid flux rises in LANL-GPS differential electron flux time series, then clustering coincident detections across spacecraft to characterize each event. The pipeline runs end-to-end from raw ASCII files to a JSON event catalog.

The detection algorithm has four stages:

1. **Per-channel inflection detection** — finds the d²J/dt² → dJ/dt signature of a flux rise in each energy channel, subject to an SNR threshold and L-shell cut.
2. **Per-platform merge** — requires ≥3 channels to trigger simultaneously on the same spacecraft.
3. **Multi-platform coincidence** — clusters candidates across spacecraft using gradient-curvature drift kinematics; dispersionless events (zero drift delay) emerge naturally as a special case.
4. **Characterization** — derives onset time, MLT, and L-shell; MLT span; penetration depth; and per-platform dispersion classification (dispersionless / dispersed / ambiguous).

The pipeline is pure Python (numpy/pandas). No machine learning is used.

## Files

| File | Purpose |
|---|---|
| `gps_reader.py` | Parse LANL-GPS v1.10 ASCII files into pandas DataFrames |
| `gps_injection_detect.py` | Detection pipeline (stages 1–4) |
| `find_injections.py` | Command-line driver: run detection over a time interval |
| `plot_injections.py` | Plot stacked flux panels for catalog events (publication style) |
| `tests/` | pytest suite (37 tests covering all four phases) |
| `catalogs/` | JSON catalog output |
| `plots/` | PDF plot output |

## Requirements

- Python ≥ 3.9
- numpy
- pandas
- matplotlib (for `plot_injections.py` only)
- pytest (for tests)

## Usage

### Run detection over a time interval

```bash
python3 find_injections.py \
    --start 2016-09-27T00:00:00 \
    --end   2016-09-28T00:00:00 \
    --output-dir ./catalogs/
```

GPS data files are expected in per-spacecraft subdirectories under a common data root, with filenames matching `{spacecraft}_{YYMMDD}_v1.10.ascii`. Pass `--data-dir /path/to/gps/data` to specify the root; the default is compiled into `find_injections.py` and should be updated for your local installation.

Key thresholds (all have defaults; run `--help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--snr-threshold` | 3.0 | Minimum SNR for an inflection candidate |
| `--l-max` | 6.0 | Maximum L-shell |
| `--min-platforms` | 3 | Minimum spacecraft in a valid cluster |
| `--n-channels` | 3 | Minimum channels per spacecraft |

### Plot events from a catalog

```bash
python3 plot_injections.py catalogs/injections_20160927T000000_20160928T000000.json
python3 plot_injections.py catalogs/injections_20160927T000000_20160928T000000.json \
    --events 0 2 5 --output-dir plots/ --before 2 --after 4
```

Output PDFs show stacked flux panels per spacecraft, sorted by MLT at onset, coloured by energy channel (plasma colormap). Format follows Figures 4 and 8 of Rogers et al. (2024).

### Use the API directly

```python
from gps_reader import read_gps_multi
from gps_injection_detect import detect_injections, write_catalog

data, _ = read_gps_multi(file_list)
events = detect_injections(data, snr_threshold=3.0, l_max=6.0, min_platforms=3)
write_catalog(events, 'catalog.json')
```

## Catalog format

Each event record is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `event_id` | string | Unique ID (`{onset_utc}_{onset_platform}`) |
| `onset_utc` | ISO 8601 | Injection onset time |
| `onset_mlt` | float | MLT of earliest platform at onset |
| `onset_l` | float | L-shell of earliest platform at onset |
| `onset_platforms` | list | Spacecraft at the earliest onset time |
| `cluster_platforms` | list | All spacecraft in the event cluster |
| `n_platforms` | int | Number of spacecraft in the cluster |
| `mlt_span` | float | MLT range covered by the cluster |
| `penetration_l` | float | Minimum L-shell in the cluster |
| `dispersionless_platforms` | list | Platforms classified as dispersionless |
| `dispersed_platforms` | list | Platforms classified as dispersed |
| `ambiguous_platforms` | list | Platforms with ambiguous classification |
| `algorithm_version` | string | Pipeline version used |
| `data_files` | list | Input files (provenance) |

## Validation

The pipeline is validated against two events from Rogers et al. (2024):

- **2016-09-27 ~14:44 UTC** — the primary case study (5/8 expected platforms detected at default thresholds; 3 platforms latch onto an earlier feature and require threshold tuning or manual review).
- **2022-04-10 ~03:54 UTC** — all 8 platforms detected; onset is stamped ~36 min early because a precursor candidate on ns68 is kinematically linked into the cluster.

Both are included in the test suite (`tests/test_stage4.py`).

Run all tests:

```bash
pytest tests/
```

## Reference

Rogers, A. J., Morley, S. K., & Gattiker, J. (2024). *Energetic Electron Injections Inside of GEO Observed by an Operational Constellation*. Earth and Space Science (submitted). LA-UR-24-29218.

## License

Apache 2.0. See `LICENSE`.

Written with able assistance by Claude (Sonnet 4.6)
