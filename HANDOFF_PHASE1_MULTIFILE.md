# Phase 1 — Multi-File GPS Loading

**Read `HANDOFF_OVERVIEW.md` first.** This phase doc assumes that context.

## Goal

Add a small extension to the existing `gps_reader.py` infrastructure that
loads multiple GPS ASCII files at once, returning a dict keyed by
spacecraft ID. This is the data structure the detection pipeline needs.

## Why this phase exists

`read_gps_ascii` reads one file. The detection pipeline needs simultaneous
data from many CXD-equipped GPS platforms for a given time interval (the
paper analyzed 8 platforms for one event). Force-merging into a single
DataFrame is wrong — the platforms have different time grids and different
quality flags. A dict-of-DataFrames is the right intermediate
representation.

## Deliverables

A function `read_gps_multi` exposed alongside `read_gps_ascii`. Two
acceptable locations:

1. **Preferred:** add to `gps_reader.py` directly, keeping the public API
   in one place.
2. **Acceptable:** a new module `gps_reader_multi.py` that imports from
   `gps_reader`, if Anthony prefers not to modify the existing reader.

Ask Anthony which he prefers if it's not already clear from context.

## API

```python
def read_gps_multi(paths, **kwargs):
    """Load multiple GPS ASCII files into a dict keyed by spacecraft ID.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Paths to GPS ASCII files. May span multiple spacecraft and/or
        multiple days. Each spacecraft's files are concatenated in time
        order.
    **kwargs
        Forwarded to `read_gps_ascii` (e.g. `replace_fill`, `parse_time`).

    Returns
    -------
    data : dict[str, pandas.DataFrame]
        Keys are spacecraft IDs (e.g. 'ns54', 'ns56'). Values are
        time-sorted DataFrames with the same columns produced by
        `read_gps_ascii`. Per-file metadata is not preserved in this
        return value (see `metadata` for that).
    metadata : dict[str, dict]
        Keys are spacecraft IDs. Values are the metadata dict from the
        *first* file loaded for that spacecraft. (The metadata is
        spacecraft-specific and version-specific; we assume it does not
        change across files for the same spacecraft within a v1.10
        release.)
    """
```

## Implementation notes

- **Spacecraft ID extraction.** GPS ASCII filenames follow the pattern
  `nsXX_YYMMDD_v1.10.ascii` (per the existing `GPS_HANDOFF.md`).
  Parse the spacecraft ID (`nsXX`) from the filename. Do not rely on a
  specific path; just the filename basename.
- **Concatenation when multiple files cover the same spacecraft.** Use
  `pd.concat([...]).sort_index()` to merge time-sorted DataFrames. Do
  *not* drop duplicates blindly — overlapping time ranges may indicate
  data the user didn't intend to merge, and silently dropping rows is
  worse than failing loudly. Instead: detect duplicate timestamps within
  a spacecraft and emit a warning (use `warnings.warn`) listing how many
  duplicates were found. Keep the first occurrence.
- **Failure modes.** If a file fails to parse, log/warn and skip; don't
  abort the whole load. Anthony will be loading dozens of files at a time
  and one bad file shouldn't take down the rest.
- **No filtering by data quality here.** Do not drop `dropped_data=1`
  rows in this function. That filtering happens in Stage 1. Keep
  loading and filtering separate.

## Tests

Create or extend `/home/buck/GPS_injection/tests/test_gps_reader_multi.py`.

Required tests:

1. **Basic load.** Pass `[ns56_030406_v1.10.ascii]` (the existing test
   file). Assert: result has key `'ns56'`, value is a DataFrame with
   2520 rows, metadata dict has 46 variables.
2. **Single-spacecraft, multi-file concatenation.** This requires a
   second NS56 file or fabrication of one. If a second NS56 file is not
   available, mark this test as skipped with a clear reason and ask
   Anthony how to proceed. Do not invent a fake file.
3. **Multi-spacecraft load.** Pass `[ns56_030406_v1.10.ascii,
   ns79_230122_v1.10.ascii]`. Assert: result has both `'ns56'` and
   `'ns79'` keys, each with the expected row counts.
4. **Bad file handling.** Pass a path that doesn't exist alongside a
   valid path. Assert: a warning is emitted, the valid file still loads,
   the bad path is absent from the result.
5. **kwargs forwarding.** Pass `parse_time=False` and verify that the
   resulting DataFrames do not have a DatetimeIndex.

## Non-goals for this phase

- **No directory-walking convenience function.** A `read_gps_dir(path,
  pattern)` would be nice but isn't needed yet. Don't build it.
- **No download functionality.** The user is responsible for having the
  files locally.
- **No quality-flag filtering.** Belongs to Stage 1.
- **No L-shell or MLT filtering.** Belongs to Stage 1.

## Done criterion

- `read_gps_multi` is importable from `gps_reader` (or `gps_reader_multi`).
- All five tests pass.
- A short docstring example shows typical usage.
- Anthony has reviewed and approved.

When done, update the status table in `HANDOFF_OVERVIEW.md` and proceed to
`HANDOFF_PHASE2_STAGE1.md` in a fresh session.
