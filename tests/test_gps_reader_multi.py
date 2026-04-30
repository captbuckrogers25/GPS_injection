"""
Tests for read_gps_multi in gps_reader.py.

Run with:
    cd /home/buck/op_code && python -m pytest tests/test_gps_reader_multi.py -v
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from gps_reader import read_gps_multi

# ---------------------------------------------------------------------------
# Paths to the two canonical test files
# ---------------------------------------------------------------------------
DATA_DIR = Path('/home/buck/data/gps')
NS56_FILE = DATA_DIR / 'ns56' / 'ns56_030406_v1.10.ascii'
NS79_FILE = DATA_DIR / 'ns79' / 'ns79_230122_v1.10.ascii'

# Second NS56 file for multi-file concatenation test (must be a different date)
NS56_FILE2 = DATA_DIR / 'ns56' / 'ns56_030413_v1.10.ascii'


# ---------------------------------------------------------------------------
# Test 1: Basic single-file load
# ---------------------------------------------------------------------------

def test_basic_load_ns56():
    """Pass a single NS56 file; verify key, row count, and metadata."""
    data, metadata = read_gps_multi([NS56_FILE])

    assert 'ns56' in data, "Expected 'ns56' key in result"
    assert 'ns56' in metadata

    df = data['ns56']
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2520, f"Expected 2520 rows, got {len(df)}"

    # The metadata should expose 46 variable definitions
    var_defs = [v for v in metadata['ns56'].values()
                if isinstance(v, dict) and 'START_COLUMN' in v]
    assert len(var_defs) == 46, f"Expected 46 variable definitions, got {len(var_defs)}"


# ---------------------------------------------------------------------------
# Test 2: Single-spacecraft, multi-file concatenation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not NS56_FILE2.exists(),
    reason=f"Second NS56 file not available at {NS56_FILE2}; "
           "ask Anthony for a second NS56 file to enable this test.",
)
def test_multifile_same_spacecraft():
    """Two files for the same spacecraft should be concatenated in time order."""
    data, metadata = read_gps_multi([NS56_FILE, NS56_FILE2])

    assert 'ns56' in data
    df = data['ns56']

    # Each file has 2520 rows; concatenated result should have >= 2520 rows.
    assert len(df) >= 2520, "Concatenated result must be at least as long as one file"

    # Index should be monotonically non-decreasing (sorted).
    assert df.index.is_monotonic_increasing, "Combined DataFrame must be time-sorted"

    # Both files' data should be present: index must span both files' time ranges.
    from gps_reader import read_gps_ascii
    df1, _ = read_gps_ascii(NS56_FILE)
    df2, _ = read_gps_ascii(NS56_FILE2)
    assert df.index.min() <= df1.index.min()
    assert df.index.max() >= df2.index.max()


# ---------------------------------------------------------------------------
# Test 3: Multi-spacecraft load
# ---------------------------------------------------------------------------

def test_multi_spacecraft():
    """Load one NS56 and one NS79 file; both keys should appear."""
    data, metadata = read_gps_multi([NS56_FILE, NS79_FILE])

    assert 'ns56' in data, "Expected 'ns56' key"
    assert 'ns79' in data, "Expected 'ns79' key"

    assert len(data['ns56']) == 2520, f"NS56: expected 2520 rows, got {len(data['ns56'])}"
    assert len(data['ns79']) == 2520, f"NS79: expected 2520 rows, got {len(data['ns79'])}"


# ---------------------------------------------------------------------------
# Test 4: Bad file handling
# ---------------------------------------------------------------------------

def test_bad_file_skipped():
    """A nonexistent path should emit a warning but not prevent valid files loading."""
    bad_path = Path('/tmp/does_not_exist_gps.ascii')

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        data, metadata = read_gps_multi([bad_path, NS56_FILE])

    # At least one warning should mention the bad file
    messages = [str(warning.message) for warning in w]
    assert any('does_not_exist' in m for m in messages), \
        f"Expected warning about bad file, got: {messages}"

    # Bad path absent from result; valid file present
    assert bad_path.stem not in data
    assert 'ns56' in data
    assert len(data['ns56']) == 2520


# ---------------------------------------------------------------------------
# Test 5: kwargs forwarding (parse_time=False)
# ---------------------------------------------------------------------------

def test_kwargs_forwarded_parse_time_false():
    """parse_time=False should propagate to read_gps_ascii; index is RangeIndex."""
    data, _ = read_gps_multi([NS56_FILE], parse_time=False)

    df = data['ns56']
    assert isinstance(df.index, pd.RangeIndex), \
        f"Expected RangeIndex when parse_time=False, got {type(df.index)}"
