"""Utilities for parsing event datetime columns with mixed source formats."""

from __future__ import annotations

import pandas as pd


def parse_datetime_column(values: pd.Series) -> pd.Series:
    """Parse a datetime-like Series using known formats with a safe fallback."""
    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    parsed = pd.to_datetime(values, format='%b %d, %Y %I:%M %p', errors='coerce')
    missing_mask = parsed.isna()

    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(values.loc[missing_mask], errors='coerce')

    return parsed