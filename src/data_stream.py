"""
Simulates a live data stream by yielding one row at a time, in chronological
order, from the cleaned dataset. Each model only ever sees rows up to "now" -
never the future - exactly like a system running in production would.
"""

from src.preprocess import FEATURE_COLS


def data_stream(df, feature_cols=FEATURE_COLS):
    """
    Generator that yields (x, y) one row at a time.

    x: dict of feature_name -> value
    y: the true label (0 = price down, 1 = price up)
    """
    for _, row in df.iterrows():
        x = row[feature_cols].to_dict()
        y = row["class"]
        yield x, y
