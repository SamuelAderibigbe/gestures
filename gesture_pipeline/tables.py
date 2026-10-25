"""Reading result tables back. Always use this instead of pd.read_csv for pipeline outputs.

Video ids are text. A video named e.g. "1012.mp4" has id "1012", which pandas would
otherwise read as the number 1012 in one file and keep as text in another, so rows
from different files silently stop matching (this once made talk time show 0).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TEXT_COLUMNS = {"video_id": str, "clip_id": str, "coder": str}


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=TEXT_COLUMNS, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
