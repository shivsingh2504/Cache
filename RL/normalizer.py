from __future__ import annotations

import numpy as np

# Columns that receive a log1p compression. Mirrors the column-index
# convention already used in baseline.py (_FREQUENCY_COL, _RECENCY_COL, ...).
_LOG1P_COLS = (0, 1, 2, 3)  # frequency, recency, key_age, predicted_popularity
_IS_EMPTY_COL = 4  # left unchanged


class StateNormalizer:
    def normalize(self, state: np.ndarray) -> np.ndarray:
        normalized = state.copy()
        normalized[:, _LOG1P_COLS] = np.log1p(normalized[:, _LOG1P_COLS])
        return normalized