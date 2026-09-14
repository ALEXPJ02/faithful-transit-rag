"""Naive persistence: the number the model has to beat.

*"This trip's delay at the next stop equals its last observed delay."*
(``docs/00-overview.md`` glossary.)

Written before the model on purpose. A baseline chosen after seeing the model's
score is not a baseline, it is a justification -- and on a network where most
trains are on time, persistence is genuinely strong. If XGBoost cannot beat it,
that is the finding, and it is the one ``docs/04-implementation-plan.md``'s
supervisor item 5 is asking about.

**It cannot predict every row.** The first stop of a trip has no previous stop,
so ``prev_stop_delay_s`` is null there -- about 6% of rows. Those rows are
excluded from *both* sides of the comparison via :func:`scoreable`, because
scoring the model on rows the baseline was never offered would flatter it.
"""

from __future__ import annotations

import pandas as pd

#: The column persistence copies forward.
PERSISTENCE_SOURCE = "prev_stop_delay_s"


def scoreable(table: pd.DataFrame) -> pd.Series:
    """Rows on which model and baseline can be compared fairly.

    A boolean mask: true where the baseline has an input to work from. Apply it
    to both predictors, never to just one.
    """
    return table[PERSISTENCE_SOURCE].notna()


def predict(table: pd.DataFrame) -> pd.Series:
    """The baseline's prediction: carry the previous stop's delay forward.

    Rows without a previous stop yield NaN rather than a filled-in zero. Filling
    would quietly assert "on time" for the first stop of every trip and make the
    baseline look better than it is.
    """
    return table[PERSISTENCE_SOURCE].astype(float)
