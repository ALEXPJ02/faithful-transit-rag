"""Tests for naive persistence and, above all, the fairness of the comparison.

The baseline cannot predict the first stop of a trip. Scoring the model on rows
the baseline was never offered would flatter it by roughly the share of first
stops, so the mask that keeps both honest is tested here rather than assumed.
"""

from __future__ import annotations

import pandas as pd

from transit_rag.prediction.model import baseline


def table_with_first_stops() -> pd.DataFrame:
    """Two trips, each starting with a stop that has no predecessor."""
    return pd.DataFrame(
        {
            "trip_id": ["a", "a", "a", "b", "b"],
            "delay_s": [0.0, 60.0, 90.0, 0.0, 30.0],
            "prev_stop_delay_s": [None, 0.0, 60.0, None, 0.0],
        }
    )


class TestPredict:
    def test_carries_the_previous_stops_delay_forward(self) -> None:
        table = table_with_first_stops()

        predicted = baseline.predict(table)

        assert list(predicted[1:3]) == [0.0, 60.0]

    def test_first_stops_are_nan_not_a_filled_in_zero(self) -> None:
        """Filling would assert 'on time' for the first stop of every trip and
        make the baseline look better than it is."""
        predicted = baseline.predict(table_with_first_stops())

        assert predicted.isna().tolist() == [True, False, False, True, False]


class TestScoreable:
    def test_masks_out_rows_the_baseline_cannot_predict(self) -> None:
        mask = baseline.scoreable(table_with_first_stops())

        assert mask.tolist() == [False, True, True, False, True]

    def test_the_masked_frame_has_no_missing_baseline_input(self) -> None:
        table = table_with_first_stops()

        comparable = table[baseline.scoreable(table)]

        assert not baseline.predict(comparable).isna().any()

    def test_the_mask_is_what_makes_the_comparison_fair(self) -> None:
        """Both predictors must be scored on the same rows. This pins the count
        so a change that widens one side without the other is caught."""
        table = table_with_first_stops()

        comparable = table[baseline.scoreable(table)]

        assert len(comparable) == 3
        assert len(baseline.predict(comparable)) == len(comparable["delay_s"])
