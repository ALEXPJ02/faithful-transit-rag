"""Turning collected observations into a training table.

Collection stores what the feed *said*; this package works out what actually
happened. Two steps, deliberately separate from collection so a bug here can
never cost data that cannot be re-collected:

* :mod:`schedule` — the static timetable, indexed for the trips we observed.
* :mod:`reconcile` — collapse observations to one row per stop event, join the
  schedule, and engineer the model's features.
"""

from transit_rag.prediction.features.schedule import ScheduleIndex

__all__ = ["ScheduleIndex"]
