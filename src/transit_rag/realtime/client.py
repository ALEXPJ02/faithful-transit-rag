"""HTTP access to the TfNSW GTFS-Realtime feeds."""

from __future__ import annotations

import logging
from typing import Any

import requests

from transit_rag.config import TfnswConfig

log = logging.getLogger(__name__)


class FeedFetchError(RuntimeError):
    """The feed could not be fetched or did not parse as a GTFS-Realtime message."""


class GtfsRealtimeClient:
    """Thin wrapper over the TfNSW realtime endpoints.

    Kept deliberately small: it fetches bytes, parses them into a
    ``FeedMessage`` and raises a single error type. Interpretation of the
    feed belongs in :mod:`transit_rag.realtime.parsing` so it can be tested
    without a network call.
    """

    def __init__(self, config: TfnswConfig, timeout_seconds: int = 30) -> None:
        self._config = config
        self._timeout = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"apikey {config.api_key}"})

    def fetch_trip_updates(self) -> Any:
        """Return the parsed Trip Update feed (predicted delays per stop)."""
        return self._fetch(self._config.trip_update_url)

    def fetch_alerts(self) -> Any:
        """Return the parsed Service Alerts feed (disruptions, planned works)."""
        return self._fetch(self._config.alerts_url)

    def _fetch(self, url: str) -> Any:
        # Imported lazily so that importing this module (for example to read
        # its docstring, or in an environment that only needs config) does not
        # require the optional `realtime` extra to be installed.
        from google.transit import gtfs_realtime_pb2

        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FeedFetchError(f"GET {url} failed: {exc}") from exc

        feed = gtfs_realtime_pb2.FeedMessage()
        try:
            feed.ParseFromString(response.content)
        except Exception as exc:
            # A wrong URL usually returns HTML or JSON with a 200, which fails
            # here rather than at raise_for_status — worth saying so plainly,
            # because a silently mis-parsing feed collects nothing for days.
            preview = response.content[:200]
            raise FeedFetchError(
                f"Response from {url} is not a GTFS-Realtime protobuf "
                f"(first bytes: {preview!r}). Confirm the endpoint path against "
                "your Open Data Hub account — see docs/05-setup-checklist.md."
            ) from exc
        return feed
