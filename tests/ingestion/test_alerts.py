"""Tests for turning collected alerts into incidents.

The alert wording in these tests is TfNSW's own, taken from the collected
alerts it stands in for, because the rule under test reads that wording.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from transit_rag.ingestion.alerts import (
    MAX_REPUBLISH_GAP,
    AlertRecord,
    Incident,
    Override,
    audit,
    classify,
    group_incidents,
    load_alerts,
    load_overrides,
)
from transit_rag.ingestion.alerts_cli import main as alerts_main
from transit_rag.prediction.collection.store import SqliteObservationStore
from transit_rag.realtime.parsing import AlertScope, ServiceAlert

START = datetime(2026, 9, 21, 6, 11, tzinfo=UTC)  # 16:11 in Sydney

TRACKWORK = (
    "There is no train running at 1:47am from Central to Hornsby due to trackwork. "
    "If you were intending to travel on this service, the following alternate travel "
    "arrangements apply."
)
EDGECLIFF = (
    "Trains are not running between Bondi Junction and Central due to an incident "
    "requiring emergency services at Edgecliff."
)
STREET_CLOSURE = (
    "Due to a police operation at Redfern causing the closure of Gibbons and Regent "
    "Streets, rail customers interchanging at Redfern for bus connections will be affected."
)
REPAIRS = "Allow extra travel time due to urgent train repairs near Martin Place earlier."


def _alert(
    alert_id: str,
    *,
    cause: str = "TECHNICAL_PROBLEM",
    description: str = REPAIRS,
    start: datetime = START,
    minutes: int = 30,
    lines: tuple[str, ...] = ("T1",),
) -> AlertRecord:
    return AlertRecord(
        alert_id=alert_id,
        cause=cause,
        effect="UNKNOWN_EFFECT",
        header_text="North Shore Line",
        description_text=description,
        first_seen=start,
        last_seen=start + timedelta(minutes=minutes),
        lines=lines,
    )


class TestClassify:
    def test_planned_maintenance_is_never_an_incident(self) -> None:
        verdict = classify(_alert("a", cause="MAINTENANCE"))
        assert not verdict.is_incident
        assert verdict.reason == "planned maintenance"

    def test_a_standing_notice_is_excluded_however_it_is_worded(self) -> None:
        verdict = classify(_alert("a", minutes=25 * 60))
        assert not verdict.is_incident
        assert verdict.reason.startswith("standing notice")

    def test_trackwork_published_as_unknown_cause_is_excluded(self) -> None:
        """The case cause alone gets wrong: planned, but not MAINTENANCE."""
        verdict = classify(_alert("a", cause="UNKNOWN_CAUSE", description=TRACKWORK))
        assert not verdict.is_incident
        assert verdict.reason == "planned work: 'trackwork'"

    def test_an_unknown_cause_closure_is_an_incident(self) -> None:
        """The case a cause filter would delete: Edgecliff, posted as unknown."""
        verdict = classify(_alert("a", cause="UNKNOWN_CAUSE", description=EDGECLIFF))
        assert verdict.is_incident

    def test_an_alert_about_streets_not_trains_is_excluded(self) -> None:
        verdict = classify(_alert("a", cause="UNKNOWN_CAUSE", description=STREET_CLOSURE))
        assert not verdict.is_incident
        assert verdict.reason == "no effect on train running stated"

    def test_matching_ignores_case(self) -> None:
        assert classify(_alert("a", description="DELAYS of 15 minutes")).is_incident

    def test_an_override_wins_and_says_so(self) -> None:
        verdict = classify(
            _alert("a", description=TRACKWORK),
            Override(is_incident=True, reason="trackwork overran into the morning peak"),
        )
        assert verdict.is_incident
        assert verdict.reason == "override: trackwork overran into the morning peak"


class TestGrouping:
    def test_a_republication_chain_is_one_incident(self) -> None:
        """Chatswood: three alerts, each on the poll after the last one left."""
        first = _alert("a", cause="UNKNOWN_CAUSE", minutes=30)
        second = _alert("b", cause="UNKNOWN_CAUSE", start=START + timedelta(minutes=61), minutes=0)
        third = _alert("c", cause="UNKNOWN_CAUSE", start=START + timedelta(minutes=91))

        [incident] = group_incidents([third, first, second])
        assert incident.alert_ids == ("a", "b", "c")

    def test_an_unknown_cause_joins_its_named_republication(self) -> None:
        """Edgecliff went from unknown to police activity; the correction is
        the incident's cause."""
        posted = _alert("a", cause="UNKNOWN_CAUSE", description=EDGECLIFF, lines=("T1", "T4"))
        named = _alert("b", cause="POLICE_ACTIVITY", minutes=121, lines=("T4",))

        [incident] = group_incidents([posted, named])
        assert incident.cause == "POLICE_ACTIVITY"
        assert incident.cause_group == "network_incident"
        assert incident.lines == ("T1", "T4")

    def test_adjacent_incidents_with_different_named_causes_stay_apart(self) -> None:
        """15 Sep: an object near the tracks, then a car through a fence."""
        repair = _alert("a", cause="TECHNICAL_PROBLEM", minutes=0)
        crash = _alert("b", cause="ACCIDENT", start=START + timedelta(minutes=30))

        assert len(group_incidents([repair, crash])) == 2

    def test_no_shared_line_means_different_incidents(self) -> None:
        t1 = _alert("a", lines=("T1",))
        t4 = _alert("b", lines=("T4",), start=START + timedelta(minutes=10))

        assert len(group_incidents([t1, t4])) == 2

    def test_a_gap_longer_than_one_alert_poll_breaks_the_chain(self) -> None:
        earlier = _alert("a", minutes=0)
        later = _alert("b", start=START + MAX_REPUBLISH_GAP + timedelta(minutes=1))

        assert len(group_incidents([earlier, later])) == 2

    def test_an_incident_never_seen_with_a_cause_stays_unknown(self) -> None:
        [incident] = group_incidents([_alert("a", cause="UNKNOWN_CAUSE")])
        assert incident.cause == "UNKNOWN_CAUSE"
        assert incident.cause_group == "other_unknown"

    def test_the_incident_id_depends_on_membership_not_order(self) -> None:
        a, b = _alert("a"), _alert("b", start=START + timedelta(minutes=31))
        assert Incident(alerts=(a, b)).incident_id == Incident(alerts=(b, a)).incident_id
        assert Incident(alerts=(a,)).incident_id != Incident(alerts=(a, b)).incident_id


class TestAudit:
    def test_only_incidents_are_grouped_and_every_alert_gets_a_row(self) -> None:
        alerts = [
            _alert("trackwork", cause="UNKNOWN_CAUSE", description=TRACKWORK),
            _alert("repairs"),
        ]
        rows, incidents = audit(alerts)

        assert [row.incident_id is None for row in rows] == [True, False]
        assert [incident.alert_ids for incident in incidents] == [("repairs",)]

    def test_alerts_first_seen_after_the_freeze_are_marked_held_out(self) -> None:
        before = _alert("old", start=datetime(2026, 9, 28, 3, 0, tzinfo=UTC))
        # 15:30 UTC on the 28th is 01:30 on the 29th in Sydney.
        after = _alert("new", start=datetime(2026, 9, 28, 15, 30, tzinfo=UTC))

        rows, _ = audit([before, after])
        assert [row.held_out for row in rows] == [False, True]


class TestOverridesFile:
    def test_a_missing_file_means_no_overrides(self, tmp_path: Path) -> None:
        assert load_overrides(tmp_path / "absent.csv") == {}

    def test_rows_are_read(self, tmp_path: Path) -> None:
        path = tmp_path / "o.csv"
        path.write_text("alert_id,is_incident,reason\nabc,no,lift notice\n", encoding="utf-8")
        assert load_overrides(path) == {"abc": Override(is_incident=False, reason="lift notice")}

    def test_a_row_without_a_reason_is_an_error_not_a_skip(self, tmp_path: Path) -> None:
        path = tmp_path / "o.csv"
        path.write_text("alert_id,is_incident,reason\nabc,no,\n", encoding="utf-8")
        with pytest.raises(ValueError, match="reason"):
            load_overrides(path)

    def test_the_committed_overrides_file_parses(self) -> None:
        load_overrides(Path(__file__).parents[2] / "data" / "alert_overrides.csv")


def _collection_db(tmp_path: Path) -> Path:
    """A collection database written by the real store, as the collector would."""
    db = tmp_path / "snapshot.db"
    seen = "2026-09-21T06:11:00+00:00"
    alerts = [
        ServiceAlert("edgecliff", "UNKNOWN_CAUSE", "UNKNOWN_EFFECT", "UNKNOWN_SEVERITY",
                     "Station Update - Bondi Junction", EDGECLIFF, "", seen),
        ServiceAlert("t8-only", "TECHNICAL_PROBLEM", "UNKNOWN_EFFECT", "UNKNOWN_SEVERITY",
                     "Airport Line", REPAIRS, "", seen),
    ]  # fmt: skip
    scopes = [
        AlertScope("edgecliff", "ESI_1a", "T4", 0, "", 0, 0, seen),
        AlertScope("edgecliff", "NSN_2a", "T1", 0, "", 0, 0, seen),
        AlertScope("t8-only", "APS_1a", "T8", 0, "", 0, 0, seen),
    ]
    with SqliteObservationStore(db) as store:
        store.record_alerts(alerts, scopes)
    return db


class TestLoadAlerts:
    def test_reads_only_alerts_on_the_tracked_lines(self, tmp_path: Path) -> None:
        [alert] = load_alerts(_collection_db(tmp_path))
        assert alert.alert_id == "edgecliff"
        assert alert.lines == ("T1", "T4")
        assert alert.description_text == EDGECLIFF

    def test_the_snapshot_is_opened_read_only(self, tmp_path: Path) -> None:
        """These files are the only copy of data that cannot be re-collected."""
        db = _collection_db(tmp_path)
        before = db.read_bytes()
        load_alerts(db)
        assert db.read_bytes() == before

    def test_a_missing_snapshot_says_what_to_do(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="pull a snapshot"):
            load_alerts(tmp_path / "absent.db")


def test_the_audit_command_prints_every_alert_and_the_incidents(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = _collection_db(tmp_path)

    exit_code = alerts_main(["audit", "--db", str(db), "--overrides", str(tmp_path / "none.csv")])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Alerts naming T1 or T4" in output
    assert "service impact: 'not running'" in output
    assert "Incidents: 1 on 1 Sydney date(s)" in output
    assert "weather_external 0" in output
