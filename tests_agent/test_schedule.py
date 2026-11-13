"""Уровень 3: проверки инструмента assign_slots."""

from pathlib import Path

import pytest

from evals import graders

from .conftest import book_approved, call, eligible_venue, letters, read_json, slot_plan, statuses


def test_slots_need_a_booking(run_dir: Path):
    plan = slot_plan(eligible_venue(run_dir))

    reply = call("assign_slots", assignments=plan)

    assert reply["ok"] is False and reply["status"] == "no_booking"
    assert not (run_dir / "schedule.json").exists()


def test_full_plan_is_recorded_without_conflicts(run_dir: Path):
    plan = slot_plan(book_approved(run_dir))

    reply = call("assign_slots", assignments=plan)

    assert reply["ok"] and set(statuses(reply)) == {"assigned"}
    assert reply["teams_without_slots"] == []
    facts = graders.collect_facts(run_dir)
    assert facts["assigned_teams"] == len(plan) == len(read_json(run_dir / "data" / "teams.json"))
    assert not any(facts.get(key) for key in ("slot_conflicts", "track_mismatch", "unbooked_venue_slots"))
    record = read_json(run_dir / "schedule.json")[plan[0]["team_id"]]
    assert record["idempotency_key"] == f"slot:{plan[0]['team_id']}"


def test_repeated_plan_changes_nothing(run_dir: Path):
    plan = slot_plan(book_approved(run_dir))
    call("assign_slots", assignments=plan)
    before = read_json(run_dir / "schedule.json")

    reply = call("assign_slots", assignments=plan)

    assert reply["ok"] and set(statuses(reply)) == {"unchanged"}
    assert read_json(run_dir / "schedule.json") == before


def other_track(plan: list[dict], item: dict) -> dict:
    return next(p for p in plan if p["hall_id"] != item["hall_id"])


@pytest.mark.parametrize(
    ("change", "status"),
    [
        (lambda plan, item: {"hall_id": other_track(plan, item)["hall_id"]}, "hall_track_mismatch"),
        (lambda plan, item: {"mentor_id": other_track(plan, item)["mentor_id"]}, "mentor_track_mismatch"),
        (lambda plan, item: {"mentor_slot": item["pitch_slot"]}, "mentor_unavailable"),
        (lambda plan, item: {"pitch_slot": "23:00-23:15"}, "hall_slot_unavailable"),
        (lambda plan, item: {"pitch_slot": item["mentor_slot"]}, "team_double_booked"),
        (lambda plan, item: {"hall_id": "H-NOPE"}, "unknown_hall"),
        (lambda plan, item: {"mentor_id": "M-NOPE"}, "unknown_mentor"),
        (lambda plan, item: {"team_id": "T-NOPE"}, "unknown_team"),
    ],
)
def test_bad_assignment_is_refused(run_dir: Path, change, status: str):
    plan = slot_plan(book_approved(run_dir))
    item = {**plan[0], **change(plan, plan[0])}

    reply = call("assign_slots", assignments=[item])

    assert reply["ok"] is False and statuses(reply) == [status]
    assert not (run_dir / "schedule.json").exists()


def test_second_team_in_same_slot_is_refused_and_first_stays(run_dir: Path):
    plan = slot_plan(book_approved(run_dir))
    first, second = plan[0], plan[1]
    assert first["hall_id"] == second["hall_id"]

    same_pitch = {**second, "pitch_slot": first["pitch_slot"]}
    same_mentor_time = {**second, "mentor_slot": first["mentor_slot"]}
    reply = call("assign_slots", assignments=[first, same_pitch, same_mentor_time])

    assert reply["status"] == "partial"
    assert statuses(reply) == ["assigned", "hall_conflict", "hall_conflict"]
    assert list(read_json(run_dir / "schedule.json")) == [first["team_id"]]


def test_slot_can_change_until_the_team_is_notified(run_dir: Path):
    plan = slot_plan(book_approved(run_dir))
    first, second = plan[0], plan[1]
    call("assign_slots", assignments=[first])

    moved = {**first, "mentor_slot": second["mentor_slot"], "pitch_slot": second["pitch_slot"]}
    assert statuses(call("assign_slots", assignments=[moved])) == ["assigned"]
    record = read_json(run_dir / "schedule.json")
    assert list(record) == [first["team_id"]] and record[first["team_id"]]["pitch_slot"] == second["pitch_slot"]

    captain = letters()[0]
    assert statuses(call("send_notifications", messages=[captain])) == ["sent"]
    assert statuses(call("assign_slots", assignments=[first])) == ["already_notified"]
    assert read_json(run_dir / "schedule.json") == record


def test_unknown_field_in_assignment_is_rejected_by_schema(run_dir: Path):
    plan = slot_plan(book_approved(run_dir))
    with pytest.raises(Exception, match="venue_id"):
        call("assign_slots", assignments=[{**plan[0], "venue_id": "V-ANY"}])
