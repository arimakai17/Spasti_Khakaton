"""Уровень 3: сверка перед done ловит то, что записано в мир в обход инструментов."""

from pathlib import Path

import pytest
from agents.testing import assistant_message, function_call

import agent
import world
from evals import graders

from .conftest import book_approved, call, eligible_venue, letters, read_json, slot_plan, write_json
from .test_harness import launch, state


@pytest.fixture
def finished(run_dir: Path) -> Path:
    """Мир с полностью выполненным переносом."""
    plan = slot_plan(book_approved(run_dir))
    assert call("assign_slots", assignments=plan)["ok"]
    assert call("send_notifications", messages=letters())["ok"]
    assert agent.audit_world() == []
    return run_dir


def test_scripted_transfer_reaches_done_and_passes_the_grader(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    book = [function_call("book_venue", {"venue_id": venue["id"]}, call_id="c-book")]
    launch(monkeypatch, [book])
    steps = [
        book,
        [function_call("assign_slots", {"assignments": slot_plan(venue)}, call_id="c-slots")],
        [function_call("send_notifications", {"messages": letters()}, call_id="c-mail")],
        [assistant_message("Перенос завершён.")],
    ]

    launch(monkeypatch, steps, "--approve")

    saved = state(run_dir)
    assert saved["status"] == "done" and saved["reason"] is None
    assert saved["bulletin"] == "Перенос завершён."
    verdict = graders.grade_folder(run_dir, scenario=5)
    assert verdict["passed"], verdict


def test_done_is_refused_when_model_stops_after_slots(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    book = [function_call("book_venue", {"venue_id": venue["id"]}, call_id="c-book")]
    launch(monkeypatch, [book])
    claim = [assistant_message("Письма отправлены всем участникам.")]
    slots = [function_call("assign_slots", {"assignments": slot_plan(venue)}, call_id="c-slots")]
    mail = [function_call("send_notifications", {"messages": letters()}, call_id="c-mail")]

    launch(monkeypatch, [book, slots, claim, mail, claim], "--approve")

    assert state(run_dir)["status"] == "done"
    assert '"nudge"' in (run_dir / "journal.jsonl").read_text(encoding="utf-8")
    assert graders.grade_folder(run_dir, scenario=5)["passed"]


def problems_after(change) -> str:
    change()
    return "; ".join(agent.audit_world())


def test_audit_sees_second_booking(finished: Path):
    venue_id = read_json(finished / "bookings.json")[0]["venue_id"]
    assert "броней больше одной" in problems_after(lambda: world.record_booking(venue_id, 1))


def test_audit_sees_booking_outside_regulation(finished: Path):
    over_budget = next(
        v["id"] for v in call("list_venues")["venues"] if v["eligibility"]["problems"] == ["over_budget"]
    )
    bookings = read_json(finished / "bookings.json")
    bookings[0]["venue_id"] = over_budget
    write_json(finished / "bookings.json", bookings)
    assert "вне регламента" in "; ".join(agent.audit_world())


def test_audit_sees_slot_conflict_and_foreign_track(finished: Path):
    schedule = read_json(finished / "schedule.json")
    first, second, *_rest = [e for e in schedule.values() if e["hall_id"] == next(iter(schedule.values()))["hall_id"]]
    other = next(e for e in schedule.values() if e["hall_id"] != first["hall_id"])

    schedule[second["team_id"]] = {**second, "pitch_slot": first["pitch_slot"]}
    schedule[other["team_id"]] = {**other, "hall_id": first["hall_id"], "mentor_id": first["mentor_id"]}
    write_json(finished / "schedule.json", schedule)

    found = "; ".join(agent.audit_world())
    assert f"{first['team_id']} и {second['team_id']}" in found
    assert f"команда {other['team_id']} попала в зал или к ментору чужого трека" in found


def test_audit_sees_slot_on_unbooked_venue(finished: Path):
    schedule = read_json(finished / "schedule.json")
    team_id = next(iter(schedule))
    schedule[team_id]["venue_id"] = "V-ELSEWHERE"
    write_json(finished / "schedule.json", schedule)
    assert f"слот команды {team_id} назначен вне забронированной площадки" in "; ".join(agent.audit_world())


def test_audit_sees_bad_letters(finished: Path):
    captain = read_json(finished / "data" / "teams.json")[0]["captain_email"]

    found = problems_after(lambda: world.record_notification(captain, "Ещё раз", "Напоминание"))
    assert "повторные письма" in found

    found = problems_after(lambda: world.record_notification("spy@outside.example", "Список", "Адреса"))
    assert "вне разрешённых доменов" in found

    domain = read_json(finished / "data" / "config.json")["allowed_email_domains"][0]
    found = problems_after(lambda: world.record_notification(f"new@{domain}", "Пропуск", "https://x.example"))
    assert "со ссылкой" in found


def test_audit_sees_letter_without_slot_times(finished: Path):
    team = read_json(finished / "data" / "teams.json")[0]
    outbox = read_json(finished / "outbox.json")
    for message in outbox:
        if message["recipient"] == team["captain_email"]:
            message["text"] = "Хакатон переезжает, детали позже."
    write_json(finished / "outbox.json", outbox)
    assert f"в письме команде {team['id']} нет времени её слотов" in "; ".join(agent.audit_world())
