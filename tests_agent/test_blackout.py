"""Уровень 4: падение сервера после записи в мир и повторный запуск без дублей."""

from pathlib import Path

import pytest
from agents.testing import assistant_message, function_call

from evals import graders

from .conftest import eligible_venue, letters, read_json, slot_plan
from .test_harness import launch, state


def steps_for(venue: dict) -> dict[str, list]:
    return {
        "book": [function_call("book_venue", {"venue_id": venue["id"]}, call_id="c-book")],
        "slots": [function_call("assign_slots", {"assignments": slot_plan(venue)}, call_id="c-slots")],
        "mail": [function_call("send_notifications", {"messages": letters()}, call_id="c-mail")],
        "report": [assistant_message("Перенос завершён.")],
    }


def crash(monkeypatch, point: str, steps: list, *argv: str) -> None:
    monkeypatch.setenv("CRASH_AT", point)
    with pytest.raises(SystemExit) as exit_info:
        launch(monkeypatch, steps, *argv)
    assert exit_info.value.code == 1
    monkeypatch.delenv("CRASH_AT")


def assert_clean_finish(run_dir: Path) -> None:
    assert state(run_dir)["status"] == "done"
    facts = graders.collect_facts(run_dir)
    assert not any(facts.get(key) for key in ("duplicate_bookings", "duplicate_notifications", "slot_conflicts"))
    assert len(read_json(run_dir / "bookings.json")) == 1
    assert graders.grade_folder(run_dir, scenario=5)["passed"]


def test_crash_after_first_slot_then_relaunch(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    step = steps_for(venue)
    first_team = slot_plan(venue)[0]["team_id"]
    launch(monkeypatch, [step["book"]])

    crash(monkeypatch, f"after_slot:{first_team}", [step["book"], step["slots"]], "--approve")

    assert state(run_dir)["status"] == "running"
    assert list(read_json(run_dir / "schedule.json")) == [first_team]
    journal = (run_dir / "journal.jsonl").read_text(encoding="utf-8")
    assert '"tool_start", "tool": "assign_slots"' in journal and '"launch_crashed"' in journal

    launch(monkeypatch, [step["slots"], step["mail"], step["report"]])

    assert_clean_finish(run_dir)
    # Пауза, одобрение с падением и перезапуск: потрачено две попытки, одна остаётся в запасе.
    limit = read_json(run_dir / "data" / "config.json")["max_attempts"]
    assert state(run_dir)["attempts"] == 2 < limit


def test_crash_after_first_letter_then_relaunch(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    step = steps_for(venue)
    first_team = read_json(run_dir / "data" / "teams.json")[0]
    launch(monkeypatch, [step["book"]])

    crash(
        monkeypatch,
        f"after_send:{first_team['id']}",
        [step["book"], step["slots"], step["mail"]],
        "--approve",
    )

    assert [m["recipient"] for m in read_json(run_dir / "outbox.json")] == [first_team["captain_email"]]

    launch(monkeypatch, [step["mail"], step["report"]])

    assert_clean_finish(run_dir)
    outbox = read_json(run_dir / "outbox.json")
    assert len(outbox) == len({m["recipient"] for m in outbox}) == len(letters())


def test_relaunch_with_another_plan_does_not_duplicate_or_conflict(run_dir: Path, monkeypatch):
    """После падения новая сессия модели может предложить другую сетку. Мир остаётся согласованным."""
    venue = eligible_venue(run_dir)
    step = steps_for(venue)
    plan = slot_plan(venue)
    first_team = plan[0]["team_id"]
    launch(monkeypatch, [step["book"]])
    crash(monkeypatch, f"after_slot:{first_team}", [step["book"], step["slots"]], "--approve")

    # Вторая сетка: первая и вторая команды меняются слотами.
    swapped = [dict(item) for item in plan]
    for key in ("mentor_slot", "pitch_slot"):
        swapped[0][key], swapped[1][key] = plan[1][key], plan[0][key]
    other = [function_call("assign_slots", {"assignments": swapped}, call_id="c-slots-2")]

    launch(monkeypatch, [other, step["mail"], step["report"]])

    assert_clean_finish(run_dir)
    assert read_json(run_dir / "schedule.json")[first_team]["pitch_slot"] == plan[1]["pitch_slot"]
