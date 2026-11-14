"""Случаи с границ: вторая площадка, ошибки модели, флаги конфига, записи в обход инструментов."""

import sys
from pathlib import Path

import pytest
from agents.testing import assistant_message, function_call

import agent
import mcp_server
import reset
import world
from mcp_server import Hall, Mentor, SlotAssignment, SlotEntry, SlotRules, Team, Venue, slot_refusal

from .conftest import (
    book_approved,
    call,
    eligible_venue,
    letters,
    read_json,
    slot_plan,
    statuses,
    venue_with,
    write_json,
)
from .modified_world import SECRET
from .test_harness import launch, state


def book(venue_id: str, call_id: str = "c-book") -> list:
    return [function_call("book_venue", {"venue_id": venue_id}, call_id=call_id)]


def open_second_venue(run_dir: Path, cost_kzt: int) -> str:
    """Делает тесную площадку раздатки пригодной по регламенту и задаёт ей цену."""
    spare = venue_with("capacity_too_small")["id"]
    path = run_dir / "data" / "venues.json"
    venues = read_json(path)
    for record in venues:
        if record["id"] == spare:
            record["total_capacity"], record["cost_kzt"] = 100, cost_kzt
    write_json(path, venues)
    return spare


# --- Вторая площадка ---------------------------------------------------------------------


def test_reject_moves_to_the_alternative_and_asks_the_operator_again(run_dir: Path, monkeypatch):
    first = eligible_venue(run_dir)["id"]
    threshold = read_json(run_dir / "data" / "config.json")["hitl_threshold_kzt"]
    second = open_second_venue(run_dir, cost_kzt=threshold + 1)
    launch(monkeypatch, [book(first)])

    launch(monkeypatch, [book(first, "c-again"), book(second, "c-second")], "--reject")

    saved = state(run_dir)
    assert saved["status"] == "waiting_approval" and saved["decision"] is None
    assert saved["rejected_venues"] == [first] and saved["checkpoint"]["venue_id"] == second
    assert not (run_dir / "bookings.json").exists()

    launch(monkeypatch, [book(second), *[[assistant_message("Готово.")]] * 3], "--approve")

    assert [b["venue_id"] for b in read_json(run_dir / "bookings.json")] == [second]
    # Две паузы у оператора идут в счёт одной попытки.
    assert state(run_dir)["attempts"] == 1


def test_refused_venue_with_alternative_does_not_end_the_task(run_dir: Path, monkeypatch):
    reset.reset_world(scenario=4, workdir=run_dir)
    broken = eligible_venue(run_dir)["id"]
    spare = open_second_venue(run_dir, cost_kzt=1)
    launch(monkeypatch, [book(broken)])

    launch(monkeypatch, [book(broken), book(spare, "c-spare"), *[[assistant_message("Готово.")]] * 3], "--approve")

    saved = state(run_dir)
    assert saved["refused_venues"] == [broken] and saved["status"] == "failed"
    assert [b["venue_id"] for b in read_json(run_dir / "bookings.json")] == [spare]


# --- Ошибки модели не роняют запуск ------------------------------------------------------


def test_unknown_tool_from_model_is_survived(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    ghost = [function_call("teleport_hackathon", {}, call_id="c-ghost")]

    launch(monkeypatch, [ghost, book(venue["id"])])

    assert state(run_dir)["status"] == "waiting_approval"
    assert '"model_error"' in (run_dir / "journal.jsonl").read_text(encoding="utf-8")


def test_arguments_outside_schema_are_journaled_and_survived(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)

    launch(monkeypatch, [book("no such id!", "c-bad"), book(venue["id"])])

    assert state(run_dir)["status"] == "waiting_approval"
    journal = (run_dir / "journal.jsonl").read_text(encoding="utf-8")
    assert '"event": "tool_error", "tool": "book_venue"' in journal


def test_unknown_skill_name_does_not_activate_anything(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    load = [function_call("load_skill", {"name": "no-such-skill"}, call_id="c-skill")]

    launch(monkeypatch, [load, book(venue["id"])])

    assert state(run_dir)["skill_activated"] is False


# --- Флаги запуска -----------------------------------------------------------------------


def test_query_respects_turn_limit(run_dir: Path, monkeypatch, capsys):
    step = [function_call("get_status", {}, call_id="c-status")]

    launch(monkeypatch, [step], "--query", "Что с переносом?", "--max-turns", "1")

    assert "Ответ не получен" in capsys.readouterr().out
    assert state(run_dir)["status"] == "new"


def test_query_says_that_decision_flag_is_skipped(run_dir: Path, monkeypatch, capsys):
    launch(monkeypatch, [[assistant_message("Ответ.")]], "--query", "Что с переносом?", "--approve")

    assert "флаг пропущен" in capsys.readouterr().out
    assert state(run_dir)["decision"] is None and state(run_dir)["attempts"] == 0


def test_zero_turn_limit_is_rejected_by_cli(run_dir: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["agent.py", "--max-turns", "0"])
    with pytest.raises(SystemExit) as exit_info:
        agent.parse_args()
    assert exit_info.value.code == 2


# --- Правила сетки на своих данных -------------------------------------------------------

TEAM_A = Team(id="T-A", size=4, track="red", captain_email="a@inside.test")
TEAM_B = Team(id="T-B", size=4, track="red", captain_email="b@inside.test")
MENTORS = {"M-R": Mentor(id="M-R", track="red", available_slots=["09:00-09:30", "09:30-10:00"], email="m@inside.test")}
SLOTS = ["09:00-09:30", "09:30-10:00", "14:00-14:15", "14:15-14:30"]
TWO_HALLS = Venue(
    id="V-X",
    available=True,
    total_capacity=50,
    cost_kzt=1,
    halls=[
        Hall(hall_id="H-1", capacity=10, tracks=["red"], available_slots=SLOTS),
        Hall(hall_id="H-2", capacity=3, tracks=["red"], available_slots=SLOTS),
    ],
)


def assignment(team: Team, hall: str, mentor_slot: str, pitch_slot: str) -> SlotAssignment:
    return SlotAssignment(
        team_id=team.id, hall_id=hall, mentor_id="M-R", mentor_slot=mentor_slot, pitch_slot=pitch_slot
    )


def entry_of(item: SlotAssignment) -> SlotEntry:
    return SlotEntry(track="red", venue_id="V-X", **item.model_dump())


def test_one_mentor_cannot_meet_two_teams_at_once_even_in_different_halls():
    taken = entry_of(assignment(TEAM_A, "H-1", SLOTS[0], SLOTS[2]))
    same_time_other_hall = assignment(TEAM_B, "H-2", SLOTS[0], SLOTS[3])
    small_team = TEAM_B.model_copy(update={"size": 3})

    refusal = slot_refusal(same_time_other_hall, small_team, TWO_HALLS, MENTORS, [taken], SlotRules())
    allowed = slot_refusal(
        same_time_other_hall, small_team, TWO_HALLS, MENTORS, [taken], SlotRules(allow_mentor_time_overlap=True)
    )

    assert refusal[0] == "mentor_conflict" and allowed is None


def test_hall_overlap_follows_the_config_flag():
    taken = entry_of(assignment(TEAM_A, "H-1", SLOTS[0], SLOTS[2]))
    same_pitch = assignment(TEAM_B, "H-1", SLOTS[1], SLOTS[2])

    assert slot_refusal(same_pitch, TEAM_B, TWO_HALLS, MENTORS, [taken], SlotRules())[0] == "hall_conflict"
    assert slot_refusal(same_pitch, TEAM_B, TWO_HALLS, MENTORS, [taken], SlotRules(allow_hall_time_overlap=True)) is None


def test_team_must_fit_into_the_hall_when_capacity_is_known():
    item = assignment(TEAM_A, "H-2", SLOTS[0], SLOTS[2])
    unknown_capacity = TWO_HALLS.model_copy(
        update={"halls": [Hall(hall_id="H-2", tracks=["red"], available_slots=SLOTS)]}
    )

    assert slot_refusal(item, TEAM_A, TWO_HALLS, MENTORS, [], SlotRules())[0] == "hall_too_small"
    assert slot_refusal(item, TEAM_A, TWO_HALLS, MENTORS, [], SlotRules(require_venue_capacity_fit=False)) is None
    assert slot_refusal(item, TEAM_A, unknown_capacity, MENTORS, [], SlotRules()) is None


def test_audit_follows_overlap_flags_from_config(run_dir: Path):
    plan = slot_plan(book_approved(run_dir))
    call("assign_slots", assignments=plan)
    schedule = read_json(run_dir / "schedule.json")
    first, second = plan[0]["team_id"], plan[1]["team_id"]
    schedule[second]["pitch_slot"] = schedule[first]["pitch_slot"]
    write_json(run_dir / "schedule.json", schedule)
    assert "пересечения в расписании" in "; ".join(agent.audit_world())

    config = read_json(run_dir / "data" / "config.json")
    config["slot_conflict_rules"]["allow_hall_time_overlap"] = True
    write_json(run_dir / "data" / "config.json", config)

    assert "пересечения в расписании" not in "; ".join(agent.audit_world())


# --- Письма ------------------------------------------------------------------------------


@pytest.fixture
def scheduled(run_dir: Path) -> dict:
    venue = book_approved(run_dir)
    plan = slot_plan(venue)
    assert call("assign_slots", assignments=plan)["ok"]
    return {"venue": venue, "plan": plan, "dir": run_dir}


def test_slot_is_frozen_after_the_mentor_letter(scheduled):
    run_dir, item = scheduled["dir"], scheduled["plan"][0]
    mentor = next(m for m in read_json(run_dir / "data" / "mentors.json") if m["id"] == item["mentor_id"])
    assert statuses(call("send_notifications", messages=[{**letters()[0], "recipient": mentor["email"]}])) == ["sent"]
    before = read_json(run_dir / "schedule.json")

    reply = call("assign_slots", assignments=[{**item, "pitch_slot": scheduled["plan"][1]["pitch_slot"]}])

    assert statuses(reply) == ["already_notified"]
    assert read_json(run_dir / "schedule.json") == before


def test_audit_sees_mentor_letter_without_session_times(scheduled):
    run_dir = scheduled["dir"]
    assert call("send_notifications", messages=letters())["ok"] and agent.audit_world() == []
    mentor = read_json(run_dir / "data" / "mentors.json")[0]
    outbox = read_json(run_dir / "outbox.json")
    for message in outbox:
        if message["recipient"] == mentor["email"]:
            message["text"] = "Хакатон переезжает, расписание пришлём позже."
    write_json(run_dir / "outbox.json", outbox)

    assert f"в письме ментору {mentor['id']} нет времени его сессий" in "; ".join(agent.audit_world())


def test_subject_with_line_break_is_rejected_by_schema(scheduled):
    captain = letters()[0]
    with pytest.raises(Exception, match="subject"):
        call("send_notifications", messages=[{**captain, "subject": "Перенос\r\nBcc: spy@outside.example"}])
    assert not (scheduled["dir"] / "outbox.json").exists()


def test_hall_name_that_looks_like_a_path_does_not_block_letters(run_dir: Path):
    """Запись ссылки без схемы ищется в тексте модели, а не в блоке, который дописывает сервер."""
    path = run_dir / "data" / "venues.json"
    venues = read_json(path)
    for record in venues:
        for hall in record["halls"]:
            hall["name"] = "Корпус B.2/3, аудитория 2.1/7"
    write_json(path, venues)
    plan = slot_plan(book_approved(run_dir))
    call("assign_slots", assignments=plan)

    reply = call("send_notifications", messages=letters())

    assert reply["ok"] and set(statuses(reply)) == {"sent"}
    assert agent.audit_world() == []


@pytest.mark.parametrize("foreign_key", [None, "made-by-someone-else"])
def test_records_made_without_the_tool_still_count_as_done(run_dir: Path, foreign_key):
    """Запись без ключа или с чужим ключом (сделана в обход инструмента) не приводит к дублю."""
    venue = book_approved(run_dir)
    call("assign_slots", assignments=slot_plan(venue))
    captain = letters()[0]
    meta = {"idempotency_key": foreign_key} if foreign_key else {}
    world.record_notification(captain["recipient"], "Раньше", "Письмо в обход инструмента", **meta)
    bookings = read_json(run_dir / "bookings.json")
    bookings[0].pop("idempotency_key")
    bookings[0].update(meta)
    write_json(run_dir / "bookings.json", bookings)

    assert statuses(call("send_notifications", messages=[captain])) == ["already_sent"]
    assert call("book_venue", venue_id=venue["id"])["status"] == "already_booked"
    assert len(read_json(run_dir / "outbox.json")) == 1 and len(read_json(run_dir / "bookings.json")) == 1


# --- Секрет внутри разрешённого поля -----------------------------------------------------


def test_token_inside_allowed_field_is_cut_out(run_dir: Path, monkeypatch):
    """Токен в тексте письма и в названии площадки, которая попадёт в контрольную точку."""
    token = f"{SECRET}inline-0001"
    venue = eligible_venue(run_dir)
    for name, field, wanted in (("inbox.json", "text", None), ("venues.json", "name", venue["id"])):
        path = run_dir / "data" / name
        records = read_json(path)
        for record in records:
            if wanted in (None, record["id"]):
                record[field] = f"{record[field]} {token}"
        write_json(path, records)

    reads = [function_call(name, {}, call_id=f"c-{name}") for name in ("list_venues", "read_inbox", "get_roster")]
    launch(monkeypatch, [reads, book(venue["id"])])

    saved = state(run_dir)
    assert saved["status"] == "waiting_approval" and "[скрыто]" in saved["checkpoint"]["venue_name"]
    assert "[скрыто]" in str(call("read_inbox")) and token not in str(call("read_inbox"))
    for path in [run_dir / "state.json", run_dir / "journal.jsonl", *run_dir.glob("sessions.db*")]:
        assert SECRET.encode() not in path.read_bytes(), path.name
    assert mcp_server.scrub({"a": [token, 1, None]}) == {"a": ["[скрыто]", 1, None]}
