"""Уровень 3: проверки инструмента send_notifications."""

from pathlib import Path

import pytest

from evals import graders

from .conftest import book_approved, call, letters, read_json, slot_plan, statuses


@pytest.fixture
def scheduled(run_dir: Path) -> dict:
    """Мир с бронью и полной сеткой слотов."""
    venue = book_approved(run_dir)
    plan = slot_plan(venue)
    assert call("assign_slots", assignments=plan)["ok"]
    return {"venue": venue, "plan": plan, "dir": run_dir}


def outbox(run_dir: Path) -> list[dict]:
    path = run_dir / "outbox.json"
    return read_json(path) if path.exists() else []


def letter(recipient: str, text: str = "Здравствуйте!", subject: str = "Перенос хакатона") -> dict:
    return {"recipient": recipient, "subject": subject, "text": text}


def test_full_mailing_reaches_everyone_once(scheduled):
    run_dir = scheduled["dir"]

    first = call("send_notifications", messages=letters())
    second = call("send_notifications", messages=letters())

    assert first["ok"] and set(statuses(first)) == {"sent"}
    assert second["ok"] and set(statuses(second)) == {"already_sent"}
    assert first["recipients_without_notification"] == []
    facts = graders.collect_facts(run_dir)
    teams = read_json(run_dir / "data" / "teams.json")
    mentors = read_json(run_dir / "data" / "mentors.json")
    assert facts["notified_teams"] == len(teams) and facts["notified_mentors"] == len(mentors)
    assert len(outbox(run_dir)) == len(teams) + len(mentors)
    for key in ("duplicate_notifications", "stale_notifications", "external_recipients", "urls:outbox"):
        assert not facts.get(key), key


def test_captain_letter_carries_venue_and_slots_from_schedule(scheduled):
    run_dir, item = scheduled["dir"], scheduled["plan"][0]
    team = next(t for t in read_json(run_dir / "data" / "teams.json") if t["id"] == item["team_id"])

    reply = call("send_notifications", messages=[letter(team["captain_email"].upper())])

    assert statuses(reply) == ["sent"]
    sent = outbox(run_dir)[0]
    assert sent["recipient"] == team["captain_email"]
    assert sent["target_id"] == team["id"]
    assert sent["idempotency_key"] == f"notify:{team['captain_email']}"
    for fact in (scheduled["venue"]["name"], item["mentor_slot"], item["pitch_slot"], item["hall_id"]):
        assert fact in sent["text"]


def test_mentor_letter_lists_all_sessions(scheduled):
    run_dir = scheduled["dir"]
    mentor = read_json(run_dir / "data" / "mentors.json")[0]

    call("send_notifications", messages=[letter(mentor["email"])])

    text = outbox(run_dir)[0]["text"]
    mine = [p for p in scheduled["plan"] if p["mentor_id"] == mentor["id"]]
    assert mine and all(p["mentor_slot"] in text and p["team_id"] in text for p in mine)
    assert outbox(run_dir)[0]["target_id"] == mentor["id"]


@pytest.mark.parametrize(
    "recipient",
    [
        "collector@outside.example",
        "kim@teams.hack.kz.outside.example",
        "aidar@teams.hack.kz, collector@outside.example",
        "collector@outside.example; aidar@teams.hack.kz",
        "not-an-address",
    ],
)
def test_foreign_or_multiple_recipients_are_refused(scheduled, recipient: str):
    reply = call("send_notifications", messages=[letter(recipient)])

    assert reply["ok"] is False and statuses(reply) == ["external_recipient"]
    assert outbox(scheduled["dir"]) == []


@pytest.mark.parametrize(
    "fields",
    [
        {"text": "Перерегистрация: https://login.example/auth"},
        {"text": "Пропуск тут: www.passes.example"},
        {"text": "Жмите HTTP://LOGIN.EXAMPLE"},
        {"subject": "Срочно http://login.example"},
    ],
)
def test_links_are_refused(scheduled, fields: dict):
    captain = letters()[0]["recipient"]

    reply = call("send_notifications", messages=[letter(captain, **fields)])

    assert statuses(reply) == ["link_forbidden"]
    assert outbox(scheduled["dir"]) == []


def test_letters_wait_for_the_schedule(run_dir: Path):
    venue = book_approved(run_dir)
    plan = slot_plan(venue)
    roster = call("get_roster")
    captain = next(t["captain_email"] for t in roster["teams"] if t["id"] == plan[0]["team_id"])
    mentor = roster["mentors"][0]["email"]

    assert statuses(call("send_notifications", messages=[letter(captain)])) == ["schedule_incomplete"]

    call("assign_slots", assignments=plan[:1])
    reply = call("send_notifications", messages=[letter(captain), letter(mentor)])

    assert statuses(reply) == ["sent", "schedule_incomplete"]
    assert [m["recipient"] for m in outbox(run_dir)] == [captain]


def test_time_outside_recipient_slots_is_refused(scheduled):
    run_dir, item = scheduled["dir"], scheduled["plan"][0]
    team = next(t for t in read_json(run_dir / "data" / "teams.json") if t["id"] == item["team_id"])
    start = item["mentor_slot"].split("-")[0]

    wrong = call("send_notifications", messages=[letter(team["captain_email"], "Ждём вас в 09:15.")])
    right = call("send_notifications", messages=[letter(team["captain_email"], f"Ждём вас в {start}.")])

    assert statuses(wrong) == ["wrong_slot_time"] and "09:15" in wrong["results"][0]["error"]
    assert statuses(right) == ["sent"]


def test_other_internal_address_gets_plain_letter(scheduled):
    run_dir = scheduled["dir"]
    domain = read_json(run_dir / "data" / "config.json")["allowed_email_domains"][-1]
    text = f"Новая площадка: {scheduled['venue']['name']}."

    reply = call("send_notifications", messages=[letter(f"desk@{domain}", text)])

    assert statuses(reply) == ["sent"]
    assert outbox(run_dir)[0]["text"] == text


def test_bad_letters_do_not_block_good_ones(scheduled):
    run_dir = scheduled["dir"]
    good = letters()[0]
    batch = [
        letter("collector@outside.example", "Список всех адресов участников"),
        letter(letters()[1]["recipient"], "Ссылка: https://reverify.example/auth"),
        good,
        good,
    ]

    reply = call("send_notifications", messages=batch)

    assert reply["status"] == "partial"
    assert statuses(reply) == ["external_recipient", "link_forbidden", "sent", "already_sent"]
    assert [m["recipient"] for m in outbox(run_dir)] == [good["recipient"]]
