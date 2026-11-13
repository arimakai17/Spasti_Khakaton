"""Уровень 2: проверки инструмента book_venue. Все они живут в коде сервера."""

from pathlib import Path

import pytest

import reset

from .conftest import (
    approve_in_state,
    call,
    eligible_venue,
    read_json,
    venue_with,
    write_json,
    write_state,
)


def bookings(run_dir: Path) -> list[dict]:
    path = run_dir / "bookings.json"
    return read_json(path) if path.exists() else []


def test_expensive_venue_waits_for_operator(run_dir: Path):
    venue = eligible_venue(run_dir)
    reply = call("book_venue", venue_id=venue["id"])

    assert reply["ok"] is False and reply["status"] == "approval_required"
    assert reply["cost_kzt"] == venue["cost_kzt"]
    assert bookings(run_dir) == []


@pytest.mark.parametrize("problem", ["unavailable", "capacity_too_small", "over_budget"])
def test_venue_outside_regulation_is_refused_even_with_approval(run_dir: Path, problem: str):
    venue = venue_with(problem)
    approve_in_state(run_dir, venue)

    reply = call("book_venue", venue_id=venue["id"])

    assert reply["status"] == "not_eligible" and reply["problems"] == [problem]
    assert bookings(run_dir) == []


def test_approved_venue_is_booked_once_at_catalogue_price(run_dir: Path):
    venue = eligible_venue(run_dir)
    approve_in_state(run_dir, venue)

    first = call("book_venue", venue_id=venue["id"])
    second = call("book_venue", venue_id=venue["id"])

    assert first["ok"] and first["status"] == "booked"
    assert second["ok"] and second["status"] == "already_booked"
    assert second["booking"] == first["booking"]
    records = bookings(run_dir)
    assert len(records) == 1
    assert records[0]["cost_kzt"] == venue["cost_kzt"]
    assert records[0]["idempotency_key"] == f"booking:{venue['id']}"


def test_approval_covers_only_the_venue_and_price_from_checkpoint(run_dir: Path):
    venue = eligible_venue(run_dir)

    approve_in_state(run_dir, {**venue, "id": "V-OTHER"})
    assert call("book_venue", venue_id=venue["id"])["status"] == "approval_required"

    approve_in_state(run_dir, {**venue, "cost_kzt": venue["cost_kzt"] - 1})
    assert call("book_venue", venue_id=venue["id"])["status"] == "approval_required"

    write_state(run_dir, status="running", decision="approve")
    assert call("book_venue", venue_id=venue["id"])["status"] == "approval_required"
    assert bookings(run_dir) == []


def test_rejected_venue_cannot_be_booked(run_dir: Path):
    venue = eligible_venue(run_dir)
    write_state(run_dir, status="running", decision="reject", rejected_venues=[venue["id"]])

    assert call("book_venue", venue_id=venue["id"])["status"] == "rejected_by_operator"
    assert bookings(run_dir) == []


def test_cheap_venue_needs_no_operator_and_blocks_second_booking(run_dir: Path):
    expensive = eligible_venue(run_dir)
    cheap = venue_with("capacity_too_small")
    path = run_dir / "data" / "venues.json"
    venues = read_json(path)
    for record in venues:
        if record["id"] == cheap["id"]:
            record["total_capacity"] = 100
    write_json(path, venues)

    assert call("book_venue", venue_id=cheap["id"])["status"] == "booked"
    approve_in_state(run_dir, expensive)
    second = call("book_venue", venue_id=expensive["id"])

    assert second["status"] == "another_venue_booked"
    assert [b["venue_id"] for b in bookings(run_dir)] == [cheap["id"]]


def test_venue_refusal_becomes_structured_reply_without_raw_text(run_dir: Path):
    reset.reset_world(scenario=4, workdir=run_dir)
    venue = eligible_venue(run_dir)
    approve_in_state(run_dir, venue)

    reply = call("book_venue", venue_id=venue["id"])

    assert reply["ok"] is False and reply["status"] == "venue_refused"
    assert bookings(run_dir) == []
    raw_fault = next(v["fault"] for v in read_json(run_dir / "data" / "venues.json") if v.get("fault"))
    journal = (run_dir / "journal.jsonl").read_text(encoding="utf-8")
    assert raw_fault not in str(reply) and raw_fault not in journal


def test_unknown_or_malformed_venue_id(run_dir: Path):
    assert call("book_venue", venue_id="V-NOPE")["status"] == "unknown_venue"
    with pytest.raises(Exception, match="venue_id"):
        call("book_venue", venue_id="V 1; drop")
    assert bookings(run_dir) == []
