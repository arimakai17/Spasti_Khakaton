"""Общие заготовки своих тестов. Папка tests/ принадлежит раздатке и не меняется."""

import asyncio
import json
from pathlib import Path

import pytest

import mcp_server
import reset


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Чистая папка прогона с полным набором входящих писем (как в сценарии 5)."""
    reset.reset_world(scenario=5, workdir=tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def call(tool: str, **arguments) -> dict:
    """Вызывает инструмент через FastMCP: со схемой аргументов и сериализацией ответа."""
    _content, structured = asyncio.run(mcp_server.mcp.call_tool(tool, arguments))
    return structured


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def eligible_venue(run_dir: Path) -> dict:
    """Площадка раздатки, которая проходит регламент. Идентификатор тесты в себя не вписывают."""
    return next(v for v in call("list_venues")["venues"] if v["eligibility"]["eligible"])


def venue_with(problem: str) -> dict:
    return next(v for v in call("list_venues")["venues"] if v["eligibility"]["problems"] == [problem])


def write_state(run_dir: Path, **fields) -> None:
    write_json(run_dir / "state.json", fields)


def approve_in_state(run_dir: Path, venue: dict, **fields) -> None:
    checkpoint = {
        "venue_id": venue["id"],
        "venue_name": venue["name"],
        "cost_kzt": venue["cost_kzt"],
        "requested_at": "2026-01-01T00:00:00",
    }
    write_state(run_dir, status="running", decision="approve", checkpoint=checkpoint, **fields)


def book_approved(run_dir: Path) -> dict:
    """Бронирует подходящую площадку с решением оператора в state.json."""
    venue = eligible_venue(run_dir)
    approve_in_state(run_dir, venue)
    assert call("book_venue", venue_id=venue["id"])["status"] == "booked"
    return venue


def slot_plan(venue: dict) -> list[dict]:
    """Корректная сетка, собранная из данных мира: зал и ментор трека, слоты по порядку."""
    roster = call("get_roster")
    plan = []
    for mentor in roster["mentors"]:
        hall = next(h for h in venue["halls"] if mentor["track"] in h["tracks"])
        mentor_slots = [s for s in mentor["available_slots"] if s in hall["available_slots"]]
        pitch_slots = [s for s in hall["available_slots"] if s not in mentor["available_slots"]]
        teams = [t for t in roster["teams"] if t["track"] == mentor["track"]]
        for team, mentor_slot, pitch_slot in zip(teams, mentor_slots, pitch_slots, strict=False):
            plan.append(
                {
                    "team_id": team["id"],
                    "hall_id": hall["hall_id"],
                    "mentor_id": mentor["id"],
                    "mentor_slot": mentor_slot,
                    "pitch_slot": pitch_slot,
                }
            )
    return plan


def letters() -> list[dict]:
    """По одному письму каждому капитану и ментору, без времени в тексте."""
    roster = call("get_roster")
    addresses = [t["captain_email"] for t in roster["teams"]] + [m["email"] for m in roster["mentors"]]
    return [
        {"recipient": address, "subject": "Перенос хакатона", "text": "Здравствуйте! Хакатон переезжает."}
        for address in addresses
    ]


def statuses(reply: dict) -> list[str]:
    return [result["status"] for result in reply["results"]]
