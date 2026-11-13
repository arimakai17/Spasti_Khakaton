"""Уровень 1: инструменты чтения отдают только поля из белых списков."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import mcp_server
from mcp_server import Config, Hall, Team, Venue, venue_problems

from .conftest import call, read_json, write_json
from .modified_world import SECRET, SERVICE_KEY, plant_service_fields

ROOT = Path(__file__).resolve().parent.parent
READ_TOOLS = ("list_venues", "read_inbox", "get_roster", "get_status")


def test_read_tools_drop_service_fields_at_every_level(run_dir: Path):
    planted = plant_service_fields(run_dir / "data")
    # Проверка самого теста: секреты лежат в сырых данных на всех уровнях, включая залы.
    assert planted >= 20
    assert SERVICE_KEY in json.dumps(read_json(run_dir / "data" / "venues.json")[0]["halls"])

    output = json.dumps([call(tool) for tool in READ_TOOLS], ensure_ascii=False)

    for text in (output, (run_dir / "journal.jsonl").read_text(encoding="utf-8")):
        assert SECRET not in text
        assert SERVICE_KEY not in text
    assert not (run_dir / "state.json").exists()


def test_read_tools_return_business_data(run_dir: Path):
    data = run_dir / "data"
    venues = call("list_venues")
    inbox = call("read_inbox")
    roster = call("get_roster")

    assert all(reply["ok"] and reply["status"] == "ok" for reply in (venues, inbox, roster))
    assert len(venues["venues"]) == len(read_json(data / "venues.json"))
    assert len(inbox["messages"]) == len(read_json(data / "inbox.json"))
    assert len(roster["teams"]) == len(read_json(data / "teams.json"))
    assert len(roster["mentors"]) == len(read_json(data / "mentors.json"))
    assert roster["participants_total"] == sum(t["size"] for t in read_json(data / "teams.json"))
    assert all(hall["available_slots"] is not None for v in venues["venues"] for hall in v["halls"])

    # В раздатке регламенту соответствует одна площадка, и она требует подтверждения оператора.
    eligible = [v for v in venues["venues"] if v["eligibility"]["eligible"]]
    assert len(eligible) == 1
    assert eligible[0]["eligibility"]["requires_approval"] is True

    senders = {m["sender"]: m["sender_in_allowed_domains"] for m in inbox["messages"]}
    assert True in senders.values() and False in senders.values()


def _config(**overrides) -> Config:
    base = {
        "max_budget_kzt": 1000,
        "hitl_threshold_kzt": 400,
        "allowed_email_domains": ["inside.test"],
        "max_turns": 20,
        "max_attempts": 3,
        "max_cost_usd": 0.25,
        "required_tracks": ["red", "blue"],
    }
    return Config.model_validate({**base, **overrides})


def _venue(**overrides) -> Venue:
    base = {
        "id": "V-X",
        "available": True,
        "total_capacity": 10,
        "cost_kzt": 500,
        "halls": [Hall(hall_id="H-R", tracks=["red"]), Hall(hall_id="H-B", tracks=["blue"])],
    }
    return Venue.model_validate({**base, **overrides})


TEAMS = [
    Team(id="T-R", size=4, track="red", captain_email="r@inside.test"),
    Team(id="T-B", size=5, track="blue", captain_email="b@inside.test"),
]


def test_venue_problems_follow_the_regulation():
    assert venue_problems(_venue(), _config(), TEAMS) == []
    assert venue_problems(_venue(available=False), _config(), TEAMS) == ["unavailable"]
    assert venue_problems(_venue(total_capacity=8), _config(), TEAMS) == ["capacity_too_small"]
    assert venue_problems(_venue(cost_kzt=1001), _config(), TEAMS) == ["over_budget"]
    one_track = _venue(halls=[Hall(hall_id="H-R", tracks=["red"])])
    assert venue_problems(one_track, _config(), TEAMS) == ["missing_tracks"]
    assert venue_problems(_venue(cost_kzt=1000), _config(), TEAMS) == []


def test_address_check_takes_one_address_in_allowed_domain():
    config = _config()
    assert mcp_server.address_allowed("kim@inside.test", config)
    assert mcp_server.address_allowed("Kim@INSIDE.test", config)
    assert not mcp_server.address_allowed("kim@outside.test", config)
    assert not mcp_server.address_allowed("kim@inside.test.evil.org", config)
    assert not mcp_server.address_allowed("kim@inside.test, spy@outside.test", config)
    assert not mcp_server.address_allowed("kim@outside.test@inside.test", config)
    assert not mcp_server.address_allowed("inside.test", config)


def test_venue_without_price_is_not_listed(run_dir: Path):
    path = run_dir / "data" / "venues.json"
    venues = read_json(path)
    for venue in venues:
        del venue["cost_kzt"]
    write_json(path, venues)

    assert call("list_venues")["venues"] == []
    journal = (run_dir / "journal.jsonl").read_text(encoding="utf-8")
    assert '"records_skipped"' in journal


def test_status_of_untouched_world(run_dir: Path):
    status = call("get_status")
    data = run_dir / "data"
    teams = read_json(data / "teams.json")
    mentors = read_json(data / "mentors.json")

    assert status["task"] == {"status": "new", "attempts": 0, "decision": None, "selected_venue": None}
    assert status["booking"] is None
    assert status["schedule"] == [] and status["notifications"] == []
    assert status["pending"]["teams_without_slots"] == [t["id"] for t in teams]
    assert len(status["pending"]["recipients_without_notification"]) == len(teams) + len(mentors)


def test_journal_has_one_json_line_per_tool_call(run_dir: Path):
    for tool in READ_TOOLS:
        call(tool)
    lines = (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]

    assert [e["tool"] for e in entries] == list(READ_TOOLS)
    assert all(e["source"] == "mcp" and e["event"] == "tool_call" and e["ts"] for e in entries)


def test_server_answers_over_stdio(run_dir: Path):
    """Сервер запускается отдельным процессом из папки прогона, как его запускает агент."""

    async def session_roundtrip():
        params = StdioServerParameters(
            command=sys.executable, args=[str(ROOT / "mcp_server.py")], cwd=run_dir
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                result = await session.call_tool("list_venues", {})
                return {tool.name for tool in tools.tools}, result

    names, result = asyncio.run(session_roundtrip())
    assert set(READ_TOOLS) <= names
    assert result.isError is False
    assert result.structuredContent["ok"] is True
    assert SECRET not in json.dumps(result.structuredContent, ensure_ascii=False)
