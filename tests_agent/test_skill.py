"""Уровень 5: пакет навыка, бюллетень и справочный режим."""

import asyncio
import sys
from pathlib import Path

import pytest
from agents.testing import assistant_message, function_call
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import agent
import mcp_server
from evals import graders

from .conftest import eligible_venue, letters, read_json, slot_plan
from .test_harness import launch, state

ROOT = Path(__file__).resolve().parent.parent
READ_TOOLS = {"list_venues", "read_inbox", "get_roster", "get_status"}


def bulletin(run_dir: Path) -> str:
    teams = read_json(run_dir / "data" / "teams.json")
    lines = ["[ПЛОЩАДКА]", "Площадка забронирована", "", "[СЕТКА СЛОТОВ]"]
    lines += [f"{team['id']} {team['name']}: слоты назначены" for team in teams]
    lines += ["Конфликтов расписания: 0", "", "[СТАТУС РАССЫЛКИ]", "Команды и менторы уведомлены"]
    return "\n".join(lines)


def transfer_steps(venue: dict, report: str) -> list:
    return [
        [function_call("book_venue", {"venue_id": venue["id"]}, call_id="c-book")],
        [function_call("assign_slots", {"assignments": slot_plan(venue)}, call_id="c-slots")],
        [function_call("send_notifications", {"messages": letters()}, call_id="c-mail")],
        [assistant_message(report)],
    ]


def tool_names(model) -> set[str]:
    return {tool.name for tool in model.calls[0].tools}


def test_skill_package_has_activation_header_procedure_and_template():
    skills = agent.find_skills()

    assert len(skills) == 1 and graders.has_valid_skill_package()
    skill = skills[0]
    assert (ROOT / "skills" / skill.name / "SKILL.md").exists()
    assert "Не применяй для справочных вопросов" in skill.description
    assert all(part in skill.body for part in agent.BULLETIN_PARTS)
    assert "assign_slots" in skill.body and "name:" not in skill.body


def test_skill_reaches_the_prompt_only_as_the_mode_allows():
    skills = agent.find_skills()
    body_marker, description = "[СЕТКА СЛОТОВ]", skills[0].description

    forced = agent.instructions_for("on", skills)
    offered = agent.instructions_for("auto", skills)
    hidden = agent.instructions_for("off", [])

    assert body_marker in forced
    assert description in offered and body_marker not in offered and "load_skill" in offered
    assert description not in hidden and body_marker not in hidden and "load_skill" not in hidden


def test_use_skill_flag_activates_skill_and_bulletin_passes_the_grader(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    steps = transfer_steps(venue, bulletin(run_dir))

    first = launch(monkeypatch, steps[:1], "--use-skill")
    assert state(run_dir)["skill_activated"] is True and "load_skill" not in tool_names(first)
    launch(monkeypatch, steps, "--approve", "--use-skill")

    saved = state(run_dir)
    assert saved["status"] == "done" and saved["skill_activated"] is True
    verdict = graders.grade_folder(run_dir, scenario=3, mode="with-skill")
    assert verdict["passed"], verdict


def test_no_skill_flag_hides_the_skill(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    steps = transfer_steps(venue, "Перенос завершён.")

    first = launch(monkeypatch, steps[:1], "--no-skill")
    launch(monkeypatch, steps, "--approve", "--no-skill")

    assert "load_skill" not in tool_names(first)
    assert "[СЕТКА СЛОТОВ]" not in first.calls[0].system_instructions
    saved = state(run_dir)
    assert saved["status"] == "done" and saved["skill_activated"] is False
    assert graders.grade_folder(run_dir, scenario=3, mode="without-skill")["passed"]


def test_model_can_load_the_skill_itself(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    name = agent.find_skills()[0].name
    load = [function_call("load_skill", {"name": name}, call_id="c-skill")]
    book = [function_call("book_venue", {"venue_id": venue["id"]}, call_id="c-book")]

    model = launch(monkeypatch, [load, book])

    assert "load_skill" in tool_names(model)
    saved = state(run_dir)
    assert saved["status"] == "waiting_approval" and saved["skill_activated"] is True
    assert '"skill_loaded"' in (run_dir / "journal.jsonl").read_text(encoding="utf-8")


def test_skill_flag_reflects_the_current_launch(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    steps = transfer_steps(venue, "Перенос завершён.")
    launch(monkeypatch, steps[:1], "--use-skill")

    launch(monkeypatch, steps, "--approve", "--no-skill")

    assert state(run_dir)["skill_activated"] is False


def test_report_outside_template_is_sent_back_to_the_model(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    steps = transfer_steps(venue, "Всё перенесено.")
    launch(monkeypatch, steps[:1], "--use-skill")

    model = launch(monkeypatch, [*steps, [assistant_message(bulletin(run_dir))]], "--approve", "--use-skill")

    saved = state(run_dir)
    assert saved["status"] == "done" and saved["bulletin"] == bulletin(run_dir)
    nudge = str(model.calls[-1].input[-1])
    assert "[ПЛОЩАДКА]" in nudge and "Конфликтов расписания: 0" in nudge
    assert graders.collect_facts(run_dir)["skill_format_valid"] == 1


def test_missing_bulletin_does_not_cancel_a_finished_transfer(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    steps = transfer_steps(venue, "Всё перенесено.")
    stubborn = [assistant_message("Всё перенесено.")]
    launch(monkeypatch, steps[:1], "--use-skill")

    launch(monkeypatch, [*steps, stubborn, stubborn], "--approve", "--use-skill")

    saved = state(run_dir)
    assert saved["status"] == "done" and saved["bulletin"] == "Всё перенесено."
    assert '"bulletin_incomplete"' in (run_dir / "journal.jsonl").read_text(encoding="utf-8")
    assert graders.collect_facts(run_dir)["false_done"] == 0


def test_query_on_fresh_world_reads_only_and_keeps_skill_off(run_dir: Path, monkeypatch, capsys):
    question = "Какой текущий статус и слот у одной команды?"
    steps = [[function_call("get_status", {}, call_id="c-status")], [assistant_message("Статус new, слотов нет.")]]

    model = launch(monkeypatch, steps, "--query", question)

    assert tool_names(model) == READ_TOOLS
    assert "Статус new, слотов нет." in capsys.readouterr().out
    saved = state(run_dir)
    assert saved["status"] == "new" and saved["attempts"] == 0 and saved["skill_activated"] is False
    assert saved["token_usage"]["requests"] == 2
    for name in ("bookings.json", "schedule.json", "outbox.json"):
        assert not (run_dir / name).exists()
    assert graders.grade_folder(run_dir, scenario=3, mode="single-query")["passed"]


def test_query_does_not_touch_a_waiting_task(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    launch(monkeypatch, [[function_call("book_venue", {"venue_id": venue["id"]}, call_id="c-book")]])
    before = state(run_dir)

    launch(monkeypatch, [[assistant_message("Ждём решения оператора.")]], "--query", "Что с переносом?")

    after = state(run_dir)
    assert {k: v for k, v in after.items() if k != "token_usage"} == {
        k: v for k, v in before.items() if k != "token_usage"
    }


def test_read_only_server_has_no_write_tools(run_dir: Path):
    venue_id = eligible_venue(run_dir)["id"]

    async def roundtrip():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "mcp_server.py")],
            cwd=run_dir,
            env={"MCP_READ_ONLY": "1"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                result = await session.call_tool("book_venue", {"venue_id": venue_id})
                return names, result

    names, result = asyncio.run(roundtrip())

    assert names == READ_TOOLS
    assert set(mcp_server.WRITE_TOOLS).isdisjoint(names)
    assert result.isError is True
    assert not (run_dir / "bookings.json").exists()


@pytest.mark.parametrize("flag", ["--use-skill", "--no-skill"])
def test_query_ignores_skill_flags(run_dir: Path, monkeypatch, flag: str):
    model = launch(monkeypatch, [[assistant_message("Ответ.")]], "--query", "Что с переносом?", flag)

    assert tool_names(model) == READ_TOOLS
    assert "[СЕТКА СЛОТОВ]" not in model.calls[0].system_instructions
    assert state(run_dir)["skill_activated"] is False
