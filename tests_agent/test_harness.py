"""Уровень 2: обвязка agent.py на скриптовой модели и настоящем MCP-сервере (stdio)."""

import sys
from pathlib import Path

import pytest
from agents import Usage
from agents.testing import ScriptedModel, assistant_message, function_call

import agent
import reset

from .conftest import approve_in_state, eligible_venue, read_json, write_state


def launch(monkeypatch: pytest.MonkeyPatch, steps: list, *argv: str, prices=(0.0, 0.0), usage=None) -> ScriptedModel:
    """Запускает agent.main() с подменённой моделью. Лишний вызов модели роняет тест."""
    model = ScriptedModel(steps, default_usage=usage)
    monkeypatch.setattr(agent, "build_model", lambda: agent.ModelSetup(model, "scripted", prices))
    monkeypatch.setattr(sys, "argv", ["agent.py", *argv])
    agent.main()
    return model


def book(venue: dict) -> list:
    return [function_call("book_venue", {"venue_id": venue["id"]}, call_id="call-book")]


def state(run_dir: Path) -> dict:
    return read_json(run_dir / "state.json")


def bookings(run_dir: Path) -> list[dict]:
    path = run_dir / "bookings.json"
    return read_json(path) if path.exists() else []


def test_first_launch_stops_before_expensive_booking(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)

    model = launch(monkeypatch, [book(venue)])

    saved = state(run_dir)
    assert saved["status"] == "waiting_approval" and saved["attempts"] == 1
    assert saved["decision"] is None and saved["selected_venue"] == venue["id"]
    assert saved["checkpoint"]["venue_id"] == venue["id"]
    assert saved["checkpoint"]["cost_kzt"] == venue["cost_kzt"]
    assert saved["bulletin"] is None
    assert bookings(run_dir) == []
    assert len(model.calls) == 1
    assert (run_dir / "sessions.db").exists()


def test_launch_without_decision_changes_nothing(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    launch(monkeypatch, [book(venue)])
    before = state(run_dir)

    launch(monkeypatch, [])

    assert state(run_dir) == before


def test_approve_books_and_done_needs_complete_world(run_dir: Path, monkeypatch):
    """Модель трижды заявляет, что всё перенесено. Без слотов и писем статус done не ставится."""
    venue = eligible_venue(run_dir)
    launch(monkeypatch, [book(venue)])
    claim = [assistant_message("Всё перенесено, расписание обновлено.")]

    model = launch(monkeypatch, [book(venue), claim, claim, claim], "--approve")

    saved = state(run_dir)
    assert [b["venue_id"] for b in bookings(run_dir)] == [venue["id"]]
    # Запуск после решения оператора продолжает ту же попытку.
    assert saved["decision"] == "approve" and saved["attempts"] == 1
    assert saved["status"] == "failed" and "команды без слотов" in saved["reason"]
    assert saved["bulletin"] is None
    assert len(model.calls) == 2 + agent.MAX_NUDGES


def test_each_launch_starts_a_new_model_session_from_the_brief(run_dir: Path, monkeypatch):
    """После паузы и --approve модель получает только сводку, без истории прошлого запуска."""
    import sqlite3

    venue = eligible_venue(run_dir)
    reads = [function_call("list_venues", {}, call_id="c-venues")]
    first = launch(monkeypatch, [reads, book(venue)])
    claim = [assistant_message("Готово.")]

    second = launch(monkeypatch, [book(venue), claim, claim, claim], "--approve")

    assert len(first.calls[0].input) == 1 and len(second.calls[0].input) == 1
    assert "Оператор одобрил бронь" in str(second.calls[0].input[0])
    sessions = sqlite3.connect(run_dir / "sessions.db").execute(
        "select distinct session_id from agent_messages order by session_id"
    )
    assert [row[0] for row in sessions] == ["launch-1", "launch-2"]
    assert state(run_dir)["attempts"] == 1


def test_reject_without_alternative_needs_attention_without_model(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    launch(monkeypatch, [book(venue)])

    launch(monkeypatch, [], "--reject")

    saved = state(run_dir)
    assert saved["status"] == "needs_attention" and saved["decision"] == "reject"
    assert saved["rejected_venues"] == [venue["id"]] and venue["id"] in saved["reason"]
    assert bookings(run_dir) == []


def test_venue_refusal_ends_in_needs_attention(run_dir: Path, monkeypatch):
    reset.reset_world(scenario=4, workdir=run_dir)
    venue = eligible_venue(run_dir)
    launch(monkeypatch, [book(venue)])

    model = launch(monkeypatch, [book(venue)], "--approve")

    saved = state(run_dir)
    assert saved["status"] == "needs_attention" and saved["refused_venues"] == [venue["id"]]
    assert saved["selected_venue"] is None
    assert bookings(run_dir) == [] and len(model.calls) == 1
    raw_fault = next(v["fault"] for v in read_json(run_dir / "data" / "venues.json") if v.get("fault"))
    for name in ("state.json", "journal.jsonl"):
        assert raw_fault not in (run_dir / name).read_text(encoding="utf-8")


def test_terminal_status_is_final(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    launch(monkeypatch, [book(venue)])
    launch(monkeypatch, [], "--reject")
    before = state(run_dir)

    launch(monkeypatch, [], "--approve")

    assert state(run_dir) == before


def test_attempt_limit_allows_exactly_max_attempts(run_dir: Path, monkeypatch):
    limit = read_json(run_dir / "data" / "config.json")["max_attempts"]
    done = [assistant_message("отчёт")]

    write_state(run_dir, status="running", attempts=limit - 1)
    model = launch(monkeypatch, [done, done, done])
    assert state(run_dir)["attempts"] == limit and len(model.calls) == 3

    write_state(run_dir, status="running", attempts=limit)
    launch(monkeypatch, [])
    saved = state(run_dir)
    assert saved["status"] == "failed" and "Лимит попыток" in saved["reason"]


def test_turn_limit_fails_the_task(run_dir: Path, monkeypatch):
    step = [function_call("list_venues", {}, call_id="call-1")]

    model = launch(monkeypatch, [step, step], "--max-turns", "2")

    saved = state(run_dir)
    assert saved["status"] == "failed" and "Лимит шагов (2)" in saved["reason"]
    assert len(model.calls) == 2


def test_cost_limit_fails_the_task(run_dir: Path, monkeypatch):
    step = [function_call("list_venues", {}, call_id="call-1")]
    heavy = Usage(requests=1, input_tokens=200_000, output_tokens=0, total_tokens=200_000)

    model = launch(monkeypatch, [step, step, step], prices=(1.0, 1.0), usage=heavy)

    saved = state(run_dir)
    assert saved["status"] == "failed" and "Стоимость" in saved["reason"]
    assert saved["token_usage"]["cost_usd"] == pytest.approx(0.4)
    assert saved["token_usage"]["requests"] == 2 == len(model.calls)


def test_unknown_price_keeps_counting_tokens(run_dir: Path, monkeypatch):
    venue = eligible_venue(run_dir)
    usage = Usage(requests=1, input_tokens=10**9, output_tokens=5, total_tokens=10**9 + 5)

    launch(monkeypatch, [book(venue)], prices=None, usage=usage)

    saved = state(run_dir)
    assert saved["status"] == "waiting_approval"
    assert saved["token_usage"]["cost_usd"] is None and saved["token_usage"]["output_tokens"] == 5
    assert '"cost_unknown"' in (run_dir / "journal.jsonl").read_text(encoding="utf-8")


def test_missing_key_does_not_spend_an_attempt(run_dir: Path, monkeypatch):
    monkeypatch.setattr(agent, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/model")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["agent.py"])

    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        agent.main()

    assert not (run_dir / "state.json").exists()


@pytest.mark.parametrize(
    ("provider", "model_var", "key_var", "model_class"),
    [
        ("openai", "OPENAI_MODEL", "OPENAI_API_KEY", agent.OpenAIResponsesModel),
        ("openrouter", "OPENROUTER_MODEL", "OPENROUTER_API_KEY", agent.OpenAIChatCompletionsModel),
        ("hf", "HF_MODEL", "HF_TOKEN", agent.OpenAIChatCompletionsModel),
        ("ollama", "OLLAMA_MODEL", None, agent.OpenAIChatCompletionsModel),
    ],
)
def test_provider_is_chosen_by_environment(monkeypatch, provider, model_var, key_var, model_class):
    monkeypatch.setenv("MODEL_PROVIDER", provider)
    monkeypatch.setenv(model_var, "vendor/gpt-5.6-luna")
    if key_var:
        monkeypatch.setenv(key_var, "test-key")
    for name in ("MODEL_PRICE_INPUT_PER_1M", "MODEL_PRICE_OUTPUT_PER_1M"):
        monkeypatch.delenv(name, raising=False)

    setup = agent.build_model()

    assert isinstance(setup.model, model_class)
    assert setup.name == f"{provider}:vendor/gpt-5.6-luna"
    # Облачный запрос ограничен по времени, чтобы повторы уложились в лимит грейдера на запуск.
    client = setup.model._client
    if provider == "ollama":
        assert client.max_retries == 2
    else:
        assert client.timeout == 120.0 and client.max_retries == 3
        assert client.timeout * (client.max_retries + 1) < 600
    assert setup.prices == ((0.0, 0.0) if provider == "ollama" else (0.20, 1.20))


def test_status_cannot_leave_the_graph(run_dir: Path):
    task = agent.TaskState(status="done")
    with pytest.raises(RuntimeError, match="done -> running"):
        agent.move(task, "running", "повтор")
    assert not (run_dir / "state.json").exists()


def test_server_crash_leaves_task_running_and_relaunch_does_not_rebook(run_dir: Path, monkeypatch):
    """Строку [сбой] в выводе агента проверяет грейдер в сценарии 2: клиент MCP запоминает
    поток stderr при импорте, и перехват вывода в pytest её не видит."""
    venue = eligible_venue(run_dir)
    approve_in_state(run_dir, venue, attempts=1)

    monkeypatch.setenv("CRASH_AT", f"after_booking:{venue['id']}")
    with pytest.raises(SystemExit) as crash:
        launch(monkeypatch, [book(venue)])
    assert crash.value.code == 1
    assert state(run_dir)["status"] == "running" and len(bookings(run_dir)) == 1

    monkeypatch.delenv("CRASH_AT")
    done = [assistant_message("отчёт")]
    launch(monkeypatch, [book(venue), done, done, done])
    assert len(bookings(run_dir)) == 1 and state(run_dir)["attempts"] == 3
