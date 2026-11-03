import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import charts
import agent
import server


def test_stopping_distance_chart_creates_png_for_wet_road(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, "CHART_DIR", tmp_path)
    result = Path(charts.stopping_distance_chart(100, "wet"))
    assert result == tmp_path / "stopping_100_wet.png"
    assert result.is_file()
    assert result.stat().st_size > 0


def test_adas_timeline_chart_creates_png(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, "CHART_DIR", tmp_path)
    result = Path(charts.adas_timeline_chart(guide_loaded=False))
    assert result == tmp_path / "adas_timeline.png"
    assert result.is_file()
    assert result.stat().st_size > 0


@pytest.mark.parametrize("speed,condition", [(-1, "wet"), (100, "mud")])
def test_invalid_chart_inputs_return_error_without_file(tmp_path, monkeypatch,
                                                       speed, condition):
    monkeypatch.setattr(charts, "CHART_DIR", tmp_path)
    with pytest.raises(ValueError):
        charts.stopping_distance_chart(speed, condition)
    assert list(tmp_path.iterdir()) == []


def test_chart_route_rejects_traversal_and_non_png():
    client = server.app.test_client()
    assert client.get("/charts/../secret.png").status_code == 404
    assert client.get("/charts/%2E%2E%5Csecret.png").status_code == 404
    assert client.get("/charts/C%3A%5Csecret.png").status_code == 404
    assert client.get("/charts/secret.jpg").status_code == 404


def test_agent_includes_chart_only_after_successful_stopping_tool(monkeypatch):
    import charts as chart_module

    monkeypatch.setattr(agent, "retrieve", lambda *args, **kwargs: [])
    monkeypatch.setattr(agent, "_passages_contain_answer", lambda *args, **kwargs: False)
    monkeypatch.setattr(agent.websearch, "search_web", lambda *args, **kwargs: [])
    monkeypatch.setattr(chart_module, "stopping_distance_chart",
                        lambda speed, condition: f"stopping_{int(speed)}_{condition}.png")
    calls = iter([
        {"tool_calls": [{"function": {"name": "stopping_distance",
          "arguments": {"speed_kmh": 100, "road_condition": "wet"}}}]},
        {"content": "Reaction and braking distance total about 216 m."},
    ])
    monkeypatch.setattr(agent, "_ollama_chat", lambda *args, **kwargs: next(calls))
    result = agent.ask_agent("ADAS stopping distance at 100 km/h on wet roads")
    assert result["answer_source"] == "calculation"
    assert result["answer"].startswith(
        "Calculated with Sentinel's simplified formula (not from the documents):")
    assert "reaction time 1.5 s" in result["answer"]
    assert "friction coefficient 0.5" in result["answer"]
    stop_trace = next(item for item in result["tool_trace"]
                      if item["tool"] == "stopping_distance")
    assert stop_trace["result"]["total_m"] == 120.3
    assert result["images"] == [{
        "url": "/charts/stopping_100_wet.png",
        "caption": "Stopping distance at 100 km/h on wet road (simplified estimate)",
        "kind": "chart",
    }]

    monkeypatch.setattr(agent, "_ollama_chat", lambda *args, **kwargs:
                        {"content": "AEB can help reduce collision severity."})
    no_chart = agent.ask_agent("Explain AEB")
    assert no_chart["images"] == []


def _mock_timeline_answer(monkeypatch, chart_function=None):
    import charts as chart_module

    class EmptyIndex:
        def get(self, **kwargs):
            return {"ids": [], "metadatas": []}

    monkeypatch.setattr(agent, "_scope_check", lambda *args, **kwargs: True)
    monkeypatch.setattr(agent, "get_collection", lambda *args, **kwargs: EmptyIndex())
    monkeypatch.setattr(agent, "run_tool", lambda name, arguments, *args, **kwargs:
                        {"hits": [], "found": False} if name == "search_docs" else
                        {"results": [], "found": False})
    monkeypatch.setattr(agent, "_ollama_chat", lambda *args, **kwargs:
                        {"content": "ADAS systems developed across several eras."})
    if chart_function is None:
        chart_function = lambda **kwargs: "adas_timeline.png"
    monkeypatch.setattr(chart_module, "adas_timeline_chart", chart_function)


def test_timeline_request_returns_chart_image(monkeypatch):
    _mock_timeline_answer(monkeypatch)
    result = agent.ask_agent("make me a graph of the evolution of ADAS")
    assert result["images"] == [{
        "url": "/charts/adas_timeline.png",
        "caption": "ADAS evolution timeline",
        "source": "2021 ADAS Guide",
        "printed_page": "4",
        "source_status": "not currently loaded",
        "kind": "chart",
    }]


def test_aeb_how_it_works_returns_no_chart(monkeypatch):
    _mock_timeline_answer(monkeypatch)
    result = agent.ask_agent("How does AEB work?")
    assert result["images"] == []


def test_ascii_box_code_block_is_stripped():
    answer = "Before.\n```text\n+---+\n| ADAS |\n+---+\n  v\n```\nAfter."
    cleaned = agent._clean_model_answer(answer)
    assert cleaned == "Before.\nAfter."


def test_timeline_chart_failure_keeps_text_answer(monkeypatch):
    def fail(**kwargs):
        raise OSError("disk unavailable")
    _mock_timeline_answer(monkeypatch, chart_function=fail)
    result = agent.ask_agent("make me a graph of the evolution of ADAS")
    assert result["answer"].endswith("ADAS systems developed across several eras.")
    assert result["images"] == []
