"""Compare configured small Ollama candidates on the ADAS tool eval set.

Run after pulling all models listed in src/config.py:
    python tools/check_tool_models.py
Each candidate is stopped before the next is loaded to stay within 8 GB RAM.
"""
import json
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent import ask_agent
from config import CHAT_MODEL_CANDIDATES, MODEL_CHECK_KEEP_ALIVE, PROJECT_ROOT

QUESTIONS_FILE = PROJECT_ROOT / "questions_adas.json"
RESULTS_DIR = PROJECT_ROOT / "results"


def _check_row(question, result):
    trace = result.get("tool_trace", [])
    names = [item.get("tool") for item in trace]
    expected = question["category"] != "off_topic"
    tool_called = (("search_docs" in names) if expected else
                   result.get("answer_source") == "declined")
    if question["category"] == "adas_not_in_docs" and not question.get("expect_clarification"):
        doc_lookup = next((item for item in trace if item.get("tool") == "search_docs"), {})
        if doc_lookup.get("result") != "results":
            tool_called = tool_called and "web_search" in names
    if "stopping distance" in question["question"].lower() or question["id"] == "adas04":
        tool_called = tool_called and "stopping_distance" in names
    if question.get("expect_clarification"):
        tool_called = tool_called and "ask_user" in names

    valid_arguments = True
    for call in trace:
        args = call.get("arguments") or {}
        if call.get("tool") == "search_docs":
            valid_arguments &= bool(args.get("query"))
        elif call.get("tool") == "web_search":
            valid_arguments &= bool(args.get("query"))
        elif call.get("tool") == "ask_user":
            valid_arguments &= bool(args.get("question"))
        elif call.get("tool") == "stopping_distance":
            valid_arguments &= {"speed_kmh", "road_condition"}.issubset(args)
        elif call.get("tool") == "generate_quiz":
            valid_arguments &= bool(args.get("system_name"))

    answer = result.get("answer", "")
    low = answer.lower()
    label_ok = True
    if result.get("answer_source") == "web":
        label_ok = "from the web" in low and any(s.get("type") == "web" for s in result.get("sources", []))
    elif result.get("answer_source") == "general_knowledge":
        label_ok = "general knowledge" in low or bool(result.get("clarification_question"))
    elif result.get("answer_source") == "documents":
        label_ok = "p." in low and any(s.get("type") == "document" for s in result.get("sources", []))
    elif result.get("answer_source") == "declined":
        label_ok = question["category"] == "off_topic"
    return {"tool_called_when_expected": bool(tool_called),
            "valid_arguments": bool(valid_arguments),
            "correct_tier_label": bool(label_ok),
            "correct_off_topic_decline": (result.get("answer_source") == "declined"
                                          if question["category"] == "off_topic" else None),
            "final_answer_produced": bool(answer.strip()),
            "answer_source": result.get("answer_source"), "answer": answer,
            "trace": trace, "sources": result.get("sources", [])}


def _stop_model(model):
    try:
        subprocess.run(["ollama", "stop", model], check=False,
                       capture_output=True, text=True, timeout=30)
    except Exception:
        pass


def run():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(exist_ok=True)
    comparison = {}
    for model in CHAT_MODEL_CANDIDATES:
        print(f"Checking {model} (one candidate loaded at a time)")
        rows = []
        for question in questions:
            started = time.perf_counter()
            try:
                result = ask_agent(question["question"], model=model,
                                   keep_alive=MODEL_CHECK_KEEP_ALIVE)
                checked = _check_row(question, result)
            except Exception as exc:
                checked = {"error": str(exc), "final_answer_produced": False,
                           "tool_called_when_expected": False,
                           "valid_arguments": False, "correct_tier_label": False,
                           "answer": "", "trace": [], "sources": []}
            checked["id"] = question["id"]
            checked["latency_seconds"] = round(time.perf_counter() - started, 3)
            rows.append(checked)
            print(f"  {question['id']}: tools={checked['tool_called_when_expected']} "
                  f"args={checked['valid_arguments']} label={checked['correct_tier_label']} "
                  f"final={checked['final_answer_produced']} "
                  f"{checked['latency_seconds']:.1f}s")
        _stop_model(model)
        denom = max(len(rows), 1)
        scores = {key: sum(bool(r.get(key)) for r in rows) / denom for key in (
            "tool_called_when_expected", "valid_arguments", "correct_tier_label",
            "final_answer_produced")}
        declines = [r["correct_off_topic_decline"] for r in rows
                    if r.get("correct_off_topic_decline") is not None]
        scores["correct_off_topic_decline"] = sum(declines) / len(declines) if declines else None
        scores["mean_latency_seconds"] = sum(r["latency_seconds"] for r in rows) / denom
        comparison[model] = {"scores": scores, "rows": rows}

    # Selection score emphasizes safe tier handling and valid tool use equally.
    def score(item):
        metrics = item["scores"]
        vals = [metrics[k] for k in ("tool_called_when_expected", "valid_arguments",
                                     "correct_tier_label", "correct_off_topic_decline",
                                     "final_answer_produced") if metrics.get(k) is not None]
        return sum(vals) / len(vals) if vals else 0
    recommended = max(comparison, key=lambda m: (score(comparison[m]),
                                                   -comparison[m]["scores"]["mean_latency_seconds"])) if comparison else None
    report = {"recommended_model": recommended,
              "note": "Recommendation is based on this placeholder question set; inspect raw answers and verify against real ADAS PDFs.",
              "models": comparison}
    out = RESULTS_DIR / f"tool_model_check_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recommended: {recommended}; report: {out}")
    return report


if __name__ == "__main__":
    run()
