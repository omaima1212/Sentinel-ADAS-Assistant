"""Compare the ADAS agent with a no-RAG/no-tools Ollama baseline.

Run: python src/eval_adas.py
The question set is explicitly a placeholder and must be checked against the
real PDFs before treating scores as project results.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent import ask_agent
from config import ADAS_COLLECTION_NAME, CHAT_MODEL, PROJECT_ROOT
from embed import get_collection
from evaluate import HEDGE_MARKERS
from generate import ask_model
from retrieve import retrieve

QUESTION_FILE = PROJECT_ROOT / "questions_adas.json"
RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_PROMPT = """You are a concise ADAS and vehicle-safety assistant. Answer ADAS,
driver assistance, and closely related automotive safety questions. Decline
unrelated questions in one sentence. You have no documents, tools, or web
search: never claim otherwise, invent a citation, or state exact car-specific
features from memory. Label ordinary answers 'From general knowledge
(unverified):'. Remind the user that ADAS assists the driver and does not
replace them. If vehicle make/model/year is missing for a car-specific answer,
ask for it."""


def _source_label(answer):
    lower = answer.lower()
    if "not found in your documents. from the web" in lower or "from the web:" in lower:
        return "web"
    if "not found in your documents. from general knowledge" in lower or "from general knowledge (unverified)" in lower:
        return "general_knowledge"
    if any(word in lower for word in ("outside my scope", "i can help with adas", "unrelated to adas")):
        return "declined"
    return "unknown"


def _contains_expected(answer, terms):
    low = answer.lower()
    if any(marker in low for marker in HEDGE_MARKERS):
        return False
    return any(term.lower() in low for term in terms) if terms else None


def _score(question, result, baseline=False):
    answer = result["answer"]
    category = question["category"]
    assigned = _source_label(answer) if baseline else result.get("answer_source", "unknown")
    if question.get("expect_clarification"):
        correct_answer = bool(result.get("clarification_question")) if not baseline else "?" in answer
    elif category == "off_topic":
        correct_answer = assigned == "declined"
    else:
        expected = question.get("expected_terms", [])
        correct_answer = _contains_expected(answer, expected) if expected else None
    expected_source = question["expected_answer_source"]
    if expected_source == "web_or_general":
        correct_tier = assigned in ("web", "general_knowledge")
    else:
        correct_tier = assigned == expected_source
    sources = result.get("sources", [])
    expected_doc = question.get("expected_source")
    citation = None
    if category == "in_docs":
        citation = bool(expected_doc and any(
            s.get("type") == "document" and expected_doc.lower() in s.get("source", "").lower()
            for s in sources
        ) and "p." in answer.lower())
    lower = answer.lower()
    if question.get("expect_clarification"):
        correct_label = bool(result.get("clarification_question")) if not baseline else "?" in answer
    elif category == "off_topic":
        correct_label = assigned == "declined"
    elif expected_source == "documents":
        correct_label = assigned == "documents"
    elif assigned == "web":
        correct_label = lower.startswith("not found in your documents. from the web:")
    else:
        correct_label = lower.startswith("not found in your documents. from general knowledge (unverified):")
    return {
        "answer_source": assigned,
        "correct_answer": correct_answer,
        "correct_tier": correct_tier,
        "correct_tier_label": correct_label,
        "correct_citation": citation,
        "correct_decline": assigned == "declined" if category == "off_topic" else None,
        "false_decline": assigned == "declined" if category != "off_topic" else None,
    }


def _metrics(rows, key):
    def rate(name):
        values = [row[key][name] for row in rows if row[key].get(name) is not None]
        return sum(values) / len(values) if values else None
    calls = [row[key].get("tool_calls", 0) for row in rows]
    times = [row[key].get("latency_seconds", 0) for row in rows]
    return {"correct_answer_rate": rate("correct_answer"),
            "correct_tier_rate": rate("correct_tier"),
            "correct_tier_label_rate": rate("correct_tier_label"),
            "correct_citation_rate": rate("correct_citation"),
            "correct_decline_rate": rate("correct_decline"),
            "false_decline_rate": rate("false_decline"),
            "mean_tool_calls": sum(calls) / len(calls) if calls else 0,
            "mean_latency_seconds": sum(times) / len(times) if times else 0}


def validate_question_set(questions):
    if not questions:
        raise RuntimeError("ADAS evaluation is refusing to run: the question set is empty.")
    if any(question.get("placeholder") is True for question in questions):
        raise RuntimeError(
            "ADAS evaluation is refusing to run: questions_adas.json contains "
            "placeholder questions. Replace them with verified questions, "
            "expected answers, and source filenames before measuring accuracy."
        )


def run():
    questions = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    validate_question_set(questions)
    RESULTS_DIR.mkdir(exist_ok=True)
    collection = get_collection(ADAS_COLLECTION_NAME)
    rows = []
    print(f"ADAS eval: {len(questions)} questions; model={CHAT_MODEL}; "
          f"shared chunks={collection.count()}")
    for question in questions:
        print(f"[{question['id']}] {question['question']}")
        row = {"question": question}
        for key, runner in (
            ("agent", lambda q=question: ask_agent(q["question"])),
            ("baseline", lambda q=question: {"answer": ask_model(
                q["question"], system_prompt=BASELINE_PROMPT), "sources": []}),
        ):
            start = time.perf_counter()
            try:
                result = runner()
                error = None
            except Exception as exc:
                result = {"answer": f"ERROR: {exc}", "sources": [], "tool_trace": []}
                error = str(exc)
            latency = time.perf_counter() - start
            scored = _score(question, result, baseline=(key == "baseline"))
            trace = result.get("tool_trace", [])
            result_row = {**scored, "answer": result.get("answer", ""),
                          "sources": result.get("sources", []),
                          "tool_trace": trace,
                          "tool_calls": sum(1 for item in trace
                                            if item.get("tool") not in ("ollama", "iteration_limit", "malformed_tool_call")),
                          "latency_seconds": round(latency, 3), "error": error}
            row[key] = result_row
            print(f"  {key}: tier={scored['answer_source']} "
                  f"tools={result_row['tool_calls']} {latency:.1f}s")
        rows.append(row)

    hybrid_result = {"status": "not measured", "reason": "shared ADAS collection is empty"}
    if collection.count():
        bench = []
        for q in (item for item in questions if item["category"] == "in_docs"):
            target = q.get("expected_source", "").lower()
            target_terms = [term.lower() for term in q.get("expected_terms", [])]
            entry = {"id": q["id"]}
            for mode, enabled in (("vector", False), ("hybrid", True)):
                hits = retrieve(q["question"], k=10, collection_name=ADAS_COLLECTION_NAME,
                                use_hybrid=enabled)
                rank = next((i for i, hit in enumerate(hits, 1)
                             if ((target and target in hit["source"].lower())
                                 or any(term in hit["text"].lower() for term in target_terms))), None)
                entry[mode] = {"rank": rank, "hit": rank is not None}
            bench.append(entry)
        hybrid_result = {"status": "measured", "questions": bench,
                         "vector_hit_rate": sum(q["vector"]["hit"] for q in bench) / len(bench) if bench else 0,
                         "hybrid_hit_rate": sum(q["hybrid"]["hit"] for q in bench) / len(bench) if bench else 0}

    metrics = {key: _metrics(rows, key) for key in ("agent", "baseline")}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"placeholder_question_set": True, "model": CHAT_MODEL,
              "shared_collection": ADAS_COLLECTION_NAME,
              "collection_count": collection.count(), "metrics": metrics,
              "hybrid_ab": hybrid_result, "rows": rows}
    out = RESULTS_DIR / f"adas_eval_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nComparison (placeholder questions; manually review raw answers):")
    print(f"{'metric':30s} {'agent':>14s} {'baseline':>14s}")
    for metric in metrics["agent"]:
        a, b = metrics["agent"][metric], metrics["baseline"][metric]
        av = "n/a" if a is None else f"{a:.3f}"
        bv = "n/a" if b is None else f"{b:.3f}"
        print(f"{metric:30s} {av:>14s} {bv:>14s}")
    print(f"Hybrid A/B: {hybrid_result}")
    print(f"Raw answers/report: {out}")
    return report


if __name__ == "__main__":
    run()
