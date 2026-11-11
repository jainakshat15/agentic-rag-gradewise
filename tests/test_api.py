"""
╭──────────────────────────────────────────────────────────────────────╮
│  🎓 Agentic RAG with LangChain & LangGraph                           │
│  🧪 tests/test_api.py: the week 6 FastAPI backend, in demo mode      │
│                                                                      │
│  Run:  uv run python -m pytest tests/test_api.py -q                  │
│  💡 Every test builds its own app on a temporary output folder with  │
│     demo=True: fake models, fake embeddings, no network, no cost.    │
╰──────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from gradewise.api.demo import DemoResultsModel
from gradewise.api.main import create_app
from gradewise.config import DATA_DIR

EXPECTED_EXAM = DATA_DIR / "exams" / "cs_midterm" / "expected_exam.json"
TRANSCRIPTS = DATA_DIR / "answer-sheets" / "synthetic" / "transcripts.json"
needs_data = pytest.mark.skipif(
    not (EXPECTED_EXAM.exists() and TRANSCRIPTS.exists()), reason="course data not generated yet"
)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path, demo=True)) as c:
        yield c


def wait_for(client: TestClient, batch_id: str, *states: str, timeout: float = 90) -> dict:
    """Poll GET /batches/{id} until the batch reaches one of `states`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/batches/{batch_id}").json()
        if body["status"] in states:
            return body
        assert body["status"] != "failed", body["error"]
        time.sleep(0.1)
    raise AssertionError(f"batch {batch_id} never reached {states}")


def read_sse(client: TestClient, url: str) -> list[tuple[str, dict]]:
    """Read a whole SSE stream into (event, data) pairs."""
    events, name, data = [], "message", []
    with client.stream("GET", url) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data.append(line.split(":", 1)[1].strip())
            elif line == "" and data:
                events.append((name, json.loads("\n".join(data))))
                name, data = "message", []
    return events


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["demo"] is True
    assert {"chat_model", "grading_model", "vision_model", "provider_key"} <= set(body)


def test_friendly_errors(client):
    missing = client.get("/batches/nope")
    assert missing.status_code == 404
    assert set(missing.json()) == {"error", "what", "cause", "fix", "help"}
    assert client.get("/reports/cs_midterm/student_99").status_code == 404
    assert client.get("/exams/..%2Fsecret").status_code in (404, 422)
    bad = client.post("/batches", data={"exam_id": "cs_midterm"})
    assert bad.status_code == 400 and bad.json()["error"] == "That upload can't be used"


@needs_data
def test_exam_upload_pauses_for_approval_then_stores(client):
    pack = DATA_DIR / "exams" / "cs_midterm"
    files = {
        key: (f"{key}.pdf", (pack / f"{key}.pdf").read_bytes(), "application/pdf")
        for key in ("question_paper", "marking_scheme", "methodology")
    }
    created = client.post("/exams", data={"exam_id": "demo_exam"}, files=files)
    assert created.status_code == 201, created.text
    review = created.json()
    assert review["status"] == "awaiting_approval" and review["exam_id"] == "demo_exam"
    assert review["summary"]["question_count"] > 0 and review["paper"]["exam_id"] == "demo_exam"
    assert [e["exam_id"] for e in client.get("/exams").json() if e["source"] == "database"] == []

    approved = client.post("/exams/demo_exam/approve", json={"approved": True}).json()
    assert approved["status"] == "stored"
    listed = {e["exam_id"]: e for e in client.get("/exams").json()}
    assert listed["demo_exam"]["source"] == "database"
    assert client.get("/exams/demo_exam").json()["paper"]["exam_id"] == "demo_exam"
    # ...and you can't approve twice
    assert client.post("/exams/demo_exam/approve", json={"approved": True}).status_code == 409


def test_exam_upload_rejects_non_pdf(client):
    files = {k: (f"{k}.pdf", b"hello", "application/pdf") for k in ("question_paper", "marking_scheme", "methodology")}
    response = client.post("/exams", data={"exam_id": "x"}, files=files)
    assert response.status_code == 400 and "not a PDF" in response.json()["what"]


@needs_data
def test_batch_from_transcripts_pauses_streams_reviews_and_reports(client):
    created = client.post("/batches", data={"exam_id": "cs_midterm", "from_transcripts": "true", "students": "2"})
    assert created.status_code == 202, created.text
    batch_id = created.json()["batch_id"]
    assert created.json()["students"] == ["student_01", "student_02"]

    # ⏸ The demo grader is unsure about low marks, so someone is flagged for Asha.
    paused = wait_for(client, batch_id, "paused", "done")
    assert paused["status"] == "paused"
    pending = paused["pending"]
    assert pending["type"] == "human_review" and pending["items"]
    item = pending["items"][0]
    assert item["question_text"] and item["answer_text"] and item["criteria"]
    assert all("justification" in c for c in item["criteria"])

    # 📡 The SSE stream replays every event so far, then says why it ended.
    events = read_sse(client, f"/batches/{batch_id}/events")
    stages = [data["stage"] for name, data in events if name == "progress"]
    for stage in ("queued", "load_exam", "transcribe", "grade_question", "aggregate", "human_review", "paused"):
        assert stage in stages
    assert events[-1] == ("end", {"batch_id": batch_id, "status": "paused", "last_event_id": len(stages)})
    assert client.post(f"/batches/{batch_id}/review", json={"student_id": "student_99"}).status_code == 409

    # 🧑‍🏫 Review every paused student (override the first mark), until the batch is done.
    first, overridden = True, None
    while paused["status"] == "paused":
        qid = paused["pending"]["items"][0]["question_id"]
        overrides = {qid: paused["pending"]["items"][0]["max_marks"]} if first else {}
        sid = paused["pending"]["student_id"]
        body = {"student_id": sid, "overrides": overrides, "approved": True}
        assert client.post(f"/batches/{batch_id}/review", json=body).status_code == 200
        if first:
            overridden = (sid, qid)
        first = False
        paused = wait_for(client, batch_id, "paused", "done")
    assert [r["student_id"] for r in paused["reports"]] == ["student_01", "student_02"]

    # 📊 Reports: JSON overview, one card as JSON / HTML / PDF.
    overview = client.get("/reports/cs_midterm").json()
    assert overview["count"] == 2 and overview["flagged"] == 0 and overview["questions"]
    sid, qid = overridden
    card = client.get(f"/reports/cs_midterm/{sid}").json()
    result = next(r for r in card["results"] if r["question_id"] == qid)
    assert result["awarded_marks"] == result["max_marks"] and "teacher override" in result["review_reason"]
    html = client.get(f"/reports/cs_midterm/{sid}", params={"format": "html"})
    assert html.headers["content-type"].startswith("text/html") and sid in html.text
    pdf = client.get(f"/reports/cs_midterm/{sid}", params={"format": "pdf"})
    assert pdf.headers["content-type"] == "application/pdf" and pdf.content.startswith(b"%PDF")

    # 💬 Chat with the results, answered by the offline fake (it calls a real tool).
    answer = client.post("/ask", json={"question": "Which question was hardest?", "exam_id": "cs_midterm"})
    assert answer.status_code == 200 and "class_statistics" in answer.json()["answer"]
    assert "students" in answer.json()["answer"]


def test_ask_without_reports_is_a_friendly_404(client):
    response = client.post("/ask", json={"question": "How did they do?", "exam_id": "cs_midterm"})
    assert response.status_code == 404 and response.json()["error"] == "No report cards yet"
    assert "POST /batches" in response.json()["fix"] and "week5/" not in response.json()["fix"]


def test_demo_results_model_picks_the_right_tool():
    pick = DemoResultsModel.choose_tool
    assert pick("What still needs my review?")[0] == "list_flagged"
    assert pick("Why did student_04 lose marks?") == ("get_student_report", {"student_id": "student_04"})
    assert pick("Who struggled with subnetting?")[0] == "search_reports"
    assert pick("Which question did the class struggle with most?")[0] == "class_statistics"
    assert pick("How did the class do on q7a?") == ("class_statistics", {"question_id": "Q7a"})


@needs_data
def test_knowledge_base_upload_and_ingest(client):
    pdf = DATA_DIR / "knowledge-base" / "unit3_databases.pdf"
    files = [("files", (pdf.name, pdf.read_bytes(), "application/pdf"))]
    response = client.post("/knowledge-bases/notes/documents", files=files)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["added"] == [pdf.name] and body["chunks"] > 0 and body["demo"] is True
    again = client.post("/knowledge-bases/notes/documents", files=files).json()
    assert again["skipped"] == [pdf.name] and again["chunks_added"] == 0  # 💡 idempotent ingest
    listed = {kb["name"]: kb for kb in client.get("/knowledge-bases").json()}
    assert listed["notes"]["chunks"] == body["chunks"]
    assert client.post("/knowledge-bases/bad..name/documents", files=files).status_code == 422
