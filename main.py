"""
main.py  ─  FastAPI backend for the Interview Agent
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import asyncio
import json as _json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from interviewer.agent import (run_agent, run_agent_streaming, score_answers, score_coding_submissions,
    generate_next_question, generate_coding_session
)
from interviewer.github_analyzer import set_github_token
from interviewer.text_extractor import extract_resume_text

app = FastAPI(title="Interview Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    resume_text: str
    job_description: str = ""
    num_questions: int = 8
    github_token: str = ""        # optional — only for private repo access
    time_budget: int = 60         # coding session minutes: 60 or 120


class AnswerItem(BaseModel):
    question_id: int
    question: str
    answer: str


class ScoreRequest(BaseModel):
    # Full state fields needed for context
    candidate_name: str = ""
    title: str = ""
    years_experience: int = 0
    difficulty: str = "mid"
    parsed_resume: dict = {}
    answers: list[AnswerItem]

    @classmethod
    def model_validator_yoe(cls, v):
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0


class CategoryScores(BaseModel):
    technical_accuracy: int = 0
    depth_of_knowledge: int = 0
    problem_solving: int = 0
    system_design: int = 0
    clarity: int = 0
    use_of_examples: int = 0
    role_relevance: int = 0
    culture_collaboration: int = 0


class CategoryNotes(BaseModel):
    technical_accuracy: str = ""
    depth_of_knowledge: str = ""
    problem_solving: str = ""
    system_design: str = ""
    clarity: str = ""
    use_of_examples: str = ""
    role_relevance: str = ""
    culture_collaboration: str = ""


class ScoredAnswer(BaseModel):
    question_id: int
    weighted_score: float
    overall_feedback: str
    strength: str
    gap: str
    categories: CategoryScores
    category_notes: CategoryNotes


class CategoryAverages(BaseModel):
    technical_accuracy: float = 0
    depth_of_knowledge: float = 0
    problem_solving: float = 0
    system_design: float = 0
    clarity: float = 0
    use_of_examples: float = 0
    role_relevance: float = 0
    culture_collaboration: float = 0


class ScoreResponse(BaseModel):
    scored_answers: list[ScoredAnswer]
    final_interview_score: int
    category_averages: CategoryAverages
    score_summary: str


class RunResponse(BaseModel):
    candidate_name: str
    title: str
    years_experience: int
    skills: list[str]
    languages: list[str]
    frameworks: list[str]
    github_profiles: list[dict]
    fit_score: int
    difficulty: str
    interview_topics: list[str]
    questions: list[dict]
    coding_challenges: list[dict]
    session_plan: dict
    is_technical_role: bool
    time_budget: int
    job_description: str
    summary: str
    parsed_resume: dict          # forwarded so frontend can pass back to /api/score
    errors: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/run-text", response_model=RunResponse)
async def run_from_text(req: RunRequest):
    """Run the agent from plain resume text."""
    if not req.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text is required")

    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set on server")

    if req.github_token:
        set_github_token(req.github_token)
    state = run_agent(
        resume_text=req.resume_text,
        job_description=req.job_description,
        num_questions=req.num_questions,
    )
    return _state_to_response(state)


@app.post("/api/run-file", response_model=RunResponse)
async def run_from_file(
    file: UploadFile = File(...),
    job_description: str = Form(""),
    num_questions: int = Form(8),
):
    """Run the agent from an uploaded PDF or DOCX resume."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set on server")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        resume_text = extract_resume_text(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file")

    state = run_agent(
        resume_text=resume_text,
        job_description=job_description,
        num_questions=num_questions,
    )
    return _state_to_response(state)




@app.post("/api/score", response_model=ScoreResponse)
async def score_interview_endpoint(req: ScoreRequest):
    """Score submitted answers against their questions."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set on server")

    if not req.answers:
        raise HTTPException(status_code=400, detail="No answers provided")

    # Reconstruct minimal state for the scoring node
    base_state = {
        "parsed_resume": req.parsed_resume,
        "difficulty": req.difficulty,
        "interview_answers": [a.model_dump() for a in req.answers],
        # unused by scorer but required by TypedDict
        "resume_text": "",
        "job_description": "",
        "num_questions": len(req.answers),
        "github_profiles": [],
        "fit_score": 0,
        "interview_topics": [],
        "questions": [],
        "summary": "",
        "scored_answers": [],
        "final_interview_score": 0,
        "score_summary": "",
        "messages": [],
        "errors": [],
    }

    result = score_answers(base_state, [a.model_dump() for a in req.answers])

    raw_scored = result.get("scored_answers", [])
    scored = [
        ScoredAnswer(
            question_id=s.get("question_id", 0),
            weighted_score=float(s.get("weighted_score", 0)),
            overall_feedback=s.get("overall_feedback", ""),
            strength=s.get("strength", ""),
            gap=s.get("gap", ""),
            categories=CategoryScores(**{
                k: int(s.get("categories", {}).get(k, 0))
                for k in CategoryScores.model_fields
            }),
            category_notes=CategoryNotes(**{
                k: str(s.get("category_notes", {}).get(k, ""))
                for k in CategoryNotes.model_fields
            }),
        )
        for s in raw_scored
    ]

    raw_avgs = result.get("category_averages", {})
    cat_avgs = CategoryAverages(**{
        k: float(raw_avgs.get(k, 0))
        for k in CategoryAverages.model_fields
    })

    return ScoreResponse(
        scored_answers=scored,
        final_interview_score=result.get("final_interview_score", 0),
        category_averages=cat_avgs,
        score_summary=result.get("score_summary", ""),
    )


class CodingSubmission(BaseModel):
    challenge_id: int
    code: str
    passed_tests: int = 0
    total_tests: int = 0
    stdout: str = ""
    stderr: str = ""


class CodingScoreItem(BaseModel):
    challenge_id: int
    correctness: int
    code_quality: int
    feedback: str
    strengths: str
    improvements: str


class CodingScoreRequest(BaseModel):
    parsed_resume: dict
    difficulty: str
    coding_challenges: list[dict]
    submissions: list[CodingSubmission]


class CodingScoreResponse(BaseModel):
    coding_scores: list[CodingScoreItem]
    coding_total_score: int


@app.post("/api/score-coding", response_model=CodingScoreResponse)
async def score_coding_endpoint(req: CodingScoreRequest):
    """Score coding challenge submissions."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set on server")
    if not req.submissions:
        raise HTTPException(status_code=400, detail="No submissions provided")

    base_state = {
        "parsed_resume": req.parsed_resume,
        "difficulty": req.difficulty,
        "coding_challenges": req.coding_challenges,
    }
    result = score_coding_submissions(base_state, [s.model_dump() for s in req.submissions])

    raw = result.get("coding_scores", [])
    submissions_by_idx = [s.model_dump() for s in req.submissions]

    def _safe_int(val, default=0):
        try:
            return int(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    scored = []
    for i, s in enumerate(raw):
        # challenge_id: LLM response → submission id → position+1
        cid = s.get("challenge_id")
        if cid is None and i < len(submissions_by_idx):
            cid = submissions_by_idx[i].get("challenge_id")
        if cid is None:
            cid = i + 1
        scored.append(CodingScoreItem(
            challenge_id=_safe_int(cid, i + 1),
            correctness=_safe_int(s.get("correctness"), 0),
            code_quality=_safe_int(s.get("code_quality"), 0),
            feedback=s.get("feedback") or "",
            strengths=s.get("strengths") or "",
            improvements=s.get("improvements") or "",
        ))
    return CodingScoreResponse(
        coding_scores=scored,
        coding_total_score=result.get("coding_total_score", 0),
    )


# ── SSE streaming endpoints ───────────────────────────────────────────────────

async def _sse_stream(resume_text: str, job_description: str, num_questions: int):
    """
    Async generator that truly streams SSE events as each LangGraph node completes.
    Uses asyncio.Queue as a bridge between the synchronous LangGraph thread
    and the async FastAPI response so events flow in real-time.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _run_in_thread():
        """Runs in a ThreadPoolExecutor — pushes events to queue as they arrive."""
        try:
            for event, data in run_agent_streaming(resume_text, job_description, num_questions):
                # Thread-safe put into the asyncio queue
                loop.call_soon_threadsafe(queue.put_nowait, (event, data))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, (_SENTINEL, None))

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(_run_in_thread)

    try:
        while True:
            event, data = await queue.get()

            if event is _SENTINEL:
                break

            if event == "error":
                yield f"event: error\ndata: {_json.dumps({'message': str(data)})}\n\n"
                break
            elif event == "step":
                payload = _json.dumps(data)
                yield f"event: step\ndata: {payload}\n\n"
            elif event == "done":
                safe = {k: v for k, v in data.items() if k != "messages"}

                def _clean(obj):
                    """Recursively make obj JSON-safe without stringifying numbers/bools."""
                    if obj is None:
                        return None
                    if isinstance(obj, bool):
                        return obj
                    if isinstance(obj, (int, float)):
                        return obj
                    if isinstance(obj, str):
                        return obj
                    if isinstance(obj, dict):
                        return {k: _clean(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_clean(i) for i in obj]
                    # Fallback for non-serialisable types (AIMessage etc.)
                    try:
                        _json.dumps(obj)
                        return obj
                    except Exception:
                        return str(obj)

                payload = _json.dumps(_clean(safe))
                yield f"event: done\ndata: {payload}\n\n"
                break
    finally:
        executor.shutdown(wait=False)


@app.get("/api/run-stream")
async def run_stream_text(
    resume_text: str,
    job_description: str = "",
    num_questions: int = 8,
    github_token: str = "",
    time_budget: int = 60,
):
    """SSE endpoint — GET with query params, streams step events then final state."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set on server")
    if not resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text is required")
    if github_token:
        set_github_token(github_token)
    else:
        set_github_token("")  # clear any previously set token

    return StreamingResponse(
        _sse_stream(resume_text, job_description, num_questions),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/run-stream-file")
async def run_stream_file(
    file: UploadFile = File(...),
    job_description: str = Form(""),
    num_questions: int = Form(8),
    github_token: str = Form(""),
):
    """SSE endpoint for file upload — returns SSE stream."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set on server")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from interviewer.text_extractor import extract_resume_text as _ext
        resume_text = _ext(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file")

    if github_token:
        set_github_token(github_token)
    else:
        set_github_token("")  # clear any previously set token

    return StreamingResponse(
        _sse_stream(resume_text, job_description, num_questions),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Server-side code execution ────────────────────────────────────────────────

class ExecuteRequest(BaseModel):
    challenge_id: int
    setup_code: str = ""          # imports / setup that runs before candidate code
    candidate_code: str           # the candidate's solution
    test_cases: list[dict]        # [{input: [...], expected: any}]
    timeout: int = 10             # seconds


class TestResult(BaseModel):
    index: int
    passed: bool
    got: str
    expected: str
    error: str = ""


class ExecuteResponse(BaseModel):
    challenge_id: int
    results: list[TestResult]
    passed: int
    total: int
    stdout: str
    stderr: str


@app.post("/api/execute", response_model=ExecuteResponse)
async def execute_code(req: ExecuteRequest):
    """
    Server-side Python execution for challenges that need pip packages
    (torch, django, sklearn, etc.). Runs in a subprocess with timeout.
    """
    import subprocess, sys, json as _json, textwrap

    # Build the complete test script
    test_harness = textwrap.dedent(f"""
import sys, json, traceback

# ── setup / imports ──
{req.setup_code or ""}

# ── candidate code ──
{req.candidate_code}

# ── test harness ──
_test_cases = json.loads({repr(_json.dumps(req.test_cases))})
_results = []
for _i, _tc in enumerate(_test_cases):
    try:
        _got = solve(*_tc["input"])
        _exp = _tc["expected"]
        # Normalise torch tensors / numpy arrays
        if hasattr(_got, "tolist"): _got = _got.tolist()
        if hasattr(_got, "item"):   _got = _got.item()
        if hasattr(_exp, "tolist"): _exp = _exp.tolist()
        _passed = (_got == _exp) or (str(_got) == str(_exp))
        _results.append({{"passed": bool(_passed), "got": repr(_got), "exp": repr(_exp), "err": ""}})
    except Exception:
        _results.append({{"passed": False, "got": "", "exp": repr(_tc.get("expected","")), "err": traceback.format_exc(limit=4)}})

print("__RESULTS__" + json.dumps(_results))
""")

    loop = asyncio.get_event_loop()

    def _run_subprocess():
        proc = subprocess.run(
            [sys.executable, "-c", test_harness],
            capture_output=True,
            text=True,
            timeout=req.timeout,
        )
        return proc

    try:
        proc = await loop.run_in_executor(None, _run_subprocess)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired:
        return ExecuteResponse(
            challenge_id=req.challenge_id,
            results=[], passed=0, total=len(req.test_cases),
            stdout="", stderr=f"⏱ Execution timed out after {req.timeout}s",
        )
    except Exception as e:
        return ExecuteResponse(
            challenge_id=req.challenge_id,
            results=[], passed=0, total=len(req.test_cases),
            stdout="", stderr=f"Execution error: {e}",
        )

    # Parse __RESULTS__ from stdout
    marker = "__RESULTS__"
    idx = stdout.find(marker)
    user_stdout = stdout[:idx].strip() if idx != -1 else stdout.strip()
    results_data = []
    if idx != -1:
        try:
            results_data = _json.loads(stdout[idx + len(marker):])
        except Exception:
            pass

    test_results = []
    passed_count = 0
    for i, r in enumerate(results_data):
        passed = bool(r.get("passed", False))
        if passed:
            passed_count += 1
        test_results.append(TestResult(
            index=i,
            passed=passed,
            got=str(r.get("got", "")),
            expected=str(r.get("exp", "")),
            error=str(r.get("err", "")),
        ))

    return ExecuteResponse(
        challenge_id=req.challenge_id,
        results=test_results,
        passed=passed_count,
        total=len(req.test_cases),
        stdout=user_stdout,
        stderr=stderr,
    )


@app.post("/api/github-token")
async def set_token(token: str = ""):
    """Set a GitHub personal access token for private repo access. Pass empty string to clear."""
    set_github_token(token)
    return {"status": "ok", "token_set": bool(token)}


# ── Adaptive interview endpoints ──────────────────────────────────────────────

class NextQuestionRequest(BaseModel):
    session_plan: dict
    conversation: list[dict]   # [{question, answer, q_id, covers_area}]
    parsed_resume: dict
    difficulty: str = "mid"
    job_description: str = ""


class CodingSessionRequest(BaseModel):
    session_plan: dict = {}
    parsed_resume: dict
    github_profiles: list[dict] = []
    difficulty: str = "mid"
    time_budget: int = 60      # 60 or 120 minutes
    job_description: str = ""


@app.post("/api/next-question")
async def next_question_endpoint(req: NextQuestionRequest):
    """Generate the next adaptive interview question based on conversation so far."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set on server")

    loop = asyncio.get_event_loop()
    import concurrent.futures

    def _gen():
        return generate_next_question(
            session_plan=req.session_plan,
            conversation=req.conversation,
            parsed_resume=req.parsed_resume,
            difficulty=req.difficulty,
            job_description=req.job_description,
        )

    with concurrent.futures.ThreadPoolExecutor() as pool:
        question = await loop.run_in_executor(pool, _gen)

    return question


@app.post("/api/coding-session")
async def coding_session_endpoint(req: CodingSessionRequest):
    """Generate time-boxed coding challenges for 60 or 120 minute session."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set on server")

    loop = asyncio.get_event_loop()
    import concurrent.futures

    def _gen():
        return generate_coding_session(
            session_plan=req.session_plan,
            parsed_resume=req.parsed_resume,
            github_profiles=req.github_profiles,
            difficulty=req.difficulty,
            time_budget_minutes=req.time_budget,
            job_description=req.job_description,
        )

    with concurrent.futures.ThreadPoolExecutor() as pool:
        challenges = await loop.run_in_executor(pool, _gen)

    return {"challenges": challenges, "time_budget": req.time_budget}

@app.get("/health")
def health():
    return {"status": "ok", "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY"))}


# ── Helper ────────────────────────────────────────────────────────────────────

def _state_to_response(state: dict) -> RunResponse:
    parsed = state.get("parsed_resume", {}) or {}

    def _str(v):
        return str(v) if v and str(v) not in ("None","null","undefined") else ""

    def _int(v):
        try: return int(v) if v is not None else 0
        except (TypeError, ValueError): return 0

    return RunResponse(
        candidate_name=_str(parsed.get("name")) or "Unknown",
        title=_str(parsed.get("title")),
        years_experience=_int(parsed.get("years_experience")),
        skills=parsed.get("skills") or [],
        languages=parsed.get("languages") or [],
        frameworks=parsed.get("frameworks") or [],
        github_profiles=state.get("github_profiles") or [],
        fit_score=state.get("fit_score") or 0,
        difficulty=state.get("difficulty") or "mid",
        interview_topics=state.get("interview_topics") or [],
        questions=state.get("questions") or [],
        coding_challenges=state.get("coding_challenges") or [],
        session_plan=state.get("session_plan") or {},
        is_technical_role=bool(state.get("is_technical_role", True)),
        time_budget=int(state.get("time_budget") or 60),
        job_description=state.get("job_description") or "",
        summary=state.get("summary") or "",
        parsed_resume=parsed,
        errors=state.get("errors") or [],
    )


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = Path(__file__).parent / "templates" / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Interview Agent API is running</h1><p>POST to /api/run-text or /api/run-file</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)