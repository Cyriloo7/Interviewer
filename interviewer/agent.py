"""
interviewer/agent.py
Full LangGraph multi-node agent for resume-aware + GitHub-aware
interview question generation.

Graph topology
──────────────
                   ┌──────────────┐
       START ──▶   │  parse_resume │
                   └──────┬───────┘
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
        ┌────────────┐       ┌──────────────────┐
        │ score_fit  │       │  fetch_github     │
        └─────┬──────┘       └────────┬─────────┘
              │                       │
              └──────────┬────────────┘
                         ▼
                  ┌─────────────┐
                  │ build_topics│
                  └──────┬──────┘
                         ▼
                  ┌──────────────┐
                  │ gen_questions│
                  └──────┬───────┘
                         ▼
                  ┌─────────────┐
                  │  finalize   │
                  └──────┬──────┘
                         ▼
                        END

  After the interview, call score_answers() (standalone, not in graph):

                  ┌───────────────────┐
      answers ──▶ │  score_interview   │──▶ scored_answers
                  │   (standalone)     │    final_interview_score
                  └───────────────────┘    score_summary
"""

from __future__ import annotations

import json
import os
import re
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from interviewer.github_analyzer import (
    extract_github_usernames,
    get_github_profile_summary,
)

# ── LLM ──────────────────────────────────────────────────────────────────────

_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=_model,
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

json_llm = ChatGoogleGenerativeAI(
    model=_model,
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Inputs
    resume_text: str
    job_description: str           # optional JD to tailor questions
    num_questions: int             # kept for API compat — adaptive mode ignores this

    # Intermediate
    parsed_resume: dict            # structured fields from resume
    github_profiles: list[dict]    # raw GitHub summaries
    fit_score: int                 # 0-100 rough resume-JD fit
    interview_topics: list[str]    # topics to probe
    difficulty: str                # "junior" | "mid" | "senior"
    is_technical_role: bool        # True → show coding challenges

    # Outputs
    questions: list[dict]          # kept for backwards compat — first question seed
    coding_challenges: list[dict]  # generated based on time_budget
    summary: str                   # overall candidate summary

    # Adaptive interview session plan (replaces static question list)
    session_plan: dict             # {topics, jd_areas, coverage_map, difficulty, candidate_ctx}

    # Scoring (populated after interview answers are submitted)
    interview_answers: list[dict]  # [{question_id, question, answer}]
    scored_answers: list[dict]     # [{question_id, weighted_score, categories, category_notes, ...}]
    final_interview_score: int     # 0-100 weighted final score
    category_averages: dict        # {category_key: avg_score} across all questions
    score_summary: str             # hiring recommendation paragraph

    # Coding challenge results (populated after code is submitted + executed)
    coding_submissions: list[dict] # [{challenge_id, code, passed_tests, stdout, stderr}]
    coding_scores: list[dict]      # [{challenge_id, correctness, code_quality, feedback}]
    coding_total_score: int        # 0-100
    time_budget: int               # coding session minutes (60 or 120)

    messages: Annotated[Sequence, add_messages]
    errors: list[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_json(text: str, fallback: Any) -> Any:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        return fallback


# ── Node 1: Parse Resume ──────────────────────────────────────────────────────

PARSE_SYSTEM = """You are a resume parser. Extract a JSON object with these exact fields:
{
  "name": string,
  "title": string,
  "years_experience": number (integer, 0 if unknown),
  "skills": [string],
  "languages": [string],
  "frameworks": [string],
  "education": string,
  "companies": [string],
  "github_mentions": [string],
  "project_highlights": [string],
  "summary_line": string
}

IMPORTANT rules:
- years_experience MUST be a number (integer), never null or a string. Use 0 if unknown.
- name MUST be the candidate's full name string, never null.
- github_mentions: extract ALL GitHub profile URLs or usernames found anywhere in the resume.
  Include the FULL URL if present (e.g. "https://github.com/johndoe") or just the username.
  This is critical — scan every line for github.com links.
- Respond with ONLY valid JSON, no markdown fences, no explanation."""


def parse_resume(state: AgentState) -> dict:
    resp = json_llm.invoke([
        SystemMessage(content=PARSE_SYSTEM),
        HumanMessage(content=state["resume_text"][:6000]),
    ])
    parsed = _safe_json(resp.content, {})
    return {
        "parsed_resume": parsed,
        "messages": [AIMessage(content="✅ Resume parsed successfully.")],
    }


# ── Node 2: Fetch GitHub ──────────────────────────────────────────────────────

def fetch_github(state: AgentState) -> dict:
    """
    Extract GitHub usernames from:
    1. Raw resume text (regex on URLs)
    2. parsed_resume.github_mentions (LLM-extracted, now available since
       fetch_github runs sequentially after parse_resume)
    3. Plain username fallback from mentions without github.com
    """
    resume_text = state["resume_text"]
    parsed      = state.get("parsed_resume", {}) or {}
    errors      = list(state.get("errors", []))
    candidates: list[str] = []

    # Source 1: Scan raw resume text for all github.com URLs
    from_text = extract_github_usernames(resume_text)
    print(f"[GitHub] From raw text: {from_text}")
    candidates.extend(from_text)

    # Source 2: LLM-extracted github_mentions (full URLs or plain usernames)
    for mention in parsed.get("github_mentions", []):
        mention = mention.strip()
        if not mention:
            continue
        if "github.com" in mention.lower():
            # Full URL — extract username from it
            extracted = extract_github_usernames(mention)
            print(f"[GitHub] From LLM mention URL {mention!r}: {extracted}")
            candidates.extend(extracted)
        else:
            # Plain username or @username
            clean = mention.lstrip("@").strip()
            if re.match(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$", clean):
                print(f"[GitHub] From LLM plain mention: {clean!r}")
                candidates.append(clean)

    # Deduplicate preserving order, cap at 3
    seen: set[str] = set()
    unique: list[str] = []
    for u in candidates:
        key = u.lower()
        if key not in seen:
            seen.add(key)
            unique.append(u)
    usernames = unique[:3]

    print(f"[GitHub] Final usernames to fetch: {usernames}")

    if not usernames:
        # Last resort: ask the LLM to extract a GitHub username from the full resume
        print("[GitHub] No URL found — trying LLM extraction from resume text...")
        try:
            llm_resp = json_llm.invoke([
                SystemMessage(content="""Extract the GitHub username from this resume text.
Look for any GitHub profile link, username, or mention.
Return ONLY the username (e.g. "johndoe"), nothing else.
If no GitHub username is found, return exactly: NONE"""),
                HumanMessage(content=resume_text[:4000]),
            ])
            extracted = llm_resp.content.strip().strip('"').strip()
            print(f"[GitHub] LLM extracted: {repr(extracted)}")
            if extracted and extracted.upper() != "NONE" and re.match(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$", extracted):
                usernames = [extracted]
            else:
                return {
                    "github_profiles": [],
                    "errors": errors,
                    "messages": [AIMessage(content="ℹ️ No GitHub username found in resume.")],
                }
        except Exception as e:
            print(f"[GitHub] LLM fallback failed: {e}")
            return {
                "github_profiles": [],
                "errors": errors,
                "messages": [AIMessage(content="ℹ️ No GitHub username found in resume.")],
            }

    profiles = []
    for username in usernames:
        print(f"[GitHub] Fetching @{username}...")
        profile = get_github_profile_summary(username)
        if "error" in profile:
            err_msg = f"GitHub @{username}: {profile['error']}"
            errors.append(err_msg)
            print(f"[GitHub] ✗ {err_msg}")
        else:
            profiles.append(profile)
            print(f"[GitHub] ✓ @{username}: {profile.get('public_repos', 0)} public repos")

    msg = (
        f"✅ Fetched {len(profiles)} GitHub profile(s): {[p['username'] for p in profiles]}"
        if profiles
        else f"⚠️ No GitHub data. Tried: {usernames}"
    )
    return {
        "github_profiles": profiles,
        "errors": errors,
        "messages": [AIMessage(content=msg)],
    }


# ── Node 3: Score Fit ─────────────────────────────────────────────────────────

SCORE_SYSTEM = """You are a technical recruiter. Given a resume summary and a job description,
return a JSON object:
{
  "fit_score": <integer 0-100>,
  "difficulty": "<junior|mid|senior>",
  "is_technical_role": <true|false>,
  "rationale": "<one sentence>"
}

is_technical_role should be true for: software engineer, data scientist, ML engineer, DevOps,
backend/frontend/fullstack engineer, data engineer, AI researcher, platform engineer, QA engineer,
embedded engineer, security engineer, or any role where writing code is a core job requirement.
is_technical_role should be false for: product manager, designer, marketing, sales, HR, finance,
operations, business analyst, project manager, customer success, and any non-coding role.

Only JSON, no fences."""


def score_fit(state: AgentState) -> dict:
    jd = state.get("job_description", "").strip()
    parsed = state.get("parsed_resume", {})

    if not jd:
        # Infer level from years of experience alone
        yoe = parsed.get("years_experience", 0) or 0
        if yoe < 2:
            difficulty = "junior"
        elif yoe < 6:
            difficulty = "mid"
        else:
            difficulty = "senior"
        # Without a JD, infer technical role from resume skills/title
        tech_keywords = {"python","java","javascript","typescript","c++","go","rust","sql",
                         "react","django","fastapi","pytorch","tensorflow","aws","docker",
                         "kubernetes","engineer","developer","programmer","data scientist"}
        resume_lower = (str(parsed.get("skills","")) + str(parsed.get("title","")) + 
                        str(parsed.get("languages",""))).lower()
        is_tech = any(kw in resume_lower for kw in tech_keywords)
        return {
            "fit_score": 75,
            "difficulty": difficulty,
            "is_technical_role": is_tech,
            "messages": [AIMessage(content=f"ℹ️ No JD — difficulty: '{difficulty}', technical: {is_tech}.")],
        }

    prompt = f"""RESUME SUMMARY:
Name: {parsed.get('name')}
Experience: {parsed.get('years_experience')} years
Skills: {parsed.get('skills')}
Title: {parsed.get('title')}

JOB DESCRIPTION (first 1500 chars):
{jd[:1500]}"""

    resp = json_llm.invoke([
        SystemMessage(content=SCORE_SYSTEM),
        HumanMessage(content=prompt),
    ])
    result = _safe_json(resp.content, {"fit_score": 70, "difficulty": "mid", "rationale": ""})

    is_tech = bool(result.get("is_technical_role", True))
    return {
        "fit_score": int(result.get("fit_score", 70)),
        "difficulty": result.get("difficulty", "mid"),
        "is_technical_role": is_tech,
        "messages": [AIMessage(content=f"✅ Fit: {result.get('fit_score')}% | Level: {result.get('difficulty')} | Technical: {is_tech}")],
    }


# ── Node 4: Build Topics ──────────────────────────────────────────────────────

TOPICS_SYSTEM = """You are a senior technical interviewer. Based on the candidate profile below,
return a JSON array of 5–8 topic strings to probe in the interview.
Topics should reflect their actual stack, projects, and gaps.
Respond with ONLY a JSON array of strings."""


def build_topics(state: AgentState) -> dict:
    parsed = state.get("parsed_resume", {})
    profiles = state.get("github_profiles", [])

    # Summarize GitHub activity using rich project analyses
    github_summary = ""
    for p in profiles:
        github_summary += f"\nGitHub @{p['username']} ({p.get('public_repos',0)} public repos): {p.get('bio','')}"
        for pa in p.get("project_analyses", [])[:5]:
            # Just the first 300 chars of each analysis for the topics node
            snippet = pa["analysis"][:300].replace("\n", " ")
            github_summary += f"\n  • {pa['name']}: {snippet}"

    prompt = f"""CANDIDATE:
Name: {parsed.get('name')}
Title: {parsed.get('title')}
Experience: {parsed.get('years_experience')} years
Skills: {parsed.get('skills')}
Languages: {parsed.get('languages')}
Frameworks: {parsed.get('frameworks')}
Projects: {parsed.get('project_highlights')}
Education: {parsed.get('education')}
{github_summary}

JD context: {state.get('job_description', 'Not provided')[:500]}
Interview difficulty: {state.get('difficulty', 'mid')}"""

    resp = json_llm.invoke([
        SystemMessage(content=TOPICS_SYSTEM),
        HumanMessage(content=prompt),
    ])
    topics = _safe_json(resp.content, ["Data Structures", "System Design", "Problem Solving"])

    return {
        "interview_topics": topics,
        "messages": [AIMessage(content=f"✅ Interview topics: {', '.join(topics)}")],
    }


# ── Node 5: Generate Questions ────────────────────────────────────────────────

SESSION_PLAN_SYSTEM = """You are a senior technical interviewer planning an adaptive interview session.
Given a candidate profile, GitHub projects, and job description, create a structured interview plan.

Return a JSON object:
{
  "jd_areas": [
    {"area": "<JD requirement>", "priority": <1-5>, "covered": false}
  ],
  "topics": [
    {
      "name": "<topic>",
      "source": "<resume|github|jd>",
      "depth": "<surface|deep>",
      "sample_angles": ["<angle1>", "<angle2>"]
    }
  ],
  "candidate_ctx": "<2-3 sentences: key strengths, notable projects, likely gaps — used to personalise each question>",
  "opening_question": {
    "id": 1,
    "category": "<category>",
    "question": "<the first question — open, warm, sets tone>",
    "rationale": "<why start here>",
    "difficulty": "easy",
    "covers_area": "<which JD area this probes>"
  }
}

The opening question should be welcoming — ask about a specific project or experience, not a quiz.
Return ONLY valid JSON."""


def gen_questions(state: AgentState) -> dict:
    """Build adaptive session plan with opening question. Subsequent questions generated live."""
    parsed   = state.get("parsed_resume", {})
    profiles = state.get("github_profiles", [])
    topics   = state.get("interview_topics", [])
    difficulty = state.get("difficulty", "mid")

    github_ctx = ""
    for p in profiles:
        github_ctx += f"\nGitHub @{p['username']}: {p.get('bio','')}\n"
        for pa in p.get("project_analyses", [])[:4]:
            github_ctx += f"  [{pa['name']}]: {pa['analysis'][:400]}\n"

    prompt = f"""CANDIDATE:
Name: {parsed.get('name')} | Title: {parsed.get('title')} | Level: {difficulty}
Experience: {parsed.get('years_experience')} years
Skills: {parsed.get('skills',[])} | Languages: {parsed.get('languages',[])}
Frameworks: {parsed.get('frameworks',[])}
Projects: {parsed.get('project_highlights',[])}
{github_ctx}

TOPICS TO COVER: {topics}
JOB DESCRIPTION: {state.get('job_description', 'General software engineering')[:1000]}

Create the interview session plan with an opening question."""

    resp = json_llm.invoke([
        SystemMessage(content=SESSION_PLAN_SYSTEM),
        HumanMessage(content=prompt),
    ])
    plan = _safe_json(resp.content, {})

    opening = plan.get("opening_question", {})
    opening.setdefault("id", 1)
    opening.setdefault("difficulty", "easy")
    opening.setdefault("follow_up", "")

    return {
        "session_plan": plan,
        "questions": [opening],   # seed with opening question for UI compat
        "messages": [AIMessage(content="✅ Interview session plan ready.")],
    }



# ── Node 6b: Generate Coding Challenges ──────────────────────────────────────

# Packages available in Pyodide (browser runner)
PYODIDE_PACKAGES = ["numpy", "pandas", "scipy", "sympy", "networkx", "matplotlib"]

# Packages that require server-side execution
SERVER_PACKAGES = ["torch", "tensorflow", "sklearn", "scikit-learn",
                   "django", "flask", "fastapi", "sqlalchemy",
                   "transformers", "cv2", "PIL", "requests"]

CODING_SYSTEM = """You are a senior Python engineer generating live coding challenges for a technical interview.
Generate exactly 3 coding challenges: one basic, one medium, one hard.

Each challenge has a "runner" field: "browser" or "server".
- runner "browser": pure Python stdlib OR numpy/pandas/scipy/sympy only. Runs in WebAssembly.
- runner "server":  any pip package including torch, django, flask, sklearn, transformers, etc.
                    Runs in a real Python subprocess on the interviewer's server.

Rules per level:
- Basic  (runner: "browser"): Python fundamentals + optionally numpy arrays or pandas Series.
  If candidate has PyTorch/ML background, use numpy to test tensor-like thinking.
- Medium (runner: "browser" or "server"): Based on their actual stack.
  Use numpy/pandas for data candidates. Use server runner for ML/DL (torch, sklearn).
- Hard   (runner: "server"): Based on a specific pattern from their GitHub projects.
  May use torch, django ORM logic, FastAPI-style validators, etc.
  For non-ML candidates, use browser runner with their stack's logic.

Each challenge MUST:
1. Be fully self-contained — the function MUST be named exactly "solve" (not calculate_iou, not solution, not anything else — ONLY "solve")
2. Have clear starter_code with the function signature
3. Have 3-5 test cases: [{"input": [...], "expected": <value>}, ...]
   where input is a list of positional args unpacked into solve()
4. For server runner: include a "setup_code" field with any imports/setup needed
   (e.g. "import torch\nimport numpy as np") that runs BEFORE the candidate's code
5. Be solvable in under 15 minutes
6. Have test cases with JSON-comparable values (numbers, strings, lists, bools)
   — no raw torch tensors as expected values, convert with .tolist() or .item()

Return ONLY a valid JSON array, no markdown fences:
[
  {
    "id": 1,
    "level": "basic",
    "runner": "browser",
    "title": "<short title>",
    "description": "<clear problem statement, 3-5 sentences, include examples>",
    "tags": ["numpy", "array"],
    "setup_code": "",
    "starter_code": "import numpy as np\n\ndef solve(...):\n    # your code here\n    pass",
    "solution_hint": "<one sentence hint, not the answer>",
    "test_cases": [
      {"input": [arg1, arg2], "expected": result},
      ...
    ],
    "rationale": "<why this tests the candidate>"
  },
  {
    "id": 2,
    "level": "medium",
    "runner": "browser",
    ...
  },
  {
    "id": 3,
    "level": "hard",
    "runner": "server",
    "setup_code": "import torch\nimport numpy as np",
    ...
  }
]"""


def gen_coding_challenges(state: AgentState) -> dict:
    """Generate 3 coding challenges (basic/medium/hard) personalised to the candidate.
    Skipped entirely for non-technical roles."""
    if not state.get("is_technical_role", True):
        return {
            "coding_challenges": [],
            "messages": [AIMessage(content="ℹ️ Non-technical role — coding challenges skipped.")],
        }

    parsed   = state.get("parsed_resume", {})
    profiles = state.get("github_profiles", [])
    difficulty = state.get("difficulty", "mid")

    # Build GitHub project context for medium/hard questions
    github_ctx = ""
    for p in profiles:
        github_ctx += f"\nGitHub @{p['username']}: {p.get('bio','')}\n"
        for pa in p.get("project_analyses", [])[:3]:
            github_ctx += f"  Project: {pa['name']}\n"
            github_ctx += f"  {pa['analysis'][:400]}\n"

    # Detect what packages the candidate actually knows
    frameworks_str = str(parsed.get('frameworks', []) + parsed.get('skills', [])[:10]).lower()
    has_torch    = any(k in frameworks_str for k in ['torch', 'pytorch', 'deep learning', 'neural'])
    has_ml       = any(k in frameworks_str for k in ['sklearn', 'scikit', 'xgboost', 'lightgbm', 'ml'])
    has_django   = any(k in frameworks_str for k in ['django', 'flask', 'fastapi'])
    has_numpy    = any(k in frameworks_str for k in ['numpy', 'pandas', 'scipy', 'data'])
    pyodide_pkgs = ', '.join(PYODIDE_PACKAGES)

    prompt = f"""Candidate: {parsed.get('name')}, {difficulty} level.
Languages: {parsed.get('languages', [])}
Frameworks: {parsed.get('frameworks', [])}
Skills: {parsed.get('skills', [])[:10]}
Has PyTorch/DL: {has_torch}
Has ML (sklearn etc): {has_ml}
Has Django/Flask/FastAPI: {has_django}
Has numpy/pandas: {has_numpy}
{github_ctx}

Generate 3 coding challenges following the spec exactly.

- Basic (runner: "browser"):
  {"Use numpy arrays to test tensor/matrix thinking — reshape, broadcasting, indexing." if has_torch or has_numpy else "Pure Python fundamentals — lists, dicts, strings, sorting, recursion."}
  Allowed imports: stdlib + numpy/pandas/scipy/sympy if relevant.

- Medium (runner: "{"server" if (has_torch or has_ml) else "browser"}"):
  {"Test PyTorch or sklearn understanding — write a function using torch tensors or sklearn API." if has_torch or has_ml else "Realistic algorithmic problem from their actual stack using stdlib or numpy."}
  {"Use torch or sklearn — mark runner as server." if has_torch or has_ml else ""}

- Hard (runner: "server"):
  {"Inspired by their GitHub projects — use torch (model logic, loss, grad), or Django ORM patterns, or FastAPI validators." if (has_torch or has_django) else "Deep algorithmic challenge inspired by their GitHub — may use numpy/pandas for data work."}
  This MUST be runner: "server" to allow any pip package.

IMPORTANT:
- browser challenges: ONLY stdlib + {pyodide_pkgs}
- server challenges: ANY pip package is fine (torch, django, sklearn, transformers, etc.)
- All expected values must be JSON-serialisable (use .tolist(), .item(), float() on tensors)
- Function must always be named exactly solve"""

    resp = json_llm.invoke([
        SystemMessage(content=CODING_SYSTEM),
        HumanMessage(content=prompt),
    ])
    challenges = _safe_json(resp.content, [])

    # Validate + normalise each challenge
    validated = []
    for i, c in enumerate(challenges[:3]):
        c.setdefault("id", i + 1)
        c.setdefault("level", ["basic", "medium", "hard"][i])
        # Hard challenges always use server runner
        default_runner = "server" if c.get("level") == "hard" else "browser"
        c.setdefault("runner", default_runner)
        c.setdefault("title", f"Challenge {i+1}")
        c.setdefault("description", "")
        c.setdefault("setup_code", "")
        c.setdefault("starter_code", "def solve():\n    # write your solution here\n    pass")
        c.setdefault("test_cases", [])
        c.setdefault("solution_hint", "")
        c.setdefault("rationale", "")
        c.setdefault("tags", [])
        validated.append(c)

    return {
        "coding_challenges": validated,
        "messages": [AIMessage(content=f"✅ Generated {len(validated)} coding challenges.")],
    }


# ── Node 8: Score Coding Submissions ─────────────────────────────────────────

CODING_SCORE_SYSTEM = """You are a Python code reviewer scoring a live interview coding submission.
Given a challenge and the candidate's code + test results, return a JSON object:
{
  "challenge_id": <int>,
  "correctness": <int 0-10>,
  "code_quality": <int 0-10>,
  "feedback": "<2-3 sentences covering correctness, style, edge cases, efficiency>",
  "strengths": "<one phrase>",
  "improvements": "<one phrase>"
}

Correctness scoring:
  10: all tests pass, handles edge cases
  7-9: most tests pass, minor issues
  4-6: partial solution, right approach
  1-3: wrong approach or major bugs
  0: blank or completely wrong

Code quality scoring (even if tests pass):
  10: clean, idiomatic, well-named, efficient
  7-9: readable, minor style issues
  4-6: works but messy or inefficient
  1-3: hard to read, bad patterns
  0: unreadable

Return ONLY valid JSON."""


def score_coding(submissions: list[dict], challenges: list[dict], parsed_resume: dict) -> list[dict]:
    """Score each coding submission for correctness + code quality."""
    challenge_map = {c["id"]: c for c in challenges}
    scored = []

    for sub in submissions:
        cid = sub.get("challenge_id")
        challenge = challenge_map.get(cid, {})

        prompt = f"""Candidate: {parsed_resume.get('name')}
Challenge: {challenge.get('title', '')} ({challenge.get('level', '')})
Problem: {challenge.get('description', '')[:400]}

Candidate's code:
```python
{sub.get('code', '')}
```

Test results:
  Passed: {sub.get('passed_tests', 0)} / {sub.get('total_tests', 0)}
  stdout: {str(sub.get('stdout', ''))[:300]}
  stderr: {str(sub.get('stderr', ''))[:200]}

Score this submission."""

        resp = json_llm.invoke([
            SystemMessage(content=CODING_SCORE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        result = _safe_json(resp.content, {})
        # Always overwrite challenge_id with the actual submission id — LLM sometimes omits it
        result["challenge_id"] = cid
        result["correctness"]  = int(result.get("correctness")  or 0)
        result["code_quality"] = int(result.get("code_quality") or 0)
        result.setdefault("feedback", "")
        result.setdefault("strengths", "")
        result.setdefault("improvements", "")
        scored.append(result)

    return scored


def score_coding_submissions(base_state: dict, submissions: list[dict]) -> dict:
    """Public function: score coding submissions and return coding score data."""
    challenges = base_state.get("coding_challenges", [])
    parsed     = base_state.get("parsed_resume", {})
    scored     = score_coding(submissions, challenges, parsed)

    if scored:
        avg = sum((s["correctness"] + s["code_quality"]) / 2 for s in scored) / len(scored)
        total = round(avg * 10)
    else:
        total = 0

    return {
        "coding_scores": scored,
        "coding_total_score": total,
        "coding_submissions": submissions,
    }

# ── Node 6: Finalize ──────────────────────────────────────────────────────────

SUMMARY_SYSTEM = """Write a 3–4 sentence candidate brief for the hiring manager.
Mention: technical strengths, GitHub activity (if any), interview readiness, level assessment.
Be direct and useful. Plain text only."""


def finalize(state: AgentState) -> dict:
    parsed = state.get("parsed_resume", {})
    profiles = state.get("github_profiles", [])
    questions = state.get("questions", [])

    github_note = (
        f"Has {len(profiles)} verified GitHub profile(s)."
        if profiles
        else "No GitHub profiles found."
    )

    prompt = f"""Candidate: {parsed.get('name')}, {parsed.get('years_experience')} years exp.
Title: {parsed.get('title')}
Top skills: {parsed.get('skills', [])[:8]}
Fit score: {state.get('fit_score')}%
Difficulty set: {state.get('difficulty')}
{github_note}
Questions generated: {len(questions)}"""

    resp = llm.invoke([
        SystemMessage(content=SUMMARY_SYSTEM),
        HumanMessage(content=prompt),
    ])

    return {
        "summary": resp.content.strip(),
        "messages": [AIMessage(content="🎯 Interview pack ready.")],
    }




# ── Node 7: Score Interview ───────────────────────────────────────────────────

# Category weights for final score
SCORE_WEIGHTS = {
    "technical_accuracy":   0.25,
    "depth_of_knowledge":   0.20,
    "problem_solving":      0.20,
    "system_design":        0.15,
    "clarity":              0.10,
    "use_of_examples":      0.05,
    "role_relevance":       0.03,
    "culture_collaboration":0.02,
}

SCORE_INTERVIEW_SYSTEM = """You are a strict technical interview evaluator.
You will be given a list of interview question-answer pairs.
For EACH answer, score it across 8 categories and return a JSON array:
[
  {
    "question_id": <int>,
    "overall_feedback": "<2 sentences: what was good and what was missing overall>",
    "strength": "<one-word or short phrase describing the strongest aspect>",
    "gap": "<one-word or short phrase describing the biggest gap, or empty string>",
    "categories": {
      "technical_accuracy":    <int 0-10>,
      "depth_of_knowledge":    <int 0-10>,
      "problem_solving":       <int 0-10>,
      "system_design":         <int 0-10>,
      "clarity":               <int 0-10>,
      "use_of_examples":       <int 0-10>,
      "role_relevance":        <int 0-10>,
      "culture_collaboration": <int 0-10>
    },
    "category_notes": {
      "technical_accuracy":    "<one short sentence>",
      "depth_of_knowledge":    "<one short sentence>",
      "problem_solving":       "<one short sentence>",
      "system_design":         "<one short sentence>",
      "clarity":               "<one short sentence>",
      "use_of_examples":       "<one short sentence>",
      "role_relevance":        "<one short sentence>",
      "culture_collaboration": "<one short sentence>"
    }
  },
  ...
]

Category definitions:
  technical_accuracy    — correctness of concepts, facts, and implementation details
  depth_of_knowledge    — goes beyond surface level; shows genuine mastery
  problem_solving       — structured reasoning, considers edge cases and trade-offs
  system_design         — architecture thinking, scalability, trade-off awareness
  clarity               — answer is well-structured and easy to follow
  use_of_examples       — backs claims with concrete real examples or analogies
  role_relevance        — answer maps to the job requirements / JD
  culture_collaboration — ownership language, team awareness, how they discuss past work

Scoring guide per category:
  9-10: Exceptional  7-8: Strong  5-6: Adequate  3-4: Weak  0-2: Incorrect/absent

Calibrate scores to the candidate difficulty level (junior / mid / senior).
Return ONLY a valid JSON array, no markdown fences."""


def _weighted_score(categories: dict) -> float:
    """Compute weighted average from category scores dict."""
    total = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        total += categories.get(key, 0) * weight
    return total  # 0-10 float


def score_interview(state: AgentState) -> dict:
    """Score each Q&A pair across 8 categories and compute a weighted final score."""
    answers = state.get("interview_answers", [])
    if not answers:
        return {
            "scored_answers": [],
            "final_interview_score": 0,
            "score_summary": "No answers were submitted for scoring.",
            "messages": [AIMessage(content="⚠️ No answers to score.")],
        }

    difficulty = state.get("difficulty", "mid")
    parsed = state.get("parsed_resume", {})
    jd_context = state.get("job_description", "") or "General software engineering role"

    qa_block = "\n".join(
        f"Q{a['question_id']}: {a['question']}\nA: {a['answer']}"
        for a in answers
    )

    prompt = f"""Candidate: {parsed.get('name')}, {difficulty.upper()} level.
Role context: {jd_context[:400]}

QUESTION-ANSWER PAIRS:
{qa_block}

Evaluate each answer across all 8 categories. Return JSON array as specified."""

    resp = json_llm.invoke([
        SystemMessage(content=SCORE_INTERVIEW_SYSTEM),
        HumanMessage(content=prompt),
    ])
    scored = _safe_json(resp.content, [])

    # Add per-question weighted score and ensure all fields present
    for s in scored:
        cats = s.get("categories", {})
        s["weighted_score"] = round(_weighted_score(cats), 2)
        s.setdefault("overall_feedback", "")
        s.setdefault("strength", "")
        s.setdefault("gap", "")
        s.setdefault("category_notes", {})

    # Final score: average weighted scores → scale to 0-100
    if scored:
        avg_weighted = sum(s["weighted_score"] for s in scored) / len(scored)
        final_score = round(avg_weighted * 10)
    else:
        final_score = 0

    # Aggregate category averages across all questions
    cat_avgs = {}
    for key in SCORE_WEIGHTS:
        vals = [s.get("categories", {}).get(key, 0) for s in scored]
        cat_avgs[key] = round(sum(vals) / len(vals), 1) if vals else 0

    # Hiring recommendation
    rec_prompt = f"""Candidate: {parsed.get('name')}, {difficulty} level.
Final interview score: {final_score}/100

Category averages (0-10):
  Technical Accuracy:    {cat_avgs.get('technical_accuracy')}
  Depth of Knowledge:    {cat_avgs.get('depth_of_knowledge')}
  Problem Solving:       {cat_avgs.get('problem_solving')}
  System Design:         {cat_avgs.get('system_design')}
  Clarity:               {cat_avgs.get('clarity')}
  Use of Examples:       {cat_avgs.get('use_of_examples')}
  Role Relevance:        {cat_avgs.get('role_relevance')}
  Culture/Collaboration: {cat_avgs.get('culture_collaboration')}

Key strengths: {[s.get('strength') for s in scored if s.get('strength')]}
Key gaps:      {[s.get('gap') for s in scored if s.get('gap')]}

Write a 2–3 sentence hiring recommendation. Be direct: hire / consider / pass, and explain the 1-2 most decisive factors."""

    rec_resp = llm.invoke([HumanMessage(content=rec_prompt)])

    return {
        "scored_answers": scored,
        "final_interview_score": final_score,
        "category_averages": cat_avgs,
        "score_summary": rec_resp.content.strip(),
        "messages": [AIMessage(content=f"✅ Interview scored: {final_score}/100")],
    }

# ── Build Graph ───────────────────────────────────────────────────────────────

def build_agent() -> Any:
    builder = StateGraph(AgentState)

    builder.add_node("parse_resume", parse_resume)
    builder.add_node("fetch_github", fetch_github)
    builder.add_node("score_fit", score_fit)
    builder.add_node("build_topics", build_topics)
    builder.add_node("gen_questions", gen_questions)
    builder.add_node("gen_coding_challenges", gen_coding_challenges)
    builder.add_node("finalize", finalize)

    # Entry
    builder.add_edge(START, "parse_resume")

    # fetch_github runs AFTER parse_resume (needs parsed github_mentions)
    builder.add_edge("parse_resume", "fetch_github")

    # score_fit also runs after parse_resume (parallel with fetch_github)
    builder.add_edge("parse_resume", "score_fit")

    # Both converge at build_topics
    builder.add_edge("fetch_github", "build_topics")
    builder.add_edge("score_fit", "build_topics")

    # gen_questions and gen_coding_challenges run in parallel after build_topics
    builder.add_edge("build_topics", "gen_questions")
    builder.add_edge("build_topics", "gen_coding_challenges")

    # Both converge at finalize
    builder.add_edge("gen_questions", "finalize")
    builder.add_edge("gen_coding_challenges", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


agent = build_agent()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(
    resume_text: str,
    job_description: str = "",
    num_questions: int = 8,
) -> AgentState:
    """Run the full agent and return final state."""
    initial: AgentState = {
        "resume_text": resume_text,
        "job_description": job_description,
        "num_questions": num_questions,
        "parsed_resume": {},
        "github_profiles": [],
        "fit_score": 0,
        "interview_topics": [],
        "difficulty": "mid",
        "is_technical_role": True,
        "questions": [],
        "coding_challenges": [],
        "session_plan": {},
        "summary": "",
        "time_budget": 60,
        "interview_answers": [],
        "scored_answers": [],
        "final_interview_score": 0,
        "category_averages": {},
        "score_summary": "",
        "coding_submissions": [],
        "coding_scores": [],
        "coding_total_score": 0,
        "messages": [],
        "errors": [],
    }
    return agent.invoke(initial)


def run_agent_streaming(
    resume_text: str,
    job_description: str = "",
    num_questions: int = 8,
):
    """
    Generator that yields (event_name, data) tuples as each node completes,
    then yields ("done", final_state) at the end.
    Used by the SSE endpoint for real-time progress.
    """
    # Map LangGraph node names → human-readable step labels
    STEP_LABELS = {
        "parse_resume":          "Parsing resume",
        "fetch_github":          "Fetching GitHub profiles",
        "score_fit":             "Scoring fit & level",
        "build_topics":          "Building interview topics",
        "gen_questions":         "Generating questions",
        "gen_coding_challenges": "Generating coding challenges",
        "finalize":              "Writing candidate brief",
    }

    initial: AgentState = {
        "resume_text": resume_text,
        "job_description": job_description,
        "num_questions": num_questions,
        "parsed_resume": {},
        "github_profiles": [],
        "fit_score": 0,
        "interview_topics": [],
        "difficulty": "mid",
        "is_technical_role": True,
        "questions": [],
        "coding_challenges": [],
        "summary": "",
        "interview_answers": [],
        "scored_answers": [],
        "final_interview_score": 0,
        "category_averages": {},
        "score_summary": "",
        "coding_submissions": [],
        "coding_scores": [],
        "coding_total_score": 0,
        "messages": [],
        "errors": [],
    }

    final_state = initial
    for chunk in agent.stream(initial, stream_mode="updates"):
        # chunk = {node_name: partial_state_update}
        for node_name, update in chunk.items():
            label = STEP_LABELS.get(node_name, node_name)
            yield ("step", {"node": node_name, "label": label})
            # Merge update into running state
            for k, v in update.items():
                if k == "messages":
                    final_state["messages"] = list(final_state.get("messages", [])) + list(v)
                else:
                    final_state[k] = v
    yield ("done", final_state)


# ── Adaptive next-question generator (called live during interview) ───────────

NEXT_QUESTION_SYSTEM = """You are conducting a live adaptive technical interview.
You see the conversation so far and must decide the next question.

Rules:
- If the last answer was WEAK (vague, wrong, missing key concepts): probe deeper on the same topic
- If the last answer was STRONG: move to the next uncovered JD area or topic
- If a topic has been explored sufficiently: pivot to an uncovered area
- Keep the conversation natural — brief acknowledgement then the question
- Never repeat a question already asked
- Vary difficulty: start easy, escalate when candidate is strong, back off when struggling
- Always tie questions to the candidate's actual experience / projects when possible

Return a JSON object:
{
  "id": <next_id>,
  "category": "<topic>",
  "question": "<the question — natural, conversational>",
  "rationale": "<why this next — internal note>",
  "difficulty": "<easy|medium|hard>",
  "covers_area": "<JD area being probed>",
  "transition": "<1 sentence natural transition from previous answer, or empty string for first>",
  "interview_complete": false,
  "completion_reason": ""
}

Set interview_complete=true when:
- All JD areas have been probed at least once AND
- At least 5 questions have been asked AND
- Either the candidate has been consistently strong (wrap up) or weak (enough data)
Include completion_reason explaining why.

Return ONLY valid JSON."""


def generate_next_question(
    session_plan: dict,
    conversation: list[dict],  # [{question, answer, q_id, covers_area}]
    parsed_resume: dict,
    difficulty: str,
    job_description: str,
) -> dict:
    """
    Generate the next adaptive question based on conversation history and coverage.
    Called live after each candidate answer via /api/next-question.
    """
    # Build coverage summary
    covered_areas = {c.get("covers_area", "") for c in conversation if c.get("covers_area")}
    jd_areas = session_plan.get("jd_areas", [])
    uncovered = [a["area"] for a in jd_areas if a["area"] not in covered_areas]
    next_id = len(conversation) + 2  # +2 because opening Q is id=1

    # Format conversation history
    history_str = ""
    for i, turn in enumerate(conversation[-6:], 1):  # last 6 turns for context
        history_str += f"Q{i}: {turn.get('question', '')}\n"
        history_str += f"A{i}: {turn.get('answer', '(no answer)')}\n\n"

    prompt = f"""CANDIDATE: {parsed_resume.get('name')}, {difficulty} level
JD: {job_description[:500]}
Candidate context: {session_plan.get('candidate_ctx', '')}

COVERAGE STATUS:
  Covered areas: {list(covered_areas) or 'None yet'}
  Uncovered JD areas: {uncovered}
  Questions asked so far: {len(conversation)}

CONVERSATION SO FAR:
{history_str}

Generate the next interview question (id={next_id})."""

    resp = json_llm.invoke([
        SystemMessage(content=NEXT_QUESTION_SYSTEM),
        HumanMessage(content=prompt),
    ])
    q = _safe_json(resp.content, {})
    q.setdefault("id", next_id)
    q.setdefault("category", "General")
    q.setdefault("difficulty", "medium")
    q.setdefault("covers_area", "")
    q.setdefault("transition", "")
    q.setdefault("interview_complete", False)
    q.setdefault("completion_reason", "")
    return q


# ── Time-boxed coding session generator ──────────────────────────────────────

TIMED_CODING_SYSTEM = """You are a senior engineer designing a timed coding interview session.
Generate coding challenges that can realistically be completed within the time budget.

Time budget guidelines:
  60 minutes:  2-3 challenges max. Basic (15min) + Medium (30min) + optional Easy Extra (15min)
  120 minutes: 4-5 challenges. Basic (15min) + Medium (30min) + Hard (45min) + 1-2 extras (30min)

Each challenge MUST:
1. Have a "solve" function, ONLY Python stdlib (browser runner) unless explicitly ML/torch needed
2. The function MUST be named exactly "solve" — no exceptions
3. Have 3-5 test cases with JSON-serialisable expected values
3. Include realistic time estimate in minutes ("estimated_minutes": <int>)
4. Be scoped to fit within its time slot — no challenge should exceed 45 minutes

Return ONLY a valid JSON array of challenges following this schema:
[
  {
    "id": 1,
    "level": "basic|medium|hard|extra",
    "runner": "browser|server",
    "estimated_minutes": <int>,
    "title": "<title>",
    "description": "<clear problem with examples>",
    "tags": ["<tag>"],
    "setup_code": "",
    "starter_code": "def solve(...):\n    pass",
    "solution_hint": "<hint>",
    "test_cases": [{"input": [...], "expected": <value>}],
    "rationale": "<why this challenge>"
  }
]"""


def generate_coding_session(
    session_plan: dict,
    parsed_resume: dict,
    github_profiles: list,
    difficulty: str,
    time_budget_minutes: int,
    job_description: str,
) -> list[dict]:
    """Generate time-boxed coding challenges for 60 or 120 minute session."""
    github_ctx = ""
    for p in github_profiles:
        github_ctx += f"GitHub @{p['username']}: {p.get('bio','')}\n"
        for pa in p.get("project_analyses", [])[:3]:
            github_ctx += f"  [{pa['name']}]: {pa['analysis'][:300]}\n"

    frameworks = str(parsed_resume.get("frameworks", []) + parsed_resume.get("skills", [])[:5]).lower()
    has_torch = any(k in frameworks for k in ["torch", "pytorch", "tensorflow", "ml", "deep learning"])
    has_data  = any(k in frameworks for k in ["numpy", "pandas", "scipy", "data"])

    prompt = f"""CANDIDATE: {parsed_resume.get('name')}, {difficulty} level
Languages: {parsed_resume.get('languages', [])}
Frameworks: {parsed_resume.get('frameworks', [])}
Has ML/PyTorch: {has_torch} | Has numpy/pandas: {has_data}
{github_ctx}

TIME BUDGET: {time_budget_minutes} minutes
JD context: {job_description[:400]}

Generate coding challenges that fit exactly within {time_budget_minutes} minutes total.
The sum of all estimated_minutes must not exceed {time_budget_minutes}.
Calibrate complexity to {difficulty} level.
{"Use torch/sklearn for ML challenges (runner: server)" if has_torch else "Use stdlib + numpy if needed (runner: browser)"}"""

    resp = json_llm.invoke([
        SystemMessage(content=TIMED_CODING_SYSTEM),
        HumanMessage(content=prompt),
    ])
    challenges = _safe_json(resp.content, [])

    # Normalise
    for i, c in enumerate(challenges):
        c.setdefault("id", i + 1)
        c.setdefault("level", ["basic", "medium", "hard"][min(i, 2)])
        c.setdefault("runner", "server" if c.get("level") == "hard" and has_torch else "browser")
        c.setdefault("estimated_minutes", 20)
        c.setdefault("setup_code", "")
        # Normalise starter_code:
        # 1. Force function name to always be "solve"
        # 2. Inject "# Write your solution here" before pass
        import re as _re
        raw_sc = c.get("starter_code", "").strip()
        if not raw_sc or raw_sc == "pass":
            raw_sc = "def solve():\n    pass"

        # Rename the first def <anything>(...) -> def solve(...)
        raw_sc = _re.sub(
            r"(?m)^(def )\w+(\s*\()",
            r"\1solve\2",
            raw_sc,
            count=1,
        )

        # Inject comment before any bare `pass` line
        sc_lines = raw_sc.split("\n")
        new_sc_lines = []
        for sc_line in sc_lines:
            if sc_line.rstrip().lstrip() == "pass":
                indent = len(sc_line) - len(sc_line.lstrip())
                new_sc_lines.append(" " * indent + "# Write your solution here")
            new_sc_lines.append(sc_line)
        c["starter_code"] = "\n".join(new_sc_lines).rstrip() + "\n"
        c.setdefault("test_cases", [])
        c.setdefault("solution_hint", "")
        c.setdefault("tags", [])
        c.setdefault("rationale", "")

    return challenges


def score_answers(
    base_state: AgentState,
    answers: list[dict],
) -> AgentState:
    """
    Take the state from run_agent plus submitted answers, run score_interview.
    answers: [{"question_id": int, "question": str, "answer": str}, ...]
    """
    updated = dict(base_state)
    updated["interview_answers"] = answers
    return score_interview(updated)