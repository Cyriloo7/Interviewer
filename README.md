# Interviewer — AI Question Generator & Scorer

A full LangGraph agent that reads a candidate's resume and live GitHub repositories to generate sharp, personalised interview questions — and scores the candidate's answers to produce a final interview score.

## Architecture

```
                     ┌──────────────┐
         START ──▶   │ parse_resume │   Structured LLM extracts skills, projects, links
                     └──────┬───────┘
                            │  fan-out (parallel)
               ┌────────────┴───────────┐
               ▼                        ▼
        ┌─────────────┐        ┌──────────────────┐
        │  score_fit  │        │  fetch_github     │
        │  LLM fit %  │        │  REST API + READMEs│
        └──────┬──────┘        └────────┬──────────┘
               └──────────┬────────────┘
                           ▼
                   ┌──────────────┐
                   │ build_topics │   5–8 topics tailored to actual stack
                   └──────┬───────┘
                           ▼
                   ┌──────────────┐
                   │ gen_questions│   N questions with follow-ups + rationale
                   └──────┬───────┘
                           ▼
                   ┌──────────────┐
                   │   finalize   │   Candidate brief for hiring manager
                   └──────┬───────┘
                           ▼
                          END

  After collecting answers (web UI or CLI --interview):

                   ┌──────────────────┐
       answers ──▶ │  score_interview  │──▶ scored_answers
                   │  (standalone)     │    final_interview_score (0–100)
                   └──────────────────┘    score_summary + verdict
```

## Setup

```bash
cd interview_agent
pip install -r requirements.txt
cp .env.example .env        # fill in GOOGLE_API_KEY
```

## Usage

### Web UI (recommended)
```bash
python main.py
# Open http://localhost:8000
```
1. Paste resume text or upload a PDF/DOCX
2. Optionally paste the job description
3. Click **Generate Questions**
4. Fill in the candidate's answer for each question
5. Click **Score Interview** → get per-question scores + final score card

### CLI — generate only
```bash
python cli.py resume.pdf
python cli.py resume.pdf --jd job_description.txt --n 10
```

### CLI — interactive interview + scoring
```bash
python cli.py resume.pdf --interview
python cli.py resume.pdf --interview --jd jd.txt --n 8
```
After each question, type the candidate's answer and press Enter on a blank line to submit.
Commands: `SKIP` to skip a question, `QUIT` to abort.
At the end you get a full per-question breakdown and a final 0–100 score with hire/consider/pass verdict, and an option to save the full report as JSON.

### CLI — raw JSON output
```bash
python cli.py resume.pdf --json
```

### Python API
```python
from interviewer.agent import run_agent, score_answers

# 1. Generate questions
state = run_agent(resume_text="...", job_description="...", num_questions=8)
print(state["questions"])

# 2. Score after collecting answers
answers = [
    {"question_id": 1, "question": "...", "answer": "candidate said..."},
    {"question_id": 2, "question": "...", "answer": "candidate said..."},
]
result = score_answers(state, answers)
print(result["final_interview_score"])   # 0-100
print(result["score_summary"])           # hire / consider / pass + why
print(result["scored_answers"])          # per-question: score, feedback, strength, gap
```

## What each question contains

```json
{
  "id": 1,
  "category": "System Design",
  "question": "Your BankingApp uses Redis caching — walk me through how you handled cache invalidation when account balances changed.",
  "follow_up": "What would you change if this cache needed to be consistent across 10 regional nodes?",
  "rationale": "Tests real distributed-systems understanding grounded in their own work.",
  "difficulty": "hard"
}
```

## What the score output contains

```json
{
  "final_interview_score": 74,
  "score_summary": "Consider for further review. Strong on implementation detail but gaps in distributed systems consistency. Recommend a second round focused on system design depth.",
  "scored_answers": [
    {
      "question_id": 1,
      "score": 8,
      "feedback": "Clearly explained cache-aside pattern with TTL. Missed discussion of write-through vs write-behind trade-offs.",
      "strength": "implementation clarity",
      "gap": "write strategies"
    }
  ]
}
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ | Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `GEMINI_MODEL` | ❌ | Model name (default: `gemini-2.0-flash`) |
| `GITHUB_TOKEN` | ❌ | GitHub PAT — raises rate limit from 60 → 5000 req/hr |
