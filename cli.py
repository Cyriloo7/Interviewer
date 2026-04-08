#!/usr/bin/env python3
"""
cli.py  —  Terminal interface for the Interview Agent

Modes
─────
  Generate only (default):
    python cli.py resume.pdf
    python cli.py resume.pdf --jd jd.txt --n 10

  Interactive interview + scoring:
    python cli.py resume.pdf --interview
    python cli.py resume.pdf --interview --jd jd.txt

  Raw JSON output:
    python cli.py resume.pdf --json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Force UTF-8 output on Windows (fixes ✓ ✅ ═ █ etc.) ─────────────────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from interviewer.agent import run_agent, score_answers
from interviewer.text_extractor import extract_resume_text


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(char="─", width=64):
    print(char * width)


def header(text: str):
    sep("═")
    print(f"  {text}")
    sep("═")


def verdict_label(score: int) -> str:
    if score >= 80:
        return "✅  RECOMMEND HIRE"
    elif score >= 60:
        return "🟡  FURTHER REVIEW"
    return "❌  DO NOT PROCEED"

def composite_verdict(score: float, interview_score: int) -> str:  # NEW
    if score >= 75 or (interview_score >= 80):
        return "✅  RECOMMEND HIRE"
    elif score >= 60 or interview_score >= 65:
        return "🟡  FURTHER REVIEW" 
    return "❌  DO NOT PROCEED"

def score_bar(score: int, width: int = 30) -> str:
    filled = round((score / 10) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score}/10"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AI Interview Question Generator & Scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("resume", help="Path to PDF or DOCX resume")
    parser.add_argument("--jd", default="", help="Path to job description text file (optional)")
    parser.add_argument("--n", type=int, default=8, help="Number of questions (default: 8)")
    parser.add_argument("--interview", action="store_true",
                        help="Run interactive interview: ask questions, collect answers, score live")
    parser.add_argument("--json", action="store_true", help="Output raw JSON (generation only)")
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        sys.exit("❌  Set GOOGLE_API_KEY in your .env file.")

    resume_path = Path(args.resume)
    if not resume_path.exists():
        sys.exit(f"❌  File not found: {resume_path}")

    jd_text = ""
    if args.jd:
        jd_path = Path(args.jd)
        if jd_path.exists():
            jd_text = jd_path.read_text(encoding="utf-8")
        else:
            print(f"⚠  JD file not found: {args.jd} — continuing without it.")

    # ── Step 1: Extract + run agent ───────────────────────────────────────────
    print("\n🔍  Extracting resume text…")
    resume_text = extract_resume_text(str(resume_path))
    if not resume_text.strip():
        sys.exit("❌  Could not extract text from file.")

    print("🤖  Running agent (this may take ~30s)…\n")
    state = run_agent(resume_text=resume_text, job_description=jd_text, num_questions=args.n)

    # ── Step 2: JSON mode ─────────────────────────────────────────────────────
    if args.json:
        print(json.dumps({
            "candidate": state.get("parsed_resume"),
            "fit_score": state.get("fit_score"),
            "difficulty": state.get("difficulty"),
            "topics": state.get("interview_topics"),
            "questions": state.get("questions"),
            "summary": state.get("summary"),
            "errors": state.get("errors"),
        }, indent=2))
        return

    # ── Step 3: Candidate overview ────────────────────────────────────────────
    parsed = state.get("parsed_resume", {})
    header(f"CANDIDATE: {parsed.get('name', 'Unknown')}")
    print(f"  Title      : {parsed.get('title', '')}")
    print(f"  Experience : {parsed.get('years_experience', 0)} years")
    print(f"  Fit Score  : {state.get('fit_score', 0)}%")
    print(f"  Level      : {state.get('difficulty', 'mid').upper()}")

    # MODIFIED GITHUB SCORING (CHANGE #1)
    github_profiles = state.get("github_profiles", [])
    github_score = 50  # Neutral baseline
    if github_profiles:
        profile = github_profiles[0]
        repos = profile.get('public_repos', 0)
        stars = profile.get('stars', 0)
        github_score = min(50 + (repos * 3) + min(stars * 0.2, 20), 90)
        print(f"  GitHub     : @{profile['username']} ({repos} repos)")
        print(f"  GitHub Score: {score_bar(int(github_score/10))}")
    else:
        print("  GitHub     : None provided")
        print("  GitHub Score: [░░░░░░░░░░] 5/10 (neutral - no profile provided)")

    print(f"\n  Topics     : {', '.join(state.get('interview_topics', []))}")
    print(f"\n  Brief      :\n  {state.get('summary', '').replace(chr(10), chr(10) + '  ')}")

    questions = state.get("questions", [])
    sep()
    print(f"  {len(questions)} INTERVIEW QUESTIONS  |  Level: {state.get('difficulty','mid').upper()}")
    sep()

    # ── Step 4a: Generate-only mode ───────────────────────────────────────────
    if not args.interview:
        print()
        for q in questions:
            diff = q.get("difficulty", "medium").upper()
            print(f"[{q.get('id','?')}] {q.get('category','').upper()} · {diff}")
            print(f"  Q: {q.get('question', '')}")
            if q.get("follow_up"):
                print(f"     ↳ Follow-up: {q['follow_up']}")
            if q.get("rationale"):
                print(f"     ℹ  {q['rationale']}")
            print()

        errs = state.get("errors", [])
        if errs:
            print("⚠  Warnings:", " | ".join(errs))
        sep()
        print("  Tip: re-run with --interview to conduct the interview and score answers.")
        sep()
        return

    # ── Step 4b: Interactive interview mode ───────────────────────────────────
    print()
    print("  INTERACTIVE MODE  —  blank line submits each answer.")
    print("  Commands: SKIP to skip a question | QUIT to abort.\n")
    sep()

    collected_answers: list[dict] = []

    for i, q in enumerate(questions):
        qid = q.get("id") or (i + 1)
        diff = q.get("difficulty", "medium").upper()

        print(f"\n[{i+1}/{len(questions)}] {q.get('category','').upper()} · {diff}")
        print(f"\n  {q.get('question', '')}")
        if q.get("follow_up"):
            print(f"\n  ↳ Follow-up (use if needed): {q['follow_up']}")
        print()

        lines = []
        print("  Answer (blank line to submit):")
        while True:
            try:
                line = input("  > ")
            except (EOFError, KeyboardInterrupt):
                print("\n\nAborted.")
                sys.exit(0)

            cmd = line.strip().upper()
            if cmd == "QUIT":
                print("\nInterview aborted.")
                sys.exit(0)
            if cmd == "SKIP":
                print("  — Skipped.\n")
                break
            if line == "" and lines:
                break
            if line != "":
                lines.append(line)

        answer_text = "\n".join(lines).strip()
        if answer_text:
            collected_answers.append({
                "question_id": qid,
                "question": q.get("question", ""),
                "answer": answer_text,
            })

    if not collected_answers:
        print("\n⚠  No answers collected. Exiting.")
        sys.exit(0)

    # ── Step 5: Score ─────────────────────────────────────────────────────────
    print(f"\n\n🧠  Scoring {len(collected_answers)} answer(s)…\n")
    result = score_answers(state, collected_answers)

    scored     = result.get("scored_answers", [])
    final      = result.get("final_interview_score", 0)
    rec        = result.get("score_summary", "")
    score_map  = {s["question_id"]: s for s in scored}

    # Per-question breakdown
    sep("═")
    print("  PER-QUESTION BREAKDOWN")
    sep("═")

    CAT_LABELS = {
        "technical_accuracy":    "Technical Accuracy",
        "depth_of_knowledge":    "Depth of Knowledge",
        "problem_solving":       "Problem Solving",
        "system_design":         "System Design",
        "clarity":               "Clarity",
        "use_of_examples":       "Use of Examples",
        "role_relevance":        "Role Relevance",
        "culture_collaboration": "Culture & Collab",
    }

    for i, q in enumerate(questions):
        qid = q.get("id") or (i + 1)
        s = score_map.get(qid)
        if not s:
            continue
        diff = q.get("difficulty", "medium").upper()
        ws = s.get("weighted_score", 0)
        q_text = q.get("question", "")
        print(f"\n[{i+1}] {q.get('category','').upper()} · {diff}")
        print(f"  Q: {q_text[:100]}{'…' if len(q_text) > 100 else ''}")
        print(f"  Weighted: {score_bar(round(ws))}")
        print(f"  Feedback : {s.get('overall_feedback', '')}")
        if s.get("strength"):
            print(f"  Strength : ✓ {s['strength']}")
        if s.get("gap"):
            print(f"  Gap      : ✗ {s['gap']}")
        cats  = s.get("categories", {})
        notes = s.get("category_notes", {})
        if cats:
            print("  ─ Categories ─")
            for key, label in CAT_LABELS.items():
                v = cats.get(key, 0)
                n = notes.get(key, "")
                bar = "█" * v + "░" * (10 - v)
                note_str = f"  {n}" if n else ""
                print(f"    {label:<26} [{bar}] {v}/10{note_str}")

    # Final score card
    sep("═")
    print("  FINAL INTERVIEW SCORE")
    sep("═")

    # Dynamic composite scoring
    fit_score = state.get('fit_score', 0)
    resume_weight = 0.25
    interview_weight = 0.50
    github_weight = 0.15
    experience_weight = 0.10

    # Adjust weights if no GitHub
    if not github_profiles:
        github_weight = 0
        interview_weight += 0.10

    experience_score = min((parsed.get('years_experience', 0) / 15.0) * 100, 90)
    
    composite_score = (
        fit_score * resume_weight +
        final * interview_weight + 
        github_score * github_weight +
        experience_score * experience_weight
    )

    bar_width = 40
    filled = round((composite_score / 100) * bar_width)
    pct_bar = "█" * filled + "░" * (bar_width - filled)
    
    print(f"\n  [{pct_bar}]  {composite_score:.0f}/100")
    print(f"  Breakdown: Interview({final:.0f}) | Resume({fit_score}) | GitHub({github_score}) | Exp({experience_score:.0f})")
    print(f"\n  {composite_verdict(composite_score, final)}")

    # [UNCHANGED STATS AND RECOMMENDATION]
    strong_count = sum(1 for s in scored if s.get("weighted_score", 0) >= 7)
    weak_count   = sum(1 for s in scored if s.get("weighted_score", 0) < 5)
    avg_score    = (sum(s.get("weighted_score", 0) for s in scored) / len(scored)) if scored else 0
    top_gaps     = list(dict.fromkeys(s["gap"] for s in scored if s.get("gap")))[:3]

    print(f"\n  Strong answers : {strong_count}/{len(scored)}")
    print(f"  Weak answers   : {weak_count}/{len(scored)}")
    print(f"  Avg per Q      : {avg_score:.1f}/10")
    if top_gaps:
        print(f"  Top gaps       : {', '.join(top_gaps)}")

    print(f"\n  Recommendation :\n  {rec.replace(chr(10), chr(10) + '  ')}")

    errs = state.get("errors", [])
    if errs:
        print(f"\n⚠  Warnings: {' | '.join(errs)}")

    sep("═")

    # Optional JSON report save
    try:
        save = input("\n  Save full report to JSON? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        save = "n"

    if save == "y":
        report = {
            "candidate": parsed.get("name"),
            "title": parsed.get("title"),
            "fit_score": state.get("fit_score"),
            "difficulty": state.get("difficulty"),
            "composite_score": composite_score,
            "interview_score": final,
            "github_score": github_score,
            "verdict": composite_verdict(composite_score, final).replace("✅  ", "").replace("🟡  ", "").replace("❌  ", ""),
            "recommendation": rec,
            "weights": {
                "resume": resume_weight, "interview": interview_weight, 
                "github": github_weight, "experience": experience_weight
            },
            "per_question": [
                {
                    "id": q.get("id") or (i+1),
                    "category": q.get("category"),
                    "question": q.get("question"),
                    "answer": next((a["answer"] for a in collected_answers
                                    if a["question_id"] == (q.get("id") or (i+1))), ""),
                    "weighted_score": score_map.get(q.get("id") or (i+1), {}).get("weighted_score", 0),
                    "overall_feedback": score_map.get(q.get("id") or (i+1), {}).get("overall_feedback", ""),
                    "strength": score_map.get(q.get("id") or (i+1), {}).get("strength", ""),
                    "gap": score_map.get(q.get("id") or (i+1), {}).get("gap", ""),
                    "categories": score_map.get(q.get("id") or (i+1), {}).get("categories", {}),
                }
                for i, q in enumerate(questions)
            ],
            "category_averages": result.get("category_averages", {}),
            "errors": state.get("errors", []),
        }
        out_path = Path(args.resume).stem + "_interview_report.json"
        Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  ✅  Saved to {out_path}")

    sep()


if __name__ == "__main__":
    main()