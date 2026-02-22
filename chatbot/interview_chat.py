import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pathlib import Path

import pymupdf
from docx import Document

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# =========================
# ENV SETUP
# =========================

load_dotenv()
key = os.getenv("gemini")

if not key:
    raise SystemExit("Missing gemini API key. Add gemini=YOUR_KEY to .env")

os.environ["GOOGLE_API_KEY"] = key


# =========================
# REUSED FUNCTION FROM main.py
# =========================

def extract_text(path: str) -> str:
    """
    Extract text from PDF or DOCX.
    Reused from main.py logic.
    """
    lower = path.lower()

    if lower.endswith(".pdf"):
        doc = pymupdf.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    if lower.endswith(".docx"):
        d = Document(path)
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())

    return ""


# =========================
# UPDATED Resume Schema
# =========================

class ResumeSchema(TypedDict):
    name: Annotated[str, "Candidate name"]
    summary: Annotated[str, "Professional summary"]
    experience_years: Annotated[int, "Years of experience"]
    skills: Annotated[list[str], "Skills list"]
    links: Annotated[list[str], "Relevant links (GitHub, LinkedIn, etc.)"]
    projects: Annotated[list[str], "Notable projects or achievements"]


# =========================
# LLM Setup
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.6,
)

structured_llm = llm.with_structured_output(ResumeSchema)


# =========================
# LangGraph State
# =========================

class InterviewState(TypedDict):
    resume: dict
    questions: list[str]
    current_index: int
    answers: list[str]
    strengths: list[str]
    weaknesses: list[str]
    messages: list


# =========================
# Node 1: Generate Questions
# =========================

def generate_questions(state: InterviewState):

    resume = state["resume"]

    prompt = f"""
    Based on this resume:

    Name: {resume.get("name")}
    Experience: {resume.get("experience_years")} years
    Skills: {resume.get("skills")}
    Projects: {resume.get("projects")}
    Summary: {resume.get("summary")}

    Generate 2 deep technical interview questions.
    Make them personalized and implementation-focused.
    Return only numbered questions.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content

    questions = []
    for line in content.split("\n"):
        clean = line.strip("12345. ").strip()
        if clean:
            questions.append(clean)

    return {
        "questions": questions[:5],
        "current_index": 0,
        "answers": [],
        "strengths": [],
        "weaknesses": [],
        "messages": [AIMessage(content=questions[0])]
    }


# =========================
# Node 2: Evaluate Answer + Follow-up
# =========================

def evaluate_answer(state: InterviewState):

    idx = state["current_index"]
    questions = state["questions"]
    answers = state["answers"]

    last_answer = answers[-1]

    evaluation_prompt = f"""
    Interview Question: {questions[idx]}
    Candidate Answer: {last_answer}

    1. Does the answer show strong technical clarity? (yes/no)
    2. Is there any conceptual mistake or weak explanation?
    3. Should we ask a follow-up? (yes/no)

    Also extract:
    - One strength if present
    - One weakness if present

    Return format:
    Strength:
    Weakness:
    FollowUp: yes/no
    """

    response = llm.invoke([HumanMessage(content=evaluation_prompt)])
    content = response.content.lower()

    strength = ""
    weakness = ""
    followup_needed = False

    for line in content.split("\n"):
        if line.startswith("strength"):
            strength = line.replace("strength:", "").strip()
        if line.startswith("weakness"):
            weakness = line.replace("weakness:", "").strip()
        if "followup: yes" in line:
            followup_needed = True

    if strength:
        state["strengths"].append(strength)
    if weakness:
        state["weaknesses"].append(weakness)

    # If follow-up needed → generate it
    if followup_needed:

        followup_prompt = f"""
        The candidate answered:
        "{last_answer}"

        There was a weakness or unclear explanation.
        Ask ONE probing follow-up question to test their understanding deeper.
        """

        followup_response = llm.invoke([HumanMessage(content=followup_prompt)])
        followup_question = followup_response.content.strip()

        return {
            "messages": [AIMessage(content=followup_question)]
        }

    # Else move to next question
    idx += 1

    if idx >= len(questions):
        return {
            "current_index": idx,
            "messages": [AIMessage(content="Interview complete.")]
        }

    return {
        "current_index": idx,
        "messages": [AIMessage(content=questions[idx])]
    }


# =========================
# Build Graph
# =========================

def build_graph():

    builder = StateGraph(InterviewState)

    builder.add_node("generate", generate_questions)
    builder.add_node("evaluate", evaluate_answer)

    builder.add_edge(START, "generate")
    builder.add_edge("generate", END)

    return builder.compile()


graph = build_graph()


# =========================
# Interview Runner
# =========================

def run_interview(file_path: str):

    # Extract text
    text = extract_text(file_path)

    if not text.strip():
        print("No readable text found.")
        return

    # Structure resume
    resume_data = structured_llm.invoke(text)

    print(f"\nInterviewing: {resume_data.get('name')}\n")

    # Start state
    state = {
        "resume": resume_data,
        "questions": [],
        "current_index": 0,
        "answers": [],
        "strengths": [],
        "weaknesses": [],
        "messages": []
    }

    result = graph.invoke(state)

    questions = result["questions"]
    current_question = result["messages"][-1].content
    idx = 0
    followup_mode = False

    while True:

        print("\nQuestion:", current_question)
        answer = input("Your Answer: ")

        state["answers"].append(answer)
        state["current_index"] = idx
        state["questions"] = questions

        result = evaluate_answer(state)

        current_question = result["messages"][-1].content

        # ✅ If interview complete → stop
        if "interview complete" in current_question.lower():
            break

        # ✅ Detect if this is follow-up
        if result.get("current_index") is None:
            # It is a follow-up question
            followup_mode = True
        else:
            # It is a next main question
            idx = result["current_index"]
            followup_mode = False

        # Safety stop (very important)
        if idx >= len(questions):
            break

    # Final analysis
    print("\n========== FINAL ANALYSIS ==========")
    print("\nStrengths:")
    for s in set(state["strengths"]):
        print("-", s)

    print("\nWeaknesses:")
    for w in set(state["weaknesses"]):
        print("-", w)

    print("\n====================================")


# =========================
# Entry Point
# =========================

if __name__ == "__main__":

    path = input("Enter resume file path (PDF/DOCX): ")

    if not Path(path).exists():
        print("Invalid file path.")
    else:
        run_interview(path)