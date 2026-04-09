from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import operator
from github import Github  # pip install PyGithub
# Assume text_extractor from repo is imported
from interviewer.text_extractor import extract_resume_text  # Adapt from repo

class AgentState(TypedDict):
    resume_text: str
    jd_text: str
    scores: dict
    github_projects: list
    flags: list
    messages: Annotated[Sequence, add_messages]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
github_client = Github("your_token")  # Secure via env

def entry_node(state: AgentState) -> AgentState:
    state["resume_text"] = extract_resume_text(state["input_resume_path"])
    return state

def skills_matcher(state: AgentState) -> AgentState:
    prompt = f"Score skills match 0-100: Resume: {state['resume_text'][:2000]} JD: {state['jd_text']}"
    score = llm.invoke(prompt).content  # Parse to int
    state["scores"]["skills"] = int(score)
    return state

def github_checker(state: AgentState) -> AgentState:
    projects = llm.invoke(f"Extract GitHub projects from: {state['resume_text']}").content
    state["github_projects"] = projects.split("\n")
    verified = []
    for proj in state["github_projects"]:
        try:
            repo = github_client.search_repositories(query=f"user:{proj.split('/')[0]} {proj.split('/')[1]}")
            if repo.totalCount > 0:
                verified.append(proj)
        except:
            pass
    state["flags"].append(f"Verified: {len(verified)}/{len(state['github_projects'])}")
    state["scores"]["projects"] = (len(verified) / len(state["github_projects"])) * 100 if state["github_projects"] else 0
    return state

# Similar for other nodes...

def aggregate(state: AgentState) -> AgentState:
    weights = {"skills":0.3, "exp":0.25, "projects":0.2, "ats":0.15, "fit":0.1}
    total = sum(state["scores"][k] * v for k,v in weights.items())
    state["final_score"] = total
    state["report"] = f"Score: {total:.1f}% {state['flags']}"
    return state

# Router
def route_scores(state: AgentState):
    if state["final_score"] > 80: return "shortlist"
    elif state["final_score"] > 60: return "review"
    return END

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("entry", entry_node)
workflow.add_node("skills", skills_matcher)
workflow.add_node("github", github_checker)
# Add others...
workflow.add_node("aggregate", aggregate)
workflow.set_entry_point("entry")
workflow.add_edge("entry", "__start__")  # Parallel via fanout
# Conditional edges...
app = workflow.compile(checkpointer=MemorySaver())  # For persistence