"""
LangChain: Load & Analyze Code from a Public GitHub Repository
Requirements: pip install langchain-core langchain-openai python-dotenv requests
"""

import os
import base64
import time
import requests
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ── Directories to always skip ───────────────────────────────────────────────
EXCLUDED_DIRS = {
    ".venv", "venv", "env",
    "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".git", ".github",
    ".tox", ".nox",
    "dist", "build",
    ".idea", ".vscode",
    "site-packages",
}


def should_skip(path):
    parts = path.split("/")
    for part in parts:
        if part in EXCLUDED_DIRS or part.endswith(".egg-info"):
            return True
    return False


def load_github_repo_api(repo_url, branch="main", file_extensions=None, max_files=100):
    parts = repo_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]

    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    print(f"Fetching file tree from {owner}/{repo} (branch: {branch})...")
    resp = requests.get(tree_url, timeout=15)
    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    files = [
        item for item in tree
        if item["type"] == "blob"
        and not should_skip(item["path"])
        and (file_extensions is None or any(item["path"].endswith(ext) for ext in file_extensions))
    ]
    files = files[:max_files]

    skipped = sum(1 for item in tree if item["type"] == "blob" and should_skip(item["path"]))
    print(f"Found {len(files)} source files (skipped {skipped} in .venv/node_modules/etc.)")

    documents = []
    for i, file_info in enumerate(files):
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_info['path']}?ref={branch}"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            documents.append(Document(
                page_content=content,
                metadata={"source": file_info["path"], "url": data.get("html_url", "")},
            ))
            if (i + 1) % 5 == 0:
                print(f"  Fetched {i + 1}/{len(files)} files...")
            time.sleep(0.5)
        except requests.exceptions.Timeout:
            print(f"  Timeout: {file_info['path']}")
        except Exception as e:
            print(f"  Skipping {file_info['path']}: {e}")

    print(f"Loaded {len(documents)} files.\n")
    return documents


def format_codebase(documents):
    parts = []
    for doc in documents:
        parts.append(f"{'='*60}\nFILE: {doc.metadata['source']}\n{'='*60}\n{doc.page_content}\n")
    text = "\n".join(parts)
    tokens = len(text) // 4
    print(f"Codebase context: ~{tokens:,} tokens ({len(documents)} files)\n")
    return text


def run_qa(codebase_context):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system = SystemMessage(content=(
        "You are an expert code analyst. Below is the COMPLETE codebase of a project. "
        "Answer questions based ONLY on this code. Be specific with file names, functions, "
        "classes, and logic. NEVER hallucinate code that isn't below.\n\n"
        "If something is not in the codebase, say so.\n\n"
        "COMPLETE CODEBASE:\n" + codebase_context
    ))

    print("=" * 60)
    print("  CODEBASE ANALYZER - Ask anything about the repo!")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    history = []
    while True:
        q = input("\nYou: ").strip()
        if not q or q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages = [system] + history + [HumanMessage(content=q)]
        response = llm.invoke(messages)
        answer = response.content

        print(f"\nAnswer:\n{answer}")

        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=answer))


if __name__ == "__main__":
    REPO_URL = "https://github.com/yajur-khanna/fluent_python"
    BRANCH = "main"
    FILE_EXTENSIONS = [".py"]
    MAX_FILES = 100

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY in your .env file")
        exit(1)

    docs = load_github_repo_api(REPO_URL, BRANCH, FILE_EXTENSIONS, MAX_FILES)
    if not docs:
        print("No files loaded. Check repo URL and branch.")
        exit(1)

    context = format_codebase(docs)
    run_qa(context)