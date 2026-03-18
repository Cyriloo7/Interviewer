"""
interviewer/github_analyzer.py
Deep analysis of a candidate's public GitHub profile.

For every public (non-fork) repo it fetches:
  - Full metadata (stars, forks, topics, last push, open issues)
  - Language breakdown (bytes per language)
  - File/folder tree (up to 2 levels deep) to understand project structure
  - Full README text
  - Last 10 commit messages (what was actively worked on)

Everything is synthesised into a `project_analysis` string per repo
that the LLM uses to generate genuinely deep, code-specific questions.
"""

from __future__ import annotations

import base64
import os
import re
import time
from typing import Optional

import requests

# ── Auth ──────────────────────────────────────────────────────────────────────

# Module-level token — None by default, set via set_github_token()
_github_token: str = ""


def set_github_token(token: str) -> None:
    """Called when user provides a token for private repo access."""
    global _github_token
    _github_token = token.strip()
    print(f"[GitHub] Token set — private repos now accessible")


def _get_headers() -> dict:
    """
    Build request headers. By default NO token (public repos only).
    Token is only used if explicitly provided by the user via set_github_token().
    """
    headers = {"Accept": "application/vnd.github+json"}
    # Use module-level token first, then env var as fallback (only if user set it)
    token = _github_token or ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # Explicitly do NOT read GITHUB_TOKEN from env automatically —
    # user must opt-in via the UI or set_github_token()
    return headers

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_REPOS          = 20      # fetch all public repos up to this many
DEEP_ANALYSE_REPOS = 6       # enrich this many with tree + commits + README
README_MAX_CHARS   = 4000    # cap README to avoid token bloat
TREE_MAX_ITEMS     = 80      # max file paths to include in tree snapshot
COMMIT_COUNT       = 10      # recent commits to fetch per repo


# ── Username extraction ───────────────────────────────────────────────────────

def extract_github_usernames(text: str) -> list[str]:
    """
    Robustly extract GitHub usernames from any resume text.

    Handles all common resume formats:
      https://github.com/username
      http://github.com/username
      github.com/username
      www.github.com/username
      https://github.com/username/repo      → extracts username only
      github: username
      github - username
      github.com: username
      GitHub Profile: https://github.com/username
    """
    SKIP = {
        "sponsors", "orgs", "about", "features", "marketplace",
        "explore", "topics", "collections", "trending", "login",
        "signup", "settings", "notifications", "pulls", "issues",
        "blog", "contact", "pricing", "security", "enterprise",
        "search", "new", "import", "codespaces", "copilot",
        # Protocol/URL fragments that regex could accidentally capture
        "https", "http", "www", "git", "raw", "blob", "tree", "refs",
    }

    found: set[str] = set()

    # Pattern 1: any URL containing github.com/username (with or without https, www)
    for m in re.finditer(
        r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9][A-Za-z0-9\-]{0,38}?)(?:/|\s|$|[^A-Za-z0-9\-])",
        text, re.IGNORECASE
    ):
        username = m.group(1).strip("-").strip()
        if username and username.lower() not in SKIP and len(username) >= 1:
            found.add(username)

    # Pattern 2: "github:" or "github -" followed by username (no URL)
    for m in re.finditer(
        r"github(?:\.com)?\s*[:\-]\s*([A-Za-z0-9][A-Za-z0-9\-]{0,38})",
        text, re.IGNORECASE
    ):
        username = m.group(1).strip("-").strip()
        if username and username.lower() not in SKIP:
            found.add(username)

    # Pattern 3: bare github.com/username at end of line or before whitespace
    for m in re.finditer(
        r"github\.com/([A-Za-z0-9][A-Za-z0-9\-]{0,38})",
        text, re.IGNORECASE
    ):
        username = m.group(1).strip("-").strip()
        if username and username.lower() not in SKIP:
            found.add(username)

    print(f"[extract_github_usernames] found: {list(found)} from text snippet: {repr(text[:200])}")
    return list(found)


# ── Low-level API helpers ─────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None, timeout: int = 12) -> Optional[requests.Response]:
    """
    GET with error handling.
    - No token by default → public repos, 60 req/hr
    - Token set by user → public + private repos, 5000 req/hr
    - On 401 (bad token): retries without auth so public repos still work
    """
    headers = _get_headers()
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)

        if r.status_code == 401 and _github_token:
            # User-provided token is invalid — retry without it
            print("[GitHub] 401 with user token — retrying as public (token may be invalid)")
            r = requests.get(
                url,
                headers={"Accept": "application/vnd.github+json"},
                params=params,
                timeout=timeout,
            )

        if r.status_code in (404, 403, 451):
            return None
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"[GitHub] GET {url} failed: {e}")
        return None


def _json(url: str, params: dict | None = None) -> Optional[dict | list]:
    r = _get(url, params)
    return r.json() if r else None


# ── Per-repo enrichment ───────────────────────────────────────────────────────

def fetch_languages(full_name: str) -> dict[str, int]:
    """Return {language: bytes} for a repo."""
    data = _json(f"https://api.github.com/repos/{full_name}/languages")
    return data if isinstance(data, dict) else {}


def fetch_readme(full_name: str) -> str:
    """Fetch and decode the README (up to README_MAX_CHARS chars)."""
    data = _json(f"https://api.github.com/repos/{full_name}/readme")
    if not data:
        return ""
    raw = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
    return raw[:README_MAX_CHARS]


def fetch_file_tree(full_name: str, default_branch: str = "main") -> list[str]:
    """
    Return a flat list of file paths (up to TREE_MAX_ITEMS).
    Tries the default branch first, falls back to 'master'.
    """
    for branch in [default_branch, "master", "main", "develop"]:
        url = f"https://api.github.com/repos/{full_name}/git/trees/{branch}?recursive=1"
        data = _json(url)
        if not data:
            continue
        tree = data.get("tree", [])
        paths = [
            item["path"] for item in tree
            if item.get("type") == "blob"
            and not _skip_path(item["path"])
        ]
        return paths[:TREE_MAX_ITEMS]
    return []


def _skip_path(path: str) -> bool:
    """Return True for generated/vendor paths we don't want cluttering the tree."""
    skip_dirs = {
        "node_modules", ".venv", "venv", "env", "__pycache__",
        ".git", ".github", "dist", "build", ".mypy_cache",
        ".pytest_cache", "site-packages", ".tox",
    }
    skip_exts = {
        ".lock", ".sum", ".png", ".jpg", ".jpeg", ".gif", ".svg",
        ".ico", ".woff", ".woff2", ".ttf", ".eot", ".map",
        ".min.js", ".min.css",
    }
    parts = path.split("/")
    if any(p in skip_dirs for p in parts):
        return True
    if any(path.endswith(ext) for ext in skip_exts):
        return True
    return False


def fetch_recent_commits(full_name: str, default_branch: str = "main") -> list[str]:
    """Return a list of recent commit messages (up to COMMIT_COUNT)."""
    for branch in [default_branch, "master", "main", "develop"]:
        data = _json(
            f"https://api.github.com/repos/{full_name}/commits",
            params={"per_page": COMMIT_COUNT, "sha": branch},
        )
        if not data or not isinstance(data, list):
            continue
        messages = []
        for c in data:
            msg = c.get("commit", {}).get("message", "").split("\n")[0].strip()
            if msg:
                messages.append(msg)
        if messages:
            return messages
    return []


def fetch_repo_topics(full_name: str) -> list[str]:
    """Fetch repo topics (requires Accept: application/vnd.github.mercy-preview+json)."""
    r = _get(
        f"https://api.github.com/repos/{full_name}/topics",
        params=None,
    )
    if not r:
        return []
    # Override accept header for topics endpoint
    try:
        r2 = requests.get(
            f"https://api.github.com/repos/{full_name}/topics",
            headers={**_get_headers(), "Accept": "application/vnd.github.mercy-preview+json"},
            timeout=10,
        )
        return r2.json().get("names", []) if r2.ok else []
    except Exception:
        return []


# ── Project analysis synthesiser ─────────────────────────────────────────────

def build_project_analysis(repo: dict) -> str:
    """
    Synthesise all fetched data into a dense text block the LLM
    can reason over to create deep, specific interview questions.
    """
    lines = []

    # Header
    lines.append(f"PROJECT: {repo['full_name']}")
    lines.append(f"URL: {repo['url']}")
    lines.append(f"Description: {repo.get('description') or 'None'}")
    lines.append(f"Stars: {repo['stars']}  Forks: {repo['forks']}  Open issues: {repo.get('open_issues', 0)}")
    lines.append(f"Last pushed: {repo.get('pushed_at', 'unknown')}")

    # Languages
    langs = repo.get("languages", {})
    if langs:
        total = sum(langs.values()) or 1
        lang_str = ", ".join(
            f"{lang} {round(bytes_/total*100)}%"
            for lang, bytes_ in sorted(langs.items(), key=lambda x: -x[1])
        )
        lines.append(f"Languages: {lang_str}")

    # Topics
    topics = repo.get("topics", [])
    if topics:
        lines.append(f"Topics: {', '.join(topics)}")

    # File tree
    tree = repo.get("file_tree", [])
    if tree:
        lines.append(f"\nProject structure ({len(tree)} files shown):")
        # Group by top-level directory for readability
        top_level: dict[str, list[str]] = {}
        for path in tree:
            parts = path.split("/")
            top = parts[0] if len(parts) > 1 else "."
            top_level.setdefault(top, []).append(path)
        for folder, files in list(top_level.items())[:15]:
            lines.append(f"  {folder}/")
            for f in files[:6]:
                lines.append(f"    {f}")
            if len(files) > 6:
                lines.append(f"    … (+{len(files)-6} more)")

    # Recent commits
    commits = repo.get("recent_commits", [])
    if commits:
        lines.append(f"\nRecent commits:")
        for msg in commits:
            lines.append(f"  - {msg}")

    # README snippet
    readme = repo.get("readme", "")
    if readme:
        lines.append(f"\nREADME (first 1500 chars):")
        lines.append(readme[:1500])

    return "\n".join(lines)


# ── Main fetch functions ──────────────────────────────────────────────────────

def fetch_all_repos(username: str) -> list[dict]:
    """
    Fetch ALL public non-fork repos for a user (paginated, up to MAX_REPOS).
    Returns basic metadata for each.
    """
    repos = []
    page = 1
    while len(repos) < MAX_REPOS:
        data = _json(
            f"https://api.github.com/users/{username}/repos",
            params={"per_page": 100, "sort": "pushed", "type": "owner", "page": page},
        )
        if not data or not isinstance(data, list) or not data:
            break
        for repo in data:
            if repo.get("fork") or repo.get("private"):
                continue
            repos.append({
                "name": repo["name"],
                "full_name": repo["full_name"],
                "description": repo.get("description") or "",
                "language": repo.get("language") or "Unknown",
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "open_issues": repo.get("open_issues_count", 0),
                "url": repo["html_url"],
                "topics": repo.get("topics", []),
                "pushed_at": repo.get("pushed_at", ""),
                "updated_at": repo.get("updated_at", ""),
                "default_branch": repo.get("default_branch", "main"),
                "size": repo.get("size", 0),  # KB
            })
        if len(data) < 100:
            break
        page += 1
        time.sleep(0.3)

    return repos[:MAX_REPOS]


def enrich_repos(repos: list[dict]) -> list[dict]:
    """
    Deep-enrich the top DEEP_ANALYSE_REPOS repos with:
    language breakdown, file tree, README, recent commits.
    Lighter repos beyond that still get language breakdown.
    """
    # Sort: non-trivial repos first (size > 0, has description)
    repos_sorted = sorted(
        repos,
        key=lambda r: (
            bool(r.get("description")),
            r.get("size", 0) > 10,
            r.get("stars", 0),
        ),
        reverse=True,
    )

    enriched = []
    for i, repo in enumerate(repos_sorted):
        fn = repo["full_name"]
        branch = repo.get("default_branch", "main")

        # Language breakdown for all repos
        repo["languages"] = fetch_languages(fn)
        time.sleep(0.2)

        if i < DEEP_ANALYSE_REPOS:
            # Full deep enrichment
            repo["readme"]         = fetch_readme(fn)
            repo["file_tree"]      = fetch_file_tree(fn, branch)
            repo["recent_commits"] = fetch_recent_commits(fn, branch)
            time.sleep(0.4)

        repo["project_analysis"] = build_project_analysis(repo)
        enriched.append(repo)

    return enriched


def get_github_profile_summary(username: str) -> dict:
    """
    Full public profile + all repos with deep analysis.
    This is the main entry point called by the agent.
    """
    # User profile
    user_data = _json(f"https://api.github.com/users/{username}")
    if not user_data:
        return {"error": f"User '{username}' not found or inaccessible"}

    print(f"[GitHub] Fetching repos for @{username} "
          f"({user_data.get('public_repos', 0)} public repos)…")

    # Fetch + enrich all public repos
    repos = fetch_all_repos(username)
    print(f"[GitHub] Found {len(repos)} public non-fork repos. Enriching top {DEEP_ANALYSE_REPOS}…")

    repos = enrich_repos(repos)

    return {
        "username": username,
        "name": user_data.get("name") or username,
        "bio": user_data.get("bio") or "",
        "public_repos": user_data.get("public_repos", 0),
        "followers": user_data.get("followers", 0),
        "profile_url": user_data.get("html_url", ""),
        "repos": repos,
        # Convenience: just the project_analysis blocks for the LLM
        "project_analyses": [
            {"name": r["name"], "analysis": r["project_analysis"]}
            for r in repos
        ],
    }