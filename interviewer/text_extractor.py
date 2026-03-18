"""
interviewer/text_extractor.py
Extract text AND hyperlinks from PDF or DOCX resume files.

Many resumes embed GitHub/LinkedIn as icon-only hyperlinks — the visible
text is just an icon character or nothing, but the underlying URL is in
the PDF link annotation or DOCX hyperlink. This extractor captures both.
"""

import re
import pymupdf
from docx import Document


def extract_resume_text(path: str) -> str:
    """
    Extract readable text + all hyperlink URLs from a PDF or DOCX file.
    URLs are appended at the end so GitHub/LinkedIn links are always present
    even when they appear as icon-only links in the original document.
    """
    lower = path.lower()

    if lower.endswith(".pdf"):
        return _extract_pdf(path)

    if lower.endswith(".docx"):
        return _extract_docx(path)

    return ""


def _extract_pdf(path: str) -> str:
    doc = pymupdf.open(path)
    text_parts = []
    urls = []

    for page in doc:
        # Visible text
        text_parts.append(page.get_text())

        # Hyperlink annotations — catches icon-only links
        for link in page.get_links():
            uri = link.get("uri", "")
            if uri:
                urls.append(uri)

    doc.close()

    text = "\n".join(text_parts).strip()

    # Append all unique URLs so the GitHub regex can find them
    if urls:
        unique_urls = list(dict.fromkeys(urls))  # dedupe, preserve order
        text += "\n\n[LINKS FROM DOCUMENT]\n" + "\n".join(unique_urls)
        print(f"[text_extractor] Found {len(unique_urls)} hyperlinks in PDF: {unique_urls}")
    else:
        print("[text_extractor] No hyperlinks found in PDF — GitHub must be in visible text")

    return text


def _extract_docx(path: str) -> str:
    d = Document(path)
    parts = []

    # Visible paragraph text
    for p in d.paragraphs:
        if p.text.strip():
            parts.append(p.text)

    # Hyperlinks embedded in the document relationships
    urls = []
    for rel in d.part.rels.values():
        if "hyperlink" in rel.reltype.lower():
            target = rel.target_ref
            if target and target.startswith("http"):
                urls.append(target)

    text = "\n".join(parts)

    if urls:
        unique_urls = list(dict.fromkeys(urls))
        text += "\n\n[LINKS FROM DOCUMENT]\n" + "\n".join(unique_urls)
        print(f"[text_extractor] Found {len(unique_urls)} hyperlinks in DOCX: {unique_urls}")
    else:
        print("[text_extractor] No hyperlinks found in DOCX")

    return text.strip()