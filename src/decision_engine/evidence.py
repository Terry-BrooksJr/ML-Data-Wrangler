"""Reviewer evidence for a manual-triage case: extracted, never stored as a file.

Two kinds, from a file or pasted text:
- jira: Jira's XML (RSS) export of one or more issues, the format of
  jira-rss.xsd (rss > channel > item). Parsed with defusedxml, so entity
  expansion and external entities in a hostile file are refused.
- aha: an Aha! idea or feature, as its page prints to PDF or copies as text
  (the History tab: Aha has no clean export). The page's day headings come
  apart from its activity entries in the text, so each entry keeps its time
  and order but not its date.

extract() returns only the fields a reviewer or the arbiter needs; anything
that can't be parsed is rejected with the reason, rather than stored half
read.
"""

import hashlib
import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from typing import List, Optional, Union
from xml.etree.ElementTree import tostring

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

JIRA, AHA = "jira", "aha"
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_JIRA_ISSUES = 50
# Jira's RSS fields the engine can use; issue keys look like LRN-1234.
_ISSUE_KEY = re.compile(r"^[A-Z][A-Z0-9_]+-\d+$")
# Aha references: ideas (PROJ-I-543) and features (PROJ-123).
_AHA_REF = re.compile(r"\b([A-Z][A-Z0-9]+-(?:I-)?\d+)\b")
_AHA_URL = re.compile(r"https://[\w.-]+\.aha\.io/\S+")
_AHA_ACTIVITY = re.compile(
    r"^(?P<time>\d{1,2}:\d{2}\s?[ap]m)\s*(?P<actor>.+?) "
    r"(?P<action>created internal comment on|deleted comment on|commented on|"
    r"added vote for|updated|described|created|added|removed|deleted)"
    r" (?:idea|feature)\b",
    re.MULTILINE,
)
_AHA_WORKFLOW = re.compile(r"Workflow status\s*(?P<old>.+?)\s*→\s*(?P<new>.+?)(?:\s+Assigned to\b|$)", re.MULTILINE)
_AHA_DAY = re.compile(r"^\s*(\d{1,2} (?:January|February|March|April|May|June|July|August|"
                      r"September|October|November|December))\s*$", re.MULTILINE)


class EvidenceError(ValueError):
    """The content can't be parsed as the evidence kind; the message says why."""


@dataclass
class Evidence:
    kind: str
    source: str
    sha256: str
    data: dict


def _text_of(content: Union[bytes, str]) -> str:
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise EvidenceError(f"The content isn't UTF-8 text ({e.reason}).") from e
    return content


def _sha256(content: Union[bytes, str]) -> str:
    raw = content if isinstance(content, bytes) else content.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# --- HTML to text -----------------------------------------------------------------


class _TextOnly(HTMLParser):
    BREAKS = {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "pre", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BREAKS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(markup: Optional[str]) -> str:
    """Plain text from Jira's HTML fields: tags dropped, entities decoded,
    blank lines collapsed."""
    if not markup:
        return ""
    parser = _TextOnly()
    parser.feed(markup)
    parser.close()
    text = html.unescape("".join(parser.parts))
    lines = [line.strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# --- Jira -------------------------------------------------------------------------


def _inner_markup(element) -> str:
    """An element's text and any child markup, as one HTML string: Jira
    sends rich text escaped, or now and then as real child elements."""
    # tostring includes each child's tail, the text that follows it.
    return (element.text or "") + "".join(tostring(child, encoding="unicode") for child in element)


def _child_text(item, name: str) -> Optional[str]:
    found = item.find(name)
    if found is None:
        return None
    return (found.text or "").strip() or None


def _jira_issue(item, index: int) -> dict:
    key = _child_text(item, "key")
    if not key or not _ISSUE_KEY.match(key):
        raise EvidenceError(f"Issue {index}: no issue key (expected something like LRN-1234).")
    summary = _child_text(item, "summary") or _child_text(item, "title")
    if not summary:
        raise EvidenceError(f"Issue {key}: no summary.")
    status = item.find("status")
    if status is None or not (status.text or "").strip():
        raise EvidenceError(f"Issue {key}: no status.")
    category = item.find("statusCategory")
    project = item.find("project")
    description = item.find("description")

    comments = []
    for comment in item.findall("comments/comment"):
        comments.append(
            {
                "author": comment.get("author"),
                "created": comment.get("created"),
                "text": html_to_text(_inner_markup(comment)),
            }
        )
    links = []
    for link_type in item.findall("issuelinks/issuelinktype"):
        name = _child_text(link_type, "name")
        for direction in ("outwardlinks", "inwardlinks"):
            group = link_type.find(direction)
            if group is None:
                continue
            links.append(
                {
                    "type": name,
                    "relation": group.get("description"),
                    "issues": [
                        (k.text or "").strip()
                        for k in group.findall("issuelink/issuekey")
                        if (k.text or "").strip()
                    ],
                }
            )
    custom_fields = []
    for field in item.findall("customfields/customfield"):
        values = [
            html_to_text(_inner_markup(v))
            for v in field.findall("customfieldvalues/customfieldvalue")
        ]
        values = [v for v in values if v]
        if values:
            custom_fields.append({"name": _child_text(field, "customfieldname"), "values": values})

    return {
        "key": key,
        "summary": summary,
        "link": _child_text(item, "link"),
        "project": (project.get("key") if project is not None else None),
        "type": _child_text(item, "type"),
        "priority": _child_text(item, "priority"),
        "status": status.text.strip(),
        "status_category": category.get("key") if category is not None else None,
        "resolution": _child_text(item, "resolution"),
        "assignee": _child_text(item, "assignee"),
        "reporter": _child_text(item, "reporter"),
        "labels": [(l.text or "").strip() for l in item.findall("labels/label") if (l.text or "").strip()],
        "parent": _child_text(item, "parent"),
        "created": _child_text(item, "created"),
        "updated": _child_text(item, "updated"),
        "due": _child_text(item, "due"),
        "environment": html_to_text(_child_text(item, "environment")),
        "description": html_to_text(_inner_markup(description)) if description is not None else "",
        "comments": comments,
        "links": links,
        "custom_fields": custom_fields,
    }


def parse_jira(content: Union[bytes, str]) -> dict:
    """The issues in a Jira XML (RSS) export: {"issues": [...]}.

    Raises:
        EvidenceError: not well-formed or unsafe XML, not an rss > channel >
            item export, or an issue without a key, summary or status.
    """
    try:
        root = ElementTree.fromstring(content)
    except DefusedXmlException as e:
        raise EvidenceError(f"The XML uses a feature that isn't allowed ({type(e).__name__}).") from e
    except ElementTree.ParseError as e:
        raise EvidenceError(f"The XML isn't well-formed: {e}.") from e
    if root.tag != "rss":
        raise EvidenceError(f"Expected a Jira RSS export (an <rss> root), got <{root.tag}>.")
    channel = root.find("channel")
    if channel is None:
        raise EvidenceError("The export has no <channel>.")
    items = channel.findall("item")
    if not items:
        raise EvidenceError("The export holds no issues (<item>).")
    if len(items) > MAX_JIRA_ISSUES:
        raise EvidenceError(f"The export holds {len(items)} issues; add at most {MAX_JIRA_ISSUES} at a time.")
    return {"issues": [_jira_issue(item, n) for n, item in enumerate(items, 1)]}


# --- Aha --------------------------------------------------------------------------


def read_pdf_text(content: bytes) -> str:
    """The text of a PDF, page by page.

    Raises:
        EvidenceError: not a readable PDF, encrypted, or without text (a scan).
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise EvidenceError("The PDF is password-protected.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except PdfReadError as e:
        raise EvidenceError(f"The PDF can't be read: {e}.") from e
    if not text.strip():
        raise EvidenceError("The PDF has no text to read (is it a scan?).")
    return text


def _aha_title(text: str, reference: str) -> Optional[str]:
    # The browser's print footer: "<ref> <title> | Aha!"
    match = re.search(rf"{re.escape(reference)} (.+?) \| Aha!", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"^Name\s*(.+?)(?:\s+Score\d|$)", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Otherwise the line after the one carrying the reference.
    lines = text.splitlines()
    for i, line in enumerate(lines[:-1]):
        if reference in line and lines[i + 1].strip():
            return lines[i + 1].strip()
    return None


def _last_value_after(text: str, label: str) -> Optional[str]:
    """The value on the line after the last bare `label` line (the
    sidebar's), or on the same line."""
    matches = re.findall(rf"^\s*{label}\s*\n\s*(\S.*)$", text, re.MULTILINE)
    return matches[-1].strip() if matches else None


def parse_aha(text: str) -> dict:
    """The fields of an Aha idea or feature page, printed to PDF or copied.

    Raises:
        EvidenceError: no Aha reference (e.g. PROJ-I-123) or no title.
    """
    # Aha's icon font prints as private-use characters; they carry nothing.
    text = re.sub(r"[\ue000-\uf8ff]", "", text.replace("\r\n", "\n"))
    ref = _AHA_REF.search(text)
    if not ref:
        raise EvidenceError("No Aha reference found (expected something like LAPIPLAT-I-543).")
    reference = ref.group(1)
    title = _aha_title(text, reference)
    if not title:
        raise EvidenceError(f"Found {reference}, but not its title.")

    created = re.search(
        r"Created by (?P<by>.+?) in (?P<where>.+?) on (?P<on>[A-Z][a-z]{2} \d{1,2}, \d{4})", text
    )
    status = re.findall(r"^Status\s+(\S.*?)\s*$", text, re.MULTILINE)
    url = _AHA_URL.search(text)
    votes = re.search(r"\bVotes\s*(\d+)", text)
    score = re.search(r"\bScore\s*(\d+)", text)

    headers = list(_AHA_ACTIVITY.finditer(text))
    description_end = headers[0].start() if headers else len(text)
    description_start = text.find(title)
    description_start = description_start + len(title) if description_start != -1 else 0
    description = text[description_start:description_end].strip()

    activity, comments, workflow, seen_comments = [], [], [], set()
    for n, header in enumerate(headers):
        body_end = headers[n + 1].start() if n + 1 < len(headers) else len(text)
        body = text[header.end():body_end]
        # The header runs on to the end of its line (the idea's title).
        body = body.split("\n", 1)[1] if "\n" in body else ""
        action = header.group("action")
        actor = header.group("actor").strip()
        entry = {"time": header.group("time"), "actor": actor, "action": action}
        if "comment" in action:
            entry["deleted"] = action.startswith("deleted")
            comment_text = re.sub(r"^\s*Comment by .+\n", "", body, count=1).strip()
            # Only the comment: the sidebar and tab strip follow the last entry.
            comment_text = comment_text.split("\nName", 1)[0].strip()
            if not entry["deleted"] and (actor, comment_text) not in seen_comments:
                seen_comments.add((actor, comment_text))
                comments.append(
                    {
                        "author": actor,
                        "internal": "internal" in action,
                        "time": header.group("time"),
                        "text": comment_text,
                    }
                )
        for change in _AHA_WORKFLOW.finditer(body):
            change = {"by": actor, "time": header.group("time"), "from": change["old"].strip(), "to": change["new"].strip()}
            workflow.append(change)
            entry["workflow_status"] = f"{change['from']} → {change['to']}"
        activity.append(entry)

    return {
        "reference": reference,
        "title": title,
        "url": url.group(0) if url else None,
        "created_by": created["by"].strip() if created else None,
        "created_in": created["where"].strip() if created else None,
        "created_on": created["on"] if created else None,
        "status": status[-1] if status else (workflow[0]["to"] if workflow else None),
        "assigned_to": _last_value_after(text, "Assigned to"),
        "votes": int(votes.group(1)) if votes else None,
        "score": int(score.group(1)) if score else None,
        "description": description,
        "workflow_changes": workflow,
        "comments": comments,
        "activity": activity,
        "activity_days": _AHA_DAY.findall(text),
        "activity_dates": "Each entry has its time only: the page's day headings "
        "don't stay with their entries when it's printed or copied.",
    }


# --- entry point ------------------------------------------------------------------


def extract(kind: str, content: Union[bytes, str], source: str = "pasted") -> Evidence:
    """Parses a file's bytes or pasted text as evidence of `kind`.

    Raises:
        EvidenceError: empty, too large, or not parseable as `kind`.
    """
    if kind not in (JIRA, AHA):
        raise ValueError(f"kind must be {JIRA!r} or {AHA!r}")
    size = len(content if isinstance(content, bytes) else content.encode("utf-8"))
    if not size or not (content.strip() if isinstance(content, (bytes, str)) else content):
        raise EvidenceError("There's nothing to add: the content is empty.")
    if size > MAX_INPUT_BYTES:
        raise EvidenceError(f"The content is {size // (1024 * 1024)} MB; the limit is {MAX_INPUT_BYTES // (1024 * 1024)} MB.")
    if kind == JIRA:
        data = parse_jira(content if isinstance(content, bytes) else content.encode("utf-8"))
    else:
        if isinstance(content, bytes) and content.lstrip().startswith(b"%PDF"):
            text = read_pdf_text(content)
        else:
            text = _text_of(content)
        data = parse_aha(text)
    return Evidence(kind=kind, source=source, sha256=_sha256(content), data=data)


def describe(entry: dict) -> str:
    """One line for a stored evidence entry, e.g. "Jira LRN-12 (Done, Fixed)"."""
    data = entry.get("data") or {}
    if entry.get("kind") == JIRA:
        issues = [
            f"{i['key']} ({', '.join(x for x in (i.get('status'), i.get('resolution')) if x)})"
            for i in data.get("issues") or []
        ]
        return "Jira " + "; ".join(issues)
    return f"Aha {data.get('reference')}: {data.get('title')} ({data.get('status') or 'status unknown'})"
