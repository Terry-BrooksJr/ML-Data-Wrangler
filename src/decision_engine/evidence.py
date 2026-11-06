"""Reviewer evidence for a manual-triage case: extracted, never stored as a file.

Two kinds, from a file or pasted text:
- jira: Jira's XML (RSS) export of one or more issues, the format of
  jira-rss.xsd (rss > channel > item). Parsed with defusedxml, so entity
  expansion and external entities in a hostile file are refused.
- aha: an Aha! idea or feature, as its page prints to PDF or copies as text
  (Aha has no clean export). Two pages are read:
  - the History tab: comments, workflow changes and activity. Its day
    headings come apart from its activity entries in the text, so each entry
    keeps its time and order but not its date.
  - the idea's main page: its description, the sidebar's fields (status,
    workspace, what it was merged into, linked records), the Zendesk ticket
    it was opened from, and the statuses of related ideas.

extract() returns only the fields a reviewer or the arbiter needs; anything
that can't be parsed is rejected with the reason, rather than stored half
read.
"""

import hashlib
import html
import re
from collections import Counter
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
# A crafted file can stay under MAX_INPUT_BYTES and still expand hugely, so
# parsing stops at these. An export of MAX_JIRA_ISSUES issues with long
# histories stays well under them.
MAX_XML_ELEMENTS = 200_000
MAX_XML_DEPTH = 64
MAX_PDF_PAGES = 40
MAX_TEXT_CHARS = 2_000_000
# Decompressed size, per stream (pypdf's own default is 75 MB) and for all
# the pages' content together, checked before any text is extracted: text
# comes out of that content, so this bounds what extraction can produce.
MAX_PDF_STREAM_BYTES = 8 * 1024 * 1024
MAX_PDF_CONTENT_BYTES = 16 * 1024 * 1024
# pypdf's limits while one of our PDFs is read (its defaults are 75 MB a
# stream, 300 MB of decoding in all, 5,000 form reuses per extraction).
PYPDF_LIMITS = {
    **dict.fromkeys(
        (
            "maximum_declared_stream_length",
            "array_based_stream_maximum_output_length",
            "jbig2_maximum_output_length",
            "lzw_maximum_output_length",
            "run_length_maximum_output_length",
            "zlib_maximum_output_length",
            "brotli_maximum_output_length",
            "image_maximum_buffer_size",
        ),
        MAX_PDF_STREAM_BYTES,
    ),
    "stream_decoding_work_maximum_length": 4 * MAX_PDF_CONTENT_BYTES,
    "xform_maximum_invocations_per_extraction": 500,
}
# Jira's RSS fields the engine can use; issue keys look like LRN-1234.
_ISSUE_KEY = re.compile(r"^[A-Z][A-Z0-9_]+-\d+$")
# Aha references: ideas (PROJ-I-543) and features (PROJ-123). A Jira key
# looks the same, so the item's own reference is read only from where the
# page names it (_aha_reference), never from its free text.
_REF = r"[A-Z][A-Z0-9]+-(?:I-)?\d+"
_AHA_URL = re.compile(r"https://[\w.-]+\.aha\.io/\S+")
# An item's link, for a page that doesn't print one (the main page): ideas
# (PROJ-I-123) and features (PROJ-123) live at different paths.
AHA_IDEA_URL = "https://questionmarkcorp.aha.io/ideas/ideas/{reference}"
AHA_FEATURE_URL = "https://questionmarkcorp.aha.io/features/{reference}"
_AHA_ACTIVITY = re.compile(
    r"^(?P<time>\d{1,2}:\d{2}\s?[ap]m)\s*(?P<actor>.+?) "
    r"(?P<action>created internal comment on|deleted comment on|commented on|"
    rf"added vote for|updated|described|created|added|removed|deleted)"
    rf" (?:idea|feature)\b(?: (?P<ref>{_REF})\b)?",
    re.MULTILINE,
)
# The page header: "Created by <name> in <where> on Dec 12, 2025 Idea<REF>".
_AHA_HEADER = re.compile(
    rf"Created by .+? on [A-Z][a-z]{{2}} \d{{1,2}}, \d{{4}}\s*(?:Idea|Feature)\s?({_REF})\b"
)
# The browser's print footer: "<REF> <title> | Aha!".
_AHA_FOOTER = re.compile(rf"\b({_REF}) [^\n]+? \| Aha!\s*$", re.MULTILINE)
_AHA_WORKFLOW = re.compile(r"Workflow status\s*(?P<old>.+?)\s*→\s*(?P<new>.+?)(?:\s+Assigned to\b|$)", re.MULTILINE)
# The idea's main page: its vote box ends at "Vote trend", and its own
# description follows.
_AHA_VOTE_TREND = re.compile(r"^Vote trend\s*$", re.MULTILINE)
# The sidebar starts at its Status line, with Workspace under it.
_AHA_SIDEBAR = re.compile(r"^[ \t]*Status [^\n]*\n[ \t]*Workspace\b", re.MULTILINE)
# What can sit between the description and the sidebar: the vote chart's axis
# labels, the merge notice, and the rank's superscript ("19th" leaves "th").
_AHA_ABOVE_SIDEBAR = re.compile(rf"^[ \t]*(?:Time|Votes|Merged into {_REF}|th|st|nd|rd)?[ \t]*$")
_AHA_MERGED_INTO = re.compile(rf"^[ \t]*Merged into ({_REF})[ \t]*$", re.MULTILINE)
# Support's wording when it opens an idea for a customer.
_AHA_REQUESTER = re.compile(
    r"Opened on behalf of (?P<name>.+?) from (?P<org>.+?)\s+Ticket created from"
)
_AHA_TICKET = re.compile(r"Ticket created from\s*#(\d+)\b")
# The related-ideas tables (Similar ideas, People also voted for) come out of
# the PDF a column at a time: each row's status, then each row's name ending
# in its vote count, then each row's reference on a line of its own.
_AHA_REF_LINE = re.compile(rf"^({_REF})$")
_AHA_NAME_END = re.compile(r"^(?:(?P<rest>.*\S)\s+)?(?P<votes>\d+)$")
# A status: up to three words, with the punctuation statuses use ("Won't do").
_AHA_STATUS = re.compile(r"^[A-Z][A-Za-z’'&/.-]*(?: [A-Za-z’'&/.-]+){0,2}$")
# Lines near the tables that a status line could be mistaken for.
# A related-ideas table shows a few dozen rows; a page has a few tables.
MAX_TABLE_ROWS = 50
MAX_TABLE_BLOCKS = 20
_AHA_PAGE_LABELS = {
    "Similar ideas", "People also voted for", "Related", "Portals", "Research",
    "Concepts", "Ideas", "Perspectives", "Linked records", "Merged ideas",
}
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


def _check_xml_shape(content: bytes) -> None:
    """Streams through the XML before it is parsed into a tree, refusing
    one too deep, with too many elements or too many issues. Each element is
    cleared once read, so this pass holds little in memory.

    Raises:
        EvidenceError: over a limit.
        DefusedXmlException, ElementTree.ParseError: as parsing would.
    """
    depth = elements = items = 0
    for event, element in ElementTree.iterparse(BytesIO(content), events=("start", "end")):
        if event == "start":
            depth += 1
            elements += 1
            if depth > MAX_XML_DEPTH:
                raise EvidenceError(f"The XML nests deeper than {MAX_XML_DEPTH} levels.")
            if elements > MAX_XML_ELEMENTS:
                raise EvidenceError(f"The XML has over {MAX_XML_ELEMENTS:,} elements.")
            if depth == 3 and element.tag == "item":
                items += 1
                if items > MAX_JIRA_ISSUES:
                    raise EvidenceError(
                        f"The export holds over {MAX_JIRA_ISSUES} issues; add at most "
                        f"{MAX_JIRA_ISSUES} at a time."
                    )
        else:
            depth -= 1
            element.clear()


_XML_SPACE = frozenset(b" \t\r\n")
# Markup copied untouched by _drop_duplicate_attributes: (opener, closer).
_SKIPPED_MARKUP = ((b"<![CDATA[", b"]]>"), (b"<!--", b"-->"), (b"<?", b"?>"), (b"<!", b">"), (b"</", b">"))


def _start_tag_without_duplicates(content: bytes, start: int):
    """Reads the start tag at `start`, forward only: (end, the tag with the
    first of each repeated attribute kept). Quoted values may hold ">".
    Anything malformed is returned unchanged up to where reading stopped,
    for the parser to report."""
    n = len(content)
    k = start + 1
    while k < n and content[k] not in _XML_SPACE and content[k] not in b"/>":
        k += 1
    parts, seen = [content[start:k]], set()
    while True:
        gap = k
        while k < n and content[k] in _XML_SPACE:
            k += 1
        if k >= n:
            return n, b"".join(parts) + content[gap:n]
        if content[k] == ord(">"):
            parts.append(content[gap : k + 1])
            return k + 1, b"".join(parts)
        if content.startswith(b"/>", k):
            parts.append(content[gap : k + 2])
            return k + 2, b"".join(parts)
        name_start = k
        while k < n and content[k] not in _XML_SPACE and content[k] not in b"=/>":
            k += 1
        name = content[name_start:k]
        while k < n and content[k] in _XML_SPACE:
            k += 1
        if k >= n or content[k] != ord("="):
            return k, b"".join(parts) + content[gap:k]
        k += 1
        while k < n and content[k] in _XML_SPACE:
            k += 1
        if k >= n or content[k] not in b"\"'":
            return k, b"".join(parts) + content[gap:k]
        close = content.find(content[k : k + 1], k + 1)
        if close < 0:
            return n, b"".join(parts) + content[gap:n]
        k = close + 1
        if name not in seen:
            seen.add(name)
            parts.append(content[gap:k])


def _drop_duplicate_attributes(content: bytes) -> bytes:
    """Keeps the first of each repeated attribute in every start tag.

    Jira's export writes an @mention link in a comment with rel twice
    (rel="<account id>" ... rel="noreferrer"), which XML forbids, so a
    comment that mentions anyone made the whole export fail to parse.

    One forward pass, so its time grows with the file's size: CDATA,
    comments, declarations and end tags are copied as they are, and only
    start tags are rewritten.
    """
    out, i, n = [], 0, len(content)
    while i < n:
        j = content.find(b"<", i)
        if j < 0:
            out.append(content[i:])
            break
        out.append(content[i:j])
        for opener, closer in _SKIPPED_MARKUP:
            if content.startswith(opener, j):
                end = content.find(closer, j + len(opener))
                i = n if end < 0 else end + len(closer)
                out.append(content[j:i])
                break
        else:
            i, tag = _start_tag_without_duplicates(content, j)
            out.append(tag)
    return b"".join(out)


# A UTF-16 file's first bytes, with or without a byte-order mark: '<?' or
# the mark, in either byte order.
_UTF16_STARTS = {
    b"\xff\xfe": "utf-16",
    b"\xfe\xff": "utf-16",
    b"<\x00": "utf-16-le",
    b"\x00<": "utf-16-be",
}
_DECLARED_ENCODING = re.compile(r"""\A(<\?xml[^>]{0,200}?encoding\s*=\s*)(["'])[^"']{0,40}\2""")


def _as_utf8(content: bytes) -> bytes:
    """A UTF-16 export re-encoded as UTF-8, with its declaration saying so,
    so _drop_duplicate_attributes can find its markup. Other files are
    returned as they are."""
    encoding = _UTF16_STARTS.get(content[:2])
    if encoding is None:
        return content
    try:
        text = content.decode(encoding).lstrip("\ufeff")
    except UnicodeDecodeError:
        return content  # left for the parser to report
    return _DECLARED_ENCODING.sub(r"\1\2UTF-8\2", text, count=1).encode("utf-8")


def _parse_xml(content: bytes):
    _check_xml_shape(content)
    return ElementTree.fromstring(content)


def parse_jira(content: Union[bytes, str]) -> dict:
    """The issues in a Jira XML (RSS) export: {"issues": [...]}.

    Raises:
        EvidenceError: not well-formed or unsafe XML, not an rss > channel >
            item export, or an issue without a key, summary or status.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    try:
        try:
            root = _parse_xml(content)
        except ElementTree.ParseError as e:
            # Only Jira's repeated attributes are repaired; any other fault,
            # or one left after the repair, is reported.
            if "duplicate attribute" not in str(e):
                raise
            root = _parse_xml(_drop_duplicate_attributes(_as_utf8(content)))
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
    """The text of a PDF, page by page, within the MAX_PDF_* limits.

    Raises:
        EvidenceError: not a readable PDF, encrypted, over a limit, or
            without text (a scan). Any failure inside pypdf is reported this
            way, so a crafted file is refused rather than breaking the app.
    """
    from pypdf import apply_configuration

    # Scoped to this read (a context variable), so nothing else is affected.
    with apply_configuration(**PYPDF_LIMITS):
        return _read_pdf_text(content)


def _read_pdf_text(content: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise EvidenceError("The PDF is password-protected.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise EvidenceError(
                f"The PDF has {len(reader.pages)} pages; an Aha page prints to a few. "
                f"The limit is {MAX_PDF_PAGES}."
            )
        # The pages' content, decompressed (and cached for extraction) before
        # any text is drawn from it.
        content_bytes = 0
        for page in reader.pages:
            contents = page.get_contents()
            content_bytes += len(contents.get_data()) if contents is not None else 0
            if content_bytes > MAX_PDF_CONTENT_BYTES:
                raise EvidenceError(
                    f"The PDF's pages hold over {MAX_PDF_CONTENT_BYTES // (1024 * 1024)} MB of content."
                )
        parts, total = [], 0
        for page in reader.pages:
            text = page.extract_text() or ""
            total += len(text)
            if total > MAX_TEXT_CHARS:
                raise EvidenceError(f"The PDF holds over {MAX_TEXT_CHARS:,} characters of text.")
            parts.append(text)
        text = "\n".join(parts)
    except EvidenceError:
        raise
    except (MemoryError, RecursionError) as e:
        raise EvidenceError("The PDF is too complex to read.") from e
    except Exception as e:  # pypdf raises many kinds for a broken or crafted file
        raise EvidenceError(f"The PDF can't be read ({type(e).__name__}: {e}).") from e
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


def _aha_reference(text: str, headers: list) -> str:
    """The item's reference, from the places the page names it: its header,
    its activity entries ("updated idea <REF>") and the print footer.

    Raises:
        EvidenceError: none of them names it, or they name different items.
    """
    found = {m.group(1) for m in _AHA_HEADER.finditer(text)}
    found |= {m.group(1) for m in _AHA_FOOTER.finditer(text)}
    found |= {h.group("ref") for h in headers if h.group("ref")}
    if not found:
        raise EvidenceError(
            "No Aha reference found in the page's header, history or footer "
            "(expected something like LAPIPLAT-I-543)."
        )
    if len(found) > 1:
        raise EvidenceError(
            f"The page names more than one Aha item ({', '.join(sorted(found))}): add one page at a time."
        )
    return found.pop()


def _last_value_after(text: str, label: str) -> Optional[str]:
    """The value on the line after the last bare `label` line (the
    sidebar's), or on the same line."""
    matches = re.findall(rf"^\s*{label}\s*\n\s*(\S.*)$", text, re.MULTILINE)
    return matches[-1].strip() if matches else None


def _idea_page_title(text: str, reference: str) -> Optional[str]:
    """The main page's title: the lines between its header ("Created by ...
    Idea <REF>") and "Description", since a long title wraps."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if reference in line and line.lstrip().startswith("Created by "):
            title = []
            for following in lines[i + 1 : i + 21]:
                if following.strip() == "Description":
                    return " ".join(title) or None
                if following.strip():
                    title.append(following.strip())
            return None
    return None


def _vote_totals(text: str) -> dict:
    """The main page's vote box, label to count: {"Votes": 1, "Org": 0,
    "In F": 19}. Aha truncates its labels."""
    box = re.search(
        r"^Description\s*\nVotes\s*\nTotals\s*\n(?P<box>.*?)^Vote trend\s*$",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not box:
        return {}
    lines = [line.strip() for line in box["box"].splitlines() if line.strip()]
    return {
        label.rstrip("…").strip(): int(value)
        for label, value in zip(lines[::2], lines[1::2])
        if value.isdigit()
    }


def _idea_page_description(text: str) -> tuple:
    """(description, what sits between it and the sidebar) on the main page.

    The description follows the vote box and runs to the sidebar's Status
    line; only the chart's axis labels, the merge notice and the rank's
    stray superscript are taken off its end, so a description line that
    starts "Status" or is short stays in it. Without that sidebar, it stops
    at the first of those."""
    rest = text[_AHA_VOTE_TREND.search(text).end() :]
    rest = re.sub(r"\A\s*Proxy votes View in report[ \t]*\n", "", rest)
    sidebar = _AHA_SIDEBAR.search(rest)
    if sidebar:
        lines = rest[: sidebar.start()].split("\n")
    else:
        end = re.search(r"^\s*(?:Time\s*\nVotes\s*$|Merged into\b|Status\s)", rest, re.MULTILINE)
        lines = (rest[: end.start()] if end else rest).split("\n")
    tail = []
    while lines and _AHA_ABOVE_SIDEBAR.match(lines[-1]):
        tail.insert(0, lines.pop())
    if not sidebar and end:
        tail.append(rest[end.start() : end.start() + 200])
    description = "\n".join(line.rstrip() for line in lines)
    return re.sub(r"\n\s*\n(?:\s*\n)+", "\n\n", description).strip(), "\n".join(tail)


def _sidebar(text: str, label: str) -> Optional[str]:
    """A sidebar field's value on its label's line, or None; Aha's "No
    themes selected" and the like count as none."""
    match = re.search(rf"^{label}[ \t]+(\S.*?)\s*$", text, re.MULTILINE)
    if not match or re.fullmatch(r"No \w+ selected", match.group(1)):
        return None
    return match.group(1)


def _count_after(text: str, label: str) -> Optional[int]:
    match = re.search(rf"^{label}\s+(\d+)\s*$", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def _table_names(lines: List[str]) -> Optional[list]:
    """(name, votes) for each row, when lines are exactly whole names: each
    name's last line ends with its vote count, or is the count alone."""
    names, current = [], []
    for line in lines:
        end = _AHA_NAME_END.match(line)
        if end:
            if end["rest"]:
                current.append(end["rest"])
            if not current:
                return None
            names.append((" ".join(current), int(end["votes"])))
            current = []
        else:
            current.append(line)
    return None if current else names


def _table_rows(before: List[str], count: int) -> Optional[list]:
    """(status, name, votes) for the `count` rows whose references follow
    `before`, or None unless exactly one reading fits: `count` status lines,
    then `count` whole names."""
    readings, longest = [], 6 * count + 10  # names wrap onto a few lines each
    for start in range(len(before) - count, count - 1, -1):
        statuses = before[start - count : start]
        if not all(_AHA_STATUS.match(s) and s not in _AHA_PAGE_LABELS for s in statuses):
            continue
        names = _table_names(before[start:])
        if names is not None and len(names) == count:
            readings.append([(s, n, v) for s, (n, v) in zip(statuses, names)])
        if start < len(before) - longest:
            break
    return readings[0] if len(readings) == 1 else None


def _related_ideas(text: str, reference: str) -> list:
    """The ideas in the main page's Similar ideas and People also voted for
    tables, from where the first of them starts. A block of references is
    kept only when its statuses and names pair up with it exactly, so a
    reference elsewhere on the page (a linked record, a description's
    mention) isn't presented as a related idea. The page doesn't say which
    table each came from."""
    heading = re.search(r"^[ \t]*(?:Similar ideas|People also voted for)[ \t]*$", text, re.MULTILINE)
    if not heading:
        return []
    lines = [line.strip() for line in text[heading.end() :].splitlines() if line.strip()]
    related, i, blocks = {}, 0, 0
    while i < len(lines) and blocks < MAX_TABLE_BLOCKS:
        if not _AHA_REF_LINE.match(lines[i]):
            i += 1
            continue
        j = i
        while j < len(lines) and _AHA_REF_LINE.match(lines[j]):
            j += 1
        blocks += 1
        count = j - i
        rows = None
        if count <= MAX_TABLE_ROWS:
            # Only the lines a block's statuses and names can fill.
            rows = _table_rows(lines[max(0, i - (7 * count + 10)) : i], count)
        for ref, (status, name, votes) in zip(lines[i:j], rows or []):
            if ref != reference and ref not in related:
                related[ref] = {"reference": ref, "name": name, "status": status, "votes": votes}
        i = j
    return list(related.values())


def _idea_page_fields(text: str, reference: str) -> dict:
    """What the main page adds to the fields both pages share."""
    description, above_sidebar = _idea_page_description(text)
    # Support's wording is in the description; the merge notice sits just
    # above the sidebar, so a description that mentions a merge isn't one.
    requester = _AHA_REQUESTER.search(description)
    ticket = _AHA_TICKET.search(description)
    merged_into = _AHA_MERGED_INTO.search(above_sidebar)
    totals = _vote_totals(text)
    return {
        "description": description,
        "merged_into": merged_into.group(1) if merged_into else None,
        "zendesk_ticket": ticket.group(1) if ticket else None,
        "requested_by": requester["name"].strip() if requester else None,
        "requested_by_org": requester["org"].strip() if requester else None,
        "workspace": _sidebar(text, "Workspace"),
        "categories": _sidebar(text, "Categories"),
        "themes": _sidebar(text, "Themes"),
        "tags": _sidebar(text, "Tags"),
        "product_value": _count_after(text, "Product value"),
        "visibility": _sidebar(text, "Visibility in portals"),
        "vote_totals": totals,
        "votes": totals.get("Votes"),
        "merged_ideas": _count_after(text, "Merged ideas"),
        "linked_records": _count_after(text, "Linked records"),
        "related_ideas": _related_ideas(text, reference),
        "related_ideas_note": "Other ideas Aha lists beside this one (similar ones, "
        "or ones the same people voted for), each with its own status: none of "
        "them is this idea's status or outcome.",
    }


def parse_aha(text: str) -> dict:
    """The fields of an Aha idea or feature page, printed to PDF or copied.

    A comment later deleted is left out: Aha's history shows the deletion
    with the comment's text, and the comment's creation separately.

    Raises:
        EvidenceError: no Aha reference (e.g. PROJ-I-123) where the page
            names its item, more than one, or no title.
    """
    if len(text) > MAX_TEXT_CHARS:
        raise EvidenceError(f"The text is over {MAX_TEXT_CHARS:,} characters; an Aha page is far shorter.")
    # Aha's icon font prints as private-use characters; they carry nothing.
    text = re.sub(r"[\ue000-\uf8ff]", "", text.replace("\r\n", "\n"))
    headers = list(_AHA_ACTIVITY.finditer(text))
    reference = _aha_reference(text, headers)
    idea_page = not headers and _AHA_VOTE_TREND.search(text) is not None
    title = (idea_page and _idea_page_title(text, reference)) or _aha_title(text, reference)
    if not title:
        raise EvidenceError(f"Found {reference}, but not its title.")

    created = re.search(
        r"Created by (?P<by>.+?) in (?P<where>.+?) on (?P<on>[A-Z][a-z]{2} \d{1,2}, \d{4})", text
    )
    status = re.findall(r"^Status\s+(\S.*?)\s*$", text, re.MULTILINE)
    # The item's own link, its reference a whole path component: a comment
    # may link other Aha items, including PROJ-I-5430 beside PROJ-I-543.
    own = re.compile(rf"/{re.escape(reference)}(?:[/?#]|$)")
    url = next((u.group(0) for u in _AHA_URL.finditer(text) if own.search(u.group(0))), None)
    if url is None:
        url = (AHA_IDEA_URL if "-I-" in reference else AHA_FEATURE_URL).format(reference=reference)
    votes = re.search(r"\bVotes\s*(\d+)", text)
    score = re.search(r"\bScore\s*(\d+)", text)

    description_end = headers[0].start() if headers else len(text)
    description_start = text.find(title)
    description_start = description_start + len(title) if description_start != -1 else 0
    description = text[description_start:description_end].strip()

    events = []
    for n, header in enumerate(headers):
        body_end = headers[n + 1].start() if n + 1 < len(headers) else len(text)
        body = text[header.end():body_end]
        # The header runs on to the end of its line (the idea's title).
        events.append((header, body.split("\n", 1)[1] if "\n" in body else ""))

    def comment_of(header, body) -> tuple:
        """(author, text): the author is on the "Comment by" line, since a
        deletion's actor may be someone else."""
        by = re.match(r"\s*Comment by (.+)\n", body)
        comment_text = body[by.end():] if by else body
        # Only the comment: the sidebar and tab strip follow the last entry.
        comment_text = comment_text.split("\nName", 1)[0].strip()
        return ((by.group(1) if by else header.group("actor")).strip(), comment_text)

    # Each deletion withdraws one comment with its author and text.
    deleted = Counter(
        comment_of(h, b) for h, b in events if h.group("action") == "deleted comment on"
    )
    activity, comments, workflow = [], [], []
    for header, body in events:
        action = header.group("action")
        actor = header.group("actor").strip()
        entry = {"time": header.group("time"), "actor": actor, "action": action}
        if "comment" in action:
            entry["deleted"] = action.startswith("deleted")
            author, comment_text = comment_of(header, body)
            if entry["deleted"]:
                pass
            elif deleted[(author, comment_text)]:
                deleted[(author, comment_text)] -= 1
                entry["deleted_later"] = True
            else:
                comments.append(
                    {
                        "author": author,
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

    data = {
        "page": "idea" if idea_page else "history",
        "reference": reference,
        "title": title,
        "url": url,
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
    if idea_page:
        data.update(_idea_page_fields(text, reference))
        del data["activity_days"], data["activity_dates"]
        data["history"] = (
            "This is the idea's main page: its comments and workflow history are "
            "on its History tab, which isn't included here."
        )
    return data


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
    """One line for a stored evidence entry, e.g. "Jira LRN-12 (Done, Fixed)".

    It quotes the export, so show it as plain text, never rich text.
    """
    data = entry.get("data") or {}
    if entry.get("kind") == JIRA:
        issues = [
            f"{i['key']} ({', '.join(x for x in (i.get('status'), i.get('resolution')) if x)})"
            for i in data.get("issues") or []
        ]
        return "Jira " + "; ".join(issues)
    state = data.get("status") or "status unknown"
    if data.get("merged_into"):
        state += f", merged into {data['merged_into']}"
    return f"Aha {data.get('reference')}: {data.get('title')} ({state})"
