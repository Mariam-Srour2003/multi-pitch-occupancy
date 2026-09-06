"""A small Markdown renderer for the documents the thesis frontend shows.

The experiment log, pre-registration and RQ matrix are the substance of the thesis, and
they are written as Markdown. Rendering them in the browser means the frontend shows the
*current* documents rather than a copy that drifts.

Deliberately not a full CommonMark implementation - it handles exactly what these files
use: headings, tables, fenced code, lists, blockquotes, bold, italic, inline code, links
and horizontal rules. A dependency would carry far more surface than that for no gain, and
these files are ours rather than user input.
"""

from __future__ import annotations

import html
import re

__all__ = ["render"]

_INLINE = (
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
)


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def _table(rows: list[str]) -> str:
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]  # rows[1] is the --- separator
    th = "".join(f"<th>{_inline(c)}</th>" for c in head)
    trs = "".join(
        "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body
    )
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def render(md: str) -> str:
    """Render a Markdown document to HTML."""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    list_open = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            close_list()
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(html.escape(lines[i]))
                i += 1
            out.append("<pre><code>" + "\n".join(block) + "</code></pre>")
            i += 1
            continue

        # a table needs a header row and a --- separator directly beneath it
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) <= set("|-: ")
            and "-" in lines[i + 1]
        ):
            close_list()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(_table(block))
            continue

        if not stripped:
            close_list()
            i += 1
            continue

        if stripped.startswith("---") and set(stripped) == {"-"}:
            close_list()
            out.append("<hr>")
            i += 1
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            close_list()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        if stripped.startswith("> "):
            close_list()
            quote = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                quote.append(lines[i].strip()[2:])
                i += 1
            out.append(f'<blockquote>{_inline(" ".join(quote))}</blockquote>')
            continue

        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet:
            if not list_open:
                out.append("<ul>")
                list_open = True
            item = [bullet.group(1)]
            i += 1
            # continuation lines of the same bullet are indented
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                    and not re.match(r"^\s*[-*]\s+", lines[i]):
                item.append(lines[i].strip())
                i += 1
            out.append(f"<li>{_inline(' '.join(item))}</li>")
            continue

        close_list()
        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^\s*(#{1,4}\s|[-*]\s|\||>|```|---)", lines[i]
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")

    close_list()
    return "\n".join(out)
