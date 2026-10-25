"""Page markup kept in files: `templates/` for one page's parts, `static/` for shared ones.

A template is the page's text exactly as sent, except for ``{{name}}`` slots that `render`
fills with text computed in Python (the navigation bar, the shared JavaScript). Slots are
filled in one pass, so text put into one slot is never searched for further slots. Files are
read once and kept; nothing in them is evaluated, so JavaScript and CSS braces need no
escaping.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

__all__ = ["render", "static_text", "template"]

_HERE = Path(__file__).parent
_SLOT = re.compile(r"\{\{(\w+)\}\}")


@cache
def template(name: str) -> str:
    """The text of ``templates/<name>``."""
    return (_HERE / "templates" / name).read_text(encoding="utf-8")


@cache
def static_text(name: str) -> str:
    """The text of ``static/<name>``: a style sheet or script more than one page includes."""
    return (_HERE / "static" / name).read_text(encoding="utf-8")


def render(name: str, **parts: str) -> str:
    """``templates/<name>`` with each ``{{key}}`` replaced by ``parts[key]``.

    The slots in the file and the keys given must match exactly, so a renamed slot fails
    at import rather than leaving ``{{...}}`` in a page.
    """
    text = template(name)
    slots = set(_SLOT.findall(text))
    if slots != parts.keys():
        raise KeyError(f"{name}: slots {sorted(slots)} but parts {sorted(parts)}")
    return _SLOT.sub(lambda m: parts[m[1]], text)
