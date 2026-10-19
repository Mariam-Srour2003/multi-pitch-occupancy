"""The shared page frame: one style sheet, one navigation bar, one way to build a page.

Every page used to carry its own copy of the header links, so adding a section meant editing
five strings and missing one. The links live in `nav.py` now, and the style sheet lives here -
cut from the clip reviewer's page rather than copied, so there is still only one.
"""

from __future__ import annotations

from html import escape

from pitch_occupancy.api.clip_page import CLIP_HTML
from pitch_occupancy.api.nav import NAV, NAV_CSS, nav_html

__all__ = ["CALL_JS", "FONTS", "JS_COMMON", "NAV", "STYLE", "nav_html", "page"]

#: The font links, from the clip page.
FONTS = CLIP_HTML[CLIP_HTML.index('<link rel="preconnect"'):CLIP_HTML.index("<style>")]
#: The style sheet, from the clip page: colour tokens, light/dark, cards, tiles, tables.
STYLE = CLIP_HTML[CLIP_HTML.index("<style>"):CLIP_HTML.index("</style>") + len("</style>")]

#: JavaScript every page gets first: ``$`` (element by id) and ``esc`` (text made safe for
#: HTML, attribute values in either quote included). `page` adds it; the stand-alone pages
#: put it at the top of their own ``<script>``.
JS_COMMON = r"""
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',
  '"':'&quot;',"'":'&#39;'}[c]));
"""

#: ``call(url, opts)``: fetch JSON, throwing the server's ``detail`` (or the HTTP status).
CALL_JS = r"""
async function call(url,opts){
  const r=await fetch(url,opts);let d=null;
  try{d=await r.json()}catch(e){}
  if(!r.ok)throw new Error((d&&d.detail)?(typeof d.detail==='string'?d.detail:
    JSON.stringify(d.detail)):('HTTP '+r.status));
  return d;
}
"""

_EXTRA = """<style>
.grid2{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.card h3{margin:0 0 4px;font-size:15px}
.card p{margin:0;color:var(--ink-2);font-size:13px}
a.card{display:block;text-decoration:none;color:inherit;transition:border-color .15s}
a.card:hover{border-color:var(--accent)}
.field{display:block;margin:0 0 12px}
.field label{display:block;font:500 10.5px 'JetBrains Mono',monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);margin-bottom:5px}
.field input,.field select,.field textarea{font:13px 'JetBrains Mono',monospace;
  padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--ground);
  color:var(--ink);width:100%;max-width:520px}
.ok{color:var(--play)}.bad{color:var(--flag)}.muted{color:var(--ink-3)}
.bar{height:8px;border-radius:4px;background:var(--surface-2);overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent);width:0;transition:width .3s}
code{font-family:'JetBrains Mono',monospace;font-size:12.5px;background:var(--surface-2);
  padding:1px 5px;border-radius:4px}
pre{font:12.5px/1.5 'JetBrains Mono',monospace;background:var(--surface-2);padding:12px 14px;
  border-radius:8px;overflow-x:auto}
@media (max-width:640px){main{padding:20px 16px 60px}.hin{padding:14px 16px}}
</style>"""


def page(title: str, subtitle: str, body: str, *, active: str = "", css: str = "",
         script: str = "") -> str:
    """A whole page: head, header with navigation, ``body`` inside ``<main>``, ``script``.

    ``css`` is extra style rules (without the ``<style>`` tag); ``script`` is JavaScript
    (without the ``<script>`` tag), run after `JS_COMMON` - so it must not redeclare ``$``
    or ``esc``.
    """
    return (
        '<!doctype html>\n<html lang="en"><head>\n<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n{FONTS}{STYLE}{NAV_CSS}{_EXTRA}"
        + (f"<style>{css}</style>" if css else "")
        + "\n</head><body>\n<header><div class=\"hin\">"
        '<div class="brand">Pitch <span>Occupancy</span></div>'
        f'<div class="sub">{escape(subtitle)}</div>{nav_html(active)}</div></header>\n'
        f"<main>\n{body}\n</main>\n"
        + (f"<script>\n{JS_COMMON}{script}\n</script>\n" if script else "")
        + "</body></html>\n"
    )
