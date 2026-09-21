# The defence presentation

One talk, three forms. They carry the same argument and the same numbers.

| File | What it is |
|---|---|
| `Thesis_Defence_25min.pptx` | **The PowerPoint.** 39 slides, speaker notes on every one. |
| [`SPEAKER_SCRIPT.md`](SPEAKER_SCRIPT.md) | **What you say.** The full 25-minute script, timed, with the lines to use if an examiner pushes back. |
| The site at `/` | The same slides in the browser, paged with the arrow keys. `uv run pitch serve` |
| [`build_deck.js`](build_deck.js) | Regenerates the .pptx. `node build_deck.js out.pptx` |
| [`defence_slides.html`](defence_slides.html) | A standalone 19-slide short version, one file, opens offline. |

## The rule the whole thing is built on

**The slide holds the point. The notes hold the talk.** Nothing on a slide is a sentence you
read out — each one is a headline, a few cards, or one number. Every word you actually say
lives in the speaker notes and in the script. If you find yourself reading the screen, the
slide has too much on it; cut the slide, not the notes.

## The site

`uv run pitch serve`, then <http://127.0.0.1:8000>.

Eleven tabs, one per section of the talk, 51 slides in total.

| Key | Does |
|---|---|
| `→` `←` or the arrow buttons | Next / previous slide |
| The dots | Jump to a slide |
| **`N`** or the Notes button | Show or hide the speaker notes |

Four sections keep their evidence collapsed below the slides — the full model comparison,
the claims ledger with its retractions, the augmentation argument, and the two search
tables. The live preprocessing search still runs from the Searches tab.

**The eleven Markdown tabs are gone** (Overview, Models, Findings, Pre-registration,
Questions, Dataset, Database, Code, Ethics, Augmentation, Searches). Those documents are
still in `thesis/` and `docs/`; they were the right *evidence* and a poor *presentation*,
which is what this page is for now.

## Structure

Follows the oral-presentation template: introduce yourself → topic → roadmap → purpose →
how the research works → what problem it tackles → how it provides a solution → summary →
questions.

| Section | Covers |
|---|---|
| Start | Title, roadmap, the system in one line |
| Purpose | What it is for, **what the owner gets**, the one design rule |
| How it works | Five stages, the pipeline, what gets stored, the scenario, method |
| The problem | The gap, leakage, **the confound**, the clock rule, the clip that reversed it |
| Data | What we have, **why it is hard to get**, **what we generated with AI**, **overfitting** |
| Models | **Why we trained nothing**, and the two novel modules |
| Rules | **The 7-row decision table**, every switch and its cost, **preprocessing** |
| Searches | Two searches, the resolution floor, the prompt span |
| Augmentation | The question, the published draw, the five draws, the lesson, the sheets |
| Solution | Three fixes, guards verified by breaking them, what it will not do |
| Summary | Limits, what would change them, the summary, thank you |

## Before you present

- **Fill the placeholders**: `[PROGRAMME]`, `[INSTITUTION]`, `[Name]` on the title slide, and
  **`[VIDEO TOOL 1]` / `[VIDEO TOOL 2]`** — the two AI video generators — in the script, on
  the site's Data section, and on the PowerPoint's "So we made some" slide.
- **Every number comes from [`thesis/claims.md`](../claims.md)**, which is regenerated from
  the result artefacts by `experiments/verify_claims.py`. The site reads its numbers live;
  the PowerPoint and the script do not. If a result moves, re-read the ledger and fix both.
- **Rewrite the script in your own words.** Phrasing you did not choose collapses on the
  first follow-up question.
- **Practise the two claims that will be challenged**: that the system is more trustworthy
  than a manual check, and that we were right not to fine-tune. Both have a careful version
  in the script — use that one.
