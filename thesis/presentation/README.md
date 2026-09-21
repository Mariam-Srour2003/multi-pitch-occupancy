# The defence presentation

One talk, three forms. They carry the same argument and the same numbers.

| File | What it is |
|---|---|
| `Thesis_Defence_25min.pptx` | **The PowerPoint.** 39 slides, speaker notes on every one. |
| [`SPEAKER_SCRIPT.md`](SPEAKER_SCRIPT.md) | **What you say.** The full 25-minute script, timed, with the lines to use if an examiner pushes back. |
| The site at `/` | **The talk as nine scrollable pages**, one per template section. `uv run pitch serve` |
| [`build_deck.js`](build_deck.js) | Regenerates the .pptx. `node build_deck.js out.pptx` |
| [`defence_slides.html`](defence_slides.html) | A standalone 19-slide short version, one file, opens offline. |

## The rule the whole thing is built on

**The page holds the point. The script holds the talk.** Nothing on the site is a sentence
you read out — each block is a headline, a few cards, or one number. Every word you actually
say lives in `SPEAKER_SCRIPT.md`. If you find yourself reading the screen, the block has too
much on it; cut the block, not the script.

## The site

`uv run pitch serve`, then <http://127.0.0.1:8000>.

**Nine tabs, one per numbered section of the oral-presentation template**, in the order you
deliver them. Each tab is an ordinary scrollable page - a coloured hero, then blocks of
cards, counts, a table or a figure. No slides and no pager: read it top to bottom, and
find-in-page works.

| Tab | Template section |
|---|---|
| 1. Introduce | Introduce yourself |
| 2. Topic | Name / topic of the research |
| 3. Roadmap | Roadmap |
| 4. Purpose | Purpose of the research |
| 5. How it works | How does the research work? (+ the scenario) |
| 6. The problem | What problem does the research tackle? |
| 7. The solution | How does the research provide a solution? |
| 8. Summary | Summary |
| 9. Thank you | Thank you & questions |

**The site carries no delivery notes.** What you say is in
[`SPEAKER_SCRIPT.md`](SPEAKER_SCRIPT.md) and nowhere on the page — the screen is what the
room looks at while you talk, and a paragraph of notes on it competes with you.

Three tabs keep their evidence collapsed underneath: the full model comparison and the
before/after preprocessing sheet under *How it works*, the claims ledger with its
retractions under *The problem*, and the augmentation argument plus both search tables -
with the live preprocessing search - under *The solution*.

**The eleven Markdown tabs are gone** (Overview, Models, Findings, Pre-registration,
Questions, Dataset, Database, Code, Ethics, Augmentation, Searches), and so is the 51-slide
deck that briefly replaced them. Those documents are still in `thesis/` and `docs/`.

## What each tab covers

| Tab | Covers |
|---|---|
| 1. Introduce | Name, programme, institution, supervisor — and the work in one sentence |
| 2. Topic | The full title, what it means, the scale, and the one design rule |
| 3. Roadmap | The five stops, and the two experiments worth dwelling on |
| 4. Purpose | The purpose sentence, what the study focuses on, **what the owner gets**, and what the system refuses to do |
| 5. How it works | Five stages, the scenario, the pipeline, what gets stored, the data, **why it is hard to get**, **what we generated with AI**, **why we trained nothing**, **the 7-row rule table**, **preprocessing**, and how it was analysed |
| 6. The problem | The verification gap, what is known and missing, leakage, **the confound**, the clock rule, the clip that reversed it, and what one missing cell blocks |
| 7. The solution | Three fixes, both searches, the augmentation retraction, guards verified by breaking them, and what it contributes |
| 8. Summary | Investigated / found / contributed / implies, the limits, and what would change them |
| 9. Thank you | The closing, and the six questions most likely to come with their short answers |

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
