# The defence presentation

One talk, several forms. They carry the same argument and the same numbers.

| File | What it is |
|---|---|
| `Thesis_Defence_25min.pptx` | **The PowerPoint.** 39 slides, speaker notes on every one. |
| [`SPEAKER_SCRIPT.md`](SPEAKER_SCRIPT.md) | **What you say.** The full 25-minute script, timed, with the lines to use if an examiner pushes back. |
| The site at `/` | **The talk as nine scrollable pages**, one per template section. `uv run pitch serve` |
| [`build_deck.js`](build_deck.js) | Regenerates the .pptx. `node build_deck.js out.pptx` |
| [`defence_slides.html`](defence_slides.html) | A standalone 19-slide short version, one file, opens offline. |
| `Thesis_Site_Deck.pptx` | **The site, as a PowerPoint.** 29 slides - every block the site renders, in the site's own palette and components, plus a title page. Morph between the pages of a section, push between sections, and every block rises in on its own. |
| [`build_site_deck.js`](build_site_deck.js) | Regenerates that deck. `node build_site_deck.js [out.pptx]` |

## The rule the whole thing is built on

**The page holds the point. The script holds the talk.** Nothing on the site is a sentence
you read out — each block is a headline, a few cards, or one number. Every word you actually
say lives in `SPEAKER_SCRIPT.md`. If you find yourself reading the screen, the block has too
much on it; cut the block, not the script.

## The PowerPoint of the site

`node build_site_deck.js` writes `Thesis_Site_Deck.pptx`: **the same eight parts, the same
words, the same palette** - one slide per block of the site, plus the title page, which is
the only thing on it that is not on the site. It reads `results/coverage.md`,
`results/model_inventory.json`, `configs/rules.json` and `results/preprocess_pairs.csv` at
build time, exactly as the pages read them at request time, so a rerun of an experiment
moves both. **The site is unchanged**; the deck is a second rendering of it, not a
replacement.

Five blocks are redrawn rather than screenshotted - the DINOv2 stack, the YOLOv8 stack, the
augmentation axes, the six preprocessing pairs and the search panel's bar chart - so every
box and bar is a real shape and animates like the rest. The deck is light-only: a projector
has one colour scheme.

**One slide per block, and the whole block on it.** The nine literature strands and the
six before/after pairs are each one page here, as they are one block on the site: split
across slides they read as separate claims, and the point of both blocks is the whole set.

**What the site appends below a tab is on the deck too**, in the site's order:

| Site | Deck |
|---|---|
| Chapter 2 + `models_view.render()` | *The full model comparison* - the four models tried, the seven columns, the arrow on each saying which way is better. A cell with no result is a dash, never a zero. |
| Chapter 4 + `_augmentation()` | *What removal looks like*, *The augmentation argument*, *The preset sheets*, *One effect at a time* |
| Chapter 4 + `ARCHIVE_HTML` | *The preprocessing search* |

The search chart is drawn from `results/search_runs/dinov2__all.json` - the cell the panel
opens on, of the six it offers - with the panel's own ranking (balanced = recall &minus;
false-play) and its own colours, including the amber for a configuration that bought recall
by answering PLAY more often. It shows the top fourteen of that run and says so, and it
says where the untouched baseline ranks rather than quietly dropping it.

**The three contact sheets are not resampled.** `preprocess_effects.jpg`,
`augmentation_grid.jpg` and `augmentation_effects.jpg` are up to four times taller than a
slide, and at full height on one they are a column of unreadable thumbnails. Each is placed
several times with `srcRect` - exactly the crop PowerPoint writes when you crop a picture
in its own UI - so the sheet reads left to right instead of top to bottom at two to four
times the tile size. The file on the slide is still the artefact on disk, every band is
there, in order, and dragging the crop open gives the whole sheet back. The cuts fall where
the sheets already divide: the night frame and the day frame.

**Not on the deck:** `/related-work`, the page the *sources* link on each strand card opens.
It is `thesis/ch2_related_work.md` with its `[CITE]` slots still open - the right thing to
have behind a link, and the wrong thing to project.

Two things pptxgenjs cannot write are injected into the slide XML afterwards, at the bottom
of the build script: the hero gradients, and the transitions and the build. Note that
**Morph lives in the 2015/09 namespace, not the 2010 one** - a `<p14:morph/>` is accepted
and then silently ignored, which looks like no transition at all.

One thing the deck does not reproduce: `talk.py` asks for `tone="amber"` on the Chapter 3
hero, but the stylesheet has no `.tk-hero.amber` rule, so that band renders with no
background on the site. The deck draws the gradient the tone names.

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
