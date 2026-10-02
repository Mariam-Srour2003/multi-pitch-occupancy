// Builds the PowerPoint of the thesis *site*: the eight parts of the report as slides,
// in the site's own visual language.
//
//   node build_site_deck.js [out.pptx]
//
// **The site is still the site.** Nothing here edits it; this file reads the same
// artefacts the pages read at request time - `results/coverage.md`,
// `results/model_inventory.json`, `configs/rules.json`, `results/preprocess_pairs.csv` -
// so a rerun of an experiment moves the deck and the page together. The wording, the
// order of the blocks and the palette are `src/pitch_occupancy/api/talk.py`, transcribed
// rather than reinterpreted: the only slide that is not on the site is the title page.
//
// Two things PowerPoint can do that a scrolling page cannot are added afterwards, in
// `postprocess()`: the hero gradients (pptxgenjs writes solid fills only) and the
// transitions and entrance animations (it writes none at all). Both are injected into the
// slide XML directly - see the bottom of this file.

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");

const ROOT = path.resolve(__dirname, "..", "..");
const RESULTS = path.join(ROOT, "results");
const OUT = process.argv[2] || path.join(__dirname, "Thesis_Site_Deck.pptx");

// --- the site's palette ---------------------------------------------------------------
// Lifted from `talk.py`'s STYLES and the shell's light scheme. The deck is light-only:
// a projector has one scheme.
const P = {
  ground: "F6F8F7", surface: "FFFFFF", surface2: "EEF2F1", line: "DDE4E2",
  ink: "111817", ink2: "4B5A58", ink3: "7A8886", accent: "0D6D78", paper: "FBFAF6",
  green: "0E8F5C", greenB: "17B978", greenS: "E6F7EF",
  amber: "B26A00", amberB: "F5A524", amberS: "FDF0D9",
  coral: "D22D4C", coralB: "EF4F6B", coralS: "FDE8EC",
  sky: "1D6FA5", skyB: "3FA7E0", skyS: "E3F2FB",
  violet: "6A57D6", violetB: "7A68E0", violetS: "ECE9FB",
};

// `.tk-hero.<tone>` - the CSS gradients, as (angle-from-east, stops). The page has no
// `.tk-hero.amber` rule even though Chapter 3 asks for `tone="amber"`, so that hero
// currently renders with no background at all on the site; the deck draws the gradient the
// tone names rather than reproducing the gap.
const HEROES = {
  ink: [50, [[0, "0E1B2C"], [60000, "16283F"], [100000, "0B2A3A"]]],
  green: [45, [[0, "0E8F5C"], [100000, "14A86D"]]],
  coral: [45, [[0, "C42746"], [100000, "EF4F6B"]]],
  sky: [45, [[0, "14567F"], [100000, "2C89C4"]]],
  violet: [45, [[0, "4A3AA8"], [100000, "7A68E0"]]],
  amber: [45, [[0, "9A5B00"], [100000, "F5A524"]]],
};
const TONE_BAR = { ink: P.accent, green: P.greenB, coral: P.coralB, sky: P.skyB,
                   violet: P.violetB, amber: P.amberB };

// Archivo and JetBrains Mono are the site's faces and are not installed on Windows; these
// are the nearest pair that is always there, so the file opens the same on any machine.
const BODY = "Segoe UI";
const MONO = "Consolas";

// --- geometry -------------------------------------------------------------------------
const W = 13.333, H = 7.5;
const M = 0.55;                    // page gutter
const CW = W - 2 * M;              // 12.233
const TOP = 1.52;                  // first line of content on a content slide
const BOT = 6.98;                  // last line of content
const CH = BOT - TOP;              // 5.46
const GAP = 0.18;

// --- html -> pptx runs ------------------------------------------------------------------
// The copy is held in `talk.py` as HTML fragments and is transcribed here unchanged, so
// the deck needs the same small subset of markup the page uses: <b>, <i>, <br>, <code>,
// and the entities. Anything else - the `sources ->` link on a strand card - is site
// navigation and is dropped rather than rendered as dead text.
const ENT = {
  mdash: "\u2014", ndash: "\u2013", amp: "&", euro: "\u20ac", ge: "\u2265", le: "\u2264",
  middot: "\u00b7", rsquo: "\u2019", lsquo: "\u2018", ldquo: "\u201c", rdquo: "\u201d",
  times: "\u00d7", ne: "\u2260", nbsp: "\u00a0", sect: "\u00a7", rarr: "\u2192",
  minus: "\u2212", hellip: "\u2026", lt: "<", gt: ">", quot: '"', deg: "\u00b0",
  sigma: "\u03c3", Sigma: "\u03a3", mu: "\u03bc", alpha: "\u03b1", beta: "\u03b2",
  plusmn: "\u00b1", divide: "\u00f7", infin: "\u221e", asymp: "\u2248", bull: "\u2022",
  larr: "\u2190", uarr: "\u2191", darr: "\u2193", harr: "\u2194", apos: "'",
  "#215": "\u00d7", "#8209": "\u2011",
};
function ents(s) {
  return s.replace(/&(#?\w+);/g, (m, k) => (k in ENT ? ENT[k] : m));
}

/** `html` as pptxgenjs text runs, every run carrying `base`. */
function rt(html, base = {}) {
  const out = [];
  let bold = 0, ital = 0, code = 0;
  const src = String(html).replace(/<a\b[^>]*>.*?<\/a>/gi, "");
  const re = /<(\/?)(b|i|code|br)\s*\/?>/gi;
  let at = 0, m;
  const push = (text, brk) => {
    if (!text && !brk) return;
    const o = Object.assign({}, base);
    if (bold) o.bold = true;
    if (ital) o.italic = true;
    if (code) { o.fontFace = MONO; o.fontSize = (base.fontSize || 12) - 0.5; }
    if (brk) o.breakLine = true;
    out.push({ text: ents(text), options: o });
  };
  while ((m = re.exec(src))) {
    const chunk = src.slice(at, m.index);
    at = m.index + m[0].length;
    const tag = m[2].toLowerCase(), close = m[1] === "/";
    if (tag === "br") { push(chunk, true); continue; }
    push(chunk, false);
    const d = close ? -1 : 1;
    if (tag === "b") bold = Math.max(0, bold + d);
    else if (tag === "i") ital = Math.max(0, ital + d);
    else code = Math.max(0, code + d);
  }
  push(src.slice(at), false);
  return out.length ? out : [{ text: "", options: base }];
}
/** Plain text, for measuring. */
const plain = (html) => ents(String(html).replace(/<[^>]+>/g, ""));

// --- measuring ----------------------------------------------------------------------------
// A card on the site is as tall as its text. A card on a slide is as tall as it was drawn,
// and a row sized for the longest of four leaves the other three half empty - which is what
// the first build of this deck looked like. So every block measures its own copy and is then
// centred in the band it was given. The estimate is deliberately generous: a card with slack
// under its last line still reads as a card, one whose last line is clipped does not.
const CPC = 0.50;      // mean glyph width as a fraction of point size, Segoe UI regular
const CPC_B = 0.545;   // ... and bold

function estLines(html, fs, widthIn, factor) {
  const t = plain(String(html).replace(/<br\s*\/?>/gi, "\n"));
  const cpl = Math.max(8, (widthIn * 72) / ((factor || CPC) * fs));
  return t.split("\n").reduce(function (n, seg) {
    return n + Math.max(1, Math.ceil(seg.trim().length / cpl));
  }, 0);
}
const lineH = function (n, fs, mult) { return (n * fs * mult) / 72; };

function cardsH(items, innerW, ts, bs) {
  return items.reduce(function (max, it) {
    return Math.max(max, lineH(estLines(it[0], ts, innerW, CPC_B), ts, 1.30)
                       + lineH(estLines(it[1], bs, innerW, CPC), bs, 1.42));
  }, 0) + 0.36;
}
function stepHs(items, innerW, ts, bs) {
  return items.map(function (it) {
    return Math.max(0.80, lineH(estLines(it[0], ts, innerW, CPC_B), ts, 1.30)
                        + lineH(estLines(it[1], bs, innerW, CPC), bs, 1.40) + 0.30);
  });
}
function tilesH(items, innerW) {
  return items.reduce(function (max, it) {
    return Math.max(max, lineH(estLines(it[1], 11.5, innerW, CPC), 11.5, 1.42));
  }, 0) + 1.10;
}
/** Centre a block of height `h` in the band `box` - never above its top. */
const centre = function (box, h) { return box.y + Math.max(0, (box.h - h) * 0.42); };

// --- artefacts ---------------------------------------------------------------------------
// The same files the pages read, read the same way. A missing one is reported on the slide
// rather than silently dropped, which is the rule `talk.py` states for the site.
const readText = (p) => (fs.existsSync(p) ? fs.readFileSync(p, "utf8") : "");
const readJson = (p) => { try { return JSON.parse(readText(p) || "{}"); } catch { return {}; } };

function coverage() {
  const t = readText(path.join(RESULTS, "coverage.md"));
  const m = t.match(/\|\s*(\d[\d,]*)\s*recorded frames\s*\|\s*(\d+)\s*venues/);
  return m ? { frames: m[1], venues: m[2] } : {};
}
function csvRows(name) {
  const t = readText(path.join(RESULTS, name));
  if (!t.trim()) return [];
  const lines = t.trim().split(/\r?\n/);
  const split = (l) => {
    const out = []; let cur = "", q = false;
    for (const ch of l) {
      if (ch === '"') q = !q;
      else if (ch === "," && !q) { out.push(cur); cur = ""; }
      else cur += ch;
    }
    out.push(cur); return out;
  };
  const head = split(lines[0]);
  return lines.slice(1).map((l) => Object.fromEntries(split(l).map((v, i) => [head[i], v])));
}

const COV = coverage();
const INV = readJson(path.join(RESULTS, "model_inventory.json"));
const RULES = readJson(path.join(ROOT, "configs", "rules.json"));

// --- deck scaffolding ---------------------------------------------------------------------
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Fady Khazaka";
pres.title = "Playground Activity Detection using Deep Learning";
pres.subject = "Master's thesis defence";

// Per-slide animation plan, filled as the slides are built and applied in `postprocess`.
// `skip` shapes are always on screen; each number in `groups` is how many consecutive
// shapes rise in together, as one beat.
const PLAN = [];
let cur = null;
let pageNo = 0;

function newSlide(kind, notes) {
  const s = pres.addSlide();
  // `--ground` behind, `--surface` on the cards, exactly as the page stacks them: white
  // cards on white lose their edge, and the site never puts them there.
  if (kind !== "hero") s.background = { color: P.ground };
  cur = { skip: 0, groups: [], transition: kind === "hero" ? "push" : "morph" };
  PLAN.push(cur);
  if (notes) s.addNotes(notes);
  return s;
}
const stat = (n) => { cur.skip += n; };          // always on screen
const grp = (n) => { cur.groups.push(n); };      // one beat of the build

// Sentinels: pptxgenjs has no gradient fill, so a hero is filled with an impossible colour
// that `postprocess` swaps for the real gradient.
const SENT = {};
let sentN = 0;
function gradient(tone) {
  const key = "FF" + String(++sentN).padStart(4, "0");
  SENT[key] = HEROES[tone];
  return key;
}

// --- components ---------------------------------------------------------------------------
// One function per `.tk-*` component in the site's stylesheet, at slide scale.

/** `.tk-hero` as a whole slide: the section breaks and the title page. */
function hero(s, tone, eyebrow, title, lead, opts) {
  opts = opts || {};
  s.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: W, h: H, fill: { color: gradient(tone) }, line: { type: "none" },
  });
  stat(1);
  // The hero is three stacked lines of type, so it is measured and centred rather than
  // placed: a two-line title with the gap of a three-line one reads as a mistake.
  const titleSize = opts.titleSize || 36;
  const titleW = opts.titleW || 10.6;
  const leadSize = opts.leadSize || 16;
  const leadW = opts.leadW || 9.8;
  const titleH = lineH(estLines(title, titleSize, titleW, CPC_B), titleSize, 1.16) + 0.12;
  const leadH = lead ? lineH(estLines(lead, leadSize, leadW, CPC), leadSize, 1.5) + 0.1 : 0;
  const block = 0.42 + titleH + (lead ? 0.26 + leadH : 0);
  const y = opts.y != null ? opts.y : (H - block) * 0.42;
  if (eyebrow) {
    s.addText(ents(eyebrow).toUpperCase(), {
      x: M + 0.35, y: y, w: CW - 0.7, h: 0.32, fontFace: MONO, fontSize: 12, bold: true,
      charSpacing: 3, color: "BBC9D6", valign: "bottom",
    });
    grp(1);
  }
  s.addText(rt(title, { fontFace: BODY, fontSize: titleSize, bold: true, color: "FFFFFF" }), {
    x: M + 0.35, y: y + 0.42, w: titleW, h: titleH,
    valign: "top", lineSpacing: titleSize * 1.16,
  });
  grp(1);
  if (lead) {
    s.addText(rt(lead, { fontFace: BODY, fontSize: leadSize, color: "DCE6EE" }), {
      x: M + 0.35, y: y + 0.42 + titleH + 0.26, w: leadW, h: leadH,
      valign: "top", lineSpacing: leadSize * 1.5,
    });
    grp(1);
  }
}

/** The chrome every content slide carries: section, title, rule, page number. */
function head(s, tone, section, title) {
  s.addText(ents(section).toUpperCase(), {
    x: M, y: 0.34, w: CW, h: 0.26, fontFace: MONO, fontSize: 10, bold: true,
    charSpacing: 2.4, color: P.ink3, valign: "middle",
  });
  s.addText(rt(title, { fontFace: BODY, fontSize: 23, bold: true, color: P.ink }), {
    x: M, y: 0.62, w: CW - 0.6, h: 0.52, valign: "middle",
  });
  s.addShape(pres.ShapeType.rect, {
    x: M, y: 1.26, w: CW, h: 0.012, fill: { color: P.line }, line: { type: "none" },
  });
  s.addShape(pres.ShapeType.rect, {
    x: M, y: 1.245, w: 1.35, h: 0.042, fill: { color: TONE_BAR[tone] }, line: { type: "none" },
  });
  pageNo += 1;
  s.addText(String(pageNo), {
    x: W - M - 0.8, y: H - 0.5, w: 0.8, h: 0.3, fontFace: MONO, fontSize: 10,
    color: P.ink3, align: "right", valign: "middle",
  });
  stat(5);
}

/** `.tk-tiles` - a row of big numbers, each with its coloured top border. */
function tiles(s, items, box) {
  const n = items.length;
  const w = (box.w - GAP * (n - 1)) / n;
  const h = tilesH(items, w - 0.36);
  const y0 = centre(box, h);
  const FG = { good: P.green, bad: P.coral, lead: P.sky, warn: P.amber, "": P.ink };
  const TOP_ = { good: P.greenB, bad: P.coralB, lead: P.skyB, warn: P.amberB, "": P.line };
  items.forEach(function (it, i) {
    const value = it[0], label = it[1], tone = it[2] || "";
    const x = box.x + i * (w + GAP);
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: y0, w: w, h: h, rectRadius: 0.1,
      fill: { color: TOP_[tone] }, line: { type: "none" },
    });
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: y0 + 0.07, w: w, h: h - 0.07, rectRadius: 0.1,
      fill: { color: P.surface }, line: { color: P.line, width: 0.75 },
    });
    s.addText(rt(value, { fontFace: MONO, fontSize: 32, bold: true, color: FG[tone] }), {
      x: x + 0.18, y: y0 + 0.24, w: w - 0.36, h: 0.62, valign: "middle",
    });
    s.addText(rt(label, { fontFace: BODY, fontSize: 11.5, color: P.ink2 }), {
      x: x + 0.18, y: y0 + 0.88, w: w - 0.36, h: h - 0.98, valign: "top", lineSpacing: 16,
    });
    grp(4);
  });
}

/** `.tk-cards` - a grid of cards, each with the accent stripe down its left edge. */
const CARD_ACCENT = [P.greenB, P.skyB, P.amberB, P.coralB];
function cards(s, items, box, opt) {
  opt = opt || {};
  const cols = opt.cols || 2;
  const rows = Math.ceil(items.length / cols);
  const w = (box.w - GAP * (cols - 1)) / cols;
  const ts = opt.titleSize || 13.5, bs = opt.bodySize || 11;
  const h = Math.min((box.h - GAP * (rows - 1)) / rows, cardsH(items, w - 0.5, ts, bs));
  const y0 = centre(box, h * rows + GAP * (rows - 1));
  items.forEach(function (it, i) {
    const x = box.x + (i % cols) * (w + GAP);
    const y = y0 + Math.floor(i / cols) * (h + GAP);
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: y, w: w, h: h, rectRadius: 0.1,
      fill: { color: CARD_ACCENT[(opt.from || 0) + i & 3] }, line: { type: "none" },
    });
    s.addShape(pres.ShapeType.roundRect, {
      x: x + 0.07, y: y, w: w - 0.07, h: h, rectRadius: 0.1,
      fill: { color: P.surface }, line: { color: P.line, width: 0.75 },
    });
    const runs = rt(it[0], { fontFace: BODY, fontSize: ts, bold: true, color: P.ink });
    runs[runs.length - 1].options.breakLine = true;
    s.addText(runs.concat(rt(it[1], { fontFace: BODY, fontSize: bs, color: P.ink2 })), {
      x: x + 0.28, y: y + 0.15, w: w - 0.5, h: h - 0.3, valign: "top",
      lineSpacing: bs * 1.4, paraSpaceBefore: 4,
    });
    grp(3);
  });
  return y0 + h * rows + GAP * (rows - 1);   // where a note under the block starts from
}

/** `.tk-steps` - cards that carry their number, because the order is the meaning. */
const STEP_DOT = [P.greenB, P.skyB, P.violet, P.amberB, P.coralB];
const STEP_FG = ["04301F", "04283D", "FFFFFF", "3A2400", "FFFFFF"];
function steps(s, items, box, opt) {
  opt = opt || {};
  const n = items.length;
  const bs = opt.bodySize || 11, ts = opt.titleSize || 13.5;
  let hs = stepHs(items, box.w - 1.05, ts, bs);
  const over = hs.reduce(function (t, v) { return t + v; }, 0) + 0.14 * (n - 1);
  if (over > box.h) hs = hs.map(function (v) { return v * (box.h - 0.14 * (n - 1)) / (over - 0.14 * (n - 1)); });
  const tot = hs.reduce(function (t, v) { return t + v; }, 0) + 0.14 * (n - 1);
  let acc = centre(box, tot);
  items.forEach(function (it, i) {
    const h = hs[i];
    const y = acc;
    acc += h + 0.14;
    const k = (opt.from || 0) + i;
    s.addShape(pres.ShapeType.roundRect, {
      x: box.x, y: y, w: box.w, h: h, rectRadius: 0.09,
      fill: { color: P.surface }, line: { color: P.line, width: 0.75 },
    });
    s.addShape(pres.ShapeType.ellipse, {
      x: box.x + 0.22, y: y + (h - 0.42) / 2, w: 0.42, h: 0.42,
      fill: { color: STEP_DOT[k % 5] }, line: { type: "none" },
    });
    s.addText(String(k + 1), {
      x: box.x + 0.22, y: y + (h - 0.42) / 2, w: 0.42, h: 0.42, fontFace: MONO,
      fontSize: 13, bold: true, color: STEP_FG[k % 5], align: "center", valign: "middle",
    });
    const runs = rt(it[0], { fontFace: BODY, fontSize: opt.titleSize || 13.5, bold: true,
                             color: P.ink });
    runs[runs.length - 1].options.breakLine = true;
    s.addText(runs.concat(rt(it[1], { fontFace: BODY, fontSize: bs, color: P.ink2 })), {
      x: box.x + 0.82, y: y + 0.12, w: box.w - 1.05, h: h - 0.24, valign: "top",
      lineSpacing: bs * 1.38, paraSpaceBefore: 4,
    });
    grp(4);
  });
}

/** `.tk-table` - the header in mono small caps, one highlighted row. */
function table(s, headers, rows, box, opt) {
  opt = opt || {};
  const headH = 0.4;
  const bodyH = (box.h - headH) / rows.length;
  const noLine = { type: "none" };
  const cellBorder = [noLine, noLine, { pt: 0.75, color: P.line }, noLine];
  const head_ = headers.map(function (t) {
    return {
      text: plain(t).toUpperCase(),
      options: { fontFace: MONO, fontSize: 9, bold: true, color: P.ink3, charSpacing: 1.2,
                 fill: { color: P.surface2 }, valign: "middle", border: cellBorder },
    };
  });
  const body = rows.map(function (r, i) {
    const hi = opt.hi === i;
    return r.map(function (c) {
      return {
        text: rt(c, { fontFace: BODY, fontSize: opt.fontSize || 11,
                      color: hi ? P.ink : P.ink2 }),
        options: { fill: { color: hi ? P.greenS : P.surface }, valign: "middle",
                   border: cellBorder },
      };
    });
  });
  s.addTable([head_].concat(body), {
    x: box.x, y: box.y, w: box.w, colW: opt.colW,
    rowH: [headH].concat(rows.map(function () { return bodyH; })),
    margin: [4, 10, 4, 10], autoPage: false,
  });
  grp(1);
}

/** `.tk-quote` - the sky bar down the left, and the sentence. */
function quote(s, text, box, opt) {
  opt = opt || {};
  s.addShape(pres.ShapeType.roundRect, {
    x: box.x, y: box.y, w: box.w, h: box.h, rectRadius: 0.08,
    fill: { color: opt.bar || P.skyB }, line: { type: "none" },
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: box.x + 0.09, y: box.y, w: box.w - 0.09, h: box.h, rectRadius: 0.08,
    fill: { color: opt.fill || P.surface }, line: { type: "none" },
  });
  const fs_ = opt.fontSize || 14.5;
  s.addText(rt(text, { fontFace: BODY, fontSize: fs_, color: P.ink }), {
    x: box.x + 0.36, y: box.y + 0.18, w: box.w - 0.72, h: box.h - 0.36, valign: "middle",
    lineSpacing: fs_ * 1.5, paraSpaceBefore: 8,
  });
  grp(3);
}

/** `.tk-note` - the small grey paragraph under a block. */
function note(s, text, box) {
  s.addText(rt(text, { fontFace: BODY, fontSize: box.fontSize || 10, color: P.ink3 }), {
    x: box.x, y: box.y, w: box.w, h: box.h, valign: "top",
    lineSpacing: (box.fontSize || 10) * 1.4,
  });
  grp(1);
}

/** A tinted panel behind a block, for `tint=` on the site. */
function tintPanel(s, colour, box) {
  s.addShape(pres.ShapeType.roundRect, {
    x: box.x, y: box.y, w: box.w, h: box.h, rectRadius: 0.11,
    fill: { color: colour }, line: { type: "none" },
  });
  stat(1);
}

/** A diagram box: title, the setting lines, then the result lines. */
function dbox(s, b) {
  s.addShape(pres.ShapeType.roundRect, {
    x: b.x, y: b.y, w: b.w, h: b.h, rectRadius: 0.08,
    fill: { color: P.surface }, line: { color: P.line, width: 0.75 },
  });
  const runs = rt(b.title, { fontFace: BODY, fontSize: 11, bold: true, color: P.ink,
                             align: "center" });
  runs[runs.length - 1].options.breakLine = true;
  const mid = [];
  (b.l1 || []).forEach(function (t) {
    mid.push.apply(mid, rt(t, { fontFace: BODY, fontSize: 9.5, color: P.ink2,
                                align: "center", breakLine: true }));
  });
  (b.l2 || []).forEach(function (t) {
    mid.push.apply(mid, rt(t, { fontFace: BODY, fontSize: 8.5, color: P.ink3,
                                align: "center", breakLine: true }));
  });
  s.addText(runs.concat(mid), {
    x: b.x + 0.06, y: b.y + 0.06, w: b.w - 0.12, h: b.h - 0.12, valign: "middle",
    align: "center", lineSpacing: 13,
  });
  grp(2);
}

/** The arrow between two diagram boxes. */
function arrow(s, x1, y1, x2, y2, colour) {
  // A shape's extent cannot be negative, so an arrow that runs up or left is drawn in a
  // positive box and mirrored into place. PowerPoint refuses to open a file with a
  // negative `cy` at all - it does not repair it.
  const w = x2 - x1, h = y2 - y1;
  s.addShape(pres.ShapeType.line, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(w), h: Math.abs(h),
    flipH: w < 0, flipV: h < 0,
    line: { color: colour || P.ink3, width: 1.25, endArrowType: "triangle" },
  });
  grp(1);   // its own beat, so the groups stay aligned with the shapes after it
}

/** The dashed or solid region drawn around a run of diagram boxes. */
function region(s, box, colour, dashed, label) {
  s.addShape(pres.ShapeType.roundRect, {
    x: box.x, y: box.y, w: box.w, h: box.h, rectRadius: 0.06,
    fill: { type: "none" },
    line: { color: colour, width: dashed ? 1 : 1.5, dashType: dashed ? "dash" : "solid" },
  });
  s.addText(rt(label, { fontFace: BODY, fontSize: 9.5, color: colour }), {
    x: box.x + 0.12, y: box.y - 0.28, w: box.w - 0.24, h: 0.26, valign: "middle",
  });
  grp(2);
}

// =========================================================================================
// The slides. One per block of the site, in the site's order, with the site's words.
// The only slide that is not on the site is the first one.
// =========================================================================================

const TITLE = "Playground Activity Detection using Deep Learning";
const LEAD = "A CPU-only system that detects activity and verifies booked pitch usage.";

// --- title page --------------------------------------------------------------------------
{
  const s = newSlide("hero");
  hero(s, "ink", "Master's thesis · Defence", TITLE, LEAD, {
    y: 1.82, titleSize: 40, titleW: 11.3, leadSize: 17, leadW: 10.4,
  });
  s.addShape(pres.ShapeType.rect, {
    x: M + 0.35, y: 4.92, w: 3.4, h: 0.022, fill: { color: "4E657C" }, line: { type: "none" },
  });
  s.addText([
    { text: "Fady Khazaka", options: { fontFace: BODY, fontSize: 20, bold: true,
                                       color: "FFFFFF", breakLine: true } },
    { text: "Supervisor: Dr. Mohammad Khalil", options: { fontFace: BODY, fontSize: 14.5,
                                                          color: "C6D3DF" } },
  ], { x: M + 0.35, y: 5.18, w: 8.0, h: 0.95, valign: "top", lineSpacing: 27 });
  grp(2);
}

// =========================== 1. Summary ==================================================
{
  const s = newSlide("content");
  head(s, "ink", "Summary", "The abstract");
  quote(s, "I developed this project after noticing that most AI solutions for football "
    + "pitches focus on athletes, not business owners. Our system uses deep learning to "
    + "automatically verify whether booked pitches were actually used, giving owners "
    + "greater trust and peace of mind without relying on human monitoring. The system "
    + "takes the CCTV camera stream and analyzes one picture every minute. Each picture is "
    + "classified as Active Play, Empty, Maintenance, or Not Playing, so the owner "
    + "doesn’t need to constantly check the cameras to know what is happening on the "
    + "pitch.<br><br>It runs on a CPU-only server, so it is affordable and deployable in "
    + "the real world.",
    { x: M, y: 1.58, w: CW, h: 2.28 }, { fontSize: 12.5 });
  cards(s, [
    ["The owner&rsquo;s problem",
     "For example: Seven pitches, one person, and no way to watch them all. Checking by eye "
     + "is a sample, not an audit."],
    ["The market",
     "~<b>1,200</b> football playgrounds in Lebanon &mdash; the same problem, multiplied."],
    ["Booking &ne; attendance",
     "An online booking says a slot was <i>sold</i>. It does not say anyone came."],
    ["Affordable by design",
     "CPU servers rather than GPUs or cloud inference. The constraint is what makes it "
     + "deployable, not a limitation to work around."],
  ], { x: M, y: 4.04, w: CW, h: 2.62 }, { cols: 4, titleSize: 12.5, bodySize: 10 });
}

{
  const s = newSlide("content");
  head(s, "ink", "Summary", "The study in numbers");
  const items = [
    [COV.frames || "?", "labelled frames", "lead"],
    ["3", "classes", "good"],
    ["7", "venue folds", "warn"],
    ["0", "GPUs &mdash; and no cloud services either", "bad"],
  ];
  const inner = { x: M + 0.34, w: CW - 0.68 };
  const th = tilesH(items, (inner.w - GAP * 3) / 4 - 0.36);
  const py = centre({ y: TOP, h: CH }, th + 0.68);
  tintPanel(s, P.skyS, { x: M, y: py, w: CW, h: th + 0.68 });
  tiles(s, items, { x: inner.x, y: py + 0.34, w: inner.w, h: th });
}

{
  const s = newSlide("content");
  head(s, "ink", "Summary", "Three findings");
  cards(s, [
    ["A high score does not always mean the model understands the pitch",
     "For example, the model learned that dark images mean playing and bright images mean "
     + "empty. So, it learned the lighting instead of the real activity."],
    ["When we tested a new video, we found a bigger problem.",
     "The original dataset did not contain enough different situations. So the model could "
     + "work well on the original data, but not always on new data."],
    ["Data augmentation did not always improve the model",
     "One experiment gave us <b>85.5%</b>, but when repeated the same experiment, the "
     + "results dropped to <b>34.8%&ndash;41.4%</b>. This showed us that the high score was "
     + "not stable, it was a random result."],
  ], { x: M, y: 2.0, w: CW, h: 3.6 }, { cols: 3, titleSize: 14, bodySize: 11.5 });
}

// =========================== 2. Plan =====================================================
{
  const s = newSlide("hero");
  hero(s, "sky", "Plan", "How this report is organised",
    "Eight parts: a summary, this plan, an introduction, four chapters, and a conclusion.");
}

const PARTS = [
  ["Summary", "The abstract, the scale of the study, and the three findings."],
  ["Plan", "This page."],
  ["Introduction",
   "The problem in general, who has worked on it, and how we resolve it."],
  ["Chapter 1 &mdash; State of the art",
   "Playgrounds: how facilities are monitored today and why each alternative fails. Deep "
   + "learning: what the field has established, and the gap."],
  ["Chapter 2 &mdash; Deep learning methods and our model",
   "The families considered, why the backbones are frozen, the architecture, and how it is "
   + "evaluated."],
  ["Chapter 3 &mdash; YOLOv8",
   "Why a detector at all, the candidates tested, the bake-off that chose one, what it "
   + "cannot see, and how it is wired to the classifier of Chapter 2: a motion check "
   + "between two frames and a head count inside the pitch boundary, each able to overturn "
   + "a match-in-progress verdict but never to create one."],
  ["Chapter 4 &mdash; Applications and data augmentation",
   "The deployed application, preprocessing, the augmentation presets, the retraction, and "
   + "the data generated with AI."],
  ["Conclusion and perspectives",
   "What was achieved, what it cannot claim, and what would change that."],
];
[0, 4].forEach(function (from) {
  const s = newSlide("content");
  head(s, "sky", "Plan", "The eight parts");
  steps(s, PARTS.slice(from, from + 4), { x: M, y: 1.62, w: CW, h: 5.12 },
    { from: from, bodySize: 11 });
});

// =========================== 3. Introduction =============================================
{
  const s = newSlide("hero");
  hero(s, "green", "The main idea of the project is:",
    "The booking says an hour was sold. Only the camera knows if anyone came.",
    "What is really being sold here is <b>trust</b>, and today this information can depend "
    + "on a paper or a manual record made by employees. The problem is that nobody can "
    + "check this information all the time.",
    { titleSize: 34, titleW: 10.9, leadSize: 15.5, leadW: 10.2 });
}

{
  const s = newSlide("content");
  head(s, "green", "Introduction", "Our system tries to answer simple questions:");
  const bottom = cards(s, [
    ["Was this hour booked?",
     "The system shows the booking, but not what actually happened on the pitch."],
    ["Did they actually come?", "A customer may book the pitch but never show up."],
    ["Did they actually play?",
     "People may be on the pitch without actually playing a game."],
    ["Is the pitch empty right now?",
     "The owner cannot know unless someone checks the pitch or the camera."],
    ["Did someone play without booking?",
     "Someone may use the pitch without a booking, so the owner could lose money."],
    ["Did staff record it correctly?",
     "Manual records can be entered incorrectly or forgotten."],
  ], { x: M, y: TOP, w: CW, h: CH - 0.5 }, { cols: 3, titleSize: 13.5, bodySize: 11 });
  note(s, "The CCTV camera gives us real visual evidence.",
    { x: M, y: bottom + 0.3, w: CW, h: 0.4, fontSize: 11 });
}

// =========================== 4. Chapter 1 ================================================
{
  const s = newSlide("hero");
  hero(s, "violet", "Chapter 1 &mdash; State of the art", "Playgrounds, and deep learning",
    "There are already different ways to monitor activity. But each solution has "
    + "limitations.");
}

{
  const s = newSlide("content");
  head(s, "violet", "Chapter 1 — State of the art",
    "Playgrounds &mdash; how occupancy is measured today");
  table(s,
    ["Approach", "Cost per pitch", "What it measures", "How it fails"],
    [["PIR / motion sensor", "~&euro;20", "Detect movements",
      "Gave us false alarm, Rain, wind-blown netting, foxes, staff crossing &mdash; and no "
      + "evidence trail"],
     ["Door counter / turnstile", "&euro;300&ndash;2,000", "Count people entering",
      "Cannot tell us if they actually played, nobody counts <i>out</i>"],
     ["Floodlight power draw", "~&euro;60", "Lights on",
      "Daylight slots invisible; lights left on; shared circuits"],
     ["Mobile App check-in", "~&euro;0", "ask customers to check in",
      "It depends on the customer, and they may not do it"],
     ["Manual logging", "Staff time", "What staff wrote down",
      "<b>It is one of the three records being audited</b>"],
     ["<b>This system</b>", "<b>Reuses existing CCTV</b>",
      "<b>Give us real images using CCTV, with evidence</b>",
      "<b>Needs a camera view; classification error</b>"]],
    { x: M, y: 1.66, w: CW, h: 5.0 },
    { hi: 5, colW: [2.1, 1.55, 2.95, 5.633], fontSize: 11 });
}

const STRANDS = [
  ["&sect;2.1 Occupancy and activity recognition from fixed cameras",
   "That classifying scene state from a fixed viewpoint is a solved problem class, and what "
   + "accuracy the field considers ordinary."],
  ["&sect;2.2 Frozen features and linear probes",
   "That a frozen backbone with a small trained head is a legitimate method rather than a "
   + "shortcut."],
  ["&sect;2.3 Dataset leakage and evaluation protocol",
   "That near-duplicate leakage is a known, recurring problem, with prior cases where a "
   + "benchmark was found to be measuring memorisation."],
  ["&sect;2.4 Trivial baselines and benchmark validity",
   "That checking a benchmark against a baseline which ignores the input is established "
   + "practice."],
  ["&sect;2.5 Edge and CPU-constrained inference",
   "What is achievable without a GPU, and at what accuracy cost."],
  ["&sect;2.6 Calibration and uncertainty for deployed classifiers",
   "That a confidence score is an operational quantity once it routes work to a person, and "
   + "how calibration is normally assessed."],
  ["&sect;2.7 Facility management, audit and record reconciliation",
   "That cross-checking a sensor against an administrative record is a recognised problem, "
   + "in this domain or an adjacent one."],
  ["&sect;2.8 Prior art &mdash; who already does this",
   "Whether a commercial or academic system already audits facility occupancy, and what it "
   + "does not do."],
  ["&sect;2.9 Selective prediction and learning to defer",
   "That abstaining and handing a case to a human is a studied design with its own metrics "
   + "&mdash; so the REVIEW band is literature-backed, not an engineering convenience."],
];
// All nine at once, as the page shows them. Split over three slides they read as three
// separate claims; the point of the block is that there are nine strands and the chapter
// has to answer every one of them, which is only visible when they are all on the screen.
{
  const s = newSlide("content");
  head(s, "violet", "Chapter 1 — State of the art",
    "Deep learning &mdash; what the field must establish");
  cards(s, STRANDS, { x: M, y: TOP, w: CW, h: CH },
    { cols: 3, titleSize: 12.5, bodySize: 10.5 });
}

// =========================== 5. Chapter 2 ================================================
{
  const s = newSlide("hero");
  hero(s, "ink", "Chapter 2 &mdash; Deep learning methods",
    "The families considered, and the model we built",
    "Three frozen backbones, a classifier small enough to read on one screen, and the "
    + "reason it is that way round.");
}

{
  const s = newSlide("content");
  head(s, "ink", "Chapter 2 — Deep learning methods",
    "We considered four main AI models");
  cards(s, [
    ["Convolutional networks",
     "ConvNeXtV2 &mdash; a modern CNN, supervised and masked-autoencoder pretraining on "
     + "ImageNet-22k."],
    ["Vision transformers", "ViT-Base &mdash; supervised ImageNet-21k, fine-tuned to 1k."],
    ["Self-supervised transformers",
     "DINOv2 &mdash; self-supervised on LVD-142M, no labels at all."],
    ["Vision-language models",
     "CLIP / OpenCLIP / SigLIP, used <b>zero-shot</b> with written class descriptions."],
  ], { x: M, y: 2.15, w: CW, h: 3.2 }, { cols: 4, titleSize: 13, bodySize: 11 });
}

{
  const s = newSlide("content");
  head(s, "ink", "Chapter 2 — Deep learning methods", "Why the Backbones Are Frozen");
  const FROZEN = [
    ["Fine-Tuning Needs Variety We Do Not Have",
     "We did not fine-tune all <b>200 million parameters</b> because we only have "
     + "<b>~150 different scenes</b>. With so little data, the model could memorize the "
     + "pictures instead of learning the game, causing <b>overfitting</b>."],
    ["More Data Is Not Easy to Collect",
     "The videos show <b>real people</b>, so collecting footage from new venues requires "
     + "<b>permission</b>."],
    ["Our Numbers Show the Risk",
     "<b>98.5%</b> of frames are very similar, and accuracy <b>dropped</b> when we "
     + "increased the data from <b>300 to 671 samples</b>."],
  ];
  const inner = { x: M + 0.3, w: CW - 0.6 };
  const ch_ = cardsH(FROZEN, (inner.w - GAP * 2) / 3 - 0.5, 13.5, 11.5);
  const py = centre({ y: TOP, h: CH - 0.7 }, ch_ + 0.6);
  tintPanel(s, P.greenS, { x: M, y: py, w: CW, h: ch_ + 0.6 });
  cards(s, FROZEN, { x: inner.x, y: py + 0.3, w: inner.w, h: ch_ },
    { cols: 3, titleSize: 13.5, bodySize: 11.5 });
  note(s, "Freezing is a <b>defence against overfitting</b> first, and an efficiency win "
    + "second.", { x: M, y: py + ch_ + 0.82, w: CW, h: 0.5, fontSize: 11 });
}

// --- the DINOv2 drawing, rebuilt as slide shapes ------------------------------------------
{
  const s = newSlide("content");
  head(s, "ink", "Chapter 2 — Deep learning methods", "DINOv2 &mdash; what actually runs");
  const dino = (INV.frozen || []).filter(function (b) { return b.key === "dinov2"; })[0];
  const probe = (INV.trained || {}).probe;
  if (!dino || !probe) {
    note(s, "No <code>results/model_inventory.json</code> yet. Run "
      + "<code>uv run python experiments/model_inventory.py</code>.",
      { x: M, y: 3.0, w: CW, h: 0.6, fontSize: 13 });
  } else {
    const a = dino.architecture;
    const patches = Math.pow(224 / a.patch | 0, 2);
    const ratio = Math.floor(dino.params / probe.params);
    const n = function (v) { return v.toLocaleString("en-US"); };
    const BOXES = [
      [118, "the frame", ["224×224, inside"], ["the pitch boundary"]],
      [150, "patch embedding", [a.patch + "×" + a.patch + " patches"],
       [patches + " tokens + CLS"]],
      [176, "transformer × " + a.layers,
       [a.heads + " heads, width " + a.hidden], ["every patch sees every other"]],
      [124, "mean over tokens", ["one " + a.output_dim + "-number"], ["description"]],
      [156, "linear probe", ["768 × 3 + 3 biases"], ["the only thing trained"]],
    ];
    const total = BOXES.reduce(function (t, b) { return t + b[0]; }, 0);
    const gap = 0.34, scale = (CW - gap * (BOXES.length - 1)) / total;
    const by = 2.72, bh = 1.42;
    const xs = []; let x = M;
    BOXES.forEach(function (b) { xs.push(x); x += b[0] * scale + gap; });
    const w_ = function (i) { return BOXES[i][0] * scale; };

    region(s, { x: xs[1] - 0.12, y: by - 0.38, w: xs[3] + w_(3) + 0.12 - (xs[1] - 0.12),
                h: bh + 0.76 }, P.ink3, true,
      "frozen · " + n(dino.params) + " parameters · "
      + (INV.frozen_gradients_received || 0) + " gradients received");
    region(s, { x: xs[4] - 0.12, y: by - 0.38, w: w_(4) + 0.24, h: bh + 0.76 },
      P.accent, false, "trained · " + n(probe.params));
    BOXES.forEach(function (b, i) {
      dbox(s, { x: xs[i], y: by, w: w_(i), h: bh, title: b[1], l1: b[2], l2: b[3] });
      if (i < BOXES.length - 1) {
        arrow(s, xs[i] + w_(i) + 0.05, by + bh / 2, xs[i + 1] - 0.05, by + bh / 2);
      }
    });
    const cx = xs[4] + w_(4) / 2;
    arrow(s, cx, by + bh + 0.06, cx, by + bh + 0.5, P.accent);
    const CHIPS = ["EMPTY", "ACTIVE PLAY", "MAINTENANCE"];
    const cw = 1.72, cgap = 0.14;
    const cx0 = xs[4] + w_(4) - (cw * 3 + cgap * 2);
    CHIPS.forEach(function (t, i) {
      const cxx = cx0 + i * (cw + cgap);
      s.addShape(pres.ShapeType.roundRect, {
        x: cxx, y: by + bh + 0.56, w: cw, h: 0.36, rectRadius: 0.18,
        fill: { color: P.surface }, line: { color: P.line, width: 0.75 },
      });
      s.addText(t, { x: cxx, y: by + bh + 0.56, w: cw, h: 0.36, fontFace: MONO,
                     fontSize: 9.5, color: P.ink2, align: "center", valign: "middle" });
    });
    grp(6);
    s.addText(n(ratio) + " frozen parameters per trained one", {
      x: M, y: by + bh + 0.56, w: 4.6, h: 0.36, fontFace: BODY, fontSize: 11.5,
      color: P.ink2, valign: "middle",
    });
    grp(1);
    note(s, "DINOv2 is pretrained on " + dino.pretraining.split(" on ").pop() + " with "
      + dino.pretraining_labels + ", so it has no notion of the three classes and no "
      + "classifier: what comes out is a description of the image. Everything inside the "
      + "dashed region is used exactly as downloaded and receives no gradient at any point "
      + "in this project. The whole of what is fitted is the box on the right.",
      { x: M, y: by + bh + 1.18, w: CW, h: 1.3, fontSize: 11.5 });
  }
}


// --- the full model comparison -------------------------------------------------------------
// `models_view.render()`, which the site appends to Chapter 2 under "The full model
// comparison". Built around the decision rather than the numbers: every model that was
// tried has a row whether or not it has been evaluated, so a blank reads as "not measured"
// and not as a stale figure. Read from the same five CSVs the view reads.
{
  const LABEL = { dinov2: "DINOv2", convnextv2: "ConvNeXtV2", vit: "ViT",
                  clock_rule: "Clock rule" };
  const NATURE = {
    dinov2: "self-supervised, no labels in pretraining",
    convnextv2: "self-supervised convnet, smallest and fastest",
    vit: "supervised on ImageNet labels",
    clock_rule: "reads the clock, never the pixels",
  };
  const mean = function (name, key) {
    const out = {};
    csvRows(name).forEach(function (r) {
      if (r.held_out_venue === "MEAN_ACROSS_FOLDS") out[r.model] = r;
    });
    return out;
  };
  const floor_ = csvRows("h1_h2_baseline_floor.csv");
  const bySplit = function (prefix) {
    const out = {};
    floor_.forEach(function (r) {
      if ((r.split || "").indexOf(prefix) === 0) out[r.model] = r;
    });
    return out;
  };
  const randomSplit = bySplit("random"), groupedSplit = bySplit("grouped");
  const cross = mean("h3_cross_venue_recall.csv");
  const fp = mean("h3_with_false_play.csv");
  const lat = {};
  csvRows("efficiency_latency.csv").forEach(function (r) { lat[r.backbone] = r; });

  const num = function (v, d) {
    const f = parseFloat(v);
    return isFinite(f) ? f.toFixed(d === undefined ? 3 : d) : "—";
  };
  const ROWS = ["dinov2", "convnextv2", "vit", "clock_rule"].map(function (k) {
    const c = cross[k] || {}, f = fp[k] || {};
    const bal = (c.play_recall && f.false_play_rate !== undefined && f.false_play_rate !== "")
      ? parseFloat(c.play_recall) - parseFloat(f.false_play_rate) : null;
    return {
      key: k,
      random: (randomSplit[k] || {}).macro_f1,
      grouped: (groupedSplit[k] || {}).macro_f1,
      cross: c.play_recall, worst: c.worst_fold,
      false_play: f.false_play_rate, balanced: bal,
      ms: (lat[k] || {}).single_median_ms,
    };
  });

  const s = newSlide("content");
  head(s, "ink", "Chapter 2 — Deep learning methods", "The full model comparison");

  const COLS = [
    ["Random split", "macro-F1, frames shuffled &mdash; the leaky protocol", "up"],
    ["Grouped split", "macro-F1, no venue in both train and test", "up"],
    ["Cross-venue recall", "playing frames found at an unseen venue", "up"],
    ["Worst fold", "the lowest single fold behind that recall", "up"],
    ["False-play", "held-out empty pitches called a match", "down"],
    ["Balanced", "recall minus false-play", "up"],
    ["ms/frame", "time to score one frame", "down"],
  ];
  const noLine = { type: "none" };
  const border = [noLine, noLine, { pt: 0.75, color: P.line }, noLine];
  const header = [{
    text: "MODEL",
    options: { fontFace: MONO, fontSize: 9, bold: true, color: P.ink3, charSpacing: 1.2,
               fill: { color: P.surface2 }, valign: "bottom", border: border },
  }].concat(COLS.map(function (c) {
    return {
      text: [
        { text: plain(c[0]), options: { fontFace: BODY, fontSize: 9.5, bold: true,
                                        color: P.ink, breakLine: true } },
        { text: plain(c[1]), options: { fontFace: BODY, fontSize: 7.5, color: P.ink3,
                                        breakLine: true } },
        { text: (c[2] === "up" ? "↑ higher" : "↓ lower") + " is better",
          options: { fontFace: MONO, fontSize: 7, bold: true,
                     color: c[2] === "up" ? P.green : P.coral } },
      ],
      options: { fill: { color: P.surface2 }, valign: "bottom", align: "right",
                 border: border },
    };
  }));

  const body = ROWS.map(function (r) {
    const bad = r.false_play !== undefined && parseFloat(r.false_play) > 0.5;
    const cell = function (v, d, opt) {
      return {
        text: num(v, d),
        options: Object.assign({ fontFace: MONO, fontSize: 11.5, color: P.ink2,
                                 align: "right", valign: "middle",
                                 fill: { color: P.surface }, border: border }, opt || {}),
      };
    };
    return [{
      text: [
        { text: LABEL[r.key], options: { fontFace: BODY, fontSize: 12.5, bold: true,
                                         color: P.ink, breakLine: true } },
        { text: ents(NATURE[r.key]), options: { fontFace: BODY, fontSize: 9,
                                                color: P.ink3 } },
      ],
      options: { valign: "middle", fill: { color: P.surface }, border: border },
    },
      cell(r.random), cell(r.grouped), cell(r.cross), cell(r.worst),
      cell(r.false_play, 3, bad ? { color: P.amber, bold: true } : {}),
      cell(r.balanced), cell(r.ms, 0)];
  });

  const headH = 0.98, rowH = 0.62;
  const y0 = centre({ y: TOP, h: CH - 0.9 }, headH + ROWS.length * rowH);
  const first = 2.55, rest = (CW - first) / COLS.length;
  s.addTable([header].concat(body), {
    x: M, y: y0, w: CW,
    colW: [first].concat(COLS.map(function () { return rest; })),
    rowH: [headH].concat(ROWS.map(function () { return rowH; })),
    margin: [4, 9, 4, 9], autoPage: false,
  });
  grp(1);
  note(s, "A dash is a result that has not been measured, not a zero. Cross-venue recall is "
    + "measured on folds that are 100% ACTIVE_PLAY, so it can be earned by answering "
    + "&ldquo;playing&rdquo; to everything &mdash; which is what the false-play column is "
    + "there to catch, and why the ranking it gives is not the ranking the grouped split "
    + "gives.",
    { x: M, y: y0 + headH + ROWS.length * rowH + 0.3, w: CW, h: 0.7, fontSize: 10.5 });
}

// =========================== 6. Chapter 3 ================================================
{
  const s = newSlide("hero");
  hero(s, "amber", "Chapter 3 &mdash; YOLOv8", "Counting what stands on the pitch",
    "The classifier says what a scene looks like. The detector says how many people are on "
    + "it, whether there is a ball, and whether anything moved &mdash; and those counts can "
    + "overrule the classifier.");
}

// --- the YOLOv8 drawing, rebuilt as slide shapes ------------------------------------------
{
  const s = newSlide("content");
  head(s, "amber", "Chapter 3 — YOLOv8",
    "The architecture &mdash; one forward pass");
  const STAGES = [
    [136, "the frame", ["long edge " + (RULES.imgsz || 1280)], ["one per camera-minute"]],
    [204, "backbone (Features &amp; layers)",
     ["CSPDarknet (architecture),", "C2f blocks", "(features representation)"],
     ["features at 3 scales (s,m,l)"]],
    [164, "neck", ["PAN-FPN"], ["small objects keep detail"]],
    [190, "head", ["anchor-free (location),", "decoupled", "(where is it?, what is it?)"],
     ["box + class, no anchors"]],
    [124, "NMS", ["person ≥ " + (RULES.person_conf != null ? RULES.person_conf : 0.25)],
     ["ball ≥ " + (RULES.ball_conf != null ? RULES.ball_conf : 0.1)]],
  ];
  const total = STAGES.reduce(function (t, b) { return t + b[0]; }, 0);
  const gap = 0.30, scale = (CW - gap * (STAGES.length - 1)) / total;
  const by = 2.44, bh = 1.62;
  const xs = []; let x = M;
  STAGES.forEach(function (b) { xs.push(x); x += b[0] * scale + gap; });
  const w_ = function (i) { return STAGES[i][0] * scale; };

  region(s, { x: xs[1] - 0.12, y: by - 0.38, w: xs[3] + w_(3) + 0.12 - (xs[1] - 0.12),
              h: bh + 0.76 }, P.ink3, true,
    "one forward pass · COCO weights, nothing trained here · "
    + (RULES.detector || "yolov8n"));
  region(s, { x: xs[4] - 0.12, y: by - 0.38, w: w_(4) + 0.24, h: bh + 0.76 },
    P.accent, false, "boxes out");
  STAGES.forEach(function (b, i) {
    dbox(s, { x: xs[i], y: by, w: w_(i), h: bh, title: b[1], l1: b[2], l2: b[3] });
    if (i < STAGES.length - 1) {
      arrow(s, xs[i] + w_(i) + 0.05, by + bh / 2, xs[i + 1] - 0.05, by + bh / 2);
    }
  });
  const cx = xs[4] + w_(4) / 2;
  arrow(s, cx, by + bh + 0.06, cx, by + bh + 0.52, P.accent);
  s.addText(rt("then the rules: foot inside the boundary, "
    + (RULES.small_group_max != null ? RULES.small_group_max : 4)
    + " or fewer is not a game, " + (RULES.play_min != null ? RULES.play_min : 5)
    + "+ with a moving ball is",
    { fontFace: BODY, fontSize: 11.5, color: P.accent }), {
    x: M, y: by + bh + 0.42, w: cx - M - 0.2, h: 0.5, align: "right", valign: "middle",
  });
  grp(1);
  note(s, "The detector ends at boxes. It is asked for two COCO classes only &mdash; "
    + "person and sports ball &mdash; and it is never asked whether a match is happening: "
    + "that is decided afterwards by counting, which is why a verdict can be shown to a "
    + "manager as the objects it was made from.",
    { x: M, y: by + bh + 1.26, w: CW, h: 1.3, fontSize: 11.5 });
}

{
  const s = newSlide("content");
  head(s, "amber", "Chapter 3 — YOLOv8", "The three cases");
  steps(s, [
    ["People &mdash; How many people are on the pitch",
     "We count the people inside the pitch.<br><b>0</b> people = empty<br>A few people may "
     + "be staff or non-playing people.<br>Many people usually means a game is happening."],
    ["Motion &mdash; Is something moving?",
     "We compare the current frame with the previous frame.<br>It tells us if something is "
     + "moving.<br>It does not tell us who is moving."],
    ["Ball &mdash; Is there a ball being played?",
     "The ball must appear in at least <b>2 of 3</b> frames.<br>This helps avoid false "
     + "detections.<br>The ball must also be moving.<br>A ball that is just lying on the "
     + "pitch does not mean a game is happening."],
  ], { x: M, y: 1.62, w: CW, h: 5.12 }, { titleSize: 14, bodySize: 11 });
}

{
  const s = newSlide("content");
  head(s, "amber", "Chapter 3 — YOLOv8",
    "From boxes to a verdict &mdash; the rule table");
  const small = RULES.small_group_max != null ? RULES.small_group_max : 4;
  const play = RULES.play_min != null ? RULES.play_min : 5;
  table(s, ["#", "Condition", "Verdict"], [
    ["1", "Detector unavailable", "Uncertain &mdash; a missing detector is not an empty pitch"],
    ["2", "No pitch boundary", "Uncertain &mdash; mandatory on the deployed path"],
    ["3", "Nobody, nothing moving", "<b>EMPTY</b>"],
    ["4", "Nobody, but something moved", "Uncertain"],
    ["5", "1&ndash;" + small + " people", "Not a game &mdash; too few, ball or no ball"],
    ["6", play + "+ people, <b>and</b> a ball in play, <b>and</b> motion",
     "<b>ACTIVE PLAY</b> &mdash; the only way in"],
    ["7", play + "+ people, otherwise", "Not a game &mdash; a crowd that is not playing"],
  ], { x: M, y: 1.62, w: CW, h: 4.2 }, { hi: 5, colW: [0.6, 5.0, 6.633], fontSize: 11.5 });
  note(s, "<b>" + play + "</b> and <b>" + small + "</b> are the <b>facility&rsquo;s</b> "
    + "numbers, not fitted ones: five make a game, four or fewer do not. Every threshold "
    + "that <i>was</i> fitted is recorded in <code>configs/rules.json</code> with the "
    + "camera it was fitted on, and a cue that cannot be measured is reported as "
    + "unmeasured, never assumed.",
    { x: M, y: 5.98, w: CW, h: 0.9, fontSize: 10.5 });
}

// =========================== 7. Chapter 4 ================================================
{
  const s = newSlide("hero");
  hero(s, "sky", "Chapter 4 &mdash; Applications and data augmentation",
    "What it does in practice, and what we did about the data",
    "The preprocessing path and the augmentation argument, in pictures.");
}

// --- the six before/after pairs -----------------------------------------------------------
const SHEET = [
  ["letterbox=False", "Pad to a square instead of squashing."],
  ["top_crop=0.35", "Cut the sky and stands off the top."],
  ["centre_crop=0.5", "Keep the middle, discard the border."],
  ["gamma=0.7", "Brightness curve."],
  ["saturation=0.0", "Colour strength &mdash; 0.0 is grayscale."],
  ["blur_sigma=4.0", "Blur &mdash; destroys fine detail, including people."],
];
{
  const rows = csvRows("preprocess_pairs.csv");
  const by = {};
  rows.forEach(function (r) { by[r.label] = r; });
  const have = SHEET.filter(function (p) { return by[p[0]]; });
  if (!have.length) {
    const s = newSlide("content");
    head(s, "sky", "Chapter 4 — Applications and data augmentation",
      "Preprocessing &mdash; six switches, before and after");
    note(s, "No <code>results/preprocess_pairs.csv</code> yet. Run "
      + "<code>uv run python experiments/preprocess_pairs.py</code>.",
      { x: M, y: 3.0, w: CW, h: 0.6, fontSize: 13 });
  }
  // All six on one sheet, three across and two down. They are one argument - that a switch
  // which passes every shape and dtype check can still have removed the people - and the
  // argument is made by seeing them beside each other.
  if (have.length) {
    const s = newSlide("content");
    head(s, "sky", "Chapter 4 — Applications and data augmentation",
      "Preprocessing &mdash; six switches, before and after");
    const cols = 3, nrows = Math.ceil(have.length / cols);
    const band = { y: 1.46, h: 7.02 - 1.46 };
    const cw = (CW - GAP * (cols - 1)) / cols;
    const ch_ = (band.h - 0.14 * (nrows - 1)) / nrows;
    // The image is sized to the height its row has left after the caption, not to the
    // card's width: six pairs stacked two deep is a height budget, and a width-fitted
    // image overflows it.
    const capH = 0.62;
    const ih = Math.min(ch_ - 0.30 - capH, (cw - 0.28) * 248 / 458);
    const iw = ih * 458 / 248;
    have.forEach(function (p, k) {
      const r = by[p[0]];
      const x = M + (k % cols) * (cw + GAP);
      const y0 = band.y + Math.floor(k / cols) * (ch_ + 0.14);
      const area = parseFloat(r.area_retained);
      const crop = area < 1.0;
      s.addShape(pres.ShapeType.roundRect, {
        x: x, y: y0, w: cw, h: ch_, rectRadius: 0.09,
        fill: { color: crop ? P.coralB : P.line }, line: { type: "none" },
      });
      s.addShape(pres.ShapeType.roundRect, {
        x: x + (crop ? 0.06 : 0.008), y: y0, w: cw - (crop ? 0.06 : 0.008), h: ch_,
        rectRadius: 0.09, fill: { color: P.surface }, line: { type: "none" },
      });
      s.addImage({
        path: path.join(RESULTS, "figs", "preproc", path.basename(r.file)),
        x: x + (cw - iw) / 2, y: y0 + 0.12, w: iw, h: ih,
      });
      const pct = function (v) { return Math.round(parseFloat(v) * 100) + "%"; };
      const kept = crop ? " · " + pct(r.area_retained) + " kept" : "";
      s.addText([
        { text: p[0], options: { fontFace: MONO, fontSize: 10.5, bold: true, color: P.ink,
                                 breakLine: true } },
      ].concat(rt(p[1], { fontFace: BODY, fontSize: 9.5, color: P.ink3, breakLine: true }))
       .concat([{ text: parseFloat(r.mean_abs_change_255).toFixed(1) + "/255 moved · "
                        + pct(r.share_pixels_changed) + " of pixels" + kept,
                  options: { fontFace: MONO, fontSize: 8.5, color: P.ink3 } }]), {
        x: x + 0.22, y: y0 + 0.12 + ih + 0.06, w: cw - 0.44, h: capH, valign: "top",
        lineSpacing: 13, paraSpaceBefore: 2,
      });
      grp(4);
    });
  }
}



// --- the three sheets of `_augmentation()` ---------------------------------------------------
// The grids are the point of that block. Augmentation and preprocessing code fails
// silently - a preset that does nothing, a crop that removes the goalmouth - and every one
// of those passes a shape and dtype check. The only reliable check is a person looking, so
// the sheets go on the slides rather than being summarised into a sentence nobody can
// check.
//
// Two of them are contact sheets several times taller than a slide. They are **not**
// resampled or redrawn: the picture is placed more than once, each copy cropped to one
// band of the original with `srcRect` (see `CROP:` in `postprocess`), so the sheet is read
// left to right instead of top to bottom. Every row of the original is on the slide, in
// order, at a size a room can see.
function sheet(s, file, pieces, caption, opt) {
  opt = opt || {};
  const src = path.join(RESULTS, "figs", file);
  if (!fs.existsSync(src)) {
    note(s, "No <code>results/figs/" + file + "</code> yet. Run "
      + "<code>uv run python experiments/augmentation_grid.py</code>.",
      { x: M, y: 3.0, w: CW, h: 0.6, fontSize: 13 });
    return;
  }
  const gap = opt.gap === undefined ? 0.26 : opt.gap;
  const capH = opt.capH === undefined ? 0.72 : opt.capH;
  const top = opt.top === undefined ? TOP : opt.top;
  const band = { y: top, h: BOT - top - capH - 0.24 };
  // Each piece keeps the aspect ratio of its own slice of the original.
  const aspect = (opt.w / (opt.h / pieces));          // width / height of one piece
  let iw = (CW - gap * (pieces - 1)) / pieces;
  let ih = iw / aspect;
  if (ih > band.h) { ih = band.h; iw = ih * aspect; }
  const total = iw * pieces + gap * (pieces - 1);
  const x0 = (W - total) / 2;
  const y0 = band.y + (band.h - ih) / 2;
  for (let k = 0; k < pieces; k++) {
    s.addImage({
      path: src, x: x0 + k * (iw + gap), y: y0, w: iw, h: ih,
      altText: "CROP:" + Math.round(k * 100000 / pieces) + ","
               + Math.round((pieces - 1 - k) * 100000 / pieces),
    });
    grp(1);
  }
  note(s, caption, { x: M, y: y0 + ih + 0.24, w: CW, h: capH, fontSize: 10.5 });
}

// "What removal looks like" - the one sheet in that block that is about preprocessing, and
// the reason the block opens with it: it is what the floor looks like.
{
  const s = newSlide("content");
  head(s, "sky", "Chapter 4 — Applications and data augmentation",
    "What removal looks like");
  quote(s, "<b>Preprocessing removes information permanently; augmentation varies it and "
    + "keeps every pixel at inference.</b> That is why both exist, and why removal has a "
    + "floor that variation cannot cross.",
    { x: M, y: 1.52, w: CW, h: 0.80 }, { fontSize: 13 });
  {
    sheet(s, "preprocess_effects.jpg", 2,
      "Every preprocessing switch the search tries, at the real 224&times;224 the backbone "
      + "receives. Grey bands are letterbox padding. <code>blur_sigma=4.0</code> has "
      + "removed the people and <code>centre_crop=0.5</code> the goalmouth. "
      + "<code>roi</code> is absent: no pitch polygon drawn yet. <i>Left: the night "
      + "active-play frame. Right: the day empty frame.</i>",
      { w: 1344, h: 1578, top: 2.48 });
  }
}

// --- the augmentation argument ------------------------------------------------------------
// `_augmentation()` on the site: the claim, then `augmentation_axes()` under it. The
// drawing is the argument - the prose can only assert that removal and variation differ,
// and the asymmetry is visible only when the two rows are stacked.
{
  const s = newSlide("content");
  head(s, "sky", "Chapter 4 — Applications and data augmentation",
    "The augmentation argument");

  // Laid out from the same coordinates as the SVG (820x300), so the proportions are the
  // page's and only the type is resized for a projector.
  const S = 0.01465, X0 = M + 0.11, Y0 = 1.70;
  const px = function (v) { return X0 + v * S; };
  const py = function (v) { return Y0 + v * S; };

  function abox(x, y, w, h, title, sub, hot) {
    s.addShape(pres.ShapeType.roundRect, {
      x: px(x), y: py(y), w: w * S, h: h * S, rectRadius: 0.05,
      fill: { color: P.surface },
      line: { color: hot ? P.accent : P.line, width: hot ? 1.4 : 0.75 },
    });
    const runs = rt(title, { fontFace: BODY, fontSize: 10, bold: true, color: P.ink,
                             align: "center" });
    if (sub) {
      runs[runs.length - 1].options.breakLine = true;
      runs.push.apply(runs, rt(sub, { fontFace: BODY, fontSize: 8.5, color: P.ink3,
                                      align: "center" }));
    }
    s.addText(runs, { x: px(x), y: py(y), w: w * S, h: h * S, align: "center",
                      valign: "middle", lineSpacing: 12 });
    grp(2);
  }
  const aline = function (x1, y1, x2, y2) {
    arrow(s, px(x1), py(y1), px(x2), py(y2));
  };

  // preprocessing - a one-way chain that ends below where it started
  s.addText("preprocessing — one way, and it has a floor", {
    x: px(40), y: py(2), w: 6.0, h: 0.26, fontFace: BODY, fontSize: 11, bold: true,
    color: P.ink, valign: "middle",
  });
  grp(1);
  abox(40, 36, 124, 40, "source frame", "everything");
  aline(170, 56, 200, 56);
  abox(206, 36, 124, 40, "grayscale", "+0.022 &middot; fp 0.02");
  aline(336, 56, 366, 56);
  abox(372, 36, 150, 40, "gray + crop50", "&minus;0.061 &middot; fp 0.99");
  s.addText([
    { text: "below the untouched", options: { breakLine: true } },
    { text: "baseline — the floor" },
  ], { x: px(532), y: py(34), w: 3.0, h: 0.52, fontFace: BODY, fontSize: 9.5,
       color: P.amber, valign: "middle", lineSpacing: 13 });
  grp(1);

  s.addShape(pres.ShapeType.rect, {
    x: px(40), y: py(104), w: 744 * S, h: 0.012,
    fill: { color: P.line }, line: { type: "none" },
  });
  stat(1);

  // augmentation - a fan for training, and the source still reaches inference
  s.addText("augmentation — varies the input, discards nothing", {
    x: px(40), y: py(118), w: 6.4, h: 0.26, fontFace: BODY, fontSize: 11, bold: true,
    color: P.ink, valign: "middle",
  });
  grp(1);
  abox(40, 170, 124, 40, "source frame", "everything");
  aline(170, 186, 200, 162); aline(170, 190, 200, 190); aline(170, 194, 200, 218);
  [148, 178, 208].forEach(function (y, i) { abox(206, y, 124, 24, "view " + (i + 1), ""); });
  aline(336, 162, 366, 186); aline(336, 190, 366, 190); aline(336, 218, 366, 194);
  abox(372, 170, 150, 40, "probe training", "sees all three");
  abox(560, 170, 224, 40, "inference", "the source frame, unchanged", true);
  // the dashed return path: the source still reaches inference untouched
  s.addShape(pres.ShapeType.line, {
    x: px(102), y: py(214), w: 0, h: (262 - 214) * S,
    line: { color: P.ink3, width: 1, dashType: "dash" },
  });
  s.addShape(pres.ShapeType.line, {
    x: px(102), y: py(262), w: (672 - 102) * S, h: 0,
    line: { color: P.ink3, width: 1, dashType: "dash" },
  });
  s.addShape(pres.ShapeType.line, {
    x: px(672), y: py(216), w: 0, h: (262 - 216) * S, flipV: true,
    line: { color: P.ink3, width: 1, dashType: "dash", endArrowType: "triangle" },
  });
  s.addText("nothing was discarded, so nothing is missing here", {
    x: px(180), y: py(234), w: 4.6, h: 0.24, fontFace: BODY, fontSize: 9,
    color: P.ink3, align: "center", valign: "middle",
  });
  grp(4);

  note(s, "Each box carries recall and the false-play rate (<code>fp</code>): grayscale "
    + "gains on both axes, the crop&rsquo;s larger apparent gain is a variant answering "
    + "PLAY more often. Augmentation&rsquo;s fan exists only in training, so there is no "
    + "floor to cross.", { x: M, y: 6.26, w: CW, h: 0.6, fontSize: 10.5 });
}


// "The preset sheets" - every preset, three draws each, on a night play frame and a day
// empty frame. The sheet's own halves are the two frames, so two pieces cut it where it
// already divides.
{
  const s = newSlide("content");
  head(s, "sky", "Chapter 4 — Applications and data augmentation",
    "The preset sheets &mdash; every preset, three draws each");
  sheet(s, "augmentation_grid.jpg", 2,
    "Every preset, three draws each, on a night play frame and a day empty frame. "
    + "Probability forced to 1, so the sheet shows the effect and not the coin flip. "
    + "<i>Left: the night active-play frame. Right: the day empty frame &mdash; the two "
    + "halves of the one sheet.</i>",
    { w: 960, h: 2074 });
}

// The same sheet one effect at a time, at the magnitude `full` uses. Twenty bands, so four
// pieces of five.
{
  const s = newSlide("content");
  head(s, "sky", "Chapter 4 — Applications and data augmentation",
    "One effect at a time");
  sheet(s, "augmentation_effects.jpg", 4,
    "One effect at a time, at the magnitude <code>full</code> uses &mdash; brightness 0.2, "
    + "contrast 0.2, saturation 0.35, hue 6&deg;, gamma 0.3, noise &sigma;=5, fog 0.3, "
    + "rain 0.3, flip. Magnitudes are read off the preset. <code>flip</code> is shown once; "
    + "every other row is three draws. <i>The sheet is read left to right: four pieces of "
    + "five bands, in the order they appear on it.</i>",
    { w: 960, h: 4134, gap: 0.22 });
}

// --- the preprocessing search ---------------------------------------------------------------
// The site's search panel, as one picture: the chart it draws from a saved run, with the
// same ranking, the same colours and the same warning. `results/search_runs/` holds six of
// these (three backbones x two frame counts); the panel opens on DINOv2 over all frames,
// so that is the cell the slide shows, and it says so.
{
  const RUN_MODEL = "dinov2", RUN_SCOPE = "all";
  const LABELS = { dinov2: "DINOv2", convnextv2: "ConvNeXtV2", vit: "ViT" };
  const SCOPE_LABEL = { all: "all frames", "500": "500 frames" };
  const run = readJson(path.join(RESULTS, "search_runs", RUN_MODEL + "__" + RUN_SCOPE + ".json"));
  const evals = run.evaluations || [];

  const s = newSlide("content");
  head(s, "sky", "Chapter 4 — Applications and data augmentation",
    "The preprocessing search &mdash; change from the untouched baseline");

  if (!evals.length) {
    note(s, "No <code>results/search_runs/" + RUN_MODEL + "__" + RUN_SCOPE
      + ".json</code> yet. Run <code>uv run python experiments/search_all.py</code>.",
      { x: M, y: 3.0, w: CW, h: 0.6, fontSize: 13 });
  } else {
    const baseline = evals.filter(function (e) { return e.label === "baseline"; })
                          .map(function (e) { return e.play_recall; })[0] || 0;
    // Ranked on balanced, never on recall alone, for the reason the endpoint gives: every
    // cross-venue fold is 100% active play, so recall is buyable by answering "playing"
    // more often. Entries that predate the repair of the false-play control carry no
    // `balanced` and are ranked after the scored ones, exactly as the panel ranks them.
    const scored = evals.filter(function (e) { return "balanced" in e; })
                        .sort(function (a, b) { return b.balanced - a.balanced; });
    const stale = evals.filter(function (e) { return !("balanced" in e); })
                       .sort(function (a, b) { return b.play_recall - a.play_recall; });
    const ranked = scored.concat(stale);
    const best = scored[0];
    const who = LABELS[RUN_MODEL] + ", " + SCOPE_LABEL[RUN_SCOPE];

    s.addShape(pres.ShapeType.roundRect, {
      x: M, y: 1.50, w: CW, h: 0.42, rectRadius: 0.08,
      fill: { color: P.surface2 }, line: { type: "none" },
    });
    s.addText([
      { text: "Best on " + who + ":  ", options: { bold: true } },
      { text: best.label, options: { fontFace: MONO } },
      { text: "  at " + best.play_recall.toFixed(4) + " recall, worst fold "
              + best.worst_fold.toFixed(3) + ".   " },
      { text: evals.length + " evaluations · " + (run.n_frames || "?") + " frames",
        options: { color: P.ink3 } },
    ], { x: M + 0.26, y: 1.50, w: CW - 0.52, h: 0.42, fontFace: BODY, fontSize: 11,
         color: P.ink, valign: "middle" });
    grp(2);

    s.addText("Cross-venue play recall. Bars run right when a switch helped and left when "
      + "it hurt; the number on the right is the absolute recall. Baseline "
      + baseline.toFixed(3) + ".", {
      x: M, y: 1.98, w: CW, h: 0.3, fontFace: BODY, fontSize: 10.5, color: P.ink3,
      valign: "middle",
    });
    grp(1);

    // the chart
    const shown = ranked.slice(0, 14);
    const padL = 2.62, padR = 0.72;
    const plotX = M + padL, plotW = CW - padL - padR;
    const zero = plotX + plotW / 2;
    const top = 2.34, rowH = 0.266, barH = 0.165;
    const span = Math.max(0.06, Math.max.apply(null, shown.map(function (e) {
      return Math.abs(e.play_recall - baseline);
    }))) * 1.15;

    s.addShape(pres.ShapeType.rect, {
      x: zero, y: top, w: 0.008, h: shown.length * rowH + 0.06,
      fill: { color: P.ink3 }, line: { type: "none" },
    });
    stat(1);

    shown.forEach(function (e, i) {
      const y = top + i * rowH;
      const delta = e.play_recall - baseline;
      const len = Math.max(Math.abs(delta / span) * (plotW / 2), 0.02);
      // gained recall but also raised false alarms -> a boundary shift, not better sight
      const flagged = delta > 0.002 && "balanced" in e && e.false_play > 0.001;
      const colour = e.label === "baseline" ? P.ink3
        : !("balanced" in e) ? P.line
        : flagged ? P.amberB : delta >= 0 ? P.green : P.coral;
      s.addText(e.label + "  · r" + e.round, {
        x: M, y: y, w: padL - 0.14, h: rowH, fontFace: MONO, fontSize: 8.5,
        color: P.ink2, align: "right", valign: "middle",
      });
      s.addShape(pres.ShapeType.roundRect, {
        x: delta >= 0 ? zero : zero - len, y: y + (rowH - barH) / 2, w: len, h: barH,
        rectRadius: 0.02, fill: { color: colour }, line: { type: "none" },
      });
      s.addText(e.play_recall.toFixed(3), {
        x: W - M - padR + 0.1, y: y, w: padR - 0.1, h: rowH, fontFace: MONO, fontSize: 8.5,
        bold: true, color: P.ink, valign: "middle",
      });
      grp(3);
    });

    const axisY = top + shown.length * rowH + 0.04;
    s.addText("worse", { x: plotX, y: axisY, w: 1.2, h: 0.24, fontFace: MONO, fontSize: 9,
                         color: P.ink3, valign: "middle" });
    s.addText("baseline", { x: zero - 0.6, y: axisY, w: 1.2, h: 0.24, fontFace: MONO,
                            fontSize: 9, color: P.ink3, align: "center", valign: "middle" });
    s.addText("better", { x: W - M - padR - 1.2, y: axisY, w: 1.2, h: 0.24, fontFace: MONO,
                          fontSize: 9, color: P.ink3, align: "right", valign: "middle" });
    grp(3);

    // the legend, and the sentence it exists for
    const KEY = [["improved", P.green], ["worse", P.coral],
                 ["gained recall but also raised false alarms", P.amberB]];
    let lx = M;
    KEY.forEach(function (k) {
      s.addShape(pres.ShapeType.rect, {
        x: lx, y: axisY + 0.30, w: 0.14, h: 0.14, rectRadius: 0.02,
        fill: { color: k[1] }, line: { type: "none" },
      });
      s.addText(k[0], { x: lx + 0.2, y: axisY + 0.24, w: 3.6, h: 0.26, fontFace: BODY,
                        fontSize: 9.5, color: P.ink3, valign: "middle" });
      lx += 0.2 + 0.14 + Math.min(3.6, plain(k[0]).length * 0.062) + 0.3;
    });
    grp(6);
    note(s, "Every cross-venue test set is entirely active play, so recall can be bought "
      + "by predicting &ldquo;playing&rdquo; more often. Amber bars did exactly that and "
      + "are not improvements. Top " + shown.length + " of " + evals.length
      + " by <b>balanced = recall &minus; false-play</b>; the untouched baseline ranks "
      + (ranked.map(function (e) { return e.label; }).indexOf("baseline") + 1)
      + ", which is why it is not on the chart. Three backbones over two frame counts: each finished run is saved under <code>results/search_runs/</code>, so the six can be compared without re-running any of them.",
      { x: M, y: axisY + 0.60, w: CW, h: 0.5, fontSize: 10 });
  }
}

// =========================== 8. Conclusion ===============================================
// The last tab is a hero and one green block; on a slide they are one page - a second
// slide carrying only the block would be a blank screen with a sentence on it while the
// floor is already open.
{
  const s = newSlide("hero");
  hero(s, "green", "Questions", "Ask anything.",
    "And then, rather than another slide: the model, running.");
  quote(s, "Now we would like to <b>show you the system</b> &mdash; the model on real "
    + "footage, how a frame becomes a verdict, and what the operator actually sees.",
    { x: M + 0.35, y: 4.62, w: CW - 0.7, h: 1.5 },
    { fontSize: 16, bar: "0B6B45", fill: P.surface });
}

// =========================================================================================
// What pptxgenjs cannot write: the hero gradients, the transitions, and the build.
// Injected straight into the slide XML afterwards.
// =========================================================================================

const P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main";
const P159 = "http://schemas.microsoft.com/office/powerpoint/2015/09/main";
const MC = "http://schemas.openxmlformats.org/markup-compatibility/2006";

function gradFill(spec) {
  const stops = spec[1].map(function (g) {
    return '<a:gs pos="' + g[0] + '"><a:srgbClr val="' + g[1] + '"/></a:gs>';
  }).join("");
  return '<a:gradFill rotWithShape="1"><a:gsLst>' + stops + '</a:gsLst><a:lin ang="'
       + spec[0] * 60000 + '" scaled="0"/></a:gradFill>';
}

/**
 * Morph for the pages inside a section, a push for the break between sections.
 *
 * Morph is a 2016 feature and lives in the **2015/09** namespace, not the 2010 one: a
 * `<p14:morph/>` inside a `Requires="p14"` choice is accepted and then silently ignored,
 * which reads as no transition at all with the right duration. Both are wrapped so a
 * reader that does not know the extension still gets the fallback.
 */
function transitionXml(kind) {
  if (kind === "morph") {
    return '<mc:AlternateContent xmlns:mc="' + MC + '">'
      + '<mc:Choice xmlns:p159="' + P159 + '" Requires="p159">'
      + '<p:transition xmlns:p14="' + P14 + '" spd="slow" p14:dur="800">'
      + '<p159:morph option="byObject"/></p:transition></mc:Choice>'
      + '<mc:Fallback><p:transition spd="med"><p:fade/></p:transition></mc:Fallback>'
      + "</mc:AlternateContent>";
  }
  return '<mc:AlternateContent xmlns:mc="' + MC + '">'
    + '<mc:Choice xmlns:p14="' + P14 + '" Requires="p14">'
    + '<p:transition spd="slow" p14:dur="700"><p:push dir="u"/></p:transition>'
    + "</mc:Choice>"
    + '<mc:Fallback><p:transition spd="slow"><p:push dir="u"/></p:transition></mc:Fallback>'
    + "</mc:AlternateContent>";
}

/**
 * The build: every group rises a little and fades in, one after the previous, with no
 * click. A defence is talked over, not clicked through - the slide should assemble itself
 * while the first sentence is being said and then stay still.
 */
function timingXml(groups, ids) {
  let id = 3;
  const next = function () { return id++; };
  // The stagger is a budget, not a constant: the search chart has twenty beats, and a fixed
  // 160ms between them would spend three seconds drawing a slide nobody is waiting for.
  const step = Math.max(45, Math.round(1100 / Math.max(1, groups.length)));
  const body = groups.map(function (g, gi) {
    const outer = next(), inner = next();
    const effects = g.map(function (spid, k) {
      const e = next(), setId = next(), animId = next(), fadeId = next();
      return '<p:par><p:cTn id="' + e + '" presetID="34" presetClass="entr" '
        + 'presetSubtype="0" fill="hold" grpId="0" nodeType="'
        + (k === 0 ? (gi === 0 ? "afterEffect" : "afterEffect") : "withEffect") + '">'
        + '<p:stCondLst><p:cond delay="' + (k === 0 ? (gi === 0 ? 100 : step) : 0)
        + '"/></p:stCondLst><p:childTnLst>'
        + '<p:set><p:cBhvr><p:cTn id="' + setId + '" dur="1" fill="hold">'
        + '<p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
        + '<p:tgtEl><p:spTgt spid="' + spid + '"/></p:tgtEl>'
        + "<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>"
        + '</p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>'
        + '<p:anim calcmode="lin" valueType="num"><p:cBhvr additive="base">'
        + '<p:cTn id="' + animId + '" dur="520" fill="hold"/>'
        + '<p:tgtEl><p:spTgt spid="' + spid + '"/></p:tgtEl>'
        + "<p:attrNameLst><p:attrName>ppt_y</p:attrName></p:attrNameLst></p:cBhvr>"
        + '<p:tavLst><p:tav tm="0"><p:val><p:strVal val="#ppt_y+0.035"/></p:val></p:tav>'
        + '<p:tav tm="100000"><p:val><p:strVal val="#ppt_y"/></p:val></p:tav></p:tavLst>'
        + "</p:anim>"
        + '<p:animEffect transition="in" filter="fade"><p:cBhvr>'
        + '<p:cTn id="' + fadeId + '" dur="520"/>'
        + '<p:tgtEl><p:spTgt spid="' + spid + '"/></p:tgtEl></p:cBhvr></p:animEffect>'
        + "</p:childTnLst></p:cTn></p:par>";
    }).join("");
    return '<p:par><p:cTn id="' + outer + '" fill="hold">'
      + '<p:stCondLst><p:cond delay="' + (gi === 0 ? 0 : "indefinite") + '"/></p:stCondLst>'
      + '<p:childTnLst><p:par><p:cTn id="' + inner + '" fill="hold">'
      + '<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
      + effects + "</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>";
  }).join("");
  const bld = ids.map(function (spid) {
    return '<p:bldP spid="' + spid + '" grpId="0" animBg="1"/>';
  }).join("");
  return "<p:timing><p:tnLst><p:par>"
    + '<p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
    + '<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" '
    + 'nodeType="mainSeq"><p:childTnLst>' + body + "</p:childTnLst></p:cTn>"
    + '<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl>'
    + "</p:cond></p:prevCondLst>"
    + '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl>'
    + "</p:cond></p:nextCondLst></p:seq>"
    + "</p:childTnLst></p:cTn></p:par></p:tnLst>"
    + (bld ? "<p:bldLst>" + bld + "</p:bldLst>" : "") + "</p:timing>";
}

async function postprocess(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const names = Object.keys(zip.files)
    .filter(function (n) { return /^ppt\/slides\/slide\d+\.xml$/.test(n); })
    .sort(function (a, b) {
      return parseInt(a.match(/(\d+)/)[1], 10) - parseInt(b.match(/(\d+)/)[1], 10);
    });
  for (let i = 0; i < names.length; i++) {
    let xml = await zip.file(names[i]).async("string");
    const plan = PLAN[i] || { skip: 0, groups: [], transition: "morph" };

    // 1. pptxgenjs numbers tables on their own counter, so a slide can carry two shapes
    //    with the same id. Renumber every shape on the slide in document order - which is
    //    the order they were added - so an animation can name one.
    let k = 0;
    const ids = [];
    xml = xml.replace(/<p:cNvPr id="\d+"/g, function () {
      k += 1;
      if (k > 1) ids.push(k);
      return '<p:cNvPr id="' + k + '"';
    });

    // 2. a picture marked CROP:<top>,<bottom> shows one band of its file. `srcRect` is
    //    exactly what PowerPoint writes when a picture is cropped in its own UI, so what
    //    is on the slide is still the artefact on disk - nothing is resampled on the way
    //    in, and the uncropped file is one drag away.
    const crops = [];
    xml.replace(/descr="CROP:(\d+),(\d+)"/g, function (m, t, b) {
      crops.push([t, b]);
      return m;
    });
    if (crops.length) {
      let ci = 0;
      xml = xml.replace(/<a:stretch><a:fillRect\/><\/a:stretch>/g, function (m) {
        const c = crops[ci++];
        return c ? '<a:srcRect t="' + c[0] + '" b="' + c[1] + '"/>' + m : m;
      });
      xml = xml.replace(/descr="CROP:\d+,\d+"/g, 'descr="one band of the sheet"');
    }

    // 3. the hero gradients
    Object.keys(SENT).forEach(function (key) {
      xml = xml.split('<a:solidFill><a:srgbClr val="' + key + '"/></a:solidFill>')
               .join(gradFill(SENT[key]));
    });

    // 4. the transition and the build
    let rest = ids.slice(plan.skip);
    const groups = [];
    plan.groups.forEach(function (n) {
      if (!rest.length) return;
      groups.push(rest.slice(0, n));
      rest = rest.slice(n);
    });
    while (rest.length) { groups.push(rest.slice(0, 1)); rest = rest.slice(1); }

    xml = xml.replace("<p:sld ", '<p:sld xmlns:mc="' + MC + '" xmlns:p14="' + P14 + '" ');
    const animated = groups.reduce(function (a, g) { return a.concat(g); }, []);
    xml = xml.replace("</p:sld>", transitionXml(plan.transition)
      + (animated.length ? timingXml(groups, animated) : "") + "</p:sld>");
    zip.file(names[i], xml);
  }
  const buf = await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
  fs.writeFileSync(file, buf);
}

pres.writeFile({ fileName: OUT })
  .then(function () { return postprocess(OUT); })
  .then(function () {
    console.log("wrote " + OUT + " - " + PLAN.length + " slides");
  })
  .catch(function (e) { console.error(e); process.exit(1); });
