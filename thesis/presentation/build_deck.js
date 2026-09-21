// Builds the 25-minute oral defence deck.
//   node build_deck.js <output.pptx>
const pptxgen = require("pptxgenjs");

const OUT = process.argv[2] || "Thesis_Defence_25min.pptx";

// --- palette: "Floodlit Pitch" -------------------------------------------------------
const DARK   = "102A1E";   // deep pitch dark - dominant on title/section/closing
const DARK2  = "1A3C2B";
const GREEN  = "2FA86B";
const GREEN_D= "1F7A4C";
const AMBER  = "F2A33C";
const CORAL  = "D6455B";
const WHITE  = "FFFFFF";
const PAPER  = "F4F7F4";
const MUTED  = "5A6B62";
const INK    = "16211C";
const SOFT_G = "E6F3EC";
const SOFT_A = "FCEFDA";
const SOFT_C = "FAE3E7";

const HEAD = "Cambria";
const BODY = "Calibri";

const W = 13.3, H = 7.5, M = 0.62;
const CW = W - 2 * M;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Mariam Srour";
pres.title = "Who Actually Used the Pitch?";

let pageNo = 0;

// --- helpers -------------------------------------------------------------------------
function slide(kind = "light") {
  const s = pres.addSlide();
  s.background = { color: kind === "dark" ? DARK : WHITE };
  return s;
}

function eyebrow(s, text, color) {
  s.addText(text.toUpperCase(), {
    x: M, y: 0.42, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, bold: true, charSpacing: 2.2,
    color: color || GREEN_D,
  });
}

function title(s, text, opts = {}) {
  s.addText(text, {
    x: M, y: opts.y || 0.78, w: opts.w || CW, h: opts.h || 0.95, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: opts.size || 38, bold: true,
    color: opts.color || INK, valign: "top", lineSpacing: opts.size ? opts.size * 1.12 : 43,
  });
}

function sub(s, text, opts = {}) {
  s.addText(text, {
    x: M, y: opts.y || 1.82, w: opts.w || 10.6, h: opts.h || 0.6, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: opts.size || 17, color: opts.color || MUTED, valign: "top",
    lineSpacing: 24,
  });
}

function foot(s, label, dark) {
  pageNo += 1;
  s.addText(label, {
    x: M, y: H - 0.62, w: 9.5, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10.5, color: dark ? "7E9D8C" : "9AAAA1",
  });
  s.addText(String(pageNo), {
    x: W - M - 1.2, y: H - 0.62, w: 1.2, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10.5, color: dark ? "7E9D8C" : "9AAAA1", align: "right",
  });
}

// A numbered circle - the deck's one repeated motif.
function badge(s, x, y, text, fill, textColor) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: 0.54, h: 0.54, fill: { color: fill || GREEN },
  });
  s.addText(String(text), {
    x, y, w: 0.54, h: 0.54, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: textColor || WHITE,
    align: "center", valign: "middle",
  });
}

// A card: tinted background, optional badge, bold head, body. No edge stripes.
function card(s, o) {
  s.addShape(pres.ShapeType.roundRect, {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.1,
    fill: { color: o.tint || PAPER },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
  });
  let ty = o.y + 0.26;
  if (o.badge !== undefined) {
    badge(s, o.x + 0.26, o.y + 0.26, o.badge, o.badgeFill || GREEN);
    ty = o.y + 0.98;
  }
  if (o.head) {
    s.addText(o.head, {
      x: o.x + 0.26, y: ty, w: o.w - 0.52, h: o.headH || 0.5, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: o.headSize || 16, bold: true, color: o.headColor || INK,
      valign: "top", lineSpacing: (o.headSize || 16) * 1.15,
    });
    ty += (o.headH || 0.5);
  }
  if (o.body) {
    s.addText(o.body, {
      x: o.x + 0.26, y: ty, w: o.w - 0.52, h: o.y + o.h - ty - 0.2, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: o.bodySize || 13.5, color: o.bodyColor || MUTED,
      valign: "top", lineSpacing: (o.bodySize || 13.5) * 1.35,
    });
  }
}

// A big number with a small label under it.
function stat(s, o) {
  s.addShape(pres.ShapeType.roundRect, {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.1,
    fill: { color: o.tint || PAPER },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
  });
  s.addText(o.value, {
    x: o.x + 0.22, y: o.y + 0.22, w: o.w - 0.44, h: 0.82, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: o.size || 40, bold: true, color: o.color || GREEN_D,
    valign: "middle",
  });
  s.addText(o.label, {
    x: o.x + 0.22, y: o.y + 1.06, w: o.w - 0.44, h: o.h - 1.26, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12.5, color: o.labelColor || MUTED, valign: "top",
    lineSpacing: 16,
  });
}

// before -> after, the shape most of this project's results take.
function beat(s, o) {
  s.addShape(pres.ShapeType.roundRect, {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.1,
    fill: { color: o.tint || PAPER },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
  });
  if (o.head) {
    s.addText(o.head, {
      x: o.x + 0.26, y: o.y + 0.22, w: o.w - 0.52, h: 0.62, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15.5, bold: true, color: INK, valign: "top",
      lineSpacing: 18,
    });
  }
  s.addText(
    [
      { text: o.from, options: { fontFace: HEAD, fontSize: 25, bold: true, color: CORAL } },
      { text: "   →   ", options: { fontFace: BODY, fontSize: 17, color: MUTED } },
      { text: o.to, options: { fontFace: HEAD, fontSize: 25, bold: true, color: GREEN_D } },
    ],
    { x: o.x + 0.26, y: o.y + 0.88, w: o.w - 0.52, h: 0.52, isTextBox: true, margin: 0,
      valign: "middle" },
  );
  s.addText(o.label, {
    x: o.x + 0.26, y: o.y + 1.46, w: o.w - 0.52, h: o.y + o.h - (o.y + 1.46) - 0.2,
    isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12.5, color: MUTED, valign: "top", lineSpacing: 16,
  });
}

function bullets(s, items, o = {}) {
  s.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: { code: "25AA" }, breakLine: i !== items.length - 1 },
    })),
    { x: o.x || M, y: o.y || 2.3, w: o.w || 7.4, h: o.h || 3.4, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: o.size || 16, color: o.color || MUTED,
      paraSpaceAfter: 10, lineSpacing: (o.size || 16) * 1.34 },
  );
}

// A statement slide: dark or coloured, one sentence, nothing else competing.
function statement(s, o) {
  s.background = { color: o.bg || DARK };
  eyebrow(s, o.eyebrow, o.eyebrowColor || GREEN);
  title(s, o.title, { y: 1.5, size: o.size || 40, color: WHITE, h: o.titleH || 2.5, w: 11.3 });
  if (o.sub) {
    sub(s, o.sub, { y: o.subY || 4.3, color: "C8DCD1", size: 17, w: 10.8, h: 1.1 });
  }
  return s;
}

// =====================================================================================
//  1 - TITLE
// =====================================================================================
{
  const s = slide("dark");
  s.addShape(pres.ShapeType.ellipse, {
    x: 10.3, y: 1.1, w: 3.4, h: 3.4, fill: { color: DARK2 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 11.2, y: 2.0, w: 1.6, h: 1.6, fill: { color: GREEN, transparency: 62 },
  });
  s.addText("MASTER'S THESIS  •  ORAL DEFENCE", {
    x: M, y: 1.45, w: 9.5, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12.5, bold: true, charSpacing: 2.6, color: GREEN,
  });
  s.addText("Who actually used\nthe pitch?", {
    x: M, y: 1.95, w: 9.2, h: 2.1, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 52, bold: true, color: WHITE, lineSpacing: 56, valign: "top",
  });
  s.addText("Multi-pitch occupancy and booking verification from cameras that are already there",
    { x: M, y: 4.18, w: 9.0, h: 0.8, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 17.5, color: "C8DCD1", lineSpacing: 24 });
  s.addShape(pres.ShapeType.ellipse, {
    x: M, y: 5.35, w: 0.16, h: 0.16, fill: { color: AMBER },
  });
  s.addText("Mariam Srour", {
    x: M + 0.34, y: 5.16, w: 8, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: WHITE,
  });
  s.addText("[Programme]  •  [Institution]  •  Supervisor: [Name]  •  [Date]", {
    x: M + 0.34, y: 5.58, w: 9, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13.5, color: "9CBBAB",
  });
  pageNo += 1;
  s.addNotes(
    "Good morning, distinguished examiners, professors and colleagues.\n\n" +
    "My name is Mariam Srour, and I am a Master's student in [Programme] at [Institution].\n\n" +
    "The title of my research is \"Multi-Pitch Occupancy and Booking Verification from " +
    "Existing Cameras\". Throughout this talk I will use the shorter version on the slide: " +
    "who actually used the pitch?\n\n" +
    "[25 minutes. Pace: about 45 seconds a slide. Do not rush the three slides on the " +
    "confound - they are the thesis.]"
  );
}

// =====================================================================================
//  2 - ROADMAP
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Roadmap");
  title(s, "Where we are going");
  const items = [
    ["1", "Purpose", "What the study is for", GREEN],
    ["2", "How it works", "The system, and a scenario", GREEN_D],
    ["3", "The problem", "And the gap that motivated it", AMBER],
    ["4", "The solution", "What we changed, and what it cost", CORAL],
    ["5", "Summary", "And your questions", GREEN],
  ];
  const w = (CW - 4 * 0.24) / 5;
  items.forEach(([n, head, body, col], i) => {
    card(s, {
      x: M + i * (w + 0.24), y: 2.25, w, h: 2.5,
      badge: n, badgeFill: col, head, body,
      tint: i % 2 === 0 ? PAPER : SOFT_G, headH: 0.42,
    });
  });
  s.addText("Two parts I will spend real time on: the configuration searches, and the augmentation experiment.",
    { x: M, y: 5.2, w: CW, h: 0.4, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14.5, italic: true, color: GREEN_D });
  foot(s, "Roadmap");
  s.addNotes(
    "In this presentation I will briefly take you through the purpose of the research; how " +
    "the research works, including a short hypothetical scenario; what problem the research " +
    "tackles; how it provides a solution; and I will conclude with a summary.\n\n" +
    "I flag the two experiments at the bottom now, because they are where the method " +
    "contribution of this thesis actually lives."
  );
}

// =====================================================================================
//  3 - PURPOSE (statement)
// =====================================================================================
{
  const s = slide("dark");
  statement(s, {
    eyebrow: "Purpose of the research",
    title: "Find out which booked hours were really used — using the cameras a facility already owns.",
    size: 36,
  });
  const chips = ["No new hardware", "No GPU — one small PC", "A human decides every case"];
  let cx = M;
  chips.forEach((t) => {
    const w = 0.34 + t.length * 0.105;
    s.addShape(pres.ShapeType.roundRect, {
      x: cx, y: 4.45, w, h: 0.52, rectRadius: 0.26,
      fill: { color: DARK2 }, line: { color: GREEN_D, width: 1 },
    });
    s.addText(t, {
      x: cx, y: 4.45, w, h: 0.52, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13.5, bold: true, color: "DCEFE4",
      align: "center", valign: "middle",
    });
    cx += w + 0.2;
  });
  foot(s, "Purpose", true);
  s.addNotes(
    "The purpose of this study is to find out which sold pitch hours were actually used, " +
    "using cameras that are already installed for match highlights.\n\n" +
    "The study focuses on a network of synthetic five-a-side football pitches, three kinds " +
    "of scene - empty, active play, and maintenance - and the facility's own booking " +
    "records.\n\n" +
    "Say this plainly: the output is not a classification. It is a billing conversation. " +
    "'This booking looks unused, and here are three frames from it.'"
  );
}

// =====================================================================================
//  4 - WHAT THE STUDY FOCUSES ON
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Purpose — the scope");
  title(s, "What the study actually covers");
  const w = (CW - 3 * 0.24) / 4;
  const tiles = [
    ["1,692", "labelled frames", GREEN_D, PAPER],
    ["3", "classes: empty, playing, maintenance", GREEN_D, SOFT_G],
    ["7", "venue folds for cross-venue testing", AMBER.replace("F2A33C", "B26A00"), SOFT_A],
    ["0", "GPUs — a CPU-only design constraint", CORAL, SOFT_C],
  ];
  tiles.forEach(([v, l, c, t], i) => {
    stat(s, { x: M + i * (w + 0.24), y: 2.2, w, h: 2.0, value: v, label: l, color: c, tint: t });
  });
  card(s, {
    x: M, y: 4.5, w: CW, h: 1.55, tint: PAPER,
    head: "One design rule carried everywhere",
    body: "The vision path must never see the booking record as an input. A model that has " +
          "read the booking flag cannot give evidence independent of the record it audits.",
    headSize: 16, bodySize: 14,
  });
  foot(s, "Purpose — scope");
  s.addNotes(
    "The scale first, so nobody has to guess it. 1,692 hand-labelled frames, three classes, " +
    "seven venue folds, and no GPU anywhere in the design.\n\n" +
    "The card at the bottom is the rule that makes this an audit rather than a rubber stamp. " +
    "If the classifier could see the booking flag, its agreement with the booking would mean " +
    "nothing. The vision path and the reconciliation layer never share a feature."
  );
}

// =====================================================================================
//  5 - HOW IT WORKS: THE PIPELINE
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "How the research works");
  title(s, "Five stages, one frame a minute");
  sub(s, "Sparse sampling, not a video stream — which is what makes it fit on one small computer.");
  const steps = [
    ["1", "Sample", "One frame per camera per minute. About 99% less network traffic."],
    ["2", "Classify", "Empty, playing, or maintenance — a small head on a frozen backbone."],
    ["3", "Check", "Pitch area, motion, people, ball. Cheap rules that can veto the model."],
    ["4", "Aggregate", "About 60 predictions an hour become one verdict: used, not used, review."],
    ["5", "Compare", "Verdict against the booking record. Disagreements are flagged with evidence."],
  ];
  const w = (CW - 4 * 0.22) / 5;
  steps.forEach(([n, head, body], i) => {
    card(s, {
      x: M + i * (w + 0.22), y: 2.72, w, h: 2.6,
      badge: n, badgeFill: i < 2 ? GREEN : i < 4 ? GREEN_D : AMBER,
      badgeText: WHITE, head, body, headH: 0.4,
      tint: i % 2 === 0 ? PAPER : SOFT_G,
    });
  });
  foot(s, "How it works");
  s.addNotes(
    "To achieve the purpose of the study I built and then evaluated a five-stage system.\n\n" +
    "Stage one is the decision that makes everything else possible. Instead of decoding " +
    "twenty to thirty continuous video streams, the system samples one frame per camera per " +
    "minute. That is roughly a 99% cut in bandwidth, from about 60-90 megabits down to under " +
    "one, and it is what lets the whole facility run on a single Intel mini-PC with no " +
    "graphics card.\n\n" +
    "Stage three is worth naming: the cheap geometric and motion checks can overrule the " +
    "neural network. I will come back to that, because it turns out to matter more than the " +
    "choice of network."
  );
}

// =====================================================================================
//  6 - HYPOTHETICAL SCENARIO
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "A hypothetical scenario");
  title(s, "Tuesday, 20:00 — Pitch 3");
  const w = (CW - 2 * 0.9) / 3;
  const cols = [
    ["The booking says", "Sold. One hour.", "Paid for, and marked “used” by staff.", SOFT_G, GREEN_D],
    ["The camera sees", "Empty pitch,\n13 of 16 minutes", "Floodlights on. Nobody on the grass.", SOFT_A, "B26A00"],
    ["The system says", "REVIEW\n+ 3 photos", "Never “do not bill”. Only “look at this”.", SOFT_G, GREEN_D],
  ];
  cols.forEach(([tag, big, body, tint, col], i) => {
    const x = M + i * (w + 0.9);
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.5, w, h: 2.6, rectRadius: 0.1, fill: { color: tint },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
    });
    s.addText(tag.toUpperCase(), {
      x: x + 0.26, y: 2.74, w: w - 0.52, h: 0.28, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11, bold: true, charSpacing: 1.6, color: col,
    });
    s.addText(big, {
      x: x + 0.26, y: 3.1, w: w - 0.52, h: 1.0, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 22, bold: true, color: INK, valign: "top", lineSpacing: 26,
    });
    s.addText(body, {
      x: x + 0.26, y: 4.2, w: w - 0.52, h: 0.7, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13, color: MUTED, valign: "top", lineSpacing: 17,
    });
    if (i < 2) {
      s.addText("→", {
        x: x + w + 0.08, y: 3.4, w: 0.74, h: 0.6, isTextBox: true, margin: 0,
        fontFace: BODY, fontSize: 26, bold: true, color: AMBER, align: "center",
      });
    }
  });
  s.addText("The output is a conversation, not a charge.", {
    x: M, y: 5.5, w: CW, h: 0.5, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 20, bold: true, color: GREEN_D,
  });
  foot(s, "How it works — scenario");
  s.addNotes(
    "Let me make that concrete. It is Tuesday evening. The booking system says pitch 3 was " +
    "sold for an hour and a member of staff marked it as used. The cameras sampled sixteen " +
    "frames across that hour, and thirteen of them show an empty pitch under floodlights.\n\n" +
    "The system does not cancel the charge. It raises one advisory with three evidence " +
    "photos, and a manager decides what to do.\n\n" +
    "In the code that is structural rather than a promise. The only output type the decision " +
    "layer can produce is an Advisory; requires_human_confirmation is a property, not a " +
    "setting; and there is no code path that acts on a verdict."
  );
}

// =====================================================================================
//  7 - METHOD: HOW THE DATA WERE COLLECTED
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "How the research works — method");
  title(s, "How the data were collected");
  const w = (CW - 3 * 0.24) / 4;
  const items = [
    ["Fixed cameras", "Existing CCTV on posts. One view forever — which is why no rotation is ever simulated."],
    ["Every 15 seconds", "Sampled from recorded footage, then hand-labelled into three classes."],
    ["Nine venues", "Seven usable as cross-venue folds, plus two cameras watching one pitch."],
    ["A written protocol", "Labelling rules fixed in advance, with the ambiguous cases decided before the runs."],
  ];
  items.forEach(([head, body], i) => {
    card(s, {
      x: M + i * (w + 0.24), y: 2.3, w, h: 2.1, head, body,
      tint: i % 2 === 0 ? PAPER : SOFT_G, headH: 0.42,
    });
  });
  card(s, {
    x: M, y: 4.66, w: CW, h: 1.32, tint: SOFT_A,
    head: "The fact that shaped the whole evaluation",
    body: "Because the cameras are fixed and frames come every 15 seconds, 98.5% of frames " +
          "have a near-duplicate somewhere in the corpus.",
    headSize: 15.5, bodySize: 14,
  });
  foot(s, "Method — data");
  s.addNotes(
    "The study involved recorded footage from fixed cameras at nine venues, sampled every " +
    "fifteen seconds and hand-labelled into three classes against a written protocol that " +
    "was fixed before the runs.\n\n" +
    "The amber card is the one to dwell on. Fixed cameras plus short intervals means almost " +
    "every frame has a near-twin. That single property is what invalidated the pilot, and " +
    "it is why the next slide is about splitting rather than about models."
  );
}

// =====================================================================================
//  8 - METHOD: HOW THE DATA WERE ANALYSED
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "How the research works — method");
  title(s, "How the data were analysed");
  const w = (CW - 0.3) / 2;
  const left = [
    ["Grouped splits, never random", "Whole venues and whole slots are held out, so a test frame never has a near-twin in training."],
    ["Four trivial baselines, every time", "A clock rule, a colour histogram, a constant predictor, a random one. If they win, the protocol is wrong."],
  ];
  const right = [
    ["Leave-one-venue-out", "Seven folds. Every comparison is reported with the smallest p-value the design could have produced."],
    ["Pre-registered, amended in the open", "Hypotheses fixed before the runs; every change since is numbered, dated and left in place."],
  ];
  left.forEach(([head, body], i) => {
    card(s, { x: M, y: 2.3 + i * 1.72, w, h: 1.5, head, body, tint: i ? SOFT_G : PAPER, headH: 0.4 });
  });
  right.forEach(([head, body], i) => {
    card(s, { x: M + w + 0.3, y: 2.3 + i * 1.72, w, h: 1.5, head, body, tint: i ? PAPER : SOFT_G, headH: 0.4 });
  });
  s.addText("Holm correction within each declared family of tests.", {
    x: M, y: 5.86, w: CW, h: 0.35, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, italic: true, color: MUTED,
  });
  foot(s, "Method — analysis");
  s.addNotes(
    "The collected data were analysed with frozen vision backbones and a small trained head, " +
    "evaluated under grouped splits and leave-one-venue-out.\n\n" +
    "Two choices here are the method contribution. First, every protocol runs four trivial " +
    "baselines alongside the models - a rule that reads only the clock, a colour histogram, a " +
    "constant predictor and a random one. If a trivial baseline wins, that is information " +
    "about the protocol, not a curiosity to leave out of the write-up.\n\n" +
    "Second, I report the resolution of each test. Over seven folds the smallest attainable " +
    "two-sided p-value is 0.0156 - so 'not significant' and 'the design could not have " +
    "produced significance' are different statements, and I say which one applies."
  );
}

// =====================================================================================
//  9 - THE PROBLEM (statement)
// =====================================================================================
{
  const s = slide("dark");
  statement(s, {
    eyebrow: "What problem does the research tackle?",
    title: "A facility sells hours.\nNobody checks which ones were used.",
    size: 40,
  });
  const w = (CW - 2 * 0.26) / 3;
  const items = [
    ["No-shows", "Paid for, never played."],
    ["Unbooked use", "Played, never paid for."],
    ["Data-entry error", "The record and the reality drift apart."],
  ];
  items.forEach(([head, body], i) => {
    const x = M + i * (w + 0.26);
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 4.42, w, h: 1.44, rectRadius: 0.1, fill: { color: DARK2 },
    });
    s.addText(head, {
      x: x + 0.26, y: 4.64, w: w - 0.52, h: 0.4, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: WHITE,
    });
    s.addText(body, {
      x: x + 0.26, y: 5.06, w: w - 0.52, h: 0.62, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13, color: "AFCBBC", valign: "top", lineSpacing: 17,
    });
  });
  foot(s, "The problem", true);
  s.addNotes(
    "This research addresses the problem that a facility sells pitch hours and has no " +
    "verified record of which ones were used. Staff write it down; the records are " +
    "unverified; three kinds of error follow, and each costs money or trust.\n\n" +
    "Concede the obvious objection before anyone raises it: a twenty-euro motion sensor is " +
    "more robust in fog, in darkness, and against a dirty lens. But a sensor answers 'did " +
    "something move'. An audit needs 'was this booking used, and here is the picture' - and " +
    "it has to tell a five-a-side match from a groundsman on a mower."
  );
}

// =====================================================================================
// 10 - WHAT IS KNOWN, AND THE GAP
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "The gap");
  title(s, "What is already known — and what is missing");
  const w = (CW - 0.4) / 2;
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 2.3, w, h: 3.3, rectRadius: 0.1, fill: { color: PAPER },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
  });
  s.addText("ALREADY KNOWN", {
    x: M + 0.3, y: 2.56, w: w - 0.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, bold: true, charSpacing: 1.8, color: GREEN_D,
  });
  bullets(s, [
    "Frozen pretrained backbones work well with very few labels.",
    "Pretraining data matters more than architecture at the same size.",
    "Occupancy and people-counting from CCTV are established tasks.",
  ], { x: M + 0.3, y: 2.98, w: w - 0.6, h: 2.4, size: 14.5 });

  s.addShape(pres.ShapeType.roundRect, {
    x: M + w + 0.4, y: 2.3, w, h: 3.3, rectRadius: 0.1, fill: { color: SOFT_C },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
  });
  s.addText("THE GAP", {
    x: M + w + 0.7, y: 2.56, w: w - 0.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, bold: true, charSpacing: 1.8, color: CORAL,
  });
  bullets(s, [
    "Almost all of it is evaluated on frames from the same scenes.",
    "So the reported scores measure memory of a place, not recognition of an activity.",
    "Nobody reports what the score would be if a trivial rule were run beside it.",
  ], { x: M + w + 0.7, y: 2.98, w: w - 0.6, h: 2.4, size: 14.5, color: "7A4048" });
  s.addText("This matters because the whole business case rests on one class — the empty pitch.",
    { x: M, y: 5.78, w: CW, h: 0.4, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14.5, italic: true, color: GREEN_D });
  foot(s, "The gap");
  s.addNotes(
    "Although a great deal is known about frozen backbones and about occupancy from CCTV, " +
    "there remains a gap: almost all of it is evaluated on frames drawn from the same scenes " +
    "that appear in training.\n\n" +
    "That gap is important here because the facility's money rests entirely on one class. " +
    "Getting 'playing' right is easy and worth nothing - if the system cannot reliably " +
    "recognise an empty pitch, it cannot flag a single unused booking."
  );
}

// =====================================================================================
// 11 - LEAKAGE
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "The gap — first consequence");
  title(s, "Honest splitting halves the score");
  const w = (CW - 2 * 0.24) / 3;
  stat(s, { x: M, y: 2.35, w, h: 1.85, value: "37.1%", tint: SOFT_C, color: CORAL,
    label: "of near-duplicate pairs cross the train/test line under a random split" });
  stat(s, { x: M + w + 0.24, y: 2.35, w, h: 1.85, value: "0.8%", tint: SOFT_G, color: GREEN_D,
    label: "under a split grouped by venue and slot" });
  stat(s, { x: M + 2 * (w + 0.24), y: 2.35, w, h: 1.85, value: "−0.49", tint: SOFT_A,
    color: "B26A00", label: "macro-F1: what being honest costs ConvNeXtV2" });
  card(s, {
    x: M, y: 4.5, w: CW, h: 1.5, tint: PAPER,
    head: "But only about two thirds of that fall is leakage",
    body: "A model that never trains cannot leak. Scoring a zero-shot model on the identical " +
          "test sets measures what changing the test set does on its own: it drops 0.183. " +
          "Overstating leakage would have been the comfortable error.",
    headSize: 15.5, bodySize: 14,
  });
  foot(s, "The gap — leakage");
  s.addNotes(
    "First consequence, and it is measurable. Under a random split, 37% of near-duplicate " +
    "pairs land on opposite sides of the train/test line. Group the split by venue and slot " +
    "and that falls to under 1%.\n\n" +
    "The price of that honesty is severe: ConvNeXtV2's macro-F1 roughly halves. And on the " +
    "leaky split, every one of its twelve errors was a frame whose near-duplicate was sitting " +
    "in the training set.\n\n" +
    "The bottom card is where I checked my own result. A model that never trains cannot leak, " +
    "so scoring a zero-shot model on the same test sets isolates what changing the test set " +
    "does by itself - 0.183 of it. The leakage-attributable part is 0.33 to 0.40, not 0.52. " +
    "Reporting the larger number would have made my story stronger and it would have been " +
    "wrong."
  );
}

// =====================================================================================
// 12 - THE CONFOUND (statement)
// =====================================================================================
{
  const s = slide("dark");
  s.background = { color: "4A1721" };
  eyebrow(s, "The gap — the real one", "F2A33C");
  title(s, "In this data, the time of day is the answer", { y: 1.4, size: 40, color: WHITE, h: 1.3, w: 11.5 });
  const w = (CW - 2 * 0.26) / 3;
  const items = [
    ["98%", "of daylight frames are an empty pitch"],
    ["99%", "of night frames are a match in progress"],
    ["99.1%", "so “night means play” is right on the whole corpus"],
  ];
  items.forEach(([v, l], i) => {
    const x = M + i * (w + 0.26);
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.9, w, h: 1.95, rectRadius: 0.1, fill: { color: "5F2029" },
    });
    s.addText(v, {
      x: x + 0.26, y: 3.12, w: w - 0.52, h: 0.8, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 42, bold: true, color: AMBER, valign: "middle",
    });
    s.addText(l, {
      x: x + 0.26, y: 3.96, w: w - 0.52, h: 0.7, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13.5, color: "F2D6DA", valign: "top", lineSpacing: 17,
    });
  });
  s.addText(
    [
      { text: "The two cells that would break the tie hold  ", options: { fontFace: BODY, fontSize: 16, color: "F2D6DA" } },
      { text: "9 frames", options: { fontFace: HEAD, fontSize: 18, bold: true, color: WHITE } },
      { text: "  (empty at night) and  ", options: { fontFace: BODY, fontSize: 16, color: "F2D6DA" } },
      { text: "6", options: { fontFace: HEAD, fontSize: 18, bold: true, color: WHITE } },
      { text: "  (play in daylight).", options: { fontFace: BODY, fontSize: 16, color: "F2D6DA" } },
    ],
    { x: M, y: 5.2, w: CW, h: 0.5, isTextBox: true, margin: 0, valign: "middle" },
  );
  foot(s, "The gap — the confound", true);
  s.addNotes(
    "This is the most important slide in the talk, so I will slow down.\n\n" +
    "Across the recorded corpus, daylight is 98% empty pitches and night is 99% football. " +
    "Which means the sentence 'night means play, day means not-play' is correct on 99.1% of " +
    "my data, without looking at a single pixel.\n\n" +
    "The consequence is that nothing measured on this data can separate a model that " +
    "recognises an empty pitch from a model that recognises the time of day. The two cells " +
    "that would break the tie hold nine frames and six frames.\n\n" +
    "I want to be precise about what kind of claim this is. It is a statement about the " +
    "dataset, not about the models - and identifying it, rather than reporting around it, is " +
    "the contribution of this thesis."
  );
}

// =====================================================================================
// 13 - THE CLOCK RULE BEATS THE MODELS (chart)
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Evidence");
  title(s, "A rule that never looks at the image wins");
  sub(s, "Tested across venues the models had never seen. Lower is better — this is how often each one calls an empty pitch a match.",
    { w: 11.4, h: 0.7 });
  s.addChart(pres.ChartType.bar, [{
    name: "False-play rate",
    labels: ["Clock rule\n(no pixels)", "DINOv2", "ViT", "ConvNeXtV2"],
    values: [0.0206, 0.3086, 0.8354, 0.9918],
  }], {
    x: M, y: 2.75, w: 7.7, h: 3.35,
    barDir: "col", chartColors: [GREEN, AMBER, CORAL, CORAL],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    dataLabelFontFace: BODY, dataLabelFontSize: 12, dataLabelColor: INK,
    showLegend: false, showTitle: false,
    catAxisLabelColor: MUTED, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 12,
    valAxisLabelColor: MUTED, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 11,
    valAxisMaxVal: 1.1, valGridLine: { color: "E4EAE6", size: 1 },
    catGridLine: { style: "none" }, barGapWidthPct: 55,
  });
  card(s, {
    x: M + 8.0, y: 2.75, w: CW - 8.0, h: 1.58, tint: SOFT_G,
    head: "And it wins on recall too",
    body: "Perfect 1.000 play-recall across unseen venues — above all three backbones.",
    headSize: 15, bodySize: 13.5,
  });
  card(s, {
    x: M + 8.0, y: 4.52, w: CW - 8.0, h: 1.58, tint: SOFT_C,
    head: "So are the models worthless?",
    body: "No. They are indistinguishable from a light meter on this dataset — which is " +
          "a statement about the dataset.",
    headSize: 15, bodySize: 13.5,
  });
  foot(s, "Evidence — the clock rule");
  s.addNotes(
    "Here is what that confound does to a benchmark. The clock rule predicts the majority " +
    "class for the lighting condition and reads no pixels at all. Across unseen venues it " +
    "has a false-play rate fifteen times lower than the best backbone, and perfect " +
    "play-recall. It beats all three on both axes at once.\n\n" +
    "Expect the question on the second card, and answer it directly: no, the backbones are " +
    "not worthless. They are indistinguishable from a light meter on this dataset.\n\n" +
    "One honesty note, and I would rather volunteer it. An earlier version of this slide said " +
    "the opposite - that the rule collapsed across venues. That number came from a label " +
    "error: a brightness threshold had filed 216 frames of floodlit night football as " +
    "daylight. Corrected, the rule wins, and the finding reversed."
  );
}

// =====================================================================================
// 14 - THE CLIP THAT REVERSED IT
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Evidence — the reversal");
  title(s, "Then one clip reversed the ranking");
  sub(s, "234 seconds of a floodlit pitch at night with nobody playing — the case the corpus does not contain.");
  const w = (CW - 2 * 0.26) / 3;
  const items = [
    ["Clock rule", "16 / 16", "minutes wrong. False alarms: 1.00.", SOFT_C, CORAL],
    ["Model alone", "0.38", "false alarm rate.", SOFT_A, "B26A00"],
    ["Full system", "0.00", "13 of 13 empty minutes right.", SOFT_G, GREEN_D],
  ];
  items.forEach(([head, big, body, tint, col], i) => {
    const x = M + i * (w + 0.26);
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.75, w, h: 2.2, rectRadius: 0.1, fill: { color: tint },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
    });
    s.addText(head, {
      x: x + 0.26, y: 2.97, w: w - 0.52, h: 0.36, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: INK,
    });
    s.addText(big, {
      x: x + 0.26, y: 3.36, w: w - 0.52, h: 0.78, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 38, bold: true, color: col, valign: "middle",
    });
    s.addText(body, {
      x: x + 0.26, y: 4.18, w: w - 0.52, h: 0.6, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13, color: MUTED, valign: "top", lineSpacing: 17,
    });
  });
  s.addText("A 1,692-frame benchmark ranked the trivial rule first. One clip reversed it completely — and the difference is not size or statistics. It is one missing case.",
    { x: M, y: 5.3, w: CW, h: 0.7, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 15.5, color: GREEN_D, lineSpacing: 21 });
  foot(s, "Evidence — the reversal");
  s.addNotes(
    "So I went and got the missing case: four minutes of a floodlit pitch at night with " +
    "nobody on it, scored minute by minute.\n\n" +
    "The clock rule calls all sixteen minutes active play. It is wrong on every single one. " +
    "The neural probe on its own sits at 0.38. The deployed path - probe plus pitch boundary " +
    "plus motion gate plus person gate - is at zero, and gets all thirteen empty minutes " +
    "right.\n\n" +
    "The point to land, and it is the most useful sentence in the thesis: the benchmark had " +
    "1,692 frames and the clip had a few hundred. The clip won, because it contained the case " +
    "the benchmark was missing. More data of the same kind would not have found this."
  );
}

// =====================================================================================
// 15 - A PROTOCOL A CONSTANT PREDICTOR WINS
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Evidence — the protocol itself");
  title(s, "One protocol has no ranking to change");
  const w = (CW - 0.34) / 2;
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 2.35, w, h: 3.2, rectRadius: 0.1, fill: { color: SOFT_C },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
  });
  s.addText("1.000", {
    x: M + 0.34, y: 2.72, w: w - 0.68, h: 1.1, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 62, bold: true, color: CORAL, valign: "middle",
  });
  s.addText("macro-F1 for a predictor that always answers “playing”", {
    x: M + 0.34, y: 3.86, w: w - 0.68, h: 0.6, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: INK, valign: "top", lineSpacing: 22,
  });
  s.addText("Ahead of every backbone — because every held-out venue is 100% active play. " +
            "There is nothing else in the fold to get wrong.", {
    x: M + 0.34, y: 4.56, w: w - 0.68, h: 0.8, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13.5, color: MUTED, valign: "top", lineSpacing: 18,
  });
  card(s, {
    x: M + w + 0.34, y: 2.35, w, h: 1.66, tint: PAPER,
    head: "A low false-play rate is not accuracy either",
    body: "On 243 held-out empty frames DINOv2 answers PLAYING 75 times and MAINTENANCE 168 " +
          "times. It is correct zero times. No model exceeds 0.165.",
    headSize: 15, bodySize: 13.5,
  });
  card(s, {
    x: M + w + 0.34, y: 4.13, w, h: 1.66, tint: SOFT_G,
    head: "We found it by checking",
    body: "A rate of 0.000 looked like success. Adding one column — what does it answer " +
          "instead? — showed it answers MAINTENANCE on all 243.",
    headSize: 15, bodySize: 13.5,
  });
  foot(s, "Evidence — the protocol");
  s.addNotes(
    "Two more cases, and the pattern is the point: each was found by checking rather than by " +
    "theorising.\n\n" +
    "On leave-one-venue-out, a constant predictor that always answers 'playing' scores a " +
    "perfect macro-F1 - ahead of every backbone. This is stronger than saying the protocol " +
    "changes the ranking. On that protocol there is no ranking to change.\n\n" +
    "The second one is subtler and I think it is the strongest result in the thesis. The " +
    "false-play rate had been quoted throughout as how well a model recognises an empty " +
    "pitch. So I asked what the models answer instead. DINOv2 says playing 75 times and " +
    "maintenance 168 times, and is correct zero times out of 243. The published numbers " +
    "stand; the inference drawn from them does not."
  );
}

// =====================================================================================
// 15b - DATA: WHY IT IS HARD TO GET
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Data");
  title(s, "Why this data is hard to get");
  const w = (CW - 3 * 0.24) / 4;
  const items = [
    ["There are people in it", "Players, staff, sometimes children. Footage of identifiable people is sensitive data."],
    ["So it cannot just be collected", "Every extra venue is a consent and data-protection conversation, not a download."],
    ["And the task needs variety", "Many venues, lights, angles, weathers. Volume of the same scene does not help."],
    ["What we got instead", "One venue with empty pitches, nine venues of night football, and no wet weather at all."],
  ];
  items.forEach(([head, body], i) => {
    card(s, {
      x: M + i * (w + 0.24), y: 2.3, w, h: 2.3, head, body,
      tint: i % 2 === 0 ? PAPER : SOFT_C, headH: 0.7, headSize: 15.5, bodySize: 13,
    });
  });
  card(s, {
    x: M, y: 4.76, w: CW, h: 1.3, tint: SOFT_A,
    head: "So the bottleneck is permission, not effort",
    body: "Not storage, not labelling time. Getting one more venue means someone agreeing to " +
          "be filmed.",
    headSize: 15, bodySize: 14,
  });
  foot(s, "Data — the constraint");
  s.addNotes(
    "The obvious question is: why not just collect more data?\n\n" +
    "Because this is footage of identifiable people on a pitch - players, staff, and " +
    "sometimes children. That is sensitive personal data. Every additional venue is a " +
    "consent and data-protection conversation with a facility, not a dataset download. The " +
    "bottleneck is permission, not storage and not labelling effort.\n\n" +
    "And note what the task needs: variety, not volume. Ten thousand more frames of the same " +
    "pitch on the same evening would add nothing, because they are near-duplicates of what I " +
    "already have."
  );
}

// =====================================================================================
// 15c - DATA: SO WE MADE SOME WITH AI
// =====================================================================================
{
  const s = slide("dark");
  s.background = { color: "14567F" };
  eyebrow(s, "Data — generated", "F2A33C");
  title(s, "So we made some, starting from something real", { y: 1.2, size: 36, color: WHITE, h: 0.9, w: 11.5 });
  const w = (CW - 3 * 0.24) / 4;
  const items = [
    ["Real anchor first", "Every generated item starts from an actual frame of an actual pitch — never a text prompt alone."],
    ["Images — Gemini, ChatGPT", "Used to produce the scene the corpus lacks: an empty pitch under floodlights."],
    ["Stills into video — 2 AI tools", "A still frame animated into motion, so the motion check has something to read. [names to be added]"],
    ["What it bought", "31 generated empty frames took cross-venue false alarms from 0.768 to 0.024 — and play-recall went up."],
  ];
  items.forEach(([head, body], i) => {
    const x = M + i * (w + 0.24);
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.35, w, h: 2.45, rectRadius: 0.1, fill: { color: "1E6FA0" },
    });
    s.addText(head, {
      x: x + 0.26, y: 2.6, w: w - 0.52, h: 0.72, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15.5, bold: true, color: WHITE, valign: "top",
      lineSpacing: 18,
    });
    s.addText(body, {
      x: x + 0.26, y: 3.38, w: w - 0.52, h: 1.2, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13, color: "D6EAF7", valign: "top", lineSpacing: 17,
    });
  });
  s.addText("Generated frames are excluded from every corpus count. They are a training aid — not evidence that the system works on real footage of that case.",
    { x: M, y: 5.1, w: CW, h: 0.9, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14.5, color: "EAF5FC", lineSpacing: 20 });
  foot(s, "Data — generated", true);
  s.addNotes(
    "So I generated some data, and I want to be careful about how I describe that.\n\n" +
    "I did not generate a dataset from text prompts. Every generated item starts from a real " +
    "anchor - a real frame of a real pitch - and the generation changes one thing about it.\n\n" +
    "For images I used Gemini and ChatGPT's image models, to produce the scene the corpus " +
    "does not contain: an empty pitch under floodlights at night. For video I took still " +
    "frames and animated them, adding motion to a still image so there is movement for the " +
    "motion check to read - using two AI video tools, which I will name.\n\n" +
    "Thirty-one generated empty frames took cross-venue false alarms from 0.768 down to " +
    "0.024, and play-recall went up rather than down, which is the sign it added signal " +
    "rather than noise.\n\n" +
    "And the guard: they are excluded from every corpus count in the thesis. Counting them " +
    "would make a gap look filled that is not. One line I did not cross - I did not generate " +
    "rain, because there is no real wet footage to check it against, and validating " +
    "synthetic rain against synthetic rain tests the generator, not the weather."
  );
}

// =====================================================================================
// 15d - DATA: SMALL DATA AND OVERFITTING
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Data — the risk");
  title(s, "Small data has a specific danger");
  sub(s, "Not “less accurate”. Overfitting — the model memorises these places instead of learning the activity.",
    { w: 11.6, h: 0.7 });
  const w = (CW - 2 * 0.26) / 3;
  const items = [
    ["The corpus is scenes, not frames", "1,692 frames are about 150 distinct scenes. The 243 held-out empty frames are 3.", SOFT_C, CORAL],
    ["More labels made it worse", "Every backbone peaks at 100–300 labels and falls at 671. The extra ones are near-duplicates.", SOFT_C, CORAL],
    ["So generalisation is what we test", "Grouped splits, leave-one-venue-out, and a trivial baseline in every protocol.", SOFT_G, GREEN_D],
  ];
  items.forEach(([head, body, tint], i) => {
    card(s, {
      x: M + i * (w + 0.26), y: 2.95, w, h: 2.3, head, body, tint,
      headH: 0.72, headSize: 15.5, bodySize: 13.5,
    });
  });
  s.addText("The risk is not that the numbers come out low. It is that they come out high for the wrong reason.",
    { x: M, y: 5.45, w: CW, h: 0.5, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 17, bold: true, color: CORAL });
  foot(s, "Data — overfitting");
  s.addNotes(
    "Here is the danger with a dataset this size, stated precisely.\n\n" +
    "The risk is not that the numbers come out low. It is that they come out high for the " +
    "wrong reason - the model memorises these specific pitches, and that looks like success. " +
    "That is overfitting, and it is the opposite of what we need, which is generalisation to " +
    "a venue the system has never seen.\n\n" +
    "The middle card is the evidence that this is real and not theoretical. Label efficiency " +
    "here is not monotone: every backbone peaks between one and three hundred labels and then " +
    "gets worse at 671, because the extra labels are near-duplicates that add redundancy " +
    "rather than information.\n\n" +
    "That is the whole reason the evaluation is built the way it is."
  );
}

// =====================================================================================
// 16 - MODELS
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "What I built");
  title(s, "Almost nothing here is trained");
  const w = (CW - 2 * 0.24) / 3;
  stat(s, { x: M, y: 2.3, w, h: 1.8, value: "200.8M", tint: PAPER, color: GREEN_D,
    label: "frozen parameters — never updated once" });
  stat(s, { x: M + w + 0.24, y: 2.3, w, h: 1.8, value: "10,932", tint: SOFT_G, color: GREEN_D,
    label: "trained parameters, all of them" });
  stat(s, { x: M + 2 * (w + 0.24), y: 2.3, w, h: 1.8, value: "18,371:1", tint: SOFT_A,
    color: "B26A00", label: "frozen to trained" });
  const items = [
    ["Three frozen backbones", "DINOv2, ConvNeXtV2 and ViT. Embed each frame once, reuse forever — which is what bought 75 experiments."],
    ["A 2,307-parameter probe", "The classifier itself. There are not enough labels to fine-tune anything larger, and pretending otherwise would leak."],
    ["The spec's model lost", "The proposal named ConvNeXtV2. Measured without leakage, DINOv2 leads cross-venue recall at 0.930."],
  ];
  items.forEach(([head, body], i) => {
    card(s, {
      x: M + i * (w + 0.24), y: 4.4, w, h: 1.68, head, body,
      tint: i === 2 ? SOFT_A : PAPER, headH: 0.4, headSize: 15, bodySize: 13,
    });
  });
  foot(s, "What I built — models");
  s.addNotes(
    "A quick word on the models, because it is the first thing anyone asks.\n\n" +
    "Two hundred million frozen parameters, and eleven thousand trained ones. The backbones " +
    "never saw this data during training - they are fixed feature extractors. That ratio is " +
    "what made the experimental breadth possible: embed the corpus once, and every downstream " +
    "experiment is a probe on cached vectors that costs seconds instead of hours.\n\n" +
    "The third card is worth saying out loud. The proposal named ConvNeXtV2. Under " +
    "leakage-free evaluation DINOv2 leads and ConvNeXtV2 does not. The decision was made on a " +
    "measurement, and the measurement was taken after the proposal. That is the process " +
    "working, not a mistake being corrected."
  );
}

// =====================================================================================
// 16b - WHY WE DID NOT TRAIN THE MODELS
// =====================================================================================
{
  const s = slide("dark");
  eyebrow(s, "Models", GREEN);
  title(s, "Why we trained nothing — and it is the data, not the compute",
    { y: 1.15, size: 34, color: WHITE, h: 0.9, w: 11.6 });
  const w = (CW - 3 * 0.24) / 4;
  const items = [
    ["Fine-tuning needs variety we do not have", "Updating 200M parameters on ~150 distinct scenes does not learn football. It learns these pitches."],
    ["The data is sensitive, so it cannot just be grown", "More venues means more consent conversations. The bottleneck is permission."],
    ["Our own numbers say so", "98.5% near-duplicates, and accuracy that falls when labels rise from 300 to 671."],
    ["Frozen also bought the breadth", "Embed once, reuse forever — which is how 75 experiments fit in one thesis."],
  ];
  items.forEach(([head, body], i) => {
    const x = M + i * (w + 0.24);
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.3, w, h: 2.5, rectRadius: 0.1, fill: { color: DARK2 },
    });
    s.addText(head, {
      x: x + 0.26, y: 2.54, w: w - 0.52, h: 0.86, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: WHITE, valign: "top",
      lineSpacing: 17,
    });
    s.addText(body, {
      x: x + 0.26, y: 3.46, w: w - 0.52, h: 1.15, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12.5, color: "AFCBBC", valign: "top", lineSpacing: 16,
    });
  });
  s.addText("Freezing is a defence against overfitting first, and an efficiency win second.", {
    x: M, y: 5.15, w: CW, h: 0.5, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: AMBER,
  });
  foot(s, "Models — why not", true);
  s.addNotes(
    "This is the slide I most want to land, because 'why didn't you fine-tune?' is a " +
    "guaranteed question.\n\n" +
    "Fine-tuning a two-hundred-million-parameter backbone on roughly a hundred and fifty " +
    "distinct scenes does not teach it football. It teaches it these pitches, under these " +
    "floodlights, from these camera angles - and then either the cross-venue number collapses, " +
    "or, far worse, it does not collapse and I believe it.\n\n" +
    "And I could not collect my way out, because this is footage of identifiable people. The " +
    "bottleneck is permission.\n\n" +
    "So freezing is a defence against overfitting first. The efficiency - embed once, reuse " +
    "forever, seventy-five experiments - is a second and very welcome consequence."
  );
}

// =====================================================================================
// 17 - THE NOVEL MODULES, REPORTED AS THEY CAME OUT
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "What I built — reported either way");
  title(s, "Two novel modules, and what they actually gave", { h: 1.3 });
  const w = (CW - 0.34) / 2;
  card(s, {
    x: M, y: 2.35, w, h: 3.2, tint: SOFT_C, headH: 0.44,
    head: "Gated multi-backbone fusion — negative",
    body: "The architecture the proposal specified, built and measured. Routing is worth " +
          "−0.024 cross-venue recall against the same head with the gate switched off, " +
          "on one informative fold of seven.\n\nThe gate puts about 0.70 of its weight on " +
          "DINOv2 in every fold: a learned constant wearing a router's costume.\n\nIt can " +
          "route — on a fixture where the useful backbone flips, it scores 1.000 against " +
          "0.671. The null is about the data, not the code.",
    headSize: 16, bodySize: 13,
  });
  card(s, {
    x: M + w + 0.34, y: 2.35, w, h: 3.2, tint: SOFT_A, headH: 0.44,
    head: "STAN — preliminary, and saturated",
    body: "A 1,651-parameter temporal model against four tuned baselines including an HMM. " +
          "It scores a perfect 1.000 on 200 composed slots, beating the HMM by 0.105.\n\n" +
          "That is an exhausted test set, not a win. The real test set is two slots.\n\n" +
          "Re-composed from four further draws: 1.000 three times, then 0.910 and 0.800. The " +
          "ordering survived; the number did not.",
    headSize: 16, bodySize: 13,
  });
  s.addText("Both are reported as they came out. The ≥30-slot rule is enforced in code, not in a caveat.",
    { x: M, y: 5.72, w: CW, h: 0.4, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, italic: true, color: GREEN_D });
  foot(s, "What I built — novel modules");
  s.addNotes(
    "The thesis proposed two novel modules, and I want to report both the way they came out.\n\n" +
    "The gated fusion head is a negative result. Routing between backbones is worth minus " +
    "0.024 cross-venue recall against the same head with the gate off. I built the ablation " +
    "as a three-rung ladder - uniform, constant, learned - because a simple on/off switch " +
    "credited routing with what is really a learned constant.\n\n" +
    "STAN scores a perfect 1.000, and I am going to argue against my own number. The composed " +
    "label is a deterministic function of five templates, so a model that reads contiguity " +
    "recovers the generating process. That is a saturated benchmark. The real test set is two " +
    "slots, and the thirty-slot threshold raises an exception in code rather than sitting in " +
    "a footnote.\n\n" +
    "I re-drew it before anyone asked - same check that retracted the augmentation headline."
  );
}

// =====================================================================================
// 17b - THE RULE TABLE
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Rules");
  title(s, "A table, not a black box");
  sub(s, "First matching row wins. n = people inside the pitch boundary; ball = a ball seen inside it; m = motion.",
    { w: 11.8, h: 0.5 });
  const rows = [
    ["1", "Detector unavailable", "UNCERTAIN — a missing detector is not an empty pitch"],
    ["2", "No pitch boundary", "UNCERTAIN — mandatory on the deployed path"],
    ["3", "n = 0, nothing moving", "EMPTY"],
    ["4", "n = 0, but something moved", "UNCERTAIN"],
    ["5", "1 ≤ n ≤ 4", "NOT A GAME — too few, ball or no ball"],
    ["6", "n > 4, and a ball, and motion", "ACTIVE PLAY — the only way in"],
    ["7", "n > 4, otherwise", "NOT A GAME — a crowd that is not playing"],
  ];
  rows.forEach(([n, cond, verdict], i) => {
    const y = 2.5 + i * 0.5;
    const hot = i === 5;
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y, w: CW, h: 0.44, rectRadius: 0.06,
      fill: { color: hot ? SOFT_G : (i % 2 === 0 ? PAPER : WHITE) },
    });
    s.addText(n, {
      x: M + 0.2, y, w: 0.5, h: 0.44, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 14, bold: true, color: hot ? GREEN_D : MUTED,
      valign: "middle",
    });
    s.addText(cond, {
      x: M + 0.8, y, w: 5.0, h: 0.44, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, bold: hot, color: INK, valign: "middle",
    });
    s.addText(verdict, {
      x: M + 6.0, y, w: CW - 6.2, h: 0.44, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, bold: hot, color: hot ? GREEN_D : MUTED,
      valign: "middle",
    });
  });
  s.addText("Five people make a game, four or fewer do not — that is the facility's rule, not something we fitted.",
    { x: M, y: 6.18, w: CW, h: 0.4, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13.5, italic: true, color: MUTED });
  foot(s, "Rules — the decision table");
  s.addNotes(
    "This is the deployed decision layer, and it is deliberately a table a facility manager " +
    "can read and argue with.\n\n" +
    "Two numbers in it are the facility's rule rather than something I fitted: five people " +
    "make a game, four or fewer do not. The rest are fitted on one camera and frozen with the " +
    "commit that froze them.\n\n" +
    "Rows six and seven changed late, and they invert what I pre-registered. Originally play " +
    "was the default above the head count. The facility's rule is the opposite - a game has a " +
    "ball in it and people moving, and a crowd standing on a pitch is not a booking being " +
    "used. Play now has to be shown, not assumed. That is stricter, and it costs recall " +
    "exactly where the detector cannot see the ball - which I measured rather than argued " +
    "about."
  );
}

// =====================================================================================
// 17c - THE SWITCHES AND WHAT THEY COST
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Rules — the switches");
  title(s, "Every switch, and what it costs");
  const w = (CW - 2 * 0.24) / 3;
  const items = [
    ["Ball required", "A game has a ball. Strict — and it costs recall where the detector cannot see one: 0.996 → 0.308."],
    ["Ball must be in play", "A ball lying on the grass is furniture beside a mower, not a match."],
    ["Motion required", "People moving, measured across a short burst of frames."],
    ["People threshold", "5 or more is a game; 1–4 is a small group."],
    ["Pitch boundary", "Feet inside the polygon. Correcting one boundary moved a result by 0.037."],
    ["Detector confidence", "Separate thresholds for people and for the ball."],
  ];
  items.forEach(([head, body], i) => {
    card(s, {
      x: M + (i % 3) * (w + 0.24), y: 2.3 + Math.floor(i / 3) * 1.62, w, h: 1.44,
      head, body, tint: i % 2 === 0 ? PAPER : SOFT_G, headH: 0.4,
      headSize: 15, bodySize: 13,
    });
  });
  card(s, {
    x: M, y: 5.6, w: CW, h: 1.02, tint: SOFT_A,
    body: "A cue that cannot be measured is reported, never assumed. If motion cannot be " +
          "computed, the clause is skipped and the trace says so — because a requirement " +
          "that silently never fires is the shape of every guard this project found not " +
          "guarding.",
    bodySize: 13.5,
  });
  foot(s, "Rules — switches");
  s.addNotes(
    "Each switch exists because its cost is a decision somebody should be able to take " +
    "deliberately.\n\n" +
    "The ball requirement is the clearest example. Cross-venue ball recall is 0.40, and " +
    "between 0.06 and 0.89 depending on the venue. Turning it on drops play recall from 0.996 " +
    "to 0.308 at the worst venues. That is the facility's rule and it is expensive; whoever " +
    "turns it off should be able to see exactly what they are buying.\n\n" +
    "The card at the bottom is a rule about rules, and it came out of finding six safeguards " +
    "in this project that were doing nothing at all."
  );
}

// =====================================================================================
// 17d - PREPROCESSING
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Preprocessing");
  title(s, "What happens to a frame before the model sees it", { h: 1.3 });
  const w = (CW - 3 * 0.22) / 4;
  const items = [
    ["Letterbox 224×224", "Pad rather than squash."],
    ["Undistort", "Take the barrel out of a wide CCTV lens."],
    ["Crop", "Centre or top — drop sky and car park."],
    ["Standardise per image", "Normalise each frame to its own statistics."],
    ["CLAHE", "Local contrast for glare. The proposal expected a gain; it cost 0.27."],
    ["Gamma & saturation", "Brightness curve, and colour down to grayscale."],
    ["Denoise & sharpen", "Sensor noise out, edges back."],
    ["Blur", "A control, not a candidate — it removes the people."],
  ];
  items.forEach(([head, body], i) => {
    card(s, {
      x: M + (i % 4) * (w + 0.22), y: 2.3 + Math.floor(i / 4) * 1.68, w, h: 1.5,
      head, body, tint: i % 2 === 0 ? PAPER : SOFT_G, headH: 0.42,
      headSize: 14.5, bodySize: 12.5,
    });
  });
  s.addText("Experiments and the live pipeline share ONE preprocessing module. The pilot proved what happens when they drift apart.",
    { x: M, y: 5.72, w: CW, h: 0.4, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, italic: true, color: GREEN_D });
  foot(s, "Preprocessing");
  s.addNotes(
    "Eight switches, and the same code path runs in the experiments and in the live system. " +
    "That shared path is not tidiness - the pilot had two, they drifted, and the numbers " +
    "stopped meaning the same thing.\n\n" +
    "Two to point at. CLAHE is the one the proposal predicted would fix floodlight glare; " +
    "measured honestly it costs 0.27 macro-F1. And blur is in there as a control rather than " +
    "a candidate: at high sigma it removes the people, and seeing what that does to the score " +
    "tells you what the score is actually reading."
  );
}

// =====================================================================================
// 18 - THE SEARCHES (statement)
// =====================================================================================
{
  const s = slide("dark");
  statement(s, {
    eyebrow: "The configuration searches",
    title: "We tuned hard. Twice.\nNeither winner meant what it looked like.",
    size: 38,
  });
  const w = (CW - 0.34) / 2;
  const items = [["88", "preprocessing runs — greedy, four rounds, two backbones"],
                 ["375", "prompt sets — scored exhaustively, with no labels at all"]];
  items.forEach(([v, l], i) => {
    const x = M + i * (w + 0.34);
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 4.3, w, h: 1.55, rectRadius: 0.1, fill: { color: DARK2 },
    });
    s.addText(v, {
      x: x + 0.3, y: 4.5, w: 2.2, h: 0.95, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 44, bold: true, color: GREEN, valign: "middle",
    });
    s.addText(l, {
      x: x + 2.6, y: 4.5, w: w - 2.9, h: 1.0, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, color: "C8DCD1", valign: "middle", lineSpacing: 19,
    });
  });
  foot(s, "The searches", true);
  s.addNotes(
    "Now the two experiments I said I would spend time on. These are the ones I would have " +
    "reported as clean successes six months ago, and both turned out to be measuring " +
    "something other than what they claimed.\n\n" +
    "Two searches with opposite cost profiles. The preprocessing search asks which image " +
    "preparation helps the backbone; every candidate needs a fresh embedding pass over the " +
    "whole dataset, so it has to be greedy. The prompt search asks how well we can do with no " +
    "labels at all, just by describing the three classes well; prompts are a few short " +
    "strings against embeddings computed once, so it can be exhaustive.\n\n" +
    "The same guard applies to both: every cross-venue fold is 100% active play, so recall " +
    "alone can be bought by answering 'playing' more often. Everything here is ranked on " +
    "recall minus false-play instead."
  );
}

// =====================================================================================
// 19 - PREPROCESSING SEARCH
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Search 1 — preprocessing");
  title(s, "The winner sits inside the noise");
  const w = (CW - 2 * 0.24) / 3;
  card(s, {
    x: M, y: 2.35, w, h: 2.3, tint: SOFT_G, headH: 0.44,
    head: "What it found",
    body: "A best recipe scoring a perfect 1.000 on the cross-venue folds, after four greedy " +
          "rounds each adding the switch that helped most.",
    headSize: 15.5, bodySize: 13.5,
  });
  stat(s, { x: M + w + 0.24, y: 2.35, w, h: 2.3, value: "0.0119", tint: SOFT_C, color: CORAL,
    size: 36, label: "One frame in the smallest venue fold moves the headline by this much." });
  stat(s, { x: M + 2 * (w + 0.24), y: 2.35, w, h: 2.3, value: "0.092", tint: SOFT_C,
    color: CORAL, size: 36,
    label: "How wide the confidence band is at the median configuration." });
  card(s, {
    x: M, y: 4.82, w: CW, h: 1.3, tint: SOFT_A,
    head: "And the switch the proposal expected to help",
    body: "CLAHE was predicted to fix floodlight glare. Measured on the honest split, it " +
          "costs DINOv2 0.27 macro-F1.",
    headSize: 15, bodySize: 14,
  });
  foot(s, "The searches — preprocessing");
  s.addNotes(
    "Eighty-eight runs over crop, undistort, gamma, saturation, sharpening, denoise and " +
    "CLAHE. It produced a clear winner with a perfect score.\n\n" +
    "Then I measured the resolution of the search itself, and that is the rest of the slide. " +
    "One frame moving in the smallest venue fold shifts the headline by 0.0119. The bootstrap " +
    "interval over the seven folds is 0.092 wide at the median configuration. Almost every " +
    "difference the search ranked on is smaller than its own error bar.\n\n" +
    "So the ranking is real arithmetic on unreal precision. The search is not wrong - it is " +
    "finer-grained than the evidence underneath it, and saying so is more useful than " +
    "reporting a winner.\n\n" +
    "The bottom card is the other half: the proposal predicted CLAHE would help with glare. " +
    "It costs 0.27. The plan was wrong and the measurement said so."
  );
}

// =====================================================================================
// 20 - PROMPT SEARCH
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Search 2 — prompts, with zero labels");
  title(s, "Wording moved the score nine times more than the model did", { h: 1.3 });
  s.addChart(pres.ChartType.bar, [{
    name: "Span of macro-F1",
    labels: ["How you word the prompt", "Which of 3 backbones you pick"],
    values: [0.7265, 0.082],
  }], {
    x: M, y: 2.5, w: 7.5, h: 2.6,
    barDir: "bar", chartColors: [AMBER, GREEN_D],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    dataLabelFontFace: BODY, dataLabelFontSize: 13, dataLabelColor: INK,
    showLegend: false, showTitle: false,
    catAxisLabelColor: INK, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 13,
    valAxisLabelColor: MUTED, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 11,
    valAxisMaxVal: 0.85, valGridLine: { color: "E4EAE6", size: 1 },
    catGridLine: { style: "none" }, barGapWidthPct: 60,
  });
  s.addText("Worst wording 0.021  →  best wording 0.747", {
    x: M, y: 5.2, w: 7.5, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 14, italic: true, color: MUTED,
  });
  card(s, {
    x: M + 7.85, y: 2.5, w: CW - 7.85, h: 1.55, tint: SOFT_A,
    head: "Only 22.9% beat the trained probe",
    body: "The median prompt set loses to it. “Zero-shot works” is only true if you " +
          "already know the wording — which needs labels.",
    headSize: 15, bodySize: 13,
  });
  card(s, {
    x: M + 7.85, y: 4.25, w: CW - 7.85, h: 1.55, tint: SOFT_C,
    head: "Picked, not proven",
    body: "The winner leads a pre-declared prompt by 0.447 — on the very folds it was " +
          "selected from. Selection bias, as a number.",
    headSize: 15, bodySize: 13,
  });
  foot(s, "The searches — prompts");
  s.addNotes(
    "375 prompt sets, no labels used at all, scored exhaustively.\n\n" +
    "Here is the number I find genuinely uncomfortable. The span from worst wording to best " +
    "wording is 0.726 macro-F1. Choosing between the three trained backbones - the decision " +
    "the entire model chapter is about - moves the score by 0.082. How you write the sentence " +
    "matters about nine times more than which model you pick.\n\n" +
    "And then the two guards on the right, both of which I want to state before anyone else " +
    "does. Only 22.9% of prompt sets beat the best trained probe and the median one loses, so " +
    "'zero-shot works' is only true if you already know which wording to use - and knowing " +
    "that requires labels. Second, the winner's lead of 0.447 is measured on the same folds " +
    "it was selected from. That is disclosed in the pre-registration as an undeclared search " +
    "family, and the hypothesis tests deliberately use a different, pre-declared prompt."
  );
}

// =====================================================================================
// 21 - AUGMENTATION: THE QUESTION
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Augmentation");
  title(s, "A good question, cheaply asked");
  sub(s, "A probe trained on camera A has to work on camera B. Can augmentation close that gap with no labels from B?",
    { w: 11.6, h: 0.7 });
  const w = (CW - 2 * 0.24) / 3;
  stat(s, { x: M, y: 2.95, w, h: 1.8, value: "0.441", tint: SOFT_C, color: CORAL,
    label: "macro-F1 on camera B, having seen no labels from it" });
  stat(s, { x: M + w + 0.24, y: 2.95, w, h: 1.8, value: "0.990", tint: SOFT_G, color: GREEN_D,
    label: "after one labelled frame of camera B" });
  stat(s, { x: M + 2 * (w + 0.24), y: 2.95, w, h: 1.8, value: "?", tint: SOFT_A, color: "B26A00",
    label: "can brightness and gamma augmentation get there for free?" });
  card(s, {
    x: M, y: 4.92, w: CW, h: 1.3, tint: PAPER,
    body: "Augmentation is not a flag in the main benchmark, and that is a budget decision. " +
          "Every other experiment fits a probe on cached embeddings; augmentation happens " +
          "before the backbone, so each augmented view needs its own forward pass and the " +
          "cache stops being a cache.",
    bodySize: 13.5,
  });
  foot(s, "Augmentation — the question");
  s.addNotes(
    "Now the augmentation experiment.\n\n" +
    "The question is a good one and it is the deployment question. A probe trained on camera " +
    "A scores 0.441 on camera B, which watches the same pitch from a different angle with " +
    "different exposure. One labelled frame of camera B takes it to 0.99. Can augmentation - " +
    "varying brightness, gamma and sensor noise during training - close that gap with no " +
    "labels from B at all?\n\n" +
    "The bottom card explains why this is one experiment on one boundary rather than a flag " +
    "on everything: augmentation happens before the backbone, so it breaks the embedding " +
    "cache that the whole experimental programme is built on."
  );
}

// =====================================================================================
// 22 - AUGMENTATION: THE FIRST ANSWER
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Augmentation — the first answer");
  title(s, "It worked. We wrote it up.");
  const w = (CW - 0.34) / 2;
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 2.4, w, h: 3.1, rectRadius: 0.1, fill: { color: SOFT_G },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: M + 0.34, y: 2.66, w: 1.5, h: 0.36, rectRadius: 0.18, fill: { color: GREEN_D },
  });
  s.addText("PUBLISHED", {
    x: M + 0.34, y: 2.66, w: 1.5, h: 0.36, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10.5, bold: true, charSpacing: 1.4, color: WHITE,
    align: "center", valign: "middle",
  });
  s.addText(
    [
      { text: "0.000", options: { fontFace: HEAD, fontSize: 40, bold: true, color: MUTED } },
      { text: "   →   ", options: { fontFace: BODY, fontSize: 20, color: MUTED } },
      { text: "0.687", options: { fontFace: HEAD, fontSize: 40, bold: true, color: GREEN_D } },
    ],
    { x: M + 0.34, y: 3.2, w: w - 0.68, h: 0.8, isTextBox: true, margin: 0, valign: "middle" },
  );
  s.addText("empty-pitch recall on a camera the model had never seen, with no labels from it",
    { x: M + 0.34, y: 4.04, w: w - 0.68, h: 0.66, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14, color: MUTED, valign: "top", lineSpacing: 18 });
  s.addText("macro-F1  0.855", {
    x: M + 0.34, y: 4.76, w: w - 0.68, h: 0.45, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 21, bold: true, color: INK,
  });
  card(s, {
    x: M + w + 0.34, y: 2.4, w, h: 1.48, tint: PAPER,
    head: "The preset",
    body: "Brightness, gamma and sensor noise — matched to the way the two cameras " +
          "actually differ.",
    headSize: 15, bodySize: 13.5,
  });
  card(s, {
    x: M + w + 0.34, y: 4.02, w, h: 1.48, tint: SOFT_A,
    head: "With one note attached",
    body: "“Only one draw has been taken.” That note turned out to be the finding.",
    headSize: 15, bodySize: 13.5,
  });
  foot(s, "Augmentation — the first answer");
  s.addNotes(
    "The first answer was yes, and it was a good result. One draw of the light preset - " +
    "brightness, gamma, sensor noise - took empty-pitch recall on the unseen camera from zero " +
    "to 0.687, with no labels from that camera at all. Macro-F1 0.855.\n\n" +
    "It went into the write-up as a finding, with one note attached: only one draw had been " +
    "taken.\n\n" +
    "Pause here. Let them sit with it as a success for a moment, because the next slide is " +
    "the whole point."
  );
}

// =====================================================================================
// 23 - AUGMENTATION: FIVE DRAWS (chart)
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Augmentation — then we ran it four more times");
  title(s, "Same code, same data. Only the random draw changed.", { h: 1.3 });
  s.addChart(pres.ChartType.bar, [{
    name: "macro-F1",
    labels: ["Draw 1\n(published)", "Draw 2", "Draw 3", "Draw 4", "Draw 5"],
    values: [0.8550, 0.4136, 0.3510, 0.3501, 0.3479],
  }], {
    x: M, y: 2.5, w: 7.9, h: 3.25,
    barDir: "col", chartColors: [GREEN_D, CORAL, CORAL, CORAL, CORAL],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    dataLabelFontFace: BODY, dataLabelFontSize: 12, dataLabelColor: INK,
    showLegend: false, showTitle: false,
    catAxisLabelColor: MUTED, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 11.5,
    valAxisLabelColor: MUTED, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 11,
    valAxisMaxVal: 1.0, valGridLine: { color: "E4EAE6", size: 1 },
    catGridLine: { style: "none" }, barGapWidthPct: 45,
  });
  card(s, {
    x: M + 8.25, y: 2.5, w: CW - 8.25, h: 1.5, tint: SOFT_A,
    body: "No augmentation at all scores 0.441. Four of the five draws fall below it, and " +
          "the published number was the maximum of five.",
    bodySize: 13.5,
  });
  stat(s, { x: M + 8.25, y: 4.15, w: CW - 8.25, h: 1.6, value: "0.221", tint: SOFT_C,
    color: CORAL, size: 32, label: "standard deviation, on a metric bounded in [0, 1]" });
  foot(s, "Augmentation — five draws");
  s.addNotes(
    "Then I ran the identical experiment four more times, changing nothing but the random " +
    "seed. Same frames, same preset, same probe seed, same test set.\n\n" +
    "0.414. 0.351. 0.350. 0.348.\n\n" +
    "The published 0.855 is the maximum of five, not a typical value. The standard deviation " +
    "is 0.221 on a metric that only runs from zero to one. And four of the five draws are " +
    "worse than using no augmentation at all, which scores 0.441.\n\n" +
    "One more detail that explains the shape. Three draws score exactly 0.3479, which is not " +
    "a coincidence: that is the macro-F1 of answering 'active play' to all 521 test frames. " +
    "What moves between draws is whether the fitted boundary reaches camera B's empty pitch " +
    "at all. It does not degrade gracefully - it is either a working classifier or the " +
    "trivial one."
  );
}

// =====================================================================================
// 24 - THE LESSON (statement)
// =====================================================================================
{
  const s = slide("dark");
  s.background = { color: "4A1721" };
  eyebrow(s, "Augmentation — the lesson", "F2A33C");
  title(s, "The number reproduced perfectly.\nThe result did not.", { y: 1.45, size: 42, color: WHITE, h: 1.7, w: 11.5 });
  const w = (CW - 2 * 0.26) / 3;
  const items = [
    ["A reproducible row", "The result file is unchanged and still reproduces exactly. That is the problem, not the defence."],
    ["A test could not catch it", "A unit test pinning the published value would have passed forever. Only re-drawing the randomness exposed it."],
    ["So: five draws, always", "Every headline now runs five times. I applied it to my own strongest result before anyone asked."],
  ];
  items.forEach(([head, body], i) => {
    const x = M + i * (w + 0.26);
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 3.5, w, h: 2.0, rectRadius: 0.1, fill: { color: "5F2029" },
    });
    s.addText(head, {
      x: x + 0.26, y: 3.74, w: w - 0.52, h: 0.42, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 16, bold: true, color: WHITE,
    });
    s.addText(body, {
      x: x + 0.26, y: 4.2, w: w - 0.52, h: 1.1, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13, color: "F2D6DA", valign: "top", lineSpacing: 17,
    });
  });
  foot(s, "Augmentation — the lesson", true);
  s.addNotes(
    "So the headline was retracted, and I want to be precise about what was retracted. The " +
    "row in the results file is unchanged and still reproduces exactly. What was withdrawn is " +
    "the claim about the method. It is now restated as a claim about that one draw, and it is " +
    "only ever quoted next to the spread.\n\n" +
    "The transferable lesson is the middle card. A unit test that fixed the published value " +
    "would have passed forever, because the artefact was reproducible. Reproducibility " +
    "checked the wrong thing. Only re-drawing the randomness caught it.\n\n" +
    "That is now standing practice in this project, and I ran it on my own strongest result - " +
    "the temporal model - and volunteered the spread before anyone asked for it. Running that " +
    "check on your own headline is the point."
  );
}

// =====================================================================================
// 25 - THE SOLUTION (statement)
// =====================================================================================
{
  const s = slide("dark");
  statement(s, {
    eyebrow: "How the research provides a solution",
    title: "The fix was never a bigger model.",
    size: 46, titleH: 1.2,
    sub: "Three interventions, each measured on the boundary it was meant to help.",
    subY: 3.1,
  });
  foot(s, "The solution", true);
  s.addNotes(
    "So what did I do about all of this?\n\n" +
    "This study seeks to address the problem in three ways, and the common thread is worth " +
    "naming before I show them: not one of them is a bigger or better neural network. Every " +
    "gain in this project came from cheap structure around the model, or from labelling a " +
    "handful of the right frames."
  );
}

// =====================================================================================
// 26 - THREE FIXES
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "The solution");
  title(s, "Three fixes, all cheap");
  const w = (CW - 2 * 0.26) / 3;
  beat(s, { x: M, y: 2.35, w, h: 2.5, tint: SOFT_G, head: "Put rules in front of the model",
    from: "0.617", to: "0.012",
    label: "False “playing” alarms on empty pitches, once the pitch boundary, motion and person checks can veto the model." });
  beat(s, { x: M + w + 0.26, y: 2.35, w, h: 2.5, tint: SOFT_G, head: "Generate the missing case",
    from: "0.768", to: "0.024",
    label: "31 generated empty-pitch frames added to training — and play-recall went up, not down." });
  beat(s, { x: M + 2 * (w + 0.26), y: 2.35, w, h: 2.5, tint: SOFT_G, head: "Label a few frames per camera",
    from: "0.441", to: "0.990",
    label: "One labelled frame of a new camera. And five from it beat those five plus 775 from the old one." });
  card(s, {
    x: M, y: 5.0, w: CW, h: 1.1, tint: SOFT_A,
    body: "What buys the accuracy is having any labels from the new camera — not a large " +
          "corpus from an old one. That is the weaker claim, and it is the true one.",
    bodySize: 14.5,
  });
  foot(s, "The solution — three fixes");
  s.addNotes(
    "First: cheap geometric and motion rules that sit in front of the model and can overrule " +
    "it. False alarms on empty pitches fall from 0.617 to 0.012. Counting every kind of wrong " +
    "verdict rather than only play-shaped ones, error falls from 1.00 to 0.11.\n\n" +
    "Second: generating the cell the data does not have. Thirty-one synthetic empty-pitch " +
    "frames take cross-venue false alarms from 0.768 to 0.024 while raising play-recall.\n\n" +
    "Third is the deployment recipe, and it is the one the facility can act on tomorrow. One " +
    "labelled frame of a new camera takes every backbone to about 0.98. And here is the " +
    "control that reframes it: from five frames, training on those five alone matches " +
    "training on those five plus 775 frames from the source camera. The source set stops " +
    "contributing.\n\n" +
    "So I state the weaker claim, because it is the true one: what buys the accuracy is any " +
    "labels from the new camera."
  );
}

// =====================================================================================
// 27 - GUARDS THAT WERE NOT GUARDING
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "The solution — the method contribution");
  title(s, "Every guard is verified by breaking it");
  sub(s, "Six safeguards in this project were doing nothing. Each was found the same way — by deliberately breaking what it protects.");
  const w = (CW - 0.3) / 2;
  const items = [
    ["A test-set lock", "Resolved against the working directory, so it locked nothing."],
    ["A confidence threshold", "Set to 0.0 — a gate that can never fire."],
    ["A gate criterion", "Grepped for a word instead of running the verifier."],
    ["Evidence selection", "Recorded a file path of None for every frame it chose."],
  ];
  items.forEach(([head, body], i) => {
    card(s, {
      x: M + (i % 2) * (w + 0.3), y: 3.0 + Math.floor(i / 2) * 1.5, w, h: 1.32,
      head, body, tint: i % 2 === 0 ? PAPER : SOFT_C, headH: 0.38,
      headSize: 15, bodySize: 13,
    });
  });
  s.addText("This practice is a contribution of the thesis — and it is what found half of the results you have just seen.",
    { x: M, y: 5.95, w: CW, h: 0.4, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 14.5, italic: true, color: GREEN_D });
  foot(s, "The solution — method");
  s.addNotes(
    "This slide is the methodological half of the contribution, and I think it is the part " +
    "that transfers beyond football pitches.\n\n" +
    "Six safeguards in this project were doing nothing at all. A test-set lock resolved " +
    "against the working directory. A processor-geometry fingerprint that never changed. A " +
    "confidence threshold set to zero, which can never fire. A gate criterion that searched " +
    "for a word instead of running the verifier. A live-camera class that raised an exception " +
    "on the attribute it needed, so it could never run. And evidence selection that recorded " +
    "a path of None for every frame it chose.\n\n" +
    "Every one of them passed whatever test existed. So each new guard is now verified by " +
    "deliberately breaking the thing it guards and confirming the guard fires. That practice " +
    "found half the results in this talk.\n\n" +
    "The root of it is one observation from early in the project: a wrong result that looks " +
    "plausible is invisible."
  );
}

// =====================================================================================
// 28 - ETHICS
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "The solution — what it will not do");
  title(s, "The system never bills anyone");
  const w = (CW - 0.34) / 2;
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 2.35, w, h: 3.25, rectRadius: 0.1, fill: { color: SOFT_G },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 1, angle: 90, opacity: 0.07 },
  });
  s.addText("Enforced, not promised", {
    x: M + 0.34, y: 2.62, w: w - 0.68, h: 0.4, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: INK,
  });
  bullets(s, [
    "Advisory is the only output type the decision layer can produce.",
    "Human confirmation is a property, not a setting — nothing can switch it off.",
    "There is no code path that acts on a verdict.",
  ], { x: M + 0.34, y: 3.12, w: w - 0.68, h: 2.2, size: 14.5, color: "3C5449" });

  const right = [
    ["Anomalies are per field, never per person", "Reconciliation has no field that could name an individual."],
    ["Review never becomes an anomaly", "The system may not convert its own uncertainty into someone else's error."],
    ["Faces are redacted before publication", "Detection and pixelation, then a whole-frame blur floor."],
  ];
  right.forEach(([head, body], i) => {
    card(s, {
      x: M + w + 0.34, y: 2.35 + i * 1.33, w, h: 1.26, head, body,
      tint: i % 2 === 0 ? PAPER : SOFT_A, headH: 0.3, headSize: 14, bodySize: 12,
    });
  });
  foot(s, "The solution — ethics");
  s.addNotes(
    "One slide on what the system deliberately will not do, because auditing how a facility's " +
    "pitches are used is close to auditing the people who work there.\n\n" +
    "The commitment is that the system never bills and never acts. What makes that credible " +
    "is that it is structural rather than a policy sentence: Advisory is the only type the " +
    "decision layer can return, human confirmation is a property rather than a parameter, and " +
    "there is no code path that acts.\n\n" +
    "The middle item on the right is the subtle one and I think it is the most important. If " +
    "the model is unsure, that uncertainty belongs to the model. Turning a REVIEW into a flag " +
    "against a member of staff would be laundering the system's own weakness into someone " +
    "else's record."
  );
}

// =====================================================================================
// 29 - LIMITS
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Limits");
  title(s, "What this study cannot claim");
  sub(s, "Named first, because an examiner will find them anyway.");
  const w = (CW - 2 * 0.26) / 3;
  const items = [
    ["1", "Empty pitches exist at one venue only", "So a cross-venue three-class evaluation is impossible on this corpus."],
    ["2", "243 empty frames are 3 scenes", "The effective sample is scenes, not frames. Recounted that way, six comparisons stop being significant."],
    ["3", "Nothing has run on the target hardware", "Every latency number in the thesis comes from a laptop."],
  ];
  items.forEach(([n, head, body], i) => {
    card(s, {
      x: M + i * (w + 0.26), y: 3.0, w, h: 2.45,
      badge: n, badgeFill: CORAL, head, body,
      tint: SOFT_C, headH: 0.7, headSize: 15.5, bodySize: 13,
    });
  });
  s.addText("Several tests could not have reached significance: over seven folds the smallest attainable p-value is 0.0156. We report which case applies.",
    { x: M, y: 5.65, w: CW, h: 0.45, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13.5, italic: true, color: MUTED });
  foot(s, "Limits");
  s.addNotes(
    "Three limits, and I would rather state them than be asked.\n\n" +
    "Every empty pitch in the corpus comes from a single venue, so a three-class cross-venue " +
    "evaluation simply cannot be run here.\n\n" +
    "The second one is the one I find most instructive. 243 held-out empty frames sound like " +
    "a healthy sample, but they are three distinct scenes - consecutive views from one fixed " +
    "camera. Six comparisons that were significant to p equals 8 times ten to the minus 53 " +
    "did not survive being recounted that way.\n\n" +
    "And nothing has run on the target mini-PC. Every speed number is from a laptop.\n\n" +
    "The line at the bottom matters to me: 'not significant' and 'the design could not have " +
    "produced significance' are different statements, and I report which one applies."
  );
}

// =====================================================================================
// 30 - WHAT WOULD CHANGE IT
// =====================================================================================
{
  const s = slide();
  eyebrow(s, "Limits — and the way out");
  title(s, "What would actually change this");
  sub(s, "Ordered by value, not by effort.");
  const items = [
    ["1", "Empty-pitch footage at a second venue",
     "Twenty to thirty minutes of an unoccupied pitch, day and night. Fixes the confound, the external-validity limit and the scope reduction at once."],
    ["2", "About 30 real labelled slots",
     "One conversation with the facility. Lifts the temporal model out of preliminary and gives reconciliation a ground truth."],
    ["3", "Five labelled frames per camera",
     "Already measured, already cheap — and it is now the onboarding recipe we would hand over."],
  ];
  items.forEach(([n, head, body], i) => {
    const y = 2.85 + i * 1.12;
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y, w: CW, h: 0.98, rectRadius: 0.1,
      fill: { color: i % 2 === 0 ? SOFT_G : PAPER },
    });
    badge(s, M + 0.26, y + 0.22, n, GREEN_D);
    s.addText(head, {
      x: M + 1.02, y: y + 0.16, w: 4.3, h: 0.66, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15.5, bold: true, color: INK, valign: "middle",
      lineSpacing: 18,
    });
    s.addText(body, {
      x: M + 5.5, y: y + 0.16, w: CW - 5.8, h: 0.66, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12.5, color: MUTED, valign: "middle", lineSpacing: 16,
    });
  });
  s.addText("The limitations are mostly a data-access problem with a known and inexpensive solution — not a methodological one.",
    { x: M, y: 6.28, w: CW, h: 0.45, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15.5, bold: true, color: GREEN_D });
  foot(s, "Limits — the way out");
  s.addNotes(
    "And here is what would change them, ordered by value rather than by effort.\n\n" +
    "Empty-pitch footage at a second venue is first because one half-hour of recording " +
    "resolves three separate limitations at once. About thirty real labelled slots is second, " +
    "and that is one conversation with the facility rather than any engineering. Five " +
    "labelled frames per camera is third, and it is already measured.\n\n" +
    "Close on the sentence at the bottom, and say it deliberately: the limitations of this " +
    "thesis are mostly a data-access problem with a known and inexpensive solution, not a " +
    "methodological one. That is a much better last word than an apology."
  );
}

// =====================================================================================
// 31 - SUMMARY
// =====================================================================================
{
  const s = slide("dark");
  eyebrow(s, "Summary", GREEN);
  title(s, "In summary", { y: 0.82, size: 40, color: WHITE, h: 0.8 });
  const items = [
    ["Investigated", "Whether cameras a facility already owns can verify which booked hours were actually used — on a CPU, with a human deciding every case."],
    ["Found", "On this data a clock beats a neural network. Lighting and occupancy are confounded so tightly that no protocol on this corpus can separate them."],
    ["Contributed", "An evaluation method: trivial baselines in every protocol, grouped splits, re-drawing randomness rather than re-running it, and every guard verified by breaking it."],
    ["Implies", "For the facility, a five-frame onboarding recipe. For the field, that a benchmark can rank a light meter first and still look healthy."],
  ];
  items.forEach(([head, body], i) => {
    const y = 1.86 + i * 1.12;
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y, w: CW, h: 0.98, rectRadius: 0.1, fill: { color: DARK2 },
    });
    s.addText(head.toUpperCase(), {
      x: M + 0.3, y: y + 0.16, w: 2.0, h: 0.66, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11.5, bold: true, charSpacing: 1.6, color: GREEN,
      valign: "middle",
    });
    s.addText(body, {
      x: M + 2.35, y: y + 0.16, w: CW - 2.65, h: 0.66, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 13.5, color: "DCEFE4", valign: "middle", lineSpacing: 17,
    });
  });
  s.addText("No protocol compensates for a case the data never contains.", {
    x: M, y: 6.26, w: CW, h: 0.44, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, italic: true, color: AMBER,
  });
  foot(s, "Summary", true);
  s.addNotes(
    "In summary, this study investigated whether the cameras a facility already owns can " +
    "verify which booked hours were actually used, on hardware it already has, with a human " +
    "deciding every case.\n\n" +
    "The findings indicate that the system works end to end - and that the evaluation is the " +
    "real result. On this corpus, lighting and occupancy are confounded so tightly that a " +
    "rule reading only the clock outperforms three modern backbones, and one short clip " +
    "containing the missing case reverses that ranking completely.\n\n" +
    "These findings contribute a set of evaluation practices that apply well beyond this " +
    "problem, and they give the facility a concrete, inexpensive recipe for onboarding a new " +
    "camera.\n\n" +
    "Then the last line, slowly, and stop talking: no protocol compensates for a case the " +
    "data never contains."
  );
}

// =====================================================================================
// 32 - THANK YOU
// =====================================================================================
{
  const s = slide("dark");
  s.addShape(pres.ShapeType.ellipse, {
    x: -1.0, y: -1.0, w: 3.6, h: 3.6, fill: { color: GREEN, transparency: 82 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 11.2, y: 5.2, w: 3.2, h: 3.2, fill: { color: AMBER, transparency: 84 },
  });
  s.addText("THANK YOU", {
    x: M, y: 1.9, w: 9, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, bold: true, charSpacing: 3, color: GREEN,
  });
  s.addText("Questions, comments\nand suggestions welcome.", {
    x: M, y: 2.45, w: 9.6, h: 1.8, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 42, bold: true, color: WHITE, lineSpacing: 50, valign: "top",
  });
  s.addText("Thank you for your time and attention.", {
    x: M, y: 4.35, w: 9, h: 0.45, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 17, color: "C8DCD1",
  });
  const chips = ["Would a colour histogram have done this?", "Is the novelty just smoothing?",
                 "How many comparisons before p < 0.05?", "What is the human ceiling?"];
  let cx = M, cy = 5.25;
  chips.forEach((t) => {
    const w = 0.34 + t.length * 0.098;
    if (cx + w > W - M) { cx = M; cy += 0.62; }
    s.addShape(pres.ShapeType.roundRect, {
      x: cx, y: cy, w, h: 0.48, rectRadius: 0.24, fill: { color: DARK2 },
    });
    s.addText(t, {
      x: cx, y: cy, w, h: 0.48, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 12, color: "AFCBBC", align: "center", valign: "middle",
    });
    cx += w + 0.16;
  });
  foot(s, "Thank you", true);
  s.addNotes(
    "Thank you for your time and attention. I welcome any questions, comments, suggestions or " +
    "feedback you may have.\n\n" +
    "The chips are the four questions most likely to come. Short answers:\n\n" +
    "Colour histogram: 0.9616 on the leaky split, and statistically indistinguishable from " +
    "ConvNeXtV2 once you count the 62 distinct scenes rather than 394 frames.\n\n" +
    "Novelty vs smoothing: tested against four tuned baselines including an HMM, beaten by " +
    "0.105 - with the saturation caveat enforced in code.\n\n" +
    "Multiple comparisons: Holm within declared families, and the undeclared search families " +
    "are disclosed as numbered amendments rather than folded in quietly.\n\n" +
    "Human ceiling: no inter-annotator figure exists. Say that plainly - it is a real gap. " +
    "Practise saying 'we do not know' without flinching; an examiner trusts a candidate who " +
    "has bounded their ignorance more than one who has not noticed it."
  );
}

pres.writeFile({ fileName: OUT }).then(() => {
  console.log("wrote " + OUT + "  (" + pageNo + " slides)");
});
