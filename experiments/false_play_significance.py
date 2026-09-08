"""Is the false-play gap between models real, or could it be sampling noise? (WP4-T4)

`h3_with_false_play.py` produced the finding that reorders the models: ConvNeXtV2 calls
99.2% of held-out empty pitches a match, ViT 83.5%, DINOv2 30.9%, and the clock rule 2.1%.
Those are point estimates on **243 frames**, and a number that reverses a recommendation
should not be quoted without an interval around it.

Three things are computed, and each answers a different question:

* **Bootstrap CIs** — how precisely is each model's own rate known?
* **McNemar, pairwise** — do two models differ on the *same* frames? They predict on an
  identical set, so the comparison is paired and the unpaired test would throw away exactly
  the information that makes it powerful.
* **Holm–Bonferroni** — six pairwise tests are a family, and "we ran six and four were
  significant" is not a finding until the correction is applied.

**The frames are not independent, and that is stated rather than assumed away.** All 243 are
consecutive samples from one camera watching one pitch across two slots; the near-duplicate
audit found that 100% of EMPTY frames have a direct near-duplicate. So the effective sample
is far smaller than 243 and every interval here is **narrower than the truth**. They are
reported as a lower bound on the uncertainty, and a gap that is not significant under this
optimistic accounting is certainly not significant.

    uv run python experiments/false_play_significance.py
"""

from __future__ import annotations

import csv
from itertools import combinations

import numpy as np

from experiments.h3_with_false_play import CACHES, load
from pitch_occupancy.config import settings
from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.data.splits import development_rows
from pitch_occupancy.db.seed import PHYSICAL_CAMERA
from pitch_occupancy.evaluation.stats import bootstrap_ci, holm_bonferroni, mcnemar
from pitch_occupancy.vision.heads import ClockRule, LinearProbe

PLAY, EMPTY = "C2_ACTIVE_PLAY", "C1_EMPTY"
SEED = 42
OUT = settings.results_dir / "false_play_significance.csv"


def correct_on_empties(model: str, rows, X) -> tuple[np.ndarray, list]:
    """Per-frame correctness on held-out empty frames: True where the model says not-PLAY."""
    pos = {r.file: i for i, r in enumerate(rows)}
    cam = lambda r: PHYSICAL_CAMERA.get(r.camera, r.camera)  # noqa: E731
    venue = [r for r in rows if r.venue == "venue_01"]
    train = [r for r in venue if cam(r) == "camera_A"]
    empties = [r for r in venue if cam(r) == "camera_B" and r.class3 == EMPTY]
    head = ClockRule() if model == "clock_rule" else LinearProbe(model, seed=SEED)
    head.fit(X[[pos[r.file] for r in train]], train)
    pred = head.predict(X[[pos[r.file] for r in empties]], empties)
    return np.array([p != PLAY for p in pred]), empties


def _repeat_on_distinct_scenes(correctness, empties):
    """Redo the pairwise tests keeping one frame per near-duplicate group.

    The independence assumption behind McNemar is false here by construction, and a caveat
    is cheaper than a check. This is the check: collapse the near-duplicates and see which
    conclusions are left. It is deliberately pessimistic - a whole slot chains into few
    groups - so a comparison that survives it is not resting on repeated frames.
    """
    import cv2

    from pitch_occupancy.data.dedup import DEFAULT_THRESHOLD, dhash, hamming

    hashes = {}
    for row in empties:
        image = cv2.imread(str(settings.dataset_dir / row.file))
        if image is not None:
            hashes[row.file] = dhash(image)
    if not hashes:
        return None

    # Greedy maximal set of *pairwise* distinct frames, not single-link groups. Chaining
    # collapses 243 consecutive frames of one empty pitch into fewer than three clusters,
    # which leaves nothing to test; requiring each kept frame to differ from every frame
    # already kept is the same idea without the transitive over-merge.
    #
    # Swept over thresholds rather than fixed at one, because "distinct" is a judgement and
    # the answer moves with it: 8-10 scenes at 2 bits, 3 at 6. The count is stable across
    # shuffles of the input order, so it is a property of the frames, not of the greed.
    index = {r.file: i for i, r in enumerate(empties)}
    pairs = list(combinations(correctness, 2))
    out = []
    for threshold in (2, DEFAULT_THRESHOLD):
        keep: list[str] = []
        for f, h in hashes.items():
            if all(hamming(h, hashes[k]) > threshold for k in keep):
                keep.append(f)
        picks = [index[f] for f in keep if f in index]
        if len(picks) < 3:
            out.append((threshold, len(picks), None, len(pairs)))
            continue
        reduced = {m: ok[picks] for m, ok in correctness.items()}
        ps = [mcnemar(reduced[a], reduced[b]).p_value for a, b in pairs]
        _, rejected = holm_bonferroni(ps)
        out.append((threshold, len(picks), sum(rejected), len(pairs)))
    return out


def main() -> None:
    all_rows = development_rows(read_manifest(settings.dataset_dir / "manifest.csv"))
    correctness: dict[str, np.ndarray] = {}
    n_frames = 0

    for model in ["clock_rule", *CACHES]:
        if model == "clock_rule":
            rows, X = all_rows, np.zeros((len(all_rows), 1), dtype=np.float32)
        else:
            path = settings.feature_cache_dir / CACHES[model]
            if not path.exists():
                continue
            rows, X = load(CACHES[model], all_rows)
        ok, empties = correct_on_empties(model, rows, X)
        correctness[model] = ok
        n_frames = len(empties)
        held_out = empties

    if len(correctness) < 2:
        raise SystemExit("need at least two models; are the feature caches built?")

    print(f"held-out EMPTY frames: {n_frames}\n")
    print(f"{'model':13} {'false-play':>11}  {'95% CI':>18}")
    intervals = {}
    for model, ok in correctness.items():
        # bootstrap the false-play indicator, which is simply the complement of correctness
        ci = bootstrap_ci(1.0 - ok.astype(float), np.mean, seed=SEED)
        intervals[model] = ci
        print(f"{model:13} {ci.estimate:11.4f}  [{ci.low:.4f}, {ci.high:.4f}]")

    print(f"\npairwise McNemar on the same {n_frames} frames "
          f"(paired: every model sees an identical set)")
    pairs = list(combinations(correctness, 2))
    results = []
    for a, b in pairs:
        r = mcnemar(correctness[a], correctness[b])
        results.append((a, b, r))
    adjusted, rejected = holm_bonferroni([r.p_value for _, _, r in results])

    print(f"{'comparison':30} {'discordant':>11} {'p':>10} {'p (Holm)':>10} {'sig':>5}")
    for (a, b, r), p_adj, rej in zip(results, adjusted, rejected, strict=True):
        disc = getattr(r, "n_discordant", None)
        disc_s = str(disc) if disc is not None else "-"
        print(f"{a + ' vs ' + b:30} {disc_s:>11} {r.p_value:10.2e} {p_adj:10.2e} "
              f"{'yes' if rej else 'no':>5}")

    print("\nThe frames are not independent - all 243 come from one camera watching one")
    print("pitch, and every EMPTY frame has a direct near-duplicate. Rather than leave")
    print("that as a caveat, the pairwise tests are repeated on one representative per")
    print("near-duplicate group, which is a deliberately pessimistic accounting.")
    survivors = _repeat_on_distinct_scenes(correctness, held_out) or []
    for threshold, kept, still_sig, total in survivors:
        if still_sig is None:
            print(f"\n  at {threshold} bits: only {kept} distinct scenes - too few to test")
            continue
        print(f"\n  at {threshold} bits: {kept} distinct scenes from {n_frames} frames; "
              f"{still_sig} of {total} comparisons stay significant after Holm")
    if survivors and all(s in (0, None) for _, _, s, _ in survivors):
        print("\n  NONE of the six comparisons survives. The p-values above are the")
        print("  arithmetic of counting 243 views of one empty pitch as 243 observations.")
        print("  The point estimates stand as descriptions of behaviour on this data;")
        print("  the claim that the models differ *significantly* does not, and needs")
        print("  empty footage from more than one scene before it can be made.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "a", "b", "estimate", "ci_low", "ci_high",
                    "p_value", "p_holm", "significant", "n_frames"])
        for model, ci in intervals.items():
            w.writerow(["false_play_rate", model, "", f"{ci.estimate:.4f}",
                        f"{ci.low:.4f}", f"{ci.high:.4f}", "", "", "", n_frames])
        for (a, b, r), p_adj, rej in zip(results, adjusted, rejected, strict=True):
            w.writerow(["mcnemar", a, b, "", "", "",
                        f"{r.p_value:.3e}", f"{p_adj:.3e}", rej, n_frames])
    print(f"\nwrote {OUT.name}")


if __name__ == "__main__":
    main()
