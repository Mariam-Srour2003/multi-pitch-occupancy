"""Per-frame latency and 20-camera throughput (WP0-T10, WP4-T11, RQ2).

The deployment claim is that one CPU-only Mini-PC serves 20-30 cameras at one frame per
minute. That is a throughput question, so this measures two things:

1. **single-frame latency** - warm-up discarded, median and p95, thread count recorded;
2. **20 concurrent streams** - measured, not extrapolated. Twenty cameras is not twenty
   times one frame: the models contend for memory bandwidth and last-level cache.

The pass/fail line is whether one round of 20 frames completes inside the 60-second
sampling cycle. Note this is a *development laptop*, not the target Mini-PC - WP7-T1
repeats it on the real hardware, and only that run settles the deployment claim.

    uv run python experiments/efficiency_latency.py
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from pitch_occupancy.data.manifest import read_manifest
from pitch_occupancy.evaluation.latency import measure_concurrent, measure_latency
from pitch_occupancy.vision.backbones import BACKBONES, embed_batch, load_backbone

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CYCLE_SECONDS = 60.0
N_CAMERAS = 20


def main() -> None:
    import platform

    import torch

    rows = read_manifest(DATASET / "manifest.csv")
    image = Image.open(DATASET / rows[0].file).convert("RGB")

    print(f"machine : {platform.processor() or platform.machine()}")
    print(f"threads : {torch.get_num_threads()}")
    print(f"torch   : {torch.__version__}  (cuda: {torch.cuda.is_available()})\n")

    records = []
    for key in sorted(BACKBONES):
        model, processor, spec = load_backbone(key)

        def one_frame() -> None:
            embed_batch(model, processor, spec, [image])

        single = measure_latency(one_frame, label=f"{key} single", repeats=30, warmup=5)
        print(f"  {single}")

        conc, wall = measure_concurrent(
            one_frame, label=f"{key} x{N_CAMERAS}", n_streams=N_CAMERAS, repeats=3, warmup=1
        )
        naive = single.median_ms * N_CAMERAS / 1000
        headroom = CYCLE_SECONDS / wall if wall else float("inf")
        print(f"  {conc}")
        print(
            f"    {N_CAMERAS} cameras: {wall:.1f}s measured vs {naive:.1f}s if simply "
            f"multiplied  ->  {headroom:.1f}x headroom in a {CYCLE_SECONDS:.0f}s cycle"
            f"  [{'OK' if wall < CYCLE_SECONDS else 'OVER BUDGET'}]\n"
        )

        records.append(
            {
                "backbone": key,
                "hf_id": spec.hf_id,
                "threads": single.threads,
                "single_median_ms": round(single.median_ms, 1),
                "single_p95_ms": round(single.p95_ms, 1),
                "single_mean_ms": round(single.mean_ms, 1),
                "concurrent_streams": N_CAMERAS,
                "concurrent_median_ms": round(conc.median_ms, 1),
                "concurrent_p95_ms": round(conc.p95_ms, 1),
                "round_wall_s": round(wall, 2),
                "naive_extrapolation_s": round(naive, 2),
                "cycle_headroom_x": round(headroom, 2),
                "fits_60s_cycle": wall < CYCLE_SECONDS,
            }
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "efficiency_latency.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"wrote {out}")

    with (RESULTS / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n- {datetime.now(timezone.utc):%Y-%m-%d} | efficiency | "
            f"`python experiments/efficiency_latency.py` | `{out.name}` | "
            f"dev laptop, {records[0]['threads']} threads\n"
        )


if __name__ == "__main__":
    main()
