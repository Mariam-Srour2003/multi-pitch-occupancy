"""Latency measurement (WP0-T10).

The central deployment claim is that one CPU-only Mini-PC can serve 20-30 cameras at one
frame per minute. That claim rests entirely on per-frame latency, so the measurement has
to be defensible:

* **warm-up runs are discarded** - the first forward pass through a fresh model pays for
  lazy allocation and kernel selection, and including it inflates the mean;
* **median and p95 are reported, not the mean** - a single scheduler hiccup moves a mean
  and the budget question is "how often is it too slow", which is a tail question;
* **thread count is recorded**, because a number measured on 8 threads says nothing about
  a machine running 20 camera streams;
* **concurrent throughput is measured, not extrapolated.** 20 cameras is not 20x the
  single-frame latency: the models contend for memory bandwidth and cache. The honest
  figure is the measured one, and it is the one that decides whether the Mini-PC is
  adequate.
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

__all__ = ["LatencyResult", "measure_latency", "measure_concurrent"]


@dataclass(frozen=True, slots=True)
class LatencyResult:
    label: str
    n: int
    threads: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    samples: tuple[float, ...] = field(default=(), repr=False)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.label:<28} median {self.median_ms:7.1f} ms   p95 {self.p95_ms:7.1f} ms   "
            f"(n={self.n}, threads={self.threads})"
        )


def _summarise(label: str, samples: list[float], threads: int) -> LatencyResult:
    arr = sorted(samples)
    return LatencyResult(
        label=label,
        n=len(arr),
        threads=threads,
        median_ms=float(statistics.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        mean_ms=float(statistics.fmean(arr)),
        min_ms=arr[0],
        max_ms=arr[-1],
        samples=tuple(arr),
    )


def measure_latency(
    fn,
    *,
    label: str,
    repeats: int = 50,
    warmup: int = 5,
    threads: int | None = None,
) -> LatencyResult:
    """Time ``fn()`` repeatedly, discarding the warm-up runs."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    actual_threads = torch.get_num_threads()

    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return _summarise(label, samples, actual_threads)


def measure_concurrent(
    fn,
    *,
    label: str,
    n_streams: int = 20,
    repeats: int = 5,
    warmup: int = 2,
) -> tuple[LatencyResult, float]:
    """Run ``n_streams`` copies of ``fn`` at once; return per-call latency and wall time.

    Answers the question the deployment actually poses - can one box keep up with 20
    cameras in a 60-second cycle - rather than multiplying a single-frame figure and
    hoping. Returns ``(latency, mean_wall_seconds_per_round)``.
    """
    import torch

    threads = torch.get_num_threads()

    def one() -> float:
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1000

    with ThreadPoolExecutor(max_workers=n_streams) as pool:
        for _ in range(warmup):
            list(pool.map(lambda _: one(), range(n_streams)))

        samples: list[float] = []
        walls: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            samples.extend(pool.map(lambda _: one(), range(n_streams)))
            walls.append(time.perf_counter() - t0)

    return _summarise(label, samples, threads), float(statistics.fmean(walls))
