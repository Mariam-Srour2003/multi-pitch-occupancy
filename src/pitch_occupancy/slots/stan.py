"""STAN: the spatio-temporal aggregation network, and the baselines it has to beat (WP5-T1).

STAN is the plan's headline novelty: instead of turning a slot's per-minute states into a
verdict with three hand-set ratio thresholds, learn the mapping from the ordered sequence. The
architecture is the one WP5-T1 specifies - a small 1-D temporal convolution over per-minute
class probabilities, well under the 100k-parameter ceiling, ending in a 3-way slot status.

**This module cannot currently produce a headline result, and it enforces that itself.** The
real test set is *two* labelled slots. WP5-T8 sets a hard gate - no headline below 30 real
labelled slots (WP2-T8) - and that rule is written here as :data:`MIN_REAL_SLOTS` and checked
by :func:`assert_preliminary`, rather than as a sentence in a plan that a later table can
quietly ignore. This project has repeatedly found guards that did not guard; a note saying
"remember to call this preliminary" is exactly that kind of guard.

**Beating unlearned constants would prove nothing**, so WP5-T7's four baselines live here
beside STAN and share its interface:

* :class:`TunedThresholds` - the same ratio rule, grid-searched on the training slots. The
  honest version of "the baseline". Of course a fitted model beats guessing; the question is
  whether it beats fitting.
* :class:`MedianSmoothing` - majority vote in a sliding window, then the tuned rule. Most of
  what a sequence model could plausibly learn is that isolated minutes are noise, and this
  captures that in four lines.
* :class:`HMMSmoothing` - a 3-state hidden Markov model whose transitions are counted from the
  training sequences and whose emissions are the classifier's own probabilities, Viterbi
  decoded, then the tuned rule. **This is the one that matters.** If STAN beats a tuned HMM
  that is a real result; if it only beats fixed thresholds, an examiner will discount it, and
  that is the most likely single point of attack on the novelty claim.
* :class:`SummaryStatsLogistic` - logistic regression on ratios, longest run, and first and
  last active minute. A sequence model that cannot beat four summary numbers is not using the
  sequence.

Every predictor takes ``fit(sequences)`` and ``predict(sequences)`` over
:class:`~pitch_occupancy.slots.synthetic.SlotSequence`, so the comparison table is a loop
rather than five special cases.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from pitch_occupancy.data.taxonomy import Class3, SlotStatus
from pitch_occupancy.slots.aggregate import Thresholds, aggregate_slot, tune_thresholds
from pitch_occupancy.slots.synthetic import SlotSequence

__all__ = [
    "majority_smooth",
    "CLASS_ORDER",
    "MIN_REAL_SLOTS",
    "NotEnoughRealSlots",
    "assert_preliminary",
    "preliminary_caveat",
    "TunedThresholds",
    "MedianSmoothing",
    "HMMSmoothing",
    "SummaryStatsLogistic",
    "STAN",
    "PREDICTORS",
]

#: The column order of every probability matrix in this module. Fixed once, here, because a
#: silently transposed or reordered matrix produces a plausible table rather than an error.
CLASS_ORDER: tuple[Class3, ...] = (
    Class3.EMPTY,
    Class3.ACTIVE_PLAY,
    Class3.MAINTENANCE_NON_SPORTING,
)

STATUS_ORDER: tuple[SlotStatus, ...] = (SlotStatus.USED, SlotStatus.NOTUSED, SlotStatus.REVIEW)

#: WP5-T8's hard gate, written down while it is still easy to be honest about it. Below this
#: many *real* labelled slots, no number from this module is a headline result. The project
#: currently has two.
MIN_REAL_SLOTS = 30


class NotEnoughRealSlots(RuntimeError):
    """Raised when a caller asks to report STAN as a headline on too few real slots."""


def count_real(sequences: Sequence[SlotSequence]) -> int:
    return sum(1 for s in sequences if not s.synthetic)


def preliminary_caveat(sequences: Sequence[SlotSequence]) -> str:
    """The sentence that must accompany any number this module produces.

    Returned rather than printed so a caller has to put it somewhere, and derived from the
    actual count so it cannot go stale the way a hardcoded "n=2" would.
    """
    n = count_real(sequences)
    if n >= MIN_REAL_SLOTS:
        return f"{n} real labelled slots; at or above the {MIN_REAL_SLOTS}-slot gate."
    return (
        f"PRELIMINARY: {n} real labelled slot(s), against the {MIN_REAL_SLOTS} that WP5-T8 "
        f"requires before STAN reports a headline number. Every figure derived from this is "
        f"a pilot result and must be labelled as one in the thesis (WP2-T8 is the blocker)."
    )


def assert_preliminary(sequences: Sequence[SlotSequence]) -> None:
    """Refuse to proceed if a caller is treating too few real slots as a headline.

    Raises:
        NotEnoughRealSlots: below :data:`MIN_REAL_SLOTS` real slots.
    """
    n = count_real(sequences)
    if n < MIN_REAL_SLOTS:
        raise NotEnoughRealSlots(
            f"{n} real labelled slot(s), below the WP5-T8 gate of {MIN_REAL_SLOTS}. "
            f"Report this as preliminary (see preliminary_caveat) or label {MIN_REAL_SLOTS - n} "
            f"more slots (WP2-T8)."
        )


# --- shared helpers -------------------------------------------------------------------


def _states(sequence: SlotSequence) -> list[Class3]:
    """The per-minute argmax states - what the threshold rule consumes."""
    return [CLASS_ORDER[i] for i in sequence.probabilities.argmax(1)]


def _labelled(sequences: Sequence[SlotSequence]) -> list[tuple[list[Class3], SlotStatus]]:
    return [(_states(s), s.truth) for s in sequences]


def majority_smooth(states: Sequence[Class3], window: int) -> list[Class3]:
    """Majority vote in a sliding window of ``window`` entries centred on each position.

    An isolated disagreeing entry between two agreeing neighbours is replaced by them, which
    is the cheapest correction for a single misread frame. It is also the cheapest way to
    erase a real brief event, so a caller showing this to a person should show what it
    changed rather than only the result - `clip_analysis.py` does.

    Lives here, at module level, because two callers need it: `MedianSmoothing`, where it is
    the STAN baseline that any sequence model has to beat, and the clip reviewer. One
    implementation rather than two that drift.

    Windows are odd by convention; an even one leans earlier, and at the sequence edges the
    window is truncated rather than padded, so the first and last entries are smoothed
    against fewer neighbours than the middle.
    """
    half = window // 2
    out = []
    for i in range(len(states)):
        chunk = states[max(0, i - half): i + half + 1]
        out.append(Counter(chunk).most_common(1)[0][0])
    return out


class _Predictor:
    name = "predictor"

    def fit(self, sequences: Sequence[SlotSequence]) -> _Predictor:
        return self

    def predict(self, sequences: Sequence[SlotSequence]) -> list[SlotStatus]:
        raise NotImplementedError


# --- WP5-T7 baselines -----------------------------------------------------------------


class TunedThresholds(_Predictor):
    """The ratio rule, grid-searched on the training slots rather than left at its defaults."""

    name = "tuned_thresholds"

    def __init__(self) -> None:
        self.thresholds = Thresholds()
        self.train_accuracy = float("nan")

    def fit(self, sequences: Sequence[SlotSequence]) -> TunedThresholds:
        self.thresholds, self.train_accuracy = tune_thresholds(_labelled(sequences))
        return self

    def predict(self, sequences: Sequence[SlotSequence]) -> list[SlotStatus]:
        return [aggregate_slot(_states(s), None, self.thresholds).status for s in sequences]


class MedianSmoothing(TunedThresholds):
    """Majority vote in a sliding window, then the tuned rule.

    The cheapest thing that uses the *order* of the minutes at all, and therefore the floor
    any sequence model has to clear before "it models the sequence" means anything. The
    window is tuned on the training slots alongside the thresholds, because leaving it at a
    guessed 5 would be the same mistake as leaving the thresholds at their defaults.
    """

    name = "median_smoothing"
    WINDOWS = (3, 5, 7, 9, 11)

    def __init__(self) -> None:
        super().__init__()
        self.window = 5

    def fit(self, sequences: Sequence[SlotSequence]) -> MedianSmoothing:
        best = (-1.0, 5, Thresholds())
        for window in self.WINDOWS:
            smoothed = [(self._smooth(_states(s), window), s.truth) for s in sequences]
            thresholds, accuracy = tune_thresholds(smoothed)
            if accuracy > best[0]:
                best = (accuracy, window, thresholds)
        self.train_accuracy, self.window, self.thresholds = best
        return self

    _smooth = staticmethod(majority_smooth)

    def predict(self, sequences: Sequence[SlotSequence]) -> list[SlotStatus]:
        return [
            aggregate_slot(self._smooth(_states(s), self.window), None, self.thresholds).status
            for s in sequences
        ]


class HMMSmoothing(TunedThresholds):
    """A 3-state HMM over the per-minute states, Viterbi decoded, then the tuned rule.

    Transitions are counted from the training slots' *true* per-minute states where they are
    known and from the argmax states otherwise; emissions are the classifier's own
    probabilities, which is what makes this a fair opponent rather than a straw man - it gets
    the same information STAN gets.

    This is the baseline the novelty claim stands or falls against. A pitch is highly
    autocorrelated minute to minute, and an HMM is the textbook model of exactly that, so
    most of what a temporal network could learn here is already in a 3x3 transition matrix.
    """

    name = "hmm"

    def __init__(self, *, smoothing: float = 1.0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.transitions = np.full((3, 3), 1 / 3)
        self.initial = np.full(3, 1 / 3)

    def fit(self, sequences: Sequence[SlotSequence]) -> HMMSmoothing:
        index = {c: i for i, c in enumerate(CLASS_ORDER)}
        counts = np.full((3, 3), self.smoothing)
        starts = np.full(3, self.smoothing)
        for s in sequences:
            states = list(s.true_states) if s.true_states else _states(s)
            starts[index[states[0]]] += 1
            for a, b in zip(states, states[1:], strict=False):
                counts[index[a], index[b]] += 1
        self.transitions = counts / counts.sum(1, keepdims=True)
        self.initial = starts / starts.sum()
        decoded = [(self._viterbi(s.probabilities), s.truth) for s in sequences]
        self.thresholds, self.train_accuracy = tune_thresholds(decoded)
        return self

    def _viterbi(self, probabilities: np.ndarray) -> list[Class3]:
        """Most likely state path. In log space - a 60-minute product of probabilities
        underflows to zero and every path then ties at -inf."""
        eps = 1e-12
        log_e = np.log(np.clip(probabilities, eps, None))
        log_t = np.log(np.clip(self.transitions, eps, None))
        delta = np.log(np.clip(self.initial, eps, None)) + log_e[0]
        back = np.zeros((len(log_e), 3), dtype=int)
        for t in range(1, len(log_e)):
            scores = delta[:, None] + log_t
            back[t] = scores.argmax(0)
            delta = scores.max(0) + log_e[t]
        path = [int(delta.argmax())]
        for t in range(len(log_e) - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return [CLASS_ORDER[i] for i in reversed(path)]

    def predict(self, sequences: Sequence[SlotSequence]) -> list[SlotStatus]:
        return [
            aggregate_slot(self._viterbi(s.probabilities), None, self.thresholds).status
            for s in sequences
        ]


class SummaryStatsLogistic(_Predictor):
    """Logistic regression on four summary numbers, ignoring the order beyond a run length.

    Ratios, longest active run, and the first and last active minute as fractions of the
    slot. If a temporal network cannot beat this, it is not using the sequence - it is using
    the ratios, which the threshold rule already had.
    """

    name = "summary_logistic"

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = seed
        self._model = None
        self._fallback = SlotStatus.REVIEW

    @staticmethod
    def features(sequence: SlotSequence) -> np.ndarray:
        states = _states(sequence)
        n = len(states)
        play = [i for i, s in enumerate(states) if s is Class3.ACTIVE_PLAY]
        longest = best = 0
        for s in states:
            best = best + 1 if s is Class3.ACTIVE_PLAY else 0
            longest = max(longest, best)
        return np.array([
            sum(s is Class3.ACTIVE_PLAY for s in states) / n,
            sum(s is Class3.EMPTY for s in states) / n,
            sum(s is Class3.MAINTENANCE_NON_SPORTING for s in states) / n,
            longest / n,
            (play[0] / n) if play else 1.0,
            (play[-1] / n) if play else 0.0,
        ])

    def fit(self, sequences: Sequence[SlotSequence]) -> SummaryStatsLogistic:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = [s.truth.value for s in sequences]
        self._fallback = SlotStatus(Counter(y).most_common(1)[0][0])
        if len(set(y)) < 2:
            return self
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.seed),
        )
        self._model.fit(np.stack([self.features(s) for s in sequences]), y)
        return self

    def predict(self, sequences: Sequence[SlotSequence]) -> list[SlotStatus]:
        if self._model is None:
            return [self._fallback] * len(sequences)
        X = np.stack([self.features(s) for s in sequences])
        return [SlotStatus(v) for v in self._model.predict(X)]


# --- STAN ------------------------------------------------------------------------------


class STAN(_Predictor):
    """A dilated 1-D temporal convolution over per-minute class probabilities.

    Two convolution layers with dilations 1 and 2 give a receptive field of about 13 minutes
    - long enough to see a late start begin, short enough not to memorise a 60-minute slot
    from 200 training examples. Pooling is mean *and* max concatenated: the mean carries the
    ratios the threshold rule uses, and the max carries "there was play at some point", which
    is the distinction between a no-show and a late start and which a mean over 60 minutes
    dilutes.

    Slots differ in length, so a batch is padded and masked rather than truncated. Truncating
    to the shortest slot would delete the end of every long one, and "was the pitch used in
    the last ten minutes" is exactly the question a late start turns on.

    Around 3k parameters, against the plan's 100k ceiling. That is not modesty: 200 composed
    slots built from 1,578 frames cannot support more, and a model that memorises the
    composition would score well here and mean nothing.
    """

    name = "stan"

    def __init__(
        self,
        *,
        channels: int = 16,
        kernel: int = 5,
        epochs: int = 300,
        lr: float = 0.01,
        weight_decay: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.channels = channels
        self.kernel = kernel
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self._model = None
        self._classes: list[SlotStatus] = []
        self._losses: list[float] = []

    @staticmethod
    def _batch(sequences: Sequence[SlotSequence]):
        """Pad to the longest slot and return the mask that says which minutes are real."""
        import torch

        longest = max(s.n_minutes for s in sequences)
        x = np.zeros((len(sequences), 3, longest), dtype=np.float32)
        mask = np.zeros((len(sequences), 1, longest), dtype=np.float32)
        for i, s in enumerate(sequences):
            x[i, :, : s.n_minutes] = s.probabilities.T
            mask[i, 0, : s.n_minutes] = 1.0
        return torch.from_numpy(x), torch.from_numpy(mask)

    def fit(self, sequences: Sequence[SlotSequence]) -> STAN:
        import torch

        self._classes = [s for s in STATUS_ORDER if any(q.truth is s for q in sequences)]
        if len(self._classes) < 2:
            raise ValueError("cannot fit STAN on slots that all share one verdict")
        index = {c: i for i, c in enumerate(self._classes)}

        torch.manual_seed(self.seed)
        self._model = _build_stan(
            channels=self.channels, kernel=self.kernel, n_classes=len(self._classes)
        )
        x, mask = self._batch(sequences)
        y = torch.tensor([index[s.truth] for s in sequences], dtype=torch.long)
        counts = torch.bincount(y, minlength=len(self._classes)).double()
        weight = (len(y) / (len(self._classes) * counts.clamp(min=1))).float()

        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        optimiser = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self._losses = []
        self._model.train()
        for _ in range(self.epochs):
            optimiser.zero_grad()
            loss = loss_fn(self._model(x, mask), y)
            loss.backward()
            optimiser.step()
            self._losses.append(float(loss.detach()))
        self._model.eval()
        return self

    def predict(self, sequences: Sequence[SlotSequence]) -> list[SlotStatus]:
        import torch

        if self._model is None:
            raise RuntimeError("STAN was not fitted")
        x, mask = self._batch(sequences)
        with torch.no_grad():
            return [self._classes[i] for i in self._model(x, mask).argmax(1).tolist()]

    def predict_proba(self, sequences: Sequence[SlotSequence]) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("STAN was not fitted")
        x, mask = self._batch(sequences)
        with torch.no_grad():
            return torch.softmax(self._model(x, mask), dim=1).numpy()

    @property
    def training_loss(self) -> list[float]:
        return list(self._losses)

    @property
    def n_parameters(self) -> int:
        if self._model is None:
            raise RuntimeError("STAN was not fitted")
        return sum(p.numel() for p in self._model.parameters())


def _build_stan(*, channels: int, kernel: int, n_classes: int):
    """Built inside a function so importing this module does not import torch."""
    import torch
    from torch import nn

    pad = kernel // 2

    class TemporalNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv1d(3, channels, kernel, padding=pad)
            self.conv2 = nn.Conv1d(channels, channels, kernel, padding=2 * pad, dilation=2)
            self.head = nn.Linear(2 * channels, n_classes)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.conv1(x * mask))
            h = torch.relu(self.conv2(h * mask))
            # Masked pooling. Without it, padding contributes zeros to the mean - so a
            # 45-minute slot in a batch whose longest is 75 would have its ratios scaled by
            # 45/75, and the model would read slot *length* as evidence about the verdict.
            h = h * mask
            lengths = mask.sum(dim=2).clamp(min=1.0)
            mean = h.sum(dim=2) / lengths
            peak = h.masked_fill(mask == 0, float("-inf")).max(dim=2).values
            peak = torch.nan_to_num(peak, neginf=0.0)
            return self.head(torch.cat([mean, peak], dim=1))

    return TemporalNet()


#: Every predictor the comparison runs, weakest first. STAN last so a table read top to
#: bottom shows what each addition bought.
PREDICTORS = (TunedThresholds, MedianSmoothing, HMMSmoothing, SummaryStatsLogistic, STAN)
