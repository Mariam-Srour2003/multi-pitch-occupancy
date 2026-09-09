"""Gated multi-backbone fusion (WP5-T2, RQ5).

**This is not** :mod:`pitch_occupancy.slots.fusion`. That module fuses the two *cameras*
watching one pitch, at verdict level, with a fixed max-activity rule. This one fuses two or
three frozen *backbones* looking at the same frame, at feature level, with learned weights.
They share a word and nothing else; the two live side by side because the plan names both
"fusion" and renaming either would break the paper trail.

The architecture is the one WP5-T2 specifies::

    cheap image statistics -> tiny MLP -> softmax weights w_k
    concat_k( w_k * standardised features_k ) -> one shared linear head -> 3 classes

The point of the design is that the gate is meant to *route*: a frame that is dark and
low-contrast might be better read by DINOv2, a bright one by ConvNeXtV2, and a router with
three cheap numbers per frame could in principle pick. Whether it does is an empirical
question, and this module is built so that question has a clean answer.

**The gate is a three-rung ladder, and each rung adds exactly one thing.** ``gate="uniform"``
fixes every weight at ``1/K``; ``gate="constant"`` learns ``K`` weights shared by every frame;
``gate="mlp"`` learns them per frame from the statistics. Head, trainer, standardisation,
regularisation and seed are identical on all three, so ``constant`` minus ``uniform`` is the
value of *learned mixing* and ``mlp`` minus ``constant`` is the value of *routing* - which is
the only thing the gate is for. Collapsing those two into one on/off comparison is how a
learned constant gets reported as a router: fitted on the venue folds this model puts about
0.70 of its weight on DINOv2 in every fold and moves it by 0.086 between frames, and against
``uniform`` alone that would have read as the gate working. The comparison against the
published logistic-regression probes crosses a trainer boundary and isolates nothing.

**The gate is also a confound risk, and a known one.** Its inputs are brightness, contrast and
edge density; at `venue_01` brightness is close to a day/night indicator and day/night is close
to the class label - the confound this project has now found four times. So the gate accepts
whatever matrix it is handed, and :func:`gate_statistics` is only the default: feeding it a
lighting indicator alone gives the ablation that detects the fifth occurrence.

Training is full-batch Adam on CPU with a fixed seed, so a fit is reproducible to the bit.
The model is small on purpose - a 3-backbone gate is about 7k parameters, two orders below
the 100k ceiling the plan sets for STAN - because 1,296 development frames over nine venues
cannot support more, and a model that overfits the venue would answer a different question.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "GATE_STATISTIC_NAMES",
    "GATE_MODES",
    "gate_statistics",
    "build_gate_statistics",
    "FusionHead",
    "GateReport",
]

#: The ablation ladder, weakest first. Each rung adds one capability to the one before it,
#: so a difference between adjacent rungs names exactly what caused it.
GATE_MODES: tuple[str, ...] = ("uniform", "constant", "mlp")

#: The three cheap statistics WP5-T2 names, in the order :func:`gate_statistics` returns
#: them. Named rather than positional because the confound ablation swaps this matrix for a
#: lighting indicator, and a silent column reordering would look like a modelling result.
GATE_STATISTIC_NAMES: tuple[str, ...] = ("brightness", "contrast", "edge_density")


def gate_statistics(image_bgr: np.ndarray) -> np.ndarray:
    """Brightness, contrast and edge density of one frame, as float32.

    All three are computed on the grayscale image and scaled to roughly unit range so the
    gate's first layer starts somewhere sensible; the standardiser inside :class:`FusionHead`
    then handles the rest.

    * **brightness** - mean intensity / 255.
    * **contrast** - standard deviation of intensity / 255. Not the CLAHE gate's percentile
      spread: that one exists to decide whether to equalise, and reusing it here would couple
      a routing decision to a preprocessing threshold that WP3 is still moving.
    * **edge density** - mean absolute Laplacian response / 255. A proxy for how much
      structure is in the frame; players, nets and shadows all raise it.
    """
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    return np.array(
        [gray.mean() / 255.0, gray.std() / 255.0, float(edges.mean()) / 255.0],
        dtype=np.float32,
    )


def build_gate_statistics(files: Sequence[str], dataset_dir) -> np.ndarray:
    """Gate statistics for ``files`` (paths relative to ``dataset_dir``), one row each.

    Raises:
        OSError: if a frame cannot be read. Zero-filling a missing frame would hand the gate
            a black, flat, edgeless image - the exact signature of a night shot - and the
            router would learn from it.
    """
    import cv2

    out = []
    for name in files:
        image = cv2.imread(str(dataset_dir / name))
        if image is None:
            raise OSError(f"cannot read {dataset_dir / name}")
        out.append(gate_statistics(image))
    return np.stack(out)


@dataclass(frozen=True, slots=True)
class GateReport:
    """What the gate did, for the diagnostic that decides whether to believe it.

    ``mean_weight`` is the average softmax weight per backbone and ``spread`` the mean
    absolute deviation of each frame's weights from that average. A gate that has learned
    nothing still produces weights; it produces the *same* weights everywhere, and ``spread``
    near zero says so where a mean over frames would hide it.

    The two numbers answer different questions and both are needed. ``mean_weight`` says which
    backbone the model prefers - a preference a ``constant`` gate can express just as well.
    ``spread`` says how much that preference *moves between frames*, which is the only part
    that is routing.
    """

    backbones: tuple[str, ...]
    mean_weight: tuple[float, ...]
    spread: float

    def is_degenerate(self, tol: float = 1e-3) -> bool:
        """Whether the gate routes every frame identically, making it an expensive constant.

        The tolerance is deliberately tight: it catches a gate that is *exactly* constant,
        not one that is merely nearly so. A gate with a spread of 0.03 is not degenerate by
        this test and is still, on the evidence, doing almost nothing - which is what the
        ``mlp`` versus ``constant`` comparison is for. A threshold on this number would be a
        guess standing in for that comparison.
        """
        return self.spread < tol


class FusionHead:
    """Gated fusion over cached frozen features, with a shared linear head.

    Args:
        backbones: keys naming the feature blocks, in the order ``fit`` will receive them.
        gate: a rung of :data:`GATE_MODES`. ``"uniform"`` fixes the weights at ``1/K`` and
            ignores the gate matrix; ``"constant"`` learns ``K`` weights shared by every
            frame; ``"mlp"`` learns them per frame. Head, trainer, standardisation,
            regularisation and seed are identical across all three, which is what makes the
            differences between them ablations rather than comparisons.
        gate_dim: width of the gate input matrix. Three for :func:`gate_statistics`, one for
            the lighting-indicator ablation. Ignored by every rung but ``"mlp"``, though the
            matrix's shape is still checked, so a variant cannot be fed the wrong one and
            silently agree.
        hidden: width of the gate's single hidden layer.
        epochs, lr, weight_decay: full-batch Adam settings. Defaults were fixed once against
            the concat ablation's training loss and then left alone for every variant; tuning
            them per variant would tune the ablation.
        balanced: weight the loss by inverse class frequency, matching
            :class:`~pitch_occupancy.vision.heads.LinearProbe`'s ``class_weight="balanced"``.
            C3 has six frames in the whole dataset and an unweighted fit never predicts it.
        seed: fixes initialisation. Full-batch training has no sampling, so a fit is
            reproducible exactly, not merely in distribution.
    """

    def __init__(
        self,
        backbones: Sequence[str],
        *,
        gate: str = "mlp",
        gate_dim: int = len(GATE_STATISTIC_NAMES),
        hidden: int = 8,
        epochs: int = 300,
        lr: float = 0.01,
        weight_decay: float = 1e-3,
        balanced: bool = True,
        seed: int = 42,
    ) -> None:
        if not backbones:
            raise ValueError("a fusion head over no backbones is not a fusion head")
        if gate not in GATE_MODES:
            raise ValueError(f"unknown gate {gate!r}; known: {', '.join(GATE_MODES)}")
        self.backbones = tuple(backbones)
        self.gate = gate
        self.gate_dim = int(gate_dim)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.balanced = bool(balanced)
        self.seed = int(seed)
        self.name = f"fusion_{gate}_" + "_".join(self.backbones)

        self._classes: list[str] = []
        self._mu: dict[str, np.ndarray] = {}
        self._sd: dict[str, np.ndarray] = {}
        self._gate_mu: np.ndarray | None = None
        self._gate_sd: np.ndarray | None = None
        self._torch_model = None
        self._losses: list[float] = []

    # --- shapes and standardisation ----------------------------------------------

    def _check(self, X: dict[str, np.ndarray], gate: np.ndarray) -> int:
        missing = [k for k in self.backbones if k not in X]
        if missing:
            raise KeyError(f"no features for {', '.join(missing)}")
        n = {len(X[k]) for k in self.backbones} | {len(gate)}
        if len(n) != 1:
            raise ValueError(
                "the backbones and the gate disagree about how many frames there are: "
                + ", ".join(f"{k}={len(X[k])}" for k in self.backbones)
                + f", gate={len(gate)}"
            )
        if gate.ndim != 2 or gate.shape[1] != self.gate_dim:
            raise ValueError(
                f"gate input is {gate.shape}, expected (n, {self.gate_dim}); the gate was "
                f"constructed for a {self.gate_dim}-column statistic matrix"
            )
        return n.pop()

    def _standardise(self, X: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [(X[k] - self._mu[k]) / self._sd[k] for k in self.backbones], axis=1
        ).astype(np.float32)

    def _standardise_gate(self, gate: np.ndarray) -> np.ndarray:
        return ((gate - self._gate_mu) / self._gate_sd).astype(np.float32)

    # --- fitting -------------------------------------------------------------------

    def fit(
        self, X: dict[str, np.ndarray], gate: np.ndarray, labels: Sequence[str]
    ) -> FusionHead:
        """Fit the gate and the head jointly on standardised features."""
        import torch

        gate = np.asarray(gate, dtype=np.float64)
        n = self._check(X, gate)
        if len(labels) != n:
            raise ValueError(f"{len(labels)} labels for {n} frames")
        self._classes = sorted(set(labels))
        if len(self._classes) < 2:
            raise ValueError(
                f"only one class ({self._classes[0]}) in the training rows; a head fitted "
                "here would predict it everywhere and the fold should have been skipped"
            )

        for key in self.backbones:
            block = np.asarray(X[key], dtype=np.float64)
            self._mu[key] = block.mean(0)
            # A constant feature contributes nothing and dividing by its zero spread would
            # produce NaNs that only surface as a silently untrained head.
            self._sd[key] = np.where(block.std(0) < 1e-8, 1.0, block.std(0))
        self._gate_mu = gate.mean(0)
        self._gate_sd = np.where(gate.std(0) < 1e-8, 1.0, gate.std(0))

        torch.manual_seed(self.seed)
        self._torch_model = _build_gated_module(
            n_backbones=len(self.backbones),
            block_dim=X[self.backbones[0]].shape[1],
            gate_dim=self.gate_dim,
            hidden=self.hidden,
            n_classes=len(self._classes),
            gate=self.gate,
        )

        index = {c: i for i, c in enumerate(self._classes)}
        y = torch.tensor([index[c] for c in labels], dtype=torch.long)
        Z = torch.from_numpy(self._standardise(X))
        G = torch.from_numpy(self._standardise_gate(gate))

        weight = None
        if self.balanced:
            counts = torch.bincount(y, minlength=len(self._classes)).double()
            weight = (len(y) / (len(self._classes) * counts.clamp(min=1))).float()
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        optimiser = torch.optim.Adam(
            self._torch_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self._losses = []
        self._torch_model.train()
        for _ in range(self.epochs):
            optimiser.zero_grad()
            loss = loss_fn(self._torch_model(Z, G), y)
            loss.backward()
            optimiser.step()
            self._losses.append(float(loss.detach()))
        self._torch_model.eval()
        return self

    # --- using -----------------------------------------------------------------------

    def _fitted(self):
        if self._torch_model is None:
            raise RuntimeError(f"{self.name} was not fitted")
        return self._torch_model

    def predict_proba(self, X: dict[str, np.ndarray], gate: np.ndarray) -> np.ndarray:
        import torch

        model = self._fitted()
        gate = np.asarray(gate, dtype=np.float64)
        self._check(X, gate)
        with torch.no_grad():
            logits = model(
                torch.from_numpy(self._standardise(X)),
                torch.from_numpy(self._standardise_gate(gate)),
            )
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X: dict[str, np.ndarray], gate: np.ndarray) -> list[str]:
        return [self._classes[i] for i in self.predict_proba(X, gate).argmax(1)]

    def gate_weights(self, gate: np.ndarray) -> np.ndarray:
        """The per-frame softmax weights, ``(n, K)``. Constant on the lower two rungs."""
        import torch

        model = self._fitted()
        gate = np.asarray(gate, dtype=np.float64)
        with torch.no_grad():
            return model.weights(torch.from_numpy(self._standardise_gate(gate))).numpy()

    def report(self, gate: np.ndarray) -> GateReport:
        """Summarise the routing, so a constant gate is visible rather than inferred."""
        w = self.gate_weights(gate)
        mean = w.mean(0)
        return GateReport(
            backbones=self.backbones,
            mean_weight=tuple(float(x) for x in mean),
            spread=float(np.abs(w - mean).mean()),
        )

    @property
    def classes_(self) -> list[str]:
        return list(self._classes)

    @property
    def training_loss(self) -> list[float]:
        """Loss per epoch. A flat tail is the only evidence that 300 epochs was enough."""
        return list(self._losses)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._fitted().parameters())


def _build_gated_module(
    *, n_backbones: int, block_dim: int, gate_dim: int, hidden: int, n_classes: int, gate: str
):
    """The torch module, built inside a function so importing this file does not import torch.

    :class:`FusionHead` is reachable from the package's ``__init__``, torch takes about a
    second to import, and the gate is used by one experiment.
    """
    import torch
    from torch import nn

    class GatedModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_backbones = n_backbones
            self.block_dim = block_dim
            self.gate_mode = gate
            if gate == "mlp":
                self.gate = nn.Sequential(
                    nn.Linear(gate_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_backbones)
                )
            elif gate == "constant":
                # K free parameters, softmaxed. Zeros rather than random values so this rung
                # *starts* at the uniform rung and can only move away from it by learning.
                self.logits = nn.Parameter(torch.zeros(n_backbones))
            self.head = nn.Linear(n_backbones * block_dim, n_classes)

        def weights(self, g: torch.Tensor) -> torch.Tensor:
            if self.gate_mode == "uniform":
                return g.new_full((g.shape[0], self.n_backbones), 1.0 / self.n_backbones)
            if self.gate_mode == "constant":
                return torch.softmax(self.logits, dim=0).expand(g.shape[0], self.n_backbones)
            return torch.softmax(self.gate(g), dim=1)

        def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            w = self.weights(g)
            # (n, K, 1) * (n, K, d) -> scale each backbone's block, then flatten back to the
            # concatenation the shared head reads. The view is the part that can silently be
            # wrong - a transposed reshape would scale interleaved coordinates rather than
            # blocks and still train - so a test pins which coordinates each weight touches.
            scaled = w.unsqueeze(2) * z.view(z.shape[0], self.n_backbones, self.block_dim)
            return self.head(scaled.reshape(z.shape[0], -1))

    return GatedModule()
