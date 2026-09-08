"""Zero-shot classification and the prompt space to search over it.

Zero-shot answers the question a trained probe cannot: **what happens at a brand-new venue
before anyone has labelled a single frame there?** That is the real onboarding cost of a
new client site, and the pilot's 75.8% for OpenCLIP was measured with one hand-written
prompt per class - a number that says as much about the prompt as the model.

The cost profile is the opposite of the preprocessing search. There, every candidate needs
a fresh pass over the images. Here the images are embedded **once**; a prompt set is a few
short strings, so thousands of combinations cost seconds. That makes an actual search
affordable where preprocessing forced a greedy one.

Prompts are built from two independent parts, so the space is a product rather than a list:

* a **template** - the framing ("a photo of {}", "a CCTV still of {}") - which mostly
  affects how the text encoder situates the phrase;
* a **descriptor** per class - what the class actually looks like - which carries the
  discriminative content.

Averaging the embeddings of several templates for one class is standard practice and
usually beats any single phrasing, so a prompt *set* is scored, not a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitch_occupancy.data.taxonomy import Class3

__all__ = [
    "PromptSet", "TEMPLATES", "DESCRIPTORS", "DECLARED_PROMPT_SET",
    "encode_prompts", "classify",
]

#: Framings. `{}` receives a descriptor.
TEMPLATES: list[str] = [
    "a photo of {}",
    "a CCTV still of {}",
    "a wide-angle security camera view of {}",
    "an aerial view of {}",
    "{}",
]

#: What each class looks like. Several per class; a prompt set picks one and the search
#: decides which. Written in the vocabulary a caption would use, not the project's.
DESCRIPTORS: dict[Class3, list[str]] = {
    Class3.EMPTY: [
        "an empty football pitch with nobody on it",
        "a deserted five-a-side pitch, no people",
        "an unused artificial turf field",
        "an empty green sports field at night",
        "a vacant football field with no players",
    ],
    Class3.ACTIVE_PLAY: [
        "people playing football on a pitch",
        "a five-a-side football match in progress",
        "footballers running on an artificial turf pitch",
        "a group of players competing in a soccer game",
        "several people playing soccer on a floodlit pitch",
    ],
    Class3.MAINTENANCE_NON_SPORTING: [
        "groundstaff maintaining a football pitch",
        "workers in high-visibility vests on a sports field",
        "a person standing on a pitch, not playing",
        "maintenance machinery on an artificial turf field",
        "a few people walking across an empty pitch",
    ],
}


@dataclass(frozen=True, slots=True)
class PromptSet:
    """One descriptor per class, rendered through one or more templates."""

    descriptors: dict[Class3, str]
    templates: tuple[str, ...]

    def phrases(self, cls: Class3) -> list[str]:
        return [t.format(self.descriptors[cls]) for t in self.templates]

    def label(self) -> str:
        n = len(self.templates)
        return " | ".join(
            f"{c.name}:{self.descriptors[c][:34]}" for c in sorted(self.descriptors, key=str)
        ) + f"  [{n} template{'s' if n != 1 else ''}]"


def encode_prompts(
    prompt_set: PromptSet, classes: list[Class3], encode_text
) -> np.ndarray:
    """Class-direction matrix ``(n_classes, dim)``, L2-normalised.

    Each class vector is the mean of its templates' embeddings - the standard prompt
    ensemble - renormalised so the later dot product is a cosine similarity.
    """
    rows = []
    for cls in classes:
        vecs = encode_text(prompt_set.phrases(cls))
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        mean = vecs.mean(axis=0)
        rows.append(mean / (np.linalg.norm(mean) or 1.0))
    return np.stack(rows)


def classify(
    image_features: np.ndarray, class_directions: np.ndarray, classes: list[Class3]
) -> list[str]:
    """Nearest class direction by cosine similarity. Image features must be normalised."""
    scores = image_features @ class_directions.T
    return [classes[i].value for i in scores.argmax(axis=1)]


def _declared() -> "PromptSet":
    """The prompt set every *hypothesis test* uses: first descriptor per class, all templates.

    Fixed by position rather than by choice, and deliberately **not** the winner of
    `prompt_search.py`. That search scored 375 prompt sets on the same folds it reports, so
    its winner's margin is optimistically biased - amendment A6 says as much - and reusing
    it would carry that bias into H6, the hypothesis about whether zero-shot lags trained
    probes. A selected number cannot test the selection.

    Defined here rather than in an experiment because two now depend on it meaning the same
    thing: `benchmark_v2.py` uses it as a composition control and `h6_zero_shot_gap.py` as
    the hypothesis's zero-shot arm. Two copies of a pre-declared constant are two that can
    drift, and the drift would be undetectable - both would still be "a fixed prompt set".
    """
    return PromptSet(
        descriptors={cls: options[0] for cls, options in DESCRIPTORS.items()},
        templates=tuple(TEMPLATES),
    )


DECLARED_PROMPT_SET = _declared()
