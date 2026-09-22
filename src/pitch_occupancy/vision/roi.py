"""Per-camera pitch boundaries: say what is inside the pitch, and ignore the rest (WP3-T1).

`vision/__init__.py` has listed this module since the project began and it has never existed,
which is why `PreprocessConfig.roi` was a switch nothing could turn on: `roi_mask` with no
polygon returns the frame untouched, so the preprocessing search left `roi` out rather than
record "ROI does not help" about a no-op. This is the missing half.

**The problem it solves is not clutter, it is a neighbour.** Five-a-side pitches are built in
rows, and a camera watching pitch 2 sees pitches 1 and 3 down the sides of its frame. A match
on the next pitch over puts real players, moving realistically, into frames where *this* pitch
is empty - so the model is right about what it sees and wrong about what was asked. No amount
of training fixes that, because the evidence genuinely is there; the question was under-
specified. A boundary specifies it.

**A polygon is per camera, not per pitch.** The two cameras on one pitch see it from opposite
ends and their neighbours fall on opposite sides of the frame, so one shared outline would be
wrong for at least one of them.

**Normalised coordinates, so a resolution change does not silently invalidate the boundary.**
Points are fractions of width and height in 0-1. A camera swapped for a 4K one keeps its
outline; a polygon in pixels would have quietly masked the wrong region.

**Filling with black is the default and is not obviously right.** It is what `roi_mask` has
always done and what the preprocessing search would have evaluated, so it stays the default -
but a large black region is a thing no pretraining set contains, and the backbone's response
to it is itself a distribution shift. `blur` keeps the frame's statistics plausible while
destroying the detail a person is made of, and `mean` is the gentlest edge. Which is right is
an empirical question this module deliberately does not answer: it offers the three and the
editor shows what each does to the prediction, on the user's own frames.

Stored in `configs/roi.json`, keyed by camera. Unlike `configs/cameras.json` - which holds
RTSP credentials and is gitignored for that reason - a boundary is geometry, carries nothing
secret, and belongs in the repository: it is part of what a site *is*, and re-deriving it by
hand after a clone is exactly the sort of setup nobody repeats identically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = [
    "Polygon", "STORE", "DERIVED_STORE", "FILLS", "DEFAULT_FILL", "FILE_KEY_CAMERA",
    "ALIASES_KEY", "load_all", "load_drawn", "load_aliases", "get", "resolve",
    "resolve_with_key",
    "save", "save_derived", "remove", "validate", "coverage", "apply", "grid_weights",
    "outline", "derive_from_frames", "derive_from_video",
]

#: A polygon is a list of ``[x, y]`` pairs, each a fraction of the frame in 0-1.
Polygon = list[list[float]]

STORE = Path(__file__).resolve().parents[3] / "configs" / "roi.json"

#: Boundaries measured from the footage by `scripts/derive_roi.py`, one per camera.
#:
#: Read **underneath** `STORE`, so a human-drawn outline always wins where one exists. They
#: are kept in a separate file rather than merged into it because their provenance differs and
#: should stay visible: one is somebody's judgement about where the pitch is, the other is a
#: convex hull of the largest green region in a median frame.
#:
#: Without these, `get` returned None for every camera in the corpus - `STORE` holds outlines
#: named `cam` and `cam2`, and the manifest has 99 cameras, none of them called that. The
#: boundary machinery was complete and applied to nothing.
DERIVED_STORE = Path(__file__).resolve().parents[3] / "configs" / "roi_derived.json"

#: How the outside of the boundary is filled. See the module docstring for why the default
#: is the least defensible of the three and is the default anyway.
FILLS = ("black", "blur", "mean")
DEFAULT_FILL = "black"

#: Below this, the boundary is almost certainly a mistake - three points clicked by accident,
#: or an outline drawn round a corner flag. Refused rather than saved, because a polygon that
#: masks 99% of the pitch produces confident nonsense rather than an obvious failure.
MIN_COVERAGE = 0.02

#: The block of `STORE` that maps a camera id *as production names it* onto the key a
#: boundary is stored under. `_read` skips every underscored key, so this block was always
#: legal in the file and never read; `resolve` reads it.
ALIASES_KEY = "_aliases"

#: The export's ``file0``/``file1`` keys against the corpus's ``camA``/``camB`` suffixes.
#:
#: `frame_source.discover_slots` is right that the ``(1)`` suffix does not name a *physical*
#: camera - it flips between recording days (`db/seed.py` ``PHYSICAL_CAMERA``). But within one
#: recording the frames were extracted with ``file0`` as ``camA`` and ``file1`` as ``camB``,
#: and every derived boundary is keyed that way. Measured on 2026-09-19 by deriving a boundary
#: from each of the four recordings and matching it against the stored ones: ``file0`` against
#: ``camA`` scored IoU 0.85 and 0.99, ``file1`` against ``camB`` 0.93 and 0.98, on both days,
#: and the crossed pairs 0.54-0.77. So the mapping holds per recording, which is the only
#: scope `resolve` applies it in - it needs the recording's ``slot_key`` to use it at all.
FILE_KEY_CAMERA = {"file0": "camA", "file1": "camB"}


def validate(polygon: Polygon) -> Polygon:
    """Return the polygon as floats, or raise ``ValueError`` saying what is wrong with it.

    Validation lives here rather than in the API layer because the file can be hand-edited
    and the live worker reads it directly - a boundary that was fine when it was saved and
    is malformed when it is read would fail at the worst moment, on a camera, at night.
    """
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        raise ValueError("a boundary needs at least 3 points")
    clean: Polygon = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"each point must be [x, y], got {point!r}")
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"points are fractions of the frame, so both must be in 0-1; got [{x}, {y}]"
            )
        clean.append([x, y])
    area = coverage(clean)
    if area < MIN_COVERAGE:
        raise ValueError(
            f"this boundary keeps {area:.1%} of the frame, which is almost certainly a "
            f"mis-click - it would leave the model nothing to look at"
        )
    return clean


def coverage(polygon: Polygon) -> float:
    """Fraction of the frame the boundary keeps, by the shoelace formula.

    Reported to the editor beside the outline: "you are about to discard 73% of what this
    camera sees" is the sentence that catches a boundary drawn round the wrong pitch, and it
    is available before anything is saved or any model is run.
    """
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def _raw(path: Path) -> dict:
    """The store as JSON, or an empty dict. Never raises - see `load_all` for why."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _read(path: Path) -> dict[str, Polygon]:
    """One store's boundaries, or nothing. Never raises - see `load_all` for why."""
    raw = _raw(path)
    out: dict[str, Polygon] = {}
    for key, value in raw.items():
        if key.startswith("_"):  # comment keys, as in configs/cameras.example.json
            continue
        try:
            out[str(key)] = validate(value)
        except (ValueError, TypeError):
            continue
    return out


def load_all() -> dict[str, Polygon]:
    """Every boundary, keyed by camera: derived ones, overlaid by hand-drawn ones.

    Malformed rather than raising, and deliberately: this is read on the live path, and a
    boundary file someone broke while editing should degrade to "no boundary" - which is the
    behaviour the system had before boundaries existed - rather than stop a night's capture.
    The editor validates on save, which is where a person is present to be told.

    A hand-drawn outline overrides a derived one for the same camera. The editor writes to
    `STORE`, so drawing one is how a person corrects a derivation they disagree with.
    """
    return {**_read(DERIVED_STORE), **_read(STORE)}


def load_drawn() -> dict[str, Polygon]:
    """Only the hand-drawn boundaries, from `STORE`.

    `load_all` deliberately hides which store an outline came from, because the masking path
    does not care. A *menu* does: `DERIVED_STORE` holds one entry per clip in the corpus, so a
    selector built from `load_all` is a hundred machine-generated ids long and the two or three
    outlines somebody actually drew are lost in it. This is the split that lets a caller put
    those first without reaching into the private readers.
    """
    return _read(STORE)


def load_aliases() -> dict[str, str]:
    """Production camera ids -> stored boundary keys, from the `_aliases` block of `STORE`.

    Malformed entries are dropped rather than raised on, for the same reason `load_all`
    never raises: this is read on the live path.
    """
    raw = _raw(STORE).get(ALIASES_KEY, {})
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v}


def get(camera: str) -> Polygon | None:
    """The boundary stored under exactly this key, or None. None means *no boundary*.

    This is the store's own vocabulary. A caller holding a camera id *as production names
    it* wants :func:`resolve`, which knows the aliases and the export's file keys - `get`
    on a production id is how the boundary machinery came to be applied to nothing (A36).
    """
    return load_all().get(camera)


def _candidates(camera: str, *, venue: str | None, slot_key: str | None) -> list[str]:
    """The keys a production camera id might be stored under, most specific first."""
    keys: list[str] = []
    if slot_key:
        keys += [f"{slot_key}/{camera}", f"{slot_key}_{camera}"]
    if venue:
        keys.append(f"{venue}/{camera}")
    keys.append(camera)
    # The export convention, scoped to one recording - see `FILE_KEY_CAMERA`.
    if slot_key and camera in FILE_KEY_CAMERA:
        keys.append(f"{slot_key}_{FILE_KEY_CAMERA[camera]}")
    return keys


def resolve_with_key(camera: str, *, venue: str | None = None,
                     slot_key: str | None = None) -> tuple[Polygon | None, str | None]:
    """The boundary for a camera as production names it, and the key it was found under.

    Production knows a camera three ways and the store knew none of them (A36): the worker
    replaying a recording calls it ``file0``, `configs/cameras.json` calls it ``camera_A``
    under ``venue_01``, and the derived store keys it ``slot_20260711_1000_camA``. Each id is
    tried at its most specific first - ``<slot>/<camera>``, ``<slot>_<camera>``,
    ``<venue>/<camera>``, then the bare id - and each of those is also looked up through the
    `_aliases` block, one hop. Last comes the export's ``file0 -> camA`` convention, which
    needs a ``slot_key`` because it holds per recording and not across days.

    Returns ``(None, None)`` when nothing matched, and the matching *key* otherwise, so a
    caller can say which boundary it applied rather than only that it applied one.
    """
    stored = load_all()
    aliases = load_aliases()
    for candidate in _candidates(camera, venue=venue, slot_key=slot_key):
        for key in (candidate, aliases.get(candidate)):
            if key and key in stored:
                return stored[key], key
    return None, None


def resolve(camera: str, *, venue: str | None = None,
            slot_key: str | None = None) -> Polygon | None:
    """:func:`resolve_with_key`, reduced to the boundary."""
    polygon, _ = resolve_with_key(camera, venue=venue, slot_key=slot_key)
    return polygon


def save(camera: str, polygon: Polygon) -> Polygon:
    """Validate and store one camera's hand-drawn boundary, preserving every other entry.

    Writes `STORE` and only `STORE`. Until 2026-09-19 this merged the derived store into the
    hand-drawn one on every save - the first outline drawn in the editor would have copied
    all 99 derived boundaries into `roi.json` and lost the provenance split the module
    docstring insists on. The two files stay two files.
    """
    clean = validate(polygon)
    if not camera or not camera.strip():
        raise ValueError("a boundary needs a camera to belong to")
    current = _read(STORE)
    current[camera.strip()] = clean
    _write(current)
    return clean


def save_derived(camera: str, polygon: Polygon) -> Polygon:
    """Store a boundary *measured* from footage under `DERIVED_STORE`.

    Kept apart from :func:`save` because a derived boundary is weaker evidence than a drawn
    one and must stay visibly so: a hand-drawn outline for the same key still wins in
    :func:`load_all`, and drawing one is how a person corrects a derivation.
    """
    clean = validate(polygon)
    if not camera or not camera.strip():
        raise ValueError("a boundary needs a camera to belong to")
    current = _read(DERIVED_STORE)
    current[camera.strip()] = clean
    _write(current, path=DERIVED_STORE, comment=_DERIVED_COMMENT)
    return clean


def remove(camera: str) -> bool:
    """Forget one camera's hand-drawn boundary. Returns whether there was one to forget.

    A derived boundary for the same key is left alone: it was measured, not drawn, and
    forgetting a person's correction should reveal the measurement, not erase it too.
    """
    current = _read(STORE)
    if camera not in current:
        return False
    del current[camera]
    _write(current)
    return True


_HAND_COMMENT = [
    "Pitch boundaries, one per camera, drawn in the editor at /roi.",
    "Points are [x, y] as fractions of the frame (0-1), so a resolution change does",
    "not invalidate them. Everything outside the outline is filled before the frame",
    "reaches the model - see src/pitch_occupancy/vision/roi.py.",
    "Committed on purpose: a boundary is geometry, not a credential, and re-drawing",
    "one by hand after a clone is setup nobody repeats identically.",
    "`_aliases` maps a camera id as production names it (venue/camera, or the worker's",
    "file0/file1 under a recording key) onto the key a boundary is stored under; see",
    "roi.resolve.",
]

_DERIVED_COMMENT = [
    "Pitch boundaries derived from the footage by scripts/derive_roi.py or",
    "roi.derive_from_video, not drawn. One per camera, [x, y] as fractions of the frame.",
    "Convex hull of the largest green region in a per-camera median frame - see",
    "src/pitch_occupancy/vision/roi_derive.py for what that does and does not achieve.",
    "A hand-drawn outline in roi.json for the same key wins over one of these.",
]


def _write(polygons: dict[str, Polygon], *, path: Path | None = None,
           comment: list[str] | None = None) -> None:
    """Write one store. The hand-drawn store's `_aliases` block survives a rewrite."""
    path = path or STORE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"_comment": comment or _HAND_COMMENT}
    if path == STORE:
        previous = _raw(STORE)
        for key in (ALIASES_KEY, "_aliases_comment"):
            if key in previous:
                payload[key] = previous[key]
    payload.update({k: polygons[k] for k in sorted(polygons)})
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def derive_from_frames(frames, *, hull: bool = True) -> Polygon | None:
    """Measure a boundary from frames of one camera. See `vision/roi_derive.py`."""
    from pitch_occupancy.vision.roi_derive import derive_from_frames as _derive

    return _derive(frames, hull=hull)


def derive_from_video(path, *, hull: bool = True) -> Polygon | None:
    """Measure a boundary from a recording or stream. See `vision/roi_derive.py`."""
    from pitch_occupancy.vision.roi_derive import derive_from_video as _derive

    return _derive(path, hull=hull)


def apply(image_bgr: np.ndarray, polygon: Polygon | None, *,
          fill: str = DEFAULT_FILL) -> np.ndarray:
    """Suppress everything outside ``polygon``. No polygon returns the frame untouched.

    That last sentence is the reason `roi` stayed out of the preprocessing search: a no-op
    that looks like a switch will be measured as "makes no difference", and the conclusion
    will be recorded about ROI masking rather than about the absence of a polygon.

    ``fill`` is one of :data:`FILLS`:

    * ``black`` - zeroes the outside. What `roi_mask` has always done, and the hardest edge.
    * ``blur`` - a heavy Gaussian outside the outline. Destroys the detail a person is made
      of, which is what the neighbouring pitch has to lose, while leaving the frame's
      statistics closer to something the backbone has seen.
    * ``mean`` - flat fill with the mean colour *inside* the outline, the gentlest edge.
    """
    import cv2
    import numpy as np

    if not polygon:
        return image_bgr
    if fill not in FILLS:
        raise ValueError(f"unknown fill {fill!r}; known: {', '.join(FILLS)}")

    h, w = image_bgr.shape[:2]
    points = np.array([[int(round(x * w)), int(round(y * h))] for x, y in polygon],
                      dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [points], 255)

    if fill == "black":
        return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    if fill == "blur":
        # Kernel scaled to the frame so the same boundary behaves the same way on a 1080p
        # and a 4K camera. Odd, because OpenCV requires it.
        k = max(21, (min(h, w) // 12) | 1)
        outside = cv2.GaussianBlur(image_bgr, (k, k), 0)
    else:
        inside_mean = cv2.mean(image_bgr, mask=mask)[:3]
        outside = np.full_like(image_bgr, np.array(inside_mean, dtype=np.uint8))

    return np.where(mask[:, :, None].astype(bool), image_bgr, outside)


def grid_weights(
    polygon: Polygon | None,
    size_wh: tuple[int, int],
    grid: tuple[int, int],
    processor=None,
    *,
    processor_geometry: bool = True,
) -> np.ndarray | None:
    """How much of each patch cell lies inside the boundary, as a ``grid``-shaped 0-1 array.

    This is the piece that makes a boundary mean something to the *model* rather than only to
    the picture. :func:`apply` fills the outside with black, which changes what those pixels
    contain but not whether the backbone looks at them: a masked frame still produces a patch
    token per position, and `backbones.embed_batch` pools with a plain mean over **all** of
    them. So the fill lands in the pooled vector, the probe scores it, and the evidence map
    shows it - which is exactly the "the model is still reading outside the pitch" that a
    boundary was drawn to stop. Weighting the pool by this array removes those positions from
    the average instead of merely darkening them.

    **The geometry is taken from the processor, not re-derived.** Two of the three backbones
    upscale to 256 and centre-crop back to 224, discarding 23.4% of the frame - and they say
    so differently, ConvNeXtV2 hiding it behind ``crop_pct`` (see `backbones.PROCESSOR_GEOMETRY`).
    Reimplementing that here would be a second preprocessing path that can drift from the
    first, which is this project's most expensive bug repeated on purpose. Instead the polygon
    is rasterised at the frame's own size and pushed through **the same processor call**, with
    rescaling and normalisation switched off so what comes back is still a mask. Whatever the
    processor does to a frame, it has now done to the boundary, including a crop that cuts part
    of the outline away.

    **Cells are fractional, not in-or-out.** A patch straddling the touchline is half pitch,
    and `INTER_AREA` gives it 0.5 rather than forcing a choice that would move the boundary by
    up to half a patch - 16 pixels at ViT's grid, which is a player's width at that distance.

    ``None`` polygon returns ``None``, meaning *pool normally*, so every caller can pass its
    polygon straight through without branching.
    """
    import cv2
    import numpy as np

    if not polygon:
        return None

    width, height = size_wh
    points = np.array([[int(round(x * width)), int(round(y * height))] for x, y in polygon],
                      dtype=np.int32)
    mask = np.zeros((height, width, 3), np.uint8)
    cv2.fillPoly(mask, [points], (255, 255, 255))

    if processor is not None:
        from PIL import Image

        kwargs: dict[str, object] = {"do_rescale": False, "do_normalize": False}
        if not processor_geometry:
            kwargs["do_resize"] = False
            if getattr(processor, "do_center_crop", False):
                kwargs["do_center_crop"] = False
        seen = processor(images=[Image.fromarray(mask)], return_tensors="np", **kwargs)
        mask = np.asarray(seen["pixel_values"])[0, 0]
    else:
        mask = mask[:, :, 0]

    cells = cv2.resize(mask.astype(np.float32), (grid[1], grid[0]),
                       interpolation=cv2.INTER_AREA)
    return np.clip(cells / 255.0, 0.0, 1.0)


def outline(image_bgr: np.ndarray, polygon: Polygon | None, *,
            colour: tuple[int, int, int] = (0, 255, 255), thickness: int = 0) -> np.ndarray:
    """Draw the boundary onto a copy of the frame, so a reader can see where it runs.

    A masked frame shows *that* something was suppressed but not what the outline was, and on
    a dark pitch at night the filled region and the real shadows are hard to tell apart. With
    the outline drawn, "the evidence is inside the pitch" becomes something the reader checks
    rather than something the caption asserts.

    Thickness scales with the frame by default so the line survives the JPEG that carries it
    to the browser at whatever size the camera happens to produce.
    """
    import cv2
    import numpy as np

    if not polygon:
        return image_bgr
    h, w = image_bgr.shape[:2]
    points = np.array([[int(round(x * w)), int(round(y * h))] for x, y in polygon],
                      dtype=np.int32)
    out = image_bgr.copy()
    cv2.polylines(out, [points], isClosed=True, color=colour,
                  thickness=thickness or max(2, min(h, w) // 300), lineType=cv2.LINE_AA)
    return out
