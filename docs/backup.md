# Backup policy (WP0-T8)

**Status: the policy is written; the backup is not yet made.** This document exists so the
gap is stated rather than implied, and so that when it is closed there is something to close
it against.

---

## What is irreplaceable

Exactly one thing.

| | size | replaceable? | where |
|---|---|---|---|
| **`data/` — the footage and extracted frames** | ~4.2 GB | **No.** It came from a client facility on one occasion. | one machine, one copy |
| `data/cache/` — feature caches | ~1.5 GB | Yes — ~25 min per backbone from the frames | same machine |
| `results/` | small | Yes — `uv run python -m experiments.reproduce_all` | git |
| code, plans, thesis text | small | Yes | git, GitHub |

Every number this project reports is regenerable from `data/` through one pipeline. **`data/`
is not regenerable from anything.** If it is lost, the thesis loses its subject: the 66 clips
were a one-off export, the two labelled venue_01 days cannot be re-recorded, and the labels
represent work that would have to be redone by hand.

That asymmetry is the whole policy. Everything else here follows from it.

## The rule

**Three copies, two media, one off-site** — the ordinary 3-2-1 rule, at the smallest scale
that satisfies it:

1. the working copy on the development machine;
2. an external drive, kept physically apart from the machine;
3. an encrypted copy in cloud storage.

`data/cache/` may be excluded from (2) and (3): it is a derived artefact, it is the largest
part of the footprint, and it rebuilds in under an hour. Excluding it turns a ~5.7 GB backup
into a ~4.2 GB one. **`data/raw/` and `data/processed/` may never be excluded** — the second
is the labelled corpus and the first is what it was extracted from.

## Encryption and access

The footage shows identifiable people at a client facility, so:

* the off-site copy is **encrypted at rest** — the cloud provider holds ciphertext only;
* the passphrase is not stored beside the backup, and not in this repository;
* the external drive is encrypted too. A drive in a bag is the copy most likely to be lost.

`thesis/ethics.md` holds the legal basis and the retention position; this document covers only
where copies live.

## Cadence

| when | what |
|---|---|
| **now, once** | the full 4.2 GB to both the drive and the cloud |
| after any new footage arrives | the same, before any processing |
| after a labelling session | `data/processed/` and the manifest |
| never | automatically on a schedule — see below |

**Deliberately manual, and deliberately not a sync.** A scheduled two-way sync propagates a
deletion as faithfully as it propagates a file, and the failure mode this policy exists to
prevent is losing footage, not losing time. A copy made by hand after a change that matters
is worth more than a mirror that would have replicated the mistake.

## Restoring

A backup nobody has restored from is a backup nobody knows the state of.

```bash
# 1. restore data/ from the external drive or the cloud copy
# 2. rebuild everything derived from it
uv sync
uv run pitch manifest                       # rebuild the index
uv run python -m experiments.reproduce_all --check   # what is missing
uv run python -m experiments.reproduce_all           # rebuild it
uv run pytest                                # the suite must pass on the restored copy
```

The reproduction pipeline is what makes this checkable: a restore is successful when
`reproduce_all --check` reports nothing missing and the committed CSVs are reproduced. That
is a stronger test than "the files are there", and it is the same guard the thesis's
reproducibility claim rests on.

**Verify the restore once, from the off-site copy, before relying on it.** Restoring from the
drive you can see proves less than restoring from the copy you cannot.

## What is deliberately not backed up

* `.venv/` — rebuilt by `uv sync` from the committed lockfile.
* `data/cache/*.npz` — see above.
* Anything already in git and pushed to GitHub. The remote is the second copy of the code;
  it is *not* a second copy of the footage, and must never become one — `.gitignore` keeps
  `data/` out, and `WP1-T5` records that nine frames reached history before it did.

## Why this is still open

The policy costs an hour and has been the top item on the critical path since the first
review. It is unblocked — the data layout stopped moving once WP0-T2 landed — so what remains
is doing it. Until then, every other risk in the register is secondary to this one: the
results are regenerable, the footage is not.
