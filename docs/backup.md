# Backup policy

**Status: the backup exists; a restore has not yet been verified** (see *Current state*).
This document is the policy that state is checked against.

---

## What is irreplaceable

Exactly one thing.

| | size | replaceable? | where |
|---|---|---|---|
| **`data/` — the footage and extracted frames** | ~4.2 GB | **No.** It came from a client facility on one occasion. | one machine, one copy |
| `data/cache/` — feature caches | ~1.5 GB | Yes — ~25 min per backbone from the frames | same machine |
| code and configuration | small | Yes | git, GitHub |

Everything derived — manifest, caches, fitted models — is regenerable from `data/`. **`data/`
is not regenerable from anything.** If it is lost, the project loses its footage: the 66 clips
were a one-off export, the two labelled venue_01 days cannot be re-recorded, and the labels
represent work that would have to be redone by hand.

That asymmetry is the whole policy. Everything else here follows from it.

## The rule

**Three copies, two media, one off-site** — the ordinary 3-2-1 rule, at the smallest scale
that satisfies it:

1. the working copy on the development machine;
2. an external drive, kept physically apart from the machine;
3. an encrypted copy in cloud storage.

## Current state (2026-09-09)

**The backup exists.** The 4.2 GB is held in **Amazon S3** and on **Google Drive**, both
off-machine. That is three copies over two independent providers, which satisfies the rule
above by a different route than the one it describes: the external drive of (2) is replaced by
a second cloud location. Two providers is not two *media*, and the tail risk that differs is an
account-level one — a billing lapse or a credential compromise reaching both — rather than the
physical failure an external drive protects against. Worth a drive eventually; not worth
blocking on, because the risk that actually mattered was having one copy.

**Not yet verified.** No file has been opened from either copy to confirm the restore works,
so this is *mitigated*, not *closed*. The verification below is the step that closes it, and an
unverified backup is one you find out about at the worst moment. Record the date here when it
is done.

| | location | verified restore |
|---|---|---|
| working copy | development machine | — |
| off-site 1 | Amazon S3 | **not yet** |
| off-site 2 | Google Drive | **not yet** |

`data/cache/` may be excluded from (2) and (3): it is a derived artefact, it is the largest
part of the footprint, and it rebuilds in under an hour. Excluding it turns a ~5.7 GB backup
into a ~4.2 GB one. **`data/raw/` and `data/processed/` may never be excluded** — the second
is the labelled corpus and the first is what it was extracted from.

## Encryption and access

The footage shows identifiable people at a client facility, so:

* the off-site copy is **encrypted at rest** — the cloud provider holds ciphertext only;
* the passphrase is not stored beside the backup, and not in this repository;
* the external drive is encrypted too. A drive in a bag is the copy most likely to be lost.

The retention worker (`uv run pitch retention`) enforces how long sampled frames and
evidence images are kept; this document covers only where copies live.

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
uv run pitch manifest --check               # the index must match the class folders
uv run pitch manifest                       # rebuild the index
uv run pitch cache                          # rebuild the feature caches
uv run pytest                               # the suite must pass on the restored copy
```

A restore is successful when the manifest rebuilds with the same frame counts as before and
the caches rebuild from it. That is a stronger test than "the files are there".

**Verify the restore once, from the off-site copy, before relying on it.** Restoring from the
drive you can see proves less than restoring from the copy you cannot.

## The verdict database: `pitch backup`

Everything above is about the footage. The running product also writes one file that cannot
be rebuilt from anything: **`data/db/pitch_monitor.db`** — every slot verdict, every operator
correction, every anomaly. It is small, it changes every few minutes, and unlike the footage it
*should* be copied on a schedule: a daily copy is a new file each day, never a sync, so the
"a sync replicates the mistake" argument above does not apply.

```bash
uv run pitch backup                              # -> data/backups/
uv run pitch backup --dest D:\pitch-backups --keep 30
uv run pitch backup --with-secrets               # also the files that hold passwords
```

What it does:

* copies the database with **SQLite's online backup API**, so it is safe while the worker is
  writing — no need to stop anything;
* names the copy `pitch_monitor-YYYYMMDD-HHMMSS.db` and checks it with
  `PRAGMA integrity_check`; a copy that fails is deleted and the command exits with an error;
* zips this system's own config files beside it as `pitch_monitor-YYYYMMDD-HHMMSS-configs.zip`:
  `rules.json`, `roi*.json`, `slots_schedule.json` (`--no-configs` skips the zip);
* keeps the newest `--keep` backups (14 by default — two weeks of daily copies) and deletes
  older ones, touching nothing else in the folder.

**Passwords.** `cameras.json`, `bookings_source.json` and `alerts.json` contain logins — the
camera streams, the booking database, the alert mail or chat account. They are left out unless
you add `--with-secrets`. If you do, **the backup folder must be private**: anyone who can read
it can log in to the cameras. Do not point `--dest` at a shared or synced folder with
`--with-secrets`.

### Running it every day

**Windows (Task Scheduler)** — run once in a terminal, adjusting the project path:

```bat
schtasks /Create /TN "Pitch backup" /SC DAILY /ST 03:30 ^
  /TR "cmd /c cd /d C:\pitch-occupancy && uv run pitch backup >> data\backups\backup.log 2>&1"
```

Or in the Task Scheduler window: *Create Basic Task* → Daily, 03:30 → *Start a program*:
program `cmd`, arguments `/c uv run pitch backup >> data\backups\backup.log 2>&1`, *Start in*
the project folder. Tick *Run whether user is logged on or not*.

**Linux (cron)** — `crontab -e`:

```cron
30 3 * * * cd /opt/pitch-occupancy && uv run pitch backup >> data/backups/backup.log 2>&1
```

Pick a time outside the recording schedule; the backup is safe during recording, but a quiet
hour makes the log easier to read. Look at the log, or at the newest file's date in the
backup folder, now and then: a scheduled job that has been failing silently is no backup.

A copy on the same disk protects against a damaged database, not against a dead disk. Copy
the backup folder off the machine too (the off-site copies above), remembering the password
rule if it holds `--with-secrets` zips.

### Restoring the database

1. **Stop the recording program** (the worker) and the server, so nothing writes to the
   database while it is replaced.
2. Keep the current file aside, in case it is wanted later:
   `data/db/pitch_monitor.db` → `data/db/pitch_monitor.before-restore.db`. Delete any
   `pitch_monitor.db-wal` / `pitch_monitor.db-shm` beside it — they belong to the old file.
3. Copy the chosen backup to **`data/db/pitch_monitor.db`**.
4. If configs are to be restored too, unzip the `-configs.zip` and copy its `configs/` files
   back into the project's `configs/` folder.
5. Start the server and the worker again, and open the dashboard: the slots should be there up
   to the time of the backup. Verdicts recorded after it are not in the copy.

Restore a copy once, on a spare machine or folder, before you need to — the same rule as for
the footage.

## What is deliberately not backed up

* `.venv/` — rebuilt by `uv sync` from the committed lockfile.
* `data/cache/*.npz` — see above.
* Anything already in git and pushed to GitHub. The remote is the second copy of the code;
  it is *not* a second copy of the footage, and must never become one — `.gitignore` keeps
  `data/` out — nine frames reached history before it did.

## Why this matters more than anything else here

Until a restore has been verified, every other operational risk is secondary to this one:
everything derived is regenerable, the footage is not.
