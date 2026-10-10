# Keeping the live worker running

The live worker is the program that records: it follows `configs/slots_schedule.json`, takes
one picture per camera per minute during each slot, and writes each slot's verdict to the
database when the slot ends. It is a separate program from the dashboard, and **nothing is
recorded while it is not running** — so on the Mini-PC it should start by itself when the
computer starts and come back if it stops.

The command it runs, from the project folder:

```
uv run python -u -m pitch_occupancy.worker --source live --model yolov8n --yes
```

| part | why |
|---|---|
| `--source live` | open the cameras in `configs/cameras.json` and follow the schedule in real time |
| `--model yolov8n` | the detector; use the one `configs/rules.json` names (the `/cameras` page shows the exact command) |
| `--yes` | skip the "type yes to connect" confirmation. A service has nobody to type it — **run it once by hand without `--yes` first**, to see what it is about to open |
| `-u` | Python writes its output straight away instead of buffering it, so the log is up to date |

Other flags that exist: `--evidence-dir data/evidence` (save evidence pictures — these show
identifiable people, so only if you need them), `--cameras PATH` (a different camera file),
`--derive-roi` (measure a missing pitch boundary from the stream), `--dry-run` (list what it
would open and exit) and `--iterations N` (stop after N one-minute ticks; for testing, not for
a service).

**It reads the schedule and the camera list only when it starts.** After changing either (on
the `/cameras` page or by hand), restart the worker. A slot that is running when you restart
is cancelled (no verdict is written for it) and recorded again from where the clock is when the
worker comes back — usually as `REVIEW`, because part of it was not watched.

## Is it running?

- **The `/cameras` page**, *Recording status* card at the top: "Running" with when it was last
  seen, the slots being recorded now (minutes done of expected, the last picture per camera),
  the next slots, the last finished slots and recent errors. It refreshes every 30 seconds.
- **The file** `data/worker_status.json`, which the worker rewrites at least once a minute:

  ```
  type data\worker_status.json        (Windows)
  cat data/worker_status.json         (Linux)
  ```

  `updated_at` is the last time the worker wrote it. If it is more than 3 minutes old the
  worker is not running, whatever `state` says: a program that crashes or is killed cannot
  write "stopped", so the old timestamp is the sign. `"state": "stopped"` means it exited
  normally (Ctrl+C or `--iterations`); `"error"` means it stopped on an error, with the message
  under `errors`.
- No status file at all means the worker has never been started on this computer (from this
  project folder).

## Windows: Task Scheduler (built in)

Task Scheduler's own "restart if the task fails" does not reliably restart a program that
exits, so the task runs a small script that restarts the worker itself. Save this as
`run_worker.bat` in the project folder (change the first line to your folder):

```bat
@echo off
cd /d C:\Users\YOU\Desktop\football
if not exist data\logs mkdir data\logs
:again
echo === worker starting %date% %time% === >> data\logs\worker.log
uv run python -u -m pitch_occupancy.worker --source live --model yolov8n --yes >> data\logs\worker.log 2>&1
echo === worker exited with %errorlevel% at %date% %time%, restarting in 60 s === >> data\logs\worker.log
timeout /t 60 /nobreak > nul
goto again
```

Then create the task (Task Scheduler > Create Task):

1. **General**: name `Pitch recording`; *Run whether user is logged on or not*; run as the
   user who installed the project (so `uv` and the project's files are found).
2. **Triggers**: New > *At startup*. Add a 1-minute delay so the network is up first.
3. **Actions**: New > *Start a program* > `C:\Users\YOU\Desktop\football\run_worker.bat`,
   *Start in*: `C:\Users\YOU\Desktop\football`.
4. **Conditions**: untick *Start the task only if the computer is on AC power*.
5. **Settings**: tick *If the task fails, restart every 1 minute*; **untick** *Stop the task if
   it runs longer than 3 days*; *If the task is already running*: *Do not start a new instance*.

Or from an administrator command prompt (it asks for the user's password itself):

```
schtasks /Create /TN "Pitch recording" /SC ONSTART /DELAY 0001:00 /RU YOU /RP ^
  /TR "C:\Users\YOU\Desktop\football\run_worker.bat"
```

then untick the 3-day limit and the AC-power condition in the task's properties as above.

Stopping it: *End* in Task Scheduler kills the script and the worker without a clean exit, so
the status card will say it stopped without saying so — expected in that case. Two copies must
never run at once: they would record every slot twice.

## Windows: NSSM (as a service)

[NSSM](https://nssm.cc) runs any program as a Windows service and restarts it when it exits.
From an administrator prompt, with `uv.exe`'s real path (`where uv`):

```
nssm install PitchRecording "C:\Users\YOU\.local\bin\uv.exe" "run python -u -m pitch_occupancy.worker --source live --model yolov8n --yes"
nssm set PitchRecording AppDirectory C:\Users\YOU\Desktop\football
nssm set PitchRecording AppStdout C:\Users\YOU\Desktop\football\data\logs\worker.log
nssm set PitchRecording AppStderr C:\Users\YOU\Desktop\football\data\logs\worker.log
nssm set PitchRecording AppExit Default Restart
nssm set PitchRecording AppRestartDelay 60000
nssm set PitchRecording Start SERVICE_DELAYED_AUTO_START
nssm start PitchRecording
```

Set the service to log on as the user who installed the project (`services.msc` > the service
> *Log On*), not *Local System*: the default system account does not have that user's `uv`, its
Python environment or its `configs/`. Create `data\logs` first. `nssm stop PitchRecording`
stops it; `nssm restart PitchRecording` picks up a changed schedule.

## Linux: systemd

`/etc/systemd/system/pitch-recording.service`:

```ini
[Unit]
Description=Pitch occupancy live recording
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pitch
WorkingDirectory=/home/pitch/football
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/pitch/.local/bin/uv run python -m pitch_occupancy.worker --source live --model yolov8n --yes
Restart=always
RestartSec=60
# Ctrl+C equivalent, so the worker writes "stopped" on the way out.
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

```
sudo systemctl daemon-reload
sudo systemctl enable --now pitch-recording
systemctl status pitch-recording
journalctl -u pitch-recording -f          # the log
sudo systemctl restart pitch-recording    # after changing the schedule or cameras
```

## Logs

The worker prints one line per finished slot (verdict, minutes captured of expected, capture
rate), any slot that failed or was cancelled, and its start-up checks (cameras, boundaries).
With the set-ups above that output goes to `data\logs\worker.log` (Windows) or the journal
(Linux). The log file is not rotated by the worker; it grows by a few lines per slot, so
clearing it now and then is enough. Camera passwords are not printed.

## What this does not do

- It does not run `pitch retention` on a schedule; add a second daily task for that
  (`docs/runbook.md`, "Disk full").
- It does not alert anyone when the worker stops. The status card shows it; checking the card
  is part of the daily check (`docs/cameras.md`).
