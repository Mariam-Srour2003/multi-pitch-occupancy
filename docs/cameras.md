# Connecting cameras

A guide for whoever installs the cameras. The same guide is on the `/cameras` page, which is
also where cameras are added, tested, checked live and given a recording schedule.

Each pitch has **two fixed cameras, each seeing about half of it**. The system takes one
short burst of pictures per camera per minute, counts the people and looks for a ball inside
each camera's pitch boundary, adds the two halves together, and records each slot as empty,
playing, or used for something else.

## 1. What you need

- The camera on the **same network** as the Mini-PC that runs this system (same switch,
  router or VLAN), or reachable over a VPN.
- The camera's **IP address** (from its app, the router's device list, or the NVR).
- A **username and password** allowed to view live video. A separate "viewer" account is
  better than the admin account.
- **RTSP turned on** in the camera's settings (often Network > Advanced, "Integration
  protocol", or "ONVIF").

## 2. Which address to use

Replace `user`, `pass` and `IP` with your own. If the password contains `@`, `/`, `?` or `#`,
write them as `%40`, `%2F`, `%3F`, `%23` - the page refuses an address where they would be
misread.

| Camera | Address pattern | Notes |
|---|---|---|
| Hikvision (and HiLook) | `rtsp://user:pass@IP:554/Streaming/Channels/101` | 101 = main stream, **102 = sub-stream** (enough here) |
| Dahua, Amcrest | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` | subtype=1 is the sub-stream |
| Axis | `rtsp://user:pass@IP/axis-media/media.amp` | `?resolution=1280x720` asks for a smaller picture |
| Reolink | `rtsp://user:pass@IP:554/h264Preview_01_main` | `h264Preview_01_sub` for the sub-stream |
| Uniview | `rtsp://user:pass@IP:554/unicast/c1/s0/live` | `s1` = sub-stream |
| TP-Link Tapo | `rtsp://user:pass@IP:554/stream1` | create a "camera account" in the Tapo app first; `stream2` = sub-stream |
| Other ONVIF cameras | varies | the free **ONVIF Device Manager** finds the camera and shows its stream address |
| Through an NVR / DVR | `rtsp://user:pass@NVR-IP:554/Streaming/Channels/301` | one address per camera; Hikvision NVR channel 3 is `301` (main) / `302` (sub); Dahua NVR uses `channel=3` |
| HTTP snapshot or MJPEG | `http://user:pass@IP/path/to/snapshot.jpg` | for cameras without RTSP; confirm with the Test button that a picture arrives |
| USB webcam (bench tests) | `usb:0` | `usb:1` for a second webcam on the same PC |
| Cloud-only cameras (Ring, Nest, Arlo, most Wi-Fi "app" cameras) | - | **not supported directly**: they need RTSP/ONVIF switched on, or to be recorded by a local NVR whose stream this system reads |

## 3. Adding and testing

1. Open `/cameras`, fill in **venue**, **camera** (e.g. `venue_01`, `camera_A`) and the
   address, and press **Save camera**. The camera ids must match the ones the schedule uses.
2. Press **Test**. The test checks, in order: that something answers at the IP and port,
   that the video stream opens, and that a picture arrives - and shows the picture.
3. Draw the pitch boundary for the camera on the **Boundaries** page (`/roi`), so people
   on the next pitch are not counted. Without one, the spot check counts the whole picture
   and the live recording marks every minute of that camera "not sure".
4. Press **Check now** under "What is happening now?" for a spot check of the pitch. This
   records nothing.

Saved addresses are kept in `configs/cameras.json` (gitignored); each save keeps the previous
file under `configs/cameras_history/` (gitignored, last 20). The page always shows the
password as `***`; when editing, leave `***` in place to keep the saved password.

## 4. Recording a day

Under **Start recording a day**, choose the venue, pitch id, cameras, the first slot's start
time (e.g. 08:00), the slot length (e.g. 90 minutes), either a number of slots or an end time,
and the days of the week. **Preview** lists the slots (08:00-09:30, 09:30-11:00, ...);
**Save schedule** replaces that pitch's slots in `configs/slots_schedule.json` and keeps every
other pitch's (the old file is kept under `configs/schedule_history/`). Slots that run past
midnight, or that overlap another pitch's slot using the same camera or starting at the same
time, are refused.

Then, on the Mini-PC, in the project folder:

```
uv run python -m pitch_occupancy.worker --source live --model yolov8n --dry-run   # check, opens nothing
uv run python -m pitch_occupancy.worker --source live --model yolov8n             # asks you to type yes
```

`--model` is the detector named in `configs/rules.json`, the same one the spot check uses;
without it the worker uses the configured default model. Add `--evidence-dir data/evidence`
only if pictures must be kept as evidence. The worker reads the schedule when it starts:
restart it after changing the schedule.

Results: each finished slot is written to the database (`data/db/pitch_monitor.db`) and
shown on the dashboard at `/`. Retention: sampled frames 7 days, evidence pictures 365 days
(`uv run pitch retention`).

## 5. Recommended settings

- **Sub-stream is enough.** One burst a minute is all this system takes; 720p or even
  640x360 works and is lighter on the network.
- **H.264** rather than H.265/H.265+ if pictures fail to arrive.
- A **fixed IP** (or a DHCP reservation in the router) so the address survives a power cut.
- The camera's clock **synced by NTP**, so its on-screen time matches the records.

## 6. Placement

- Two cameras per pitch, **each seeing about half**. A small overlap in the middle is fine;
  a large one counts the same people twice.
- Include the touchlines and goal areas; avoid pointing into floodlights or the low sun.
- Draw each camera's boundary at `/roi`.

## 7. Network

- Check an address in **VLC**: Media > Open Network Stream, paste it. If VLC shows video, this
  system can read it.
- The Mini-PC must reach each camera on its RTSP port (usually **554**); check firewalls
  and VLAN rules.
- Remote sites: connect over a **VPN**. **Never** forward the RTSP port to the internet.

## 8. Troubleshooting

What the Test button reports, and what to do:

| The test says | Likely cause | What to do |
|---|---|---|
| The address is not usable | typo, unsupported scheme, unencoded character in the password, a host name that does not resolve | write it as `rtsp://user:password@192.168.1.64:554/...`; %-encode `@ / ? #` in the password |
| Nothing answered at that address | camera off, wrong IP, different network, firewall dropping traffic | ping the IP from the Mini-PC, check the cable / PoE light, check the IP in the camera app or NVR |
| The camera answered but the video port is closed | RTSP disabled, or a non-standard port | turn on RTSP in the camera settings, check the port (554 is usual) |
| The camera is reachable but the video would not open | wrong path, wrong user name or password, user not allowed live view | check the pattern for your brand (section 2), check the password on the camera's own web page |
| The video opened but no picture arrived | H.265, too high a resolution, overloaded network, camera still booting | use the sub-stream, switch to H.264, lower the resolution, try again in a minute |

If a camera stops working during a recorded day, the slot is recorded as partly or not
observed rather than guessed - see `docs/runbook.md` ("Camera offline").

## 9. Privacy

- The cameras film **identifiable people**. Put up signs and follow local CCTV rules.
- Camera addresses and passwords stay in `configs/cameras.json` on the Mini-PC only (it and
  its history folder are gitignored). The page and the API never return a saved password,
  and the live worker prints only the camera's host.
- Pictures are not stored unless the live worker is started with `--evidence-dir`; the
  `/cameras` page's snapshots and spot checks are never saved.
