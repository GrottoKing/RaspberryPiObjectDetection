# Camera Sightings Log

A running log of everything a Raspberry Pi camera sees, served as a web page
you can open from any device on your network.

Each object gets **one entry**, not one per frame — it says what the thing is
and what colour it is ("Person with dark hair wearing blue", "Blue bus",
"Red cup"), files it under a category, and keeps **a single snapshot** of it.
Hover any entry and that snapshot follows your cursor until you move away.

Categories are not a fixed list. They appear in the sidebar as the camera
meets new kinds of thing, and the new ones flash when they first show up.

```
People          6
Vehicles        1
Personal Items  1
```

---

## Quick start on the Pi

```bash
git clone https://github.com/GrottoKing/RaspberryPiObjectDetection.git
cd RaspberryPiObjectDetection
bash scripts/install.sh
./.venv/bin/python run.py
```

It prints the address to open, e.g. `http://192.168.1.42:8010/`. Open that on
your phone or laptop — anything on the same network works.

### Try it before the camera is set up

No camera and no model needed:

```bash
python3 run.py --camera synthetic --backend mock
```

Or point it at a folder of photos to see real detections:

```bash
python3 run.py --camera folder --folder samples
```

---

## What it can actually recognise

The detector is **YOLO11n**, which knows 80 everyday object types — person,
cat, dog, car, bus, bicycle, bottle, cup, chair, laptop, phone, TV, book,
backpack, and so on. That is the honest limit: it will not tell you the make of
a car or who a person is.

The colour descriptions are added on top by sampling the pixels inside the box:

| What the camera sees | What the log says |
|---|---|
| A person in a blue jacket, dark hair | `Person with dark hair wearing blue` |
| A blue bus | `Blue bus` |
| A tie | `Dark grey tie` |
| Something too busy to call | `Laptop` (no colour invented) |

For a person it samples the head band for hair tone and the torso band for
clothing. It is a heuristic, not magic — but it refuses to guess: if the pixels
do not clearly support a colour, the attribute is simply left out rather than
made up.

**On the example in the original request** — "Person wearing blue jacket": you
get `Person with dark hair wearing blue`. The garment *type* ("jacket" vs
"shirt") is not classified, because nothing in this stack can tell those apart
reliably, and inventing it would make the log untrustworthy.

### Want better descriptions?

Swap in a bigger model — same code, just a different file:

```bash
python3 scripts/fetch_model.py --model yolo11s   # more accurate, ~2-3x slower
```

Then set `detector.model: models/yolo11s.onnx` in `config.yaml`.

---

## The web page

- **Sidebar** — categories, with live counts. New categories appear as they are
  discovered. "Most seen" lists the raw labels.
- **Hover preview** — point at any entry and its snapshot appears next to the
  cursor and tracks it. It stays up while you drift between nearby rows and
  fades once you move ~170px away. On a phone (no hover), tap for a full-screen
  view instead.
- **Live view** — a button in the header opens an MJPEG stream of what the
  camera sees right now, with boxes drawn on the tracked objects. The stream
  only runs while the panel is open.
- **Filter and search** — click a category, or type to filter by description,
  label or colour.
- **Pause updates** — freezes the list so it does not shuffle under you while
  you are reading.

Snapshots are the full frame with a box drawn round the object, scaled down to
720px wide, so you get context rather than a cryptic crop.

---

## Configuration

Everything has a working default. To change something:

```bash
cp config.example.yaml config.yaml
```

The knobs you are most likely to want:

| Setting | Why you would change it |
|---|---|
| `camera.rotation` | Camera mounted upside down → `180` |
| `camera.fps_limit` | Lower on a Pi 4/Zero (try `2`), raise on a Pi 5 |
| `detector.confidence` | Log full of nonsense → raise to `0.5`+. Missing obvious things → lower to `0.3` |
| `detector.classes` | Only care about some things → `[person, cat, dog]` |
| `detector.min_box_area` | Distant specks being logged → raise it |
| `tracker.min_hits` | Flickery one-off sightings → raise to `5` |
| `storage.max_entries` | How much history to keep before old rows and their images are deleted |

Command-line flags override the file: `run.py --fps 2 --conf 0.5 --port 8080`.
Run `python3 run.py --help` for the full list.

---

## Running it as a service

So it starts on boot and restarts if it falls over:

```bash
sudo cp scripts/objectlog.service /etc/systemd/system/
# edit User= and the two paths inside if you did not clone to /home/pi
sudo systemctl enable --now objectlog
journalctl -u objectlog -f      # watch the logs
```

---

## How it works

```
camera ──► detector ──► tracker ──► describe ──► SQLite + JPEG
(picamera2)  (YOLO11n)    (IoU)      (colour)          │
                                                       ▼
                                            Flask ──► the web page
```

- **`objectlog/camera.py`** — frame sources: `picamera2` (the real one),
  `opencv` (USB webcams), `folder` (replay photos), `synthetic` (test pattern).
  `auto` tries them in order, so it always starts.
- **`objectlog/backends/`** — detectors. `onnx` (recommended: small wheel, no
  PyTorch), `ultralytics` (if you already have it), `mock` (no model needed).
- **`objectlog/tracker.py`** — greedy IoU tracking. This is what turns "person
  seen in 40 consecutive frames" into one log entry with a duration.
- **`objectlog/describe.py`** — the colour/attribute pass.
- **`objectlog/store.py`** — SQLite log plus snapshot files, with pruning so an
  SD card cannot fill up.
- **`objectlog/pipeline.py`** — the loop that ties it together, on its own
  thread. A failure backs off and retries rather than killing the service.

A track's snapshot is **replaced** while the object is still in view and the
detector gets more confident about it (up to four times), so the image you end
up with is a decent look at the thing rather than the blurry frame it first
appeared in.

### Performance

Rough guide at 1280x720:

| Board | Detection time | Sensible `fps_limit` |
|---|---|---|
| Pi 5 | ~120–200ms | 3–5 |
| Pi 4 | ~400–600ms | 1–2 |
| Pi Zero 2 W | ~1.5–2s | 0.5 |

The ONNX model is fed a **rectangular** letterbox (640x384 for a 16:9 frame)
rather than a padded square, which is ~40% less work per frame and slightly
more accurate, since no pixels are wasted on grey bars.

---

## API

The page is just a client for these:

| Endpoint | What it gives you |
|---|---|
| `GET /api/state` | Entries, categories, label counts, live objects, detector status. Takes `since_id` and `since_ts` for incremental polling. |
| `GET /api/log?before_id=&category=&limit=` | Paging back through history |
| `POST /api/clear` | Wipe the log and all snapshots |
| `GET /snapshots/<file>` | A snapshot image |
| `GET /stream.mjpg` | Live MJPEG preview |
| `GET /healthz` | 200 when healthy, 503 with the error when not |

---

## Troubleshooting

**"could not start the detector"** — run
`python3 run.py --camera synthetic --backend mock` to confirm the web side
works, then deal with the camera separately.

**Camera not found.** Check `rpicam-hello --list-cameras` sees it. If that
works but Python does not, `picamera2` is probably missing from the venv —
`scripts/install.sh` creates it with `--system-site-packages` for exactly this
reason, since `picamera2` must come from `apt`, not `pip`.

**No model.** `python3 scripts/fetch_model.py`. If the download is blocked,
export it on another machine with `pip install ultralytics` and copy the
`.onnx` into `models/`.

**Colours look wrong.** The camera needs a few seconds of auto-white-balance
after start (it waits 2s). In poor light everything reads as grey or black —
that is the sensor, not the code.

**Page reachable locally but not from other devices.** `web.host` must be
`0.0.0.0` (the default), and check the Pi's firewall.

---

## Tests

```bash
python3 -m unittest discover -s tests -v
```

55 tests, no camera or model required — the ONNX decoding is checked against a
synthetic model with planted detections, and the pipeline runs end to end on
the synthetic camera.
