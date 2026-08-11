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

---

## Using a Hailo AI HAT+ / AI Kit

If you have a Hailo accelerator, use it. It runs models far larger than the Pi's
CPU can manage, in a fraction of the time, and the false positives that plague
`yolo11n` largely go away because you can afford a proper model.

### 1. Install the runtime — mind which chip you have

**`hailo-all` is the Hailo-8 package**, despite the name. On a Hailo-10H it
installs a driver that finds the card and then cannot start it.

```bash
sudo apt update
sudo apt install -y hailo-h10-all      # Hailo-10H  (AI HAT+, 40 TOPS)
# sudo apt install -y hailo-all        # Hailo-8 / 8L
sudo reboot
```

Not sure which you have? `lspci | grep -i hailo` names the chip.

For full bandwidth to the accelerator, enable PCIe Gen 3 — add this to
`/boot/firmware/config.txt` and reboot:

```
dtparam=pciex1_gen=3
```

### 2. Check what you have

```bash
sudo systemctl stop objectlog     # the device allows one user at a time
python3 scripts/hailo_probe.py
```

This reports the firmware, the PCIe link, whether the Python bindings import,
which compiled models (`.hef`) are on the system, and — importantly — the exact
shape of what one inference returns. **If the Hailo backend misbehaves, send
this output.** A model's result layout is fixed when it is compiled, so this is
the only reliable way to know how to read it.

A common trap: the bindings are installed by `apt`, into the system Python. If
your virtualenv cannot see system packages, `import hailo_platform` fails.
`scripts/install.sh` creates it correctly; a venv made by hand may not:

```bash
python3 -m venv --system-site-packages .venv
```

### 3. Turn it on

```yaml
detector:
  backend: hailo
  hef: /usr/share/hailo-models/yolov8m.hef   # or leave empty to auto-discover
  confidence: 0.45
```

`backend: auto` also tries Hailo first and falls back to the CPU quietly, so
you can leave it on auto if you prefer.

### Getting models

`hailo-all` installs a couple of dozen into `/usr/share/hailo-models/`. More
come from the [Hailo model zoo](https://github.com/hailo-ai/hailo_model_zoo).

**Match your chip.** A model compiled for one architecture will not load on
another, and the filename says which it is built for:

| Board | Chip | Filename suffix |
|---|---|---|
| AI HAT+ (newer, 40 TOPS) | Hailo-10H | `_h10` |
| AI HAT+ 26 TOPS | Hailo-8 | `_h8` |
| AI HAT+ 13 TOPS / AI Kit | Hailo-8L | `_h8l` |

The probe reports your architecture and marks every installed model as usable,
"not an object detector", or "built for a different chip". Note that most of
what ships is *not* a general detector — there are classifiers (`resnet`),
pose models (`_pose`), segmentation (`_seg`) and face detectors (`scrfd`) in
there, and pointing the backend at one of those will not work.

With a Hailo-8 there is no reason to run a nano model — `yolov8m` or larger is
comfortably real-time, and that is where the accuracy gain comes from.

Once it is working, raise the frame rate too. `fps_limit: 4` exists because CPU
inference is slow; the accelerator has no such problem, and a higher rate means
objects passing quickly are less likely to be missed:

```yaml
camera:
  fps_limit: 15
```

One thing that does *not* change: the tracker still collapses an object into a
single log entry however many frames it appears in, so a faster rate makes the
log more accurate, not longer.

### If boxes land in the wrong place

The one thing that varies between compiled models is whether they expect a
letterboxed frame (aspect preserved, grey bars) or a stretched one. If
detections are consistently offset or squashed, flip it:

```yaml
detector:
  hailo_letterbox: false
```

### Wrong software stack: `hailo-all` is Hailo-8 only

The most likely reason a Hailo-10H never starts. Despite the name, the
`hailo-all` metapackage is **"Hailo-8 support"** — installing it on a
Hailo-10H gives you a driver that finds the card and then cannot load firmware
into it, because only the Hailo-8 firmware is on disk.

```bash
ls -l /lib/firmware/hailo/     # only hailo8_fw*.bin? that is the problem
```

```bash
sudo apt install -y hailo-h10-all
sudo reboot
```

apt will remove `hailo-all`, `hailort` and `python3-hailort` in the process.
That is expected — the two stacks provide the same Python module and cannot be
installed together. `hailo-models`, which supplies the `.hef` files, is a
separate package and is unaffected.

| Chip | Metapackage | Runtime | Driver module |
|---|---|---|---|
| Hailo-10H | `hailo-h10-all` | `h10-hailort` | `hailo1x_pci` |
| Hailo-8 family | `hailo-all` | `hailort` | `hailo_pci` |

### "Firmware load failed" / no /dev/hailo0

If `dmesg | grep -i hailo` shows something like:

```
hailo1x 0001:01:00.0: Writing file hailo/hailo10h/customer_certificate.bin
hailo1x 0001:01:00.0: Failed with error -2 to write file ...
hailo1x 0001:01:00.0: Firmware load failed
hailo1x 0001:01:00.0: probe with driver hailo1x failed with error -2
```

then the driver found the card and could not load firmware into it. Error `-2`
is "file not found": the firmware simply is not on disk. The board never
starts, so no device node is created and everything downstream fails with
`HAILO_OUT_OF_PHYSICAL_DEVICES`. Nothing is holding the device.

```bash
ls -l /lib/firmware/hailo/          # what is actually installed
dpkg -l | grep -i hailo             # which packages you have
apt-cache search hailo              # what is available
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

**Note the two driver families.** `hailo_pci` drives Hailo-8 parts;
`hailo1x_pci` drives Hailo-10H and Hailo-15H. Loading the wrong one appears to
succeed and achieves nothing. `lsmod | grep hailo` shows which you have.

A Hailo-10H also expects the HailoRT 5.x runtime. If `hailo_platform` reports
4.x while the loaded driver is 5.x, the packages are mismatched and a
full-upgrade is what resolves it.

### "not enough free devices ... found: 0"

`HAILO_OUT_OF_PHYSICAL_DEVICES` with **found: 0** does *not* mean the device is
busy — it means the runtime can see no accelerator at all. Check whether the
kernel driver has created a device node:

```bash
ls -l /dev/hailo*
```

If there is none, the card is on the PCIe bus but unclaimed by the driver:

```bash
sudo modprobe hailo_pci            # load it by hand
dmesg | grep -i hailo | tail -20   # what did it say?
dkms status                        # did the module build for this kernel?
```

A missing module is usually a kernel update the driver was not rebuilt for:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install --reinstall hailo-all
sudo reboot
```

If `/dev/hailo0` *does* exist and you still get this error, then something is
genuinely holding it — the probe will name the process.

### Two HailoRT APIs

HailoRT 5.x (Hailo-10H) implements the `InferModel` interface; the older
vstream calls return `HAILO_NOT_IMPLEMENTED`. HailoRT 4.x (Hailo-8) is the
other way round. The backend tries `InferModel` first and falls back, and the
probe reports which one was used, so either generation works unchanged.

### Honest status

The decoding logic is unit-tested against every output shape HailoRT is known
to produce, and the runner is tested end to end against a stand-in for the
5.x API. **No test touches a real accelerator** — HailoRT cannot be installed
off a Pi. `hailo_probe.py` runs the same code path the detector does, so what
it reports is what actually happens.

---

## When it sees things that aren't there

A nano-sized COCO model will confidently hallucinate. Indoors it is especially
fond of finding cars in filing cabinets and people in wood grain. This is the
model's limit, not a bug in the plumbing — but it is very fixable.

### First, measure. Don't guess.

```bash
sudo systemctl stop objectlog        # free the camera
python3 scripts/tune.py
```

Point the camera at the room as it normally sits, with nothing in it you
actually want logged. Anything it reports is by definition a false positive.
It samples 20 frames, then prints what it saw, at what confidence, how a higher
threshold would change things, and a `config.yaml` block you can paste in.

```
class            frames    max  median  typical size
------------------------------------------------------------------------
car                  14   0.58    0.49          22.1%
person                9   0.71    0.63          31.4%
```

That tells you something a guess cannot: whether raising the threshold is even
capable of fixing it. If a false "person" peaks at 0.71, no threshold below
that will clear it without also throwing away real people.

### Then apply the fix that matches

In rough order of effectiveness:

**1. Exclude the classes that cannot be there.** The single most reliable fix.
There are no cars in your office, so no confidence score should ever produce
one:

```yaml
detector:
  exclude_classes: [car, truck, bus, train, boat, airplane]
```

Or invert it, which is stronger still — list only what you care about:

```yaml
detector:
  classes: [person, cat, dog, cup, bottle, laptop, cell phone, book, chair]
```

**2. Use a bigger model.** `yolo11n` is the smallest one there is; false
positives are the price. `yolo11s` has roughly three times the parameters and
noticeably fewer of them. It is **committed to this repository**, so there is
nothing to download:

```yaml
detector:
  model: models/yolo11s.onnx
```

It is about 1.5–2× slower per frame. At `fps_limit: 4` on a Pi 5 there is room;
on a Pi 4, drop `fps_limit` to 1–2 at the same time.

(Only `yolo11n` is published as a ready-made ONNX file, and converting the
others needs PyTorch — not something worth installing on a Pi to convert one
file, hence the committed copy. `scripts/fetch_model.py` still handles the
download or conversion if you want a different model.)

**3. Raise the confidence threshold** — but only as far as `tune.py` says is
useful:

```yaml
detector:
  confidence: 0.55
```

**4. Cap the box size.** A detector that has locked onto a desk usually reports
a box covering a third of the view. Real objects at desk distance rarely do:

```yaml
detector:
  max_box_area: 0.45
```

**5. Mask the offending area.** Last resort, because you lose real detections
there too:

```yaml
detector:
  ignore_regions:
    - [0.0, 0.75, 1.0, 1.0]     # bottom quarter of the frame
```

**6. Make the scene easier.** More light helps more than any setting: these
models were trained on well-exposed photographs, and a dim, noisy frame is
outside what they have seen. Aiming the camera so a large textured surface does
not dominate the view helps too — a desktop filling the frame is exactly the
kind of thing that gets misread.

After any change: `sudo systemctl restart objectlog`.

### What about actually training it?

You can fine-tune YOLO on your own images, and the specific technique for this
problem is to include "negative" images — photos of your office with no
labelled objects — so the model learns that your desk is background. It works.

It also means labelling a few hundred images, access to a GPU for a few hours,
and a retraining loop every time the room changes. For suppressing known-wrong
classes in one room, an `exclude_classes` line gets you the same outcome in
thirty seconds. Fine-tuning earns its keep when you need to detect something
COCO does not know about at all — a specific product, a particular animal —
not when you need it to stop seeing cars indoors.

If you do want a genuine capability jump rather than a tuning fix, the hardware
route is a **Hailo AI HAT** (26 TOPS, around £70): it runs far larger models in
real time on a Pi 5, and the accuracy difference is a different category
entirely from anything in this section.

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
| `detector.exclude_classes` | Silence specific false positives → `[car, truck, bus]` |
| `detector.min_box_area` | Distant specks being logged → raise it |
| `detector.max_box_area` | Something large and static keeps being logged → lower to `0.45` |
| `detector.ignore_regions` | One patch of the view keeps fooling it → mask it |
| `tracker.min_hits` | Flickery one-off sightings → raise to `5` |
| `tracker.min_seconds` | Same object logged repeatedly → raise to `1.0` |
| `tracker.rejoin_seconds` | How long a fixture is remembered so it is not re-logged. Raise for a static room, `0` to log every reappearance |
| `tracker.max_missing_seconds` | Object re-logged when it briefly vanishes → raise to `4` |
| `storage.max_entries` | How much history to keep before old rows and their images are deleted |

Command-line flags override the file: `run.py --fps 2 --conf 0.5 --port 8080`.
Run `python3 run.py --help` for the full list.

---

## Starting on boot

```bash
sudo bash scripts/install-service.sh
```

That works out your username and the repo path itself, adds you to the `video`
group if needed, installs a systemd service, starts it, and then tells you
whether it actually came up — printing the log if it did not.

From then on it starts at boot and restarts by itself if it crashes.

```bash
journalctl -u objectlog -f          # watch what it is doing
sudo systemctl restart objectlog    # after changing config.yaml
sudo systemctl stop objectlog
sudo systemctl disable objectlog    # stop starting on boot
```

Note that the service and a hand-started `run.py` will fight over the port. If
you want to run it manually for a bit, `sudo systemctl stop objectlog` first.

`scripts/objectlog.service` is the plain template, if you would rather install
it by hand — it assumes user `pi` and `/home/pi/RaspberryPiObjectDetection`, so
edit `User=` and the two paths to match your setup.

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
- **`objectlog/backends/`** — detectors. `hailo` (AI HAT+ accelerator), `onnx`
  (recommended on CPU: small wheel, no PyTorch), `ultralytics` (if you already
  have it), `mock` (no model needed).
- **`objectlog/tracker.py`** — greedy IoU tracking. This is what turns "person
  seen in 40 consecutive frames" into one log entry with a duration.
- **`objectlog/describe.py`** — the colour/attribute pass.
- **`objectlog/store.py`** — SQLite log plus snapshot files, with pruning so an
  SD card cannot fill up.
- **`objectlog/pipeline.py`** — the loop that ties it together, on its own
  thread. A failure backs off and retries rather than killing the service.

### Not logging the same thing twice

The room's fixtures — a shelf, a clock, a chair — should be logged once, not
every time the detector blinks. When a track closes, the tracker remembers it
if the object **stayed put** during its life, and a later detection in the same
place resumes that entry instead of creating a new one. The memory is reseeded
from the database at startup, so restarting the service does not re-log the
furniture either.

The "stayed put" test is what makes this safe. A person who walks through the
frame twice is genuinely two sightings and gets two entries; only things that
did not move are treated as the same object returning. Tune with
`tracker.rejoin_seconds` (0 disables it).

A track's snapshot is **replaced** while the object is still in view and the
detector gets more confident about it (up to four times), so the image you end
up with is a decent look at the thing rather than the blurry frame it first
appeared in.

### Performance

Rough guide at 1280x720:

| Board | Detection time | Sensible `fps_limit` |
|---|---|---|
| Pi 5 + Hailo-8 | ~10–20ms | 10–15 |
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

112 tests, no camera, model or accelerator required — the ONNX decoding is checked against a
synthetic model with planted detections, and the pipeline runs end to end on
the synthetic camera.
