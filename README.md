# Automatic Parking Violation Detector

Detect vehicles that are parked illegally — crossing over painted parking-space
lines into driving lanes — in **top-down / aerial parking-lot video**, using
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for vehicle
detection and OpenCV for geometry and visualization.

![status](https://img.shields.io/badge/status-work--in--progress-yellow)
![python](https://img.shields.io/badge/python-3.8%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What it does

1. Reads an aerial video (drone or fixed overhead camera) of a parking lot.
2. Runs YOLOv8 to detect and track cars, trucks, and motorcycles frame-to-frame.
3. Compares each vehicle against a grid of **individual parking-stall boxes**
   you define once, matching the real white-line markings in your footage.
4. Flags a vehicle as a violation only if it is **both**:
   - sitting mostly outside its stall (into the driving lane), **and**
   - stationary there for **more than 5 seconds** (configurable),

   which filters out cars that are simply driving past.
5. Draws stall outlines, green boxes for legally parked vehicles, and a
   flashing red box + live timer for violators.
6. Logs every confirmed violation to the console with a frame number,
   timestamp, and the specific stall it occurred near.

---

## Demo output

| Legal parking | In-progress crossing | Confirmed violation |
|---|---|---|
| Green box | Orange box + countdown | Flashing red box + "LINE VIOLATION" + timer |

Console output looks like:

```
Video: 1920x1080 @ 30.0 FPS | reference size assumed: (1522, 709)
Total stalls defined: 88
Violation requires crossing + stationary for 5.0s (~150 frames).

[debug] frame 30: 42 vehicle(s) detected this frame
[VIOLATION] track_id=17 | frame=214 | time=7.13s | stall=Row-3-Stall-7 | bbox=(410,201,478,266)
```

---

## Requirements

- Python 3.8+
- [Ultralytics](https://pypi.org/project/ultralytics/) (YOLOv8)
- OpenCV
- NumPy

```bash
pip install ultralytics opencv-python numpy
```

The first run will automatically download the `yolov8n.pt` weights (~6 MB)
if they aren't already present.

---

## Quick start

1. Place your source clip in the project folder as `parking_input.mp4`
   (or pass a different path with `--input`).
2. **Trace your parking rows** (see below — this step matters more than any
   other setting).
3. Run the detector:

```bash
python parking_violation_detector.py
```

The annotated result is written to `parking_result.mp4`.

### Custom input/output paths

```bash
python parking_violation_detector.py --input my_clip.mp4 --output my_result.mp4 --model yolov8s.pt
```

---

## Step 1 (important): define your parking rows

The script has to know where the real white lines are in **your** footage —
it can't guess this from the video alone. There's an interactive tool for
this, so you never have to hand-measure pixel coordinates:

```bash
python parking_violation_detector.py --pick-zones
```

This opens the first frame of your video in a window. For each row of
parking spaces:

1. Left-click its **4 corners in order**: top-left → top-right →
   bottom-right → bottom-left.
2. Press **`n`** — you'll be prompted in the terminal for how many
   individual stalls that row contains.
3. Repeat for every row in the lot.
4. Press **`s`** to print a ready-to-paste code block, or **`q`** to quit
   without saving.

Copy the printed `REFERENCE_FRAME_SIZE` and `PARKING_ROWS` block into the
top of `parking_violation_detector.py`, replacing the placeholder values.

The script automatically slices each row's quadrilateral into that many
equal stall boxes (using bilinear interpolation across the row, so a
mild perspective skew in the aerial shot is handled fine) — you don't need
to click every individual space by hand.

> Coordinates are stored relative to the frame size you traced on
> (`REFERENCE_FRAME_SIZE`) and are automatically rescaled to whatever
> resolution your actual video is at runtime, so they won't drift or
> distort even if you re-encode or resize the source clip later.

---

## Configuration reference

All tunable values live at the top of `parking_violation_detector.py`:

| Constant | Purpose | Notes |
|---|---|---|
| `VEHICLE_CLASSES` | COCO class ids to detect | Default `[2, 5, 7]`. Real COCO ids: `2`=car, `3`=motorcycle, `5`=bus, `7`=truck — add `3` if motorcycles aren't detected. |
| `CONF_THRESHOLD` | Minimum YOLO confidence to count a detection | Aerial vehicles score lower than ground-level ones; default `0.20`. Lower to `0.10–0.15` if recall is poor. |
| `IMG_SIZE` | Inference resolution passed to YOLO | Default `1280`. Raise to `1920` for very wide/high-res footage so small cars don't disappear during downscaling. |
| `VIOLATION_SECONDS` | How long a vehicle must be crossing + stationary before it's flagged | Default `5.0` seconds. |
| `INSIDE_RATIO_THRESHOLD` | Fraction of a vehicle's bounding-box area that must be inside its stall to count as "parked legally" | Default `0.65`. Raise for stricter enforcement, lower to reduce false positives. |
| `STATIONARY_PIXEL_THRESHOLD` | Max centroid movement (px) across recent frames to count as "not moving" | Default `15`. Increase if your camera isn't perfectly fixed. |
| `STATIONARY_HISTORY_FRAMES` | How many recent frames are used to judge stationarity | Default `10`. |
| `DEBUG_DETECTION_LOG_EVERY` | How often (in frames) to print a detection-count debug line | Default `30`. |
| `REFERENCE_FRAME_SIZE` / `PARKING_ROWS` | Your traced parking-row geometry | Set via `--pick-zones` (recommended) or by hand. |

---

## How the violation logic works

```
for each tracked vehicle:
    inside_ratio = % of its bounding box that overlaps a legal stall
    is_crossing  = inside_ratio < INSIDE_RATIO_THRESHOLD
    is_stationary = centroid hasn't moved more than STATIONARY_PIXEL_THRESHOLD
                    over the last STATIONARY_HISTORY_FRAMES frames

    if is_crossing and is_stationary:
        violation_streak += 1
    else:
        violation_streak = 0

    if violation_streak >= fps * VIOLATION_SECONDS:
        -> confirmed violation (flashing red box, console log, once per vehicle)
    elif is_crossing:
        -> orange "watching" box with live countdown
    else:
        -> green box (parked legally)
```

Vehicle identity across frames comes from YOLOv8's built-in ByteTrack
tracker (`model.track(..., persist=True)`), which is what makes the
"how long has this car been sitting there" timer possible in the first
place — a single-frame detector alone has no concept of time.

---

## Troubleshooting

**No bounding boxes appear at all / very few detections**
- Lower `CONF_THRESHOLD` (try `0.10`–`0.15`).
- Raise `IMG_SIZE` (try `1920`).
- Try a larger checkpoint: `yolov8s.pt` or `yolov8m.pt` instead of `yolov8n.pt`.
- Watch the `[debug] frame N: X vehicle(s) detected` console lines to confirm
  whether detection is the bottleneck before touching anything else.

**Zone outlines look skewed, shifted, or only partially appear**
- This means `REFERENCE_FRAME_SIZE` doesn't match the frame you actually
  traced coordinates on. Re-run `--pick-zones` on the exact video you're
  processing — it reads and prints the real resolution automatically.

**Vehicles briefly driving through a lane get flagged**
- Increase `VIOLATION_SECONDS` and/or `STATIONARY_PIXEL_THRESHOLD` isn't the
  fix here — check `STATIONARY_HISTORY_FRAMES` covers enough real time
  (frames ÷ fps) to reliably distinguish "passing through" from "parked."

**A parked car right at the edge of two stalls keeps flip-flopping**
- Lower `INSIDE_RATIO_THRESHOLD` slightly, or nudge that row's traced
  corners in `--pick-zones` to better match the real line position.

---

## Known limitations

- Assumes a **fixed, non-moving camera** — a panning/zooming drone shot will
  break both the stall geometry and the stationarity check.
- YOLOv8n is trained on COCO's mostly ground-level imagery; recall on
  small, top-down vehicles is inherently lower than on street-level video.
  For production use, consider fine-tuning on an aerial dataset (e.g.
  VisDrone or DOTA) for meaningfully better accuracy.
- Stall geometry is a straight bilinear grid — rows with strong curvature
  or highly irregular spacing may need to be split into multiple
  shorter `PARKING_ROWS` entries for a better fit.

---

## License

MIT — use, modify, and adapt freely.
