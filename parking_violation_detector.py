"""
====================================================================
 AUTOMATIC PARKING VIOLATION DETECTOR (Top-Down Aerial View)
====================================================================
What this script does
----------------------
1. Reads an aerial (drone/top-down) parking-lot video.
2. Runs YOLOv8 to detect vehicles (cars, trucks, motorcycles).
3. Builds a grid of INDIVIDUAL parking-stall boxes (not just one big
   rectangle per row) so every real white-line-bounded space is
   represented, the same way an actual parking lot is painted.
4. Flags a vehicle as a "LINE VIOLATION" if it sits mostly outside its
   stall (i.e. into the driving lane) AND stays there, unmoving, for
   more than 5 seconds.
5. Draws stall outlines, green boxes for legal parking, flashing red
   boxes + timer for violators, and logs every confirmed violation to
   the console with a timestamp and stall ID.

Requirements
------------
    pip install ultralytics opencv-python numpy

Usage
-----
    python parking_violation_detector.py
        -> processes "parking_input.mp4" -> writes "parking_result.mp4"

    python parking_violation_detector.py --pick-zones
        -> interactive tool: click the 4 corners of one row (top-left,
           top-right, bottom-right, bottom-left, in that order),
           then type how many stalls that row contains. Repeat for
           each row, then press 's' to print ready-to-paste code.
====================================================================
"""

import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ====================================================================
# 1. CONFIGURATION
# ====================================================================

INPUT_VIDEO = "parking_input.mp4"
OUTPUT_VIDEO = "parking_result.mp4"
MODEL_PATH = "yolov8n.pt"

# COCO ids: 2=car, 3=motorcycle, 5=bus, 7=truck. Kept as originally
# specified; add 3 if motorcycles/scooters aren't being picked up.
VEHICLE_CLASSES = [2, 5, 7]

# --------------------------------------------------------------
# DETECTION TUNING -- this is the #1 thing that was broken before.
# Aerial vehicles are small relative to the whole frame, so:
#   - imgsz is raised well above the YOLO default (640) so cars don't
#     shrink into a handful of pixels before the model even sees them.
#   - conf is lowered because aerial/top-down cars look nothing like
#     the mostly side/front-view cars YOLOv8 was trained on, so
#     confidence scores run lower even on correct detections.
# If you STILL get few/no boxes, try imgsz=1920, conf=0.10, or a
# bigger checkpoint (yolov8s.pt / yolov8m.pt) -- accuracy on aerial
# imagery is inherently harder for a COCO-trained model.
# --------------------------------------------------------------
CONF_THRESHOLD = 0.20
IMG_SIZE = 1280

VIOLATION_SECONDS = 5.0
INSIDE_RATIO_THRESHOLD = 0.65
STATIONARY_PIXEL_THRESHOLD = 15
STATIONARY_HISTORY_FRAMES = 10

# Print a running "how many vehicles did YOLO see this frame" line
# every N frames, purely so you can confirm detection is working.
DEBUG_DETECTION_LOG_EVERY = 30


# ====================================================================
# 2. PARKING ROWS -> AUTO-GENERATED INDIVIDUAL STALL BOXES
# ====================================================================
# Instead of one big rectangle per row, each row below is defined by
# its 4 CORNERS (top-left, top-right, bottom-right, bottom-left) plus
# how many individual stalls sit inside it. The script slices that
# quadrilateral into that many equal boxes automatically -- one per
# real parking space, the same way the white lines actually divide it.
#
# COORDINATES ARE MEASURED AGAINST A "REFERENCE" FRAME SIZE
# -----------------------------------------------------------------
# You do NOT need to hand-convert pixels to fractions. Just:
#   1. Grab one frame of your video as an image (e.g. a screenshot,
#      or cv2.imwrite("frame.jpg", frame)).
#   2. Note that image's width x height -> set REFERENCE_FRAME_SIZE.
#   3. Open that image in any editor / MS Paint / even the Windows
#      photo viewer (it shows cursor pixel coords), and read off the
#      4 corners of each row of parking spaces.
#   4. Paste those pixel numbers straight into PARKING_ROWS below.
# The script automatically rescales them to your real video's actual
# resolution at runtime, so they will never be skewed or clipped again
# even if your real video isn't the same size as your reference image.
#
# EASIER OPTION: run `python parking_violation_detector.py --pick-zones`
# and just click the corners on your real video -- it prints this
# block for you, already correct, no manual measuring needed.
# --------------------------------------------------------------------

REFERENCE_FRAME_SIZE = (1522, 709)  # (width, height) of the frame you measured corners on

PARKING_ROWS = [
    # Each row: corners = [top_left, top_right, bottom_right, bottom_left], stalls = count
    {"label": "Row-1", "corners": [(45, 35), (185, 35), (185, 690), (45, 690)], "stalls": 11},
    {"label": "Row-2", "corners": [(225, 35), (365, 35), (365, 690), (225, 690)], "stalls": 11},
    {"label": "Row-3", "corners": [(405, 35), (545, 35), (545, 690), (405, 690)], "stalls": 11},
    {"label": "Row-4", "corners": [(585, 35), (725, 35), (725, 690), (585, 690)], "stalls": 11},
    {"label": "Row-5", "corners": [(765, 35), (905, 35), (905, 690), (765, 690)], "stalls": 11},
    {"label": "Row-6", "corners": [(945, 35), (1090, 35), (1090, 690), (945, 690)], "stalls": 11},
    {"label": "Row-7", "corners": [(1130, 35), (1275, 35), (1275, 690), (1130, 690)], "stalls": 11},
    {"label": "Row-8", "corners": [(1315, 35), (1465, 35), (1465, 690), (1315, 690)], "stalls": 11},
]


# ====================================================================
# 3. GEOMETRY HELPERS
# ====================================================================

def lerp_point(p0, p1, t):
    return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def generate_stall_polygons(corners, n_stalls):
    """Slices a (possibly slightly skewed) quadrilateral row into
    n_stalls equal boxes stacked from top to bottom, using bilinear
    interpolation along the left and right edges. Works even if the
    row isn't perfectly rectangular (handles mild perspective)."""
    tl, tr, br, bl = corners
    stalls = []
    for i in range(n_stalls):
        t0, t1 = i / n_stalls, (i + 1) / n_stalls
        left_top = lerp_point(tl, bl, t0)
        right_top = lerp_point(tr, br, t0)
        left_bottom = lerp_point(tl, bl, t1)
        right_bottom = lerp_point(tr, br, t1)
        stalls.append([left_top, right_top, right_bottom, left_bottom])
    return stalls


def scale_polygon(poly, scale_x, scale_y):
    return [(x * scale_x, y * scale_y) for (x, y) in poly]


def build_stall_list(parking_rows, actual_width, actual_height):
    """Converts PARKING_ROWS (in reference-frame pixel coords) into a
    flat list of (label, polygon) for the ACTUAL video resolution."""
    ref_w, ref_h = REFERENCE_FRAME_SIZE
    scale_x, scale_y = actual_width / ref_w, actual_height / ref_h

    stalls = []
    for row in parking_rows:
        row_polys = generate_stall_polygons(row["corners"], row["stalls"])
        for i, poly in enumerate(row_polys):
            scaled = scale_polygon(poly, scale_x, scale_y)
            stalls.append((f'{row["label"]}-Stall-{i + 1}', scaled))
    return stalls


def build_zone_mask(frame_shape, stall_list):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for _, poly in stall_list:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def bbox_inside_ratio(mask, x1, y1, x2, y2):
    h, w = mask.shape[:2]
    x1c, y1c = max(int(x1), 0), max(int(y1), 0)
    x2c, y2c = min(int(x2), w - 1), min(int(y2), h - 1)
    if x2c <= x1c or y2c <= y1c:
        return 0.0
    roi = mask[y1c:y2c, x1c:x2c]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)


def nearest_stall_label(cx, cy, stall_list):
    best_label, best_dist = "Unknown", float("inf")
    for label, poly in stall_list:
        pts = np.array(poly, dtype=np.float32)
        zx, zy = pts[:, 0].mean(), pts[:, 1].mean()
        dist = (zx - cx) ** 2 + (zy - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def draw_stalls(frame, stall_list):
    """Thin blue lines for each individual stall, slightly bolder
    yellow around each full row, so you can see both the row and the
    per-space divisions -- exactly like painted lot markings."""
    for _, poly in stall_list:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 200, 0), thickness=1)


def format_timer(seconds):
    return f"{seconds:0.1f}s"


# ====================================================================
# 4. INTERACTIVE ZONE-PICKER (--pick-zones mode)
# ====================================================================

def run_zone_picker(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Could not read a frame from '{video_path}'.")
        return

    h, w = frame.shape[:2]
    rows = []
    current_corners = []
    display = frame.copy()

    def redraw():
        nonlocal display
        display = frame.copy()
        for row in rows:
            pts = np.array(row["corners"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)
        for pt in current_corners:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
        if len(current_corners) > 1:
            pts = np.array(current_corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts], False, (0, 255, 0), 2)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(current_corners) < 4:
            current_corners.append((x, y))
            redraw()

    win = "Click 4 corners per row: TL,TR,BR,BL (n=finish row, s=save, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    print(f"\nVideo frame size: {w}x{h}")
    print("--- ZONE PICKER ---")
    print("Click 4 corners of ONE parking row in this order:")
    print("  1) top-left   2) top-right   3) bottom-right   4) bottom-left")
    print("Press 'n' once you've placed all 4 -- you'll be asked how many")
    print("stalls that row has (type it in the terminal).")
    print("Repeat for each row. Press 's' to print the final code. 'q' to quit.\n")

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n"):
            if len(current_corners) == 4:
                n = input(f"How many stalls does row #{len(rows) + 1} have? ")
                try:
                    n = int(n)
                except ValueError:
                    n = 10
                rows.append({"label": f"Row-{len(rows) + 1}", "corners": current_corners[:], "stalls": n})
                current_corners = []
                redraw()
            else:
                print("Need exactly 4 points before starting a new row.")
        elif key == ord("s"):
            print(f"\nREFERENCE_FRAME_SIZE = ({w}, {h})")
            print("PARKING_ROWS = [")
            for row in rows:
                print(f'    {{"label": "{row["label"]}", "corners": {row["corners"]}, "stalls": {row["stalls"]}}},')
            print("]\n")
            print("Paste the two blocks above into the script, replacing the")
            print("existing REFERENCE_FRAME_SIZE and PARKING_ROWS.")
            break
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


# ====================================================================
# 5. MAIN PROCESSING LOOP
# ====================================================================

def run_detector(video_path, output_path, model_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video '{video_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    violation_frames_needed = max(1, int(round(fps * VIOLATION_SECONDS)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stall_list = build_stall_list(PARKING_ROWS, width, height)
    zone_mask = build_zone_mask((height, width), stall_list)

    track_state = defaultdict(lambda: {
        "centroid_history": deque(maxlen=STATIONARY_HISTORY_FRAMES),
        "violation_streak": 0,
        "confirmed": False,
    })

    frame_idx = 0
    print(f"Video: {width}x{height} @ {fps:.1f} FPS | reference size assumed: {REFERENCE_FRAME_SIZE}")
    print(f"Total stalls defined: {len(stall_list)}")
    print(f"Violation requires crossing + stationary for {VIOLATION_SECONDS}s "
          f"(~{violation_frames_needed} frames).\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            classes=VEHICLE_CLASSES,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        draw_stalls(frame, stall_list)

        result = results[0]
        n_detections = 0

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            n_detections = len(boxes)

            # ids may be None for the first frame or two while the
            # tracker warms up -- handle that gracefully instead of
            # silently dropping every detection.
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.arange(len(boxes)) * -1 - 1  # temporary negative placeholder ids

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                state = track_state[track_id]
                state["centroid_history"].append((cx, cy))

                inside_ratio = bbox_inside_ratio(zone_mask, x1, y1, x2, y2)
                is_crossing_line = inside_ratio < INSIDE_RATIO_THRESHOLD

                is_stationary = False
                hist = state["centroid_history"]
                if len(hist) == hist.maxlen:
                    xs = [p[0] for p in hist]
                    ys = [p[1] for p in hist]
                    movement = max(max(xs) - min(xs), max(ys) - min(ys))
                    is_stationary = movement < STATIONARY_PIXEL_THRESHOLD

                if is_crossing_line and is_stationary:
                    state["violation_streak"] += 1
                else:
                    state["violation_streak"] = 0
                    state["confirmed"] = False

                seconds_in_violation = state["violation_streak"] / fps
                is_confirmed_violation = state["violation_streak"] >= violation_frames_needed

                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

                if is_confirmed_violation:
                    flash_on = (frame_idx // 5) % 2 == 0
                    color = (0, 0, 255) if flash_on else (0, 0, 140)
                    thickness = 4 if flash_on else 3
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, thickness)
                    label = f"LINE VIOLATION {format_timer(seconds_in_violation)}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1i, y1i - th - 10), (x1i + tw + 6, y1i), color, -1)
                    cv2.putText(frame, label, (x1i + 3, y1i - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if not state["confirmed"]:
                        state["confirmed"] = True
                        timestamp_sec = frame_idx / fps
                        stall_label = nearest_stall_label(cx, cy, stall_list)
                        print(
                            f"[VIOLATION] track_id={track_id} | frame={frame_idx} | "
                            f"time={timestamp_sec:0.2f}s | stall={stall_label} | "
                            f"bbox=({x1i},{y1i},{x2i},{y2i})"
                        )

                elif is_crossing_line:
                    color = (0, 165, 255)
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
                    label = f"Watching {format_timer(seconds_in_violation)}/{VIOLATION_SECONDS:.0f}s"
                    cv2.putText(frame, label, (x1i, y1i - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                else:
                    color = (0, 200, 0)
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 2)
                    cv2.putText(frame, f"OK #{track_id}", (x1i, y1i - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if frame_idx % DEBUG_DETECTION_LOG_EVERY == 0:
            print(f"[debug] frame {frame_idx}: {n_detections} vehicle(s) detected this frame")

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\nDone. Output saved to '{output_path}'.")


# ====================================================================
# 6. ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic Parking Violation Detector")
    parser.add_argument("--pick-zones", action="store_true",
                         help="Interactive tool to trace your own PARKING_ROWS by clicking.")
    parser.add_argument("--input", default=INPUT_VIDEO)
    parser.add_argument("--output", default=OUTPUT_VIDEO)
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    if args.pick_zones:
        run_zone_picker(args.input)
    else:
        run_detector(args.input, args.output, args.model)
