"""
AI Traffic Management System
=============================
Single-file Flask application. Upload a traffic video from the browser and
watch it processed live:

  1. Vehicle detection            -> YOLOv8 (ultralytics), auto-falls back to
                                      an OpenCV background-subtraction blob
                                      detector if ultralytics/weights aren't
                                      available (so the demo always runs).
  2. Traffic density calculation  -> vehicle counts per approach, smoothed
                                      over a rolling window, bucketed into
                                      LOW / MEDIUM / HIGH.
  3. Emergency vehicle detection  -> HSV red/blue light-bar thresholding +
                                      flicker/strobe-rate analysis per lane.
  4. Signal decision logic        -> adaptive round-robin controller: green
                                      duration scales with density, and any
                                      lane with a confirmed emergency vehicle
                                      immediately pre-empts the cycle.
  5. Browser UI                   -> single page, drag-and-drop upload, live
                                      MJPEG annotated stream, live dashboard.

Run:
    pip install -r requirements.txt
    python app.py
    open http://127.0.0.1:5000

No external files, templates, or model weights are required to start the
app. If `ultralytics` is installed, YOLOv8n weights are auto-downloaded by
that library on first use (needs internet once). Without it, the app uses a
pure-OpenCV motion-based vehicle detector automatically.
"""

import os
import cv2
import time
import json
import uuid
import threading
import sqlite3
import numpy as np
from collections import deque, defaultdict
from flask import Flask, request, Response, render_template_string, jsonify

# =============================================================================
# CONFIG
# =============================================================================

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "traffic_management.db")


# =============================================================================
# SQLITE DATABASE
# =============================================================================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Create the SQLite database and tables if they do not already exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS vehicle_counts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL, lane TEXT NOT NULL,
        car INTEGER DEFAULT 0, motorcycle INTEGER DEFAULT 0,
        bus INTEGER DEFAULT 0, truck INTEGER DEFAULT 0, total INTEGER DEFAULT 0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS traffic_density (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL, lane TEXT NOT NULL,
        vehicle_count INTEGER NOT NULL, density_level TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS emergency_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
        lane TEXT NOT NULL, detection_type TEXT NOT NULL, action_taken TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS signal_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
        current_lane TEXT, phase TEXT NOT NULL, signal_state TEXT NOT NULL, reason TEXT
    )""")
    conn.commit()
    conn.close()


def save_traffic_snapshot(lane_dets, densities):
    """Save aggregated traffic data once per second, not once per frame."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    cur = conn.cursor()
    for lane in LANES:
        type_counts = defaultdict(int)
        for det in lane_dets.get(lane, []):
            type_counts[det.get("label", "unknown")] += 1
        total = len(lane_dets.get(lane, []))
        cur.execute(
            "INSERT INTO vehicle_counts (timestamp,lane,car,motorcycle,bus,truck,total) VALUES (?,?,?,?,?,?,?)",
            (timestamp, lane, type_counts["car"], type_counts["motorcycle"], type_counts["bus"], type_counts["truck"], total)
        )
        cur.execute(
            "INSERT INTO traffic_density (timestamp,lane,vehicle_count,density_level) VALUES (?,?,?,?)",
            (timestamp, lane, total, densities.get(lane, "LOW"))
        )
    conn.commit()
    conn.close()


def save_emergency_event(lane):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO emergency_events (timestamp,lane,detection_type,action_taken) VALUES (?,?,?,?)",
        (time.strftime('%Y-%m-%d %H:%M:%S'), lane, "Emergency vehicle signal detected", "Emergency pre-emption")
    )
    conn.commit()
    conn.close()


def save_signal_log(snapshot, reason=""):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO signal_logs (timestamp,current_lane,phase,signal_state,reason) VALUES (?,?,?,?,?)",
        (time.strftime('%Y-%m-%d %H:%M:%S'), snapshot.get("current_lane"),
         snapshot.get("phase", ""), json.dumps(snapshot.get("state", {})), reason)
    )
    conn.commit()
    conn.close()


init_database()

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# COCO class ids ultralytics YOLOv8 uses for road vehicles
YOLO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

LANES = ["North", "East", "South", "West"]

# Rolling-average window (frames) used to smooth density counts
DENSITY_WINDOW = 12

# Vehicle-count thresholds -> density level
DENSITY_THRESHOLDS = {"LOW": 3, "MEDIUM": 7}  # < LOW -> LOW, < MEDIUM -> MEDIUM, else HIGH

# Green-light duration (seconds) per density level
GREEN_TIME = {"LOW": 90, "MEDIUM": 90, "HIGH": 120}
YELLOW_TIME = 3
ALL_RED_TIME = 1
EMERGENCY_HOLD_TIME = 18  # minimum seconds an emergency pre-emption holds green

# Emergency (siren light-bar) color detection thresholds
EMERGENCY_COLOR_RATIO = 1.10     # fraction of red+blue pixels in a vehicle crop to count as a "hit"
EMERGENCY_HISTORY_LEN = 12       # frames of history kept per lane
EMERGENCY_MIN_TRANSITIONS = 4    # on/off flicker transitions within history -> confirmed emergency

FRAME_TARGET_FPS = 16  # pacing for the MJPEG stream / processing loop


# =============================================================================
# VEHICLE DETECTOR  (YOLOv8 primary, OpenCV motion-detection fallback)
# =============================================================================

class VehicleDetector:
    """Detects vehicles in a frame and returns a list of
    {box: (x1,y1,x2,y2), label: str, conf: float, centroid: (cx,cy)}.
    Uses YOLOv8 if ultralytics is importable; otherwise falls back to a
    background-subtraction + contour blob detector so the app always runs.
    """

    def __init__(self):
        self.backend = "motion"
        self.model = None
        try:
            from ultralytics import YOLO  # noqa: local import, optional dependency
            self.model = YOLO("yolov8n.pt")
            self.backend = "yolov8"
        except Exception:
            self.backend = "motion"
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=40, detectShadows=False
            )

    def detect(self, frame):
        if self.backend == "yolov8":
            return self._detect_yolo(frame)
        return self._detect_motion(frame)

    def _detect_yolo(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in YOLO_VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf < 0.25:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append({
                "box": (x1, y1, x2, y2),
                "label": YOLO_VEHICLE_CLASSES[cls_id],
                "conf": conf,
                "centroid": (cx, cy),
            })
        return detections

    def _detect_motion(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8), iterations=1)
        fg_mask = cv2.dilate(fg_mask, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        frame_area = frame.shape[0] * frame.shape[1]
        for c in contours:
            area = cv2.contourArea(c)
            # ignore tiny noise blobs and blobs bigger than a third of the frame
            if area < 450 or area > frame_area * 0.33:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            detections.append({
                "box": (x, y, x + w, y + h),
                "label": "vehicle",
                "conf": 1.0,
                "centroid": (cx, cy),
            })
        return detections


# =============================================================================
# EMERGENCY VEHICLE DETECTOR  (HSV light-bar color + strobe flicker rate)
# =============================================================================

class EmergencyDetector:
    """Flags a lane as having an emergency vehicle when vehicles in that lane
    show a high red/blue pixel ratio (siren light bar) that flickers on/off
    across consecutive frames (a strobe), rather than a single static red
    or blue vehicle body which would stay constantly "on"."""

    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=EMERGENCY_HISTORY_LEN))

    @staticmethod
    def _red_blue_ratio(crop):
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 120, 120), (180, 255, 255))
        blue = cv2.inRange(hsv, (100, 120, 120), (130, 255, 255))
        mask = red1 | red2 | blue
        return float(np.count_nonzero(mask)) / mask.size

    def update_lane(self, frame, lane, lane_detections):
        """Call once per processed frame per lane. Returns True if this lane
        is confirmed to currently contain an emergency vehicle."""
        hit = False
        for det in lane_detections:
            x1, y1, x2, y2 = det["box"]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
            crop = frame[y1:y2, x1:x2]
            if self._red_blue_ratio(crop) >= EMERGENCY_COLOR_RATIO:
                hit = True
                break

        hist = self.history[lane]
        hist.append(hit)

        if len(hist) < 4:
            return False
        transitions = sum(1 for i in range(1, len(hist)) if hist[i] != hist[i - 1])
        # require at least one recent "on" reading plus enough flicker to
        # distinguish a strobing light bar from a plain red/blue vehicle
        return hit and transitions >= EMERGENCY_MIN_TRANSITIONS


# =============================================================================
# TRAFFIC DENSITY ANALYZER
# =============================================================================

class DensityAnalyzer:
    def __init__(self):
        self.rolling = defaultdict(lambda: deque(maxlen=DENSITY_WINDOW))

    def update(self, lane, count):
        self.rolling[lane].append(count)

    def avg_count(self, lane):
        q = self.rolling[lane]
        return sum(q) / len(q) if q else 0.0

    def level(self, lane):
        avg = self.avg_count(lane)
        if avg < DENSITY_THRESHOLDS["LOW"]:
            return "LOW"
        if avg < DENSITY_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "HIGH"


# =============================================================================
# SIGNAL DECISION LOGIC (adaptive round-robin + emergency pre-emption)
# =============================================================================

class SignalController:
    def __init__(self):
        self.state = {lane: "RED" for lane in LANES}
        self.cycle_idx = 0
        self.phase = "GREEN"
        self.state[LANES[0]] = "GREEN"
        self.phase_end = time.time() + GREEN_TIME["LOW"]
        self.emergency_active_lane = None
        self.emergency_end = 0
        self.log = deque(maxlen=40)
        self.lock = threading.Lock()

    def _log(self, msg):
        self.log.appendleft(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _set_all_red(self):
        self.state = {lane: "RED" for lane in LANES}

    def _start_green(self, lane, duration, reason=""):
        self._set_all_red()
        self.state[lane] = "GREEN"
        self.phase = "GREEN"
        self.phase_end = time.time() + duration
        self.cycle_idx = LANES.index(lane)
        if reason:
            self._log(f"{lane} -> GREEN ({duration}s) {reason}")
        else:
            self._log(f"{lane} -> GREEN ({duration}s, density-based)")

    def _start_yellow(self, lane):
        self.state = {l: ("YELLOW" if l == lane else "RED") for l in LANES}
        self.phase = "YELLOW"
        self.phase_end = time.time() + YELLOW_TIME
        self._log(f"{lane} -> YELLOW")

    def tick(self, densities, emergency_lane):
        """Advance the signal state machine. Called ~1x/second."""
        with self.lock:
            now = time.time()

            # --- Emergency pre-emption takes priority over everything ---
            if emergency_lane and self.emergency_active_lane != emergency_lane:
                self.emergency_active_lane = emergency_lane
                self.emergency_end = now + EMERGENCY_HOLD_TIME
                self._start_green(emergency_lane, EMERGENCY_HOLD_TIME,
                                   reason="EMERGENCY VEHICLE PRE-EMPTION")
                return self.snapshot()

            if self.emergency_active_lane:
                still_present = (emergency_lane == self.emergency_active_lane)
                if now < self.emergency_end or still_present:
                    return self.snapshot()  # hold emergency green
                else:
                    self._log(f"{self.emergency_active_lane} emergency cleared, resuming normal cycle")
                    self.emergency_active_lane = None
                    next_idx = (self.cycle_idx + 1) % len(LANES)
                    self._start_yellow(LANES[self.cycle_idx])
                    self.cycle_idx = next_idx
                    return self.snapshot()

            # --- Normal adaptive round robin ---
            if now >= self.phase_end:
                if self.phase == "GREEN":
                    cur_lane = LANES[self.cycle_idx]
                    self._start_yellow(cur_lane)
                elif self.phase == "YELLOW":
                    self._set_all_red()
                    self.phase = "ALL_RED"
                    self.phase_end = now + ALL_RED_TIME
                    self._log("ALL RED (clearance interval)")
                else:  # ALL_RED -> advance to next lane's green
                    self.cycle_idx = (self.cycle_idx + 1) % len(LANES)
                    next_lane = LANES[self.cycle_idx]
                    level = densities.get(next_lane, "LOW")
                    self._start_green(next_lane, GREEN_TIME[level])

            return self.snapshot()

    def snapshot(self):
        return {
            "state": dict(self.state),
            "phase": self.phase,
            "current_lane": LANES[self.cycle_idx],
            "seconds_left": max(0, int(self.phase_end - time.time())),
            "emergency_lane": self.emergency_active_lane,
            "log": list(self.log)[:12],
        }


# =============================================================================
# SHARED APP STATE + BACKGROUND PROCESSING THREAD
# =============================================================================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_jpeg = None
        self.detector = None
        self.emergency_detector = None
        self.density_analyzer = None
        self.controller = None
        self.densities = {lane: "LOW" for lane in LANES}
        self.counts = {lane: 0 for lane in LANES}
        self.emergency_lane = None
        self.backend = "not started"
        self.running = False
        self.video_path = None
        self.thread = None
        self.signal_thread = None
        self.frame_wh = (0, 0)


SHARED = SharedState()


def lane_zones(w, h):
    """Split the frame into 4 quadrants, one per approach lane. This is a
    simplification for a single-camera demo; a real deployment would use one
    camera per approach."""
    return {
        "North": (0, 0, w // 2, h // 2),
        "East": (w // 2, 0, w, h // 2),
        "South": (0, h // 2, w // 2, h),
        "West": (w // 2, h // 2, w, h),
    }


def assign_lane(centroid, zones):
    cx, cy = centroid
    for lane, (x1, y1, x2, y2) in zones.items():
        if x1 <= cx < x2 and y1 <= cy < y2:
            return lane
    return None


LANE_COLORS = {
    "North": (255, 140, 0), "East": (0, 200, 255),
    "South": (60, 200, 60), "West": (200, 60, 200),
}
SIGNAL_COLORS = {"RED": (0, 0, 255), "YELLOW": (0, 220, 255), "GREEN": (0, 200, 0)}


def annotate_frame(frame, detections, zones, lane_hits, controller_snapshot):
    h, w = frame.shape[:2]
    out = frame.copy()

    # lane quadrant grid
    cv2.line(out, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)
    cv2.line(out, (0, h // 2), (w, h // 2), (90, 90, 90), 1)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        lane = assign_lane(det["centroid"], zones)
        color = LANE_COLORS.get(lane, (200, 200, 200))
        is_emerg = lane_hits.get(lane, False)
        box_color = (0, 0, 255) if is_emerg else color
        thickness = 3 if is_emerg else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)
        label = det["label"] + (" !SIREN" if is_emerg else "")
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    # lane info + signal-light chips
    state = controller_snapshot["state"]
    y0 = 20
    for i, lane in enumerate(LANES):
        txt = f"{lane}: {SHARED.counts.get(lane, 0)} veh | {SHARED.densities.get(lane, 'LOW')} | {state[lane]}"
        cv2.putText(out, txt, (8, y0 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    SIGNAL_COLORS[state[lane]], 2, cv2.LINE_AA)

    if controller_snapshot["emergency_lane"]:
        cv2.putText(out, f"EMERGENCY PRE-EMPTION: {controller_snapshot['emergency_lane']}",
                    (8, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return out


def signal_loop():
    """Runs the signal controller state machine at ~1Hz for as long as the
    session is active, independent of video frame rate."""
    while True:
        with SHARED.lock:
            running = SHARED.running
            controller = SHARED.controller
            densities = dict(SHARED.densities)
            emergency_lane = SHARED.emergency_lane
        if not running or controller is None:
            time.sleep(0.5)
            continue
        previous = controller.snapshot()
        snapshot = controller.tick(densities, emergency_lane)
        if (snapshot.get("state") != previous.get("state") or
                snapshot.get("phase") != previous.get("phase") or
                snapshot.get("emergency_lane") != previous.get("emergency_lane")):
            reason = "EMERGENCY" if snapshot.get("emergency_lane") else "NORMAL"
            save_signal_log(snapshot, reason)
        time.sleep(1.0)


def video_processing_loop(path):
    detector = VehicleDetector()
    edetector = EmergencyDetector()
    danalyzer = DensityAnalyzer()
    controller = SignalController()

    with SHARED.lock:
        SHARED.detector = detector
        SHARED.emergency_detector = edetector
        SHARED.density_analyzer = danalyzer
        SHARED.controller = controller
        SHARED.backend = detector.backend
        SHARED.running = True

    cap = cv2.VideoCapture(path)
    frame_interval = 1.0 / FRAME_TARGET_FPS
    last_db_save = 0.0
    last_emergency_lane = None

    while True:
        with SHARED.lock:
            if not SHARED.running:
                break
        loop_start = time.time()
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video for a continuous demo
            continue

        h, w = frame.shape[:2]
        zones = lane_zones(w, h)
        detections = detector.detect(frame)

        lane_dets = defaultdict(list)
        for det in detections:
            lane = assign_lane(det["centroid"], zones)
            if lane:
                lane_dets[lane].append(det)

        lane_hits = {}
        emergency_lane = None
        for lane in LANES:
            count = len(lane_dets[lane])
            danalyzer.update(lane, count)
            is_emerg = edetector.update_lane(frame, lane, lane_dets[lane])
            lane_hits[lane] = is_emerg
            if is_emerg and emergency_lane is None:
                emergency_lane = lane

        with SHARED.lock:
            SHARED.counts = {lane: len(lane_dets[lane]) for lane in LANES}
            SHARED.densities = {lane: danalyzer.level(lane) for lane in LANES}
            SHARED.emergency_lane = emergency_lane
            SHARED.frame_wh = (w, h)
            snapshot = controller.snapshot()

        # Persist aggregated traffic data once per second instead of every frame.
        now_db = time.time()
        if now_db - last_db_save >= 1.0:
            with SHARED.lock:
                current_densities = dict(SHARED.densities)
            save_traffic_snapshot(lane_dets, current_densities)
            last_db_save = now_db

        # Save only a new emergency event, avoiding duplicate rows every frame.
        if emergency_lane and emergency_lane != last_emergency_lane:
            save_emergency_event(emergency_lane)
        last_emergency_lane = emergency_lane

        annotated = annotate_frame(frame, detections, zones, lane_hits, snapshot)
        ok2, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok2:
            with SHARED.lock:
                SHARED.frame_jpeg = buf.tobytes()

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, frame_interval - elapsed))

    cap.release()


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300MB upload cap


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "No file part"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"ok": False, "error": "No file selected"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": f"Unsupported file type {ext}"}), 400

    # stop any previous run
    with SHARED.lock:
        SHARED.running = False
    if SHARED.thread and SHARED.thread.is_alive():
        SHARED.thread.join(timeout=2)

    fname = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    f.save(path)

    with SHARED.lock:
        SHARED.video_path = path

    t = threading.Thread(target=video_processing_loop, args=(path,), daemon=True)
    SHARED.thread = t
    t.start()

    return jsonify({"ok": True})


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with SHARED.lock:
                frame = SHARED.frame_jpeg
                running = SHARED.running
            if frame is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            if not running and frame is None:
                time.sleep(0.3)
            time.sleep(1.0 / FRAME_TARGET_FPS)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with SHARED.lock:
        controller = SHARED.controller
        snap = controller.snapshot() if controller else {
            "state": {l: "RED" for l in LANES}, "phase": "IDLE",
            "current_lane": None, "seconds_left": 0,
            "emergency_lane": None, "log": [],
        }
        payload = {
            "running": SHARED.running,
            "backend": SHARED.backend,
            "counts": dict(SHARED.counts),
            "densities": dict(SHARED.densities),
            "signal": snap,
        }
    return jsonify(payload)


@app.route("/reset", methods=["POST"])
def reset():
    with SHARED.lock:
        SHARED.running = False
        SHARED.frame_jpeg = None
    return jsonify({"ok": True})


# =============================================================================
# EMBEDDED FRONTEND (single page, no external template files)
# =============================================================================

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Traffic Management System</title>
<style>
  :root{
    --bg:#0b1220; --panel:#131c2f; --panel2:#0f1729; --text:#e7ecf7; --muted:#8ea0c3;
    --accent:#4da3ff; --red:#ff4d5e; --yellow:#ffcc33; --green:#33d17a; --border:#223252;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--text);font-family:'Segoe UI',Roboto,Arial,sans-serif;}
  header{padding:18px 28px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
  header h1{font-size:19px;margin:0;font-weight:600;letter-spacing:.3px;}
  header span{color:var(--muted);font-size:12px;}
  .wrap{display:grid;grid-template-columns:1.5fr 1fr;gap:18px;padding:20px 28px;max-width:1400px;margin:0 auto;}
  @media(max-width:980px){.wrap{grid-template-columns:1fr;}}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px;}
  .card h2{margin:0 0 12px 0;font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
  .upload-zone{border:2px dashed var(--border);border-radius:10px;padding:22px;text-align:center;cursor:pointer;transition:.15s;}
  .upload-zone:hover{border-color:var(--accent);background:#0f192e;}
  .upload-zone input{display:none;}
  #videoBox{position:relative;background:#000;border-radius:10px;overflow:hidden;min-height:280px;display:flex;align-items:center;justify-content:center;}
  #videoBox img{width:100%;display:block;}
  #placeholder{color:var(--muted);font-size:13px;padding:40px;text-align:center;}
  .pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:600;background:#1c2942;color:var(--accent);margin-left:8px;}
  table.lanes{width:100%;border-collapse:collapse;font-size:13px;}
  table.lanes th, table.lanes td{padding:8px 6px;border-bottom:1px solid var(--border);text-align:left;}
  table.lanes th{color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase;}
  .badge{display:inline-flex;align-items:center;gap:6px;font-weight:600;padding:3px 9px;border-radius:6px;font-size:12px;}
  .badge.RED{background:rgba(255,77,94,.15);color:var(--red);}
  .badge.YELLOW{background:rgba(255,204,51,.15);color:var(--yellow);}
  .badge.GREEN{background:rgba(51,209,122,.15);color:var(--green);}
  .dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
  .dot.RED{background:var(--red);} .dot.YELLOW{background:var(--yellow);} .dot.GREEN{background:var(--green);}
  .dens{font-size:11px;padding:2px 8px;border-radius:6px;font-weight:600;}
  .dens.LOW{background:rgba(77,163,255,.15);color:var(--accent);}
  .dens.MEDIUM{background:rgba(255,204,51,.15);color:var(--yellow);}
  .dens.HIGH{background:rgba(255,77,94,.15);color:var(--red);}
  #emergencyBanner{display:none;background:linear-gradient(90deg,#3a0d14,#5c0f18);border:1px solid var(--red);color:#ffb3ba;padding:10px 14px;border-radius:10px;margin-bottom:14px;font-weight:600;font-size:13px;animation:pulse 1s infinite;}
  @keyframes pulse{0%{opacity:1;}50%{opacity:.55;}100%{opacity:1;}}
  #log{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:var(--muted);max-height:170px;overflow-y:auto;line-height:1.6;}
  .statrow{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-top:10px;}
  .btnrow{margin-top:10px;display:flex;gap:8px;}
  button{background:var(--panel2);color:var(--text);border:1px solid var(--border);padding:7px 14px;border-radius:8px;font-size:12px;cursor:pointer;}
  button:hover{border-color:var(--accent);}
</style>
</head>
<body>
<header>
  <h1>🚦 AI Traffic Management System</h1>
  <span>YOLO vehicle detection &middot; density-adaptive signals &middot; emergency pre-emption <span class="pill" id="backendPill">backend: -</span></span>
</header>

<div class="wrap">
  <div>
    <div class="card">
      <h2>Live Feed</h2>
      <div id="videoBox">
        <div id="placeholder">Upload a traffic video below to start the live annotated feed.</div>
        <img id="videoImg" style="display:none;">
      </div>
      <div class="upload-zone" id="dropZone">
        <input type="file" id="fileInput" accept="video/*">
        <div id="dzLabel">📤 Click or drag a traffic video here (.mp4, .avi, .mov, .webm)</div>
      </div>
      <div class="btnrow">
        <button id="resetBtn">Reset session</button>
      </div>
    </div>
  </div>

  <div>
    <div id="emergencyBanner">🚨 EMERGENCY VEHICLE DETECTED — signal pre-empted</div>

    <div class="card">
      <h2>Lane Status</h2>
      <table class="lanes">
        <thead><tr><th>Lane</th><th>Vehicles</th><th>Density</th><th>Signal</th></tr></thead>
        <tbody id="laneRows"></tbody>
      </table>
      <div class="statrow"><span id="phaseText">Phase: -</span><span id="timerText">-- s</span></div>
    </div>

    <div class="card" style="margin-top:16px;">
      <h2>Decision Log</h2>
      <div id="log">Waiting for video…</div>
    </div>
  </div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const videoImg = document.getElementById('videoImg');
const placeholder = document.getElementById('placeholder');
const dzLabel = document.getElementById('dzLabel');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#4da3ff'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = ''; });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.style.borderColor = '';
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) uploadFile(fileInput.files[0]); });

function uploadFile(file) {
  dzLabel.textContent = 'Uploading ' + file.name + ' …';
  const fd = new FormData();
  fd.append('video', file);
  fetch('/upload', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        dzLabel.textContent = '✅ Processing ' + file.name + ' — drop another video to replace it';
        placeholder.style.display = 'none';
        videoImg.style.display = 'block';
        videoImg.src = '/video_feed?ts=' + Date.now();
      } else {
        dzLabel.textContent = '❌ ' + d.error;
      }
    })
    .catch(() => { dzLabel.textContent = '❌ Upload failed'; });
}

document.getElementById('resetBtn').addEventListener('click', () => {
  fetch('/reset', { method: 'POST' }).then(() => {
    videoImg.style.display = 'none';
    placeholder.style.display = 'block';
    dzLabel.textContent = '📤 Click or drag a traffic video here (.mp4, .avi, .mov, .webm)';
  });
});

function poll() {
  fetch('/status').then(r => r.json()).then(d => {
    document.getElementById('backendPill').textContent = 'backend: ' + d.backend;
    const rows = Object.keys(d.densities).map(lane => {
      const sig = d.signal.state[lane] || 'RED';
      const cnt = d.counts[lane] ?? 0;
      const dens = d.densities[lane] ?? 'LOW';
      return `<tr>
        <td>${lane}</td>
        <td>${cnt}</td>
        <td><span class="dens ${dens}">${dens}</span></td>
        <td><span class="badge ${sig}"><span class="dot ${sig}"></span>${sig}</span></td>
      </tr>`;
    }).join('');
    document.getElementById('laneRows').innerHTML = rows || '<tr><td colspan=4>No data yet</td></tr>';
    document.getElementById('phaseText').textContent = 'Phase: ' + d.signal.phase + ' (' + (d.signal.current_lane || '-') + ')';
    document.getElementById('timerText').textContent = d.signal.seconds_left + ' s left';
    document.getElementById('log').innerHTML = (d.signal.log || []).map(l => l.replace(/</g,'&lt;')).join('<br>') || 'Waiting for video…';

    const banner = document.getElementById('emergencyBanner');
    if (d.signal.emergency_lane) {
      banner.style.display = 'block';
      banner.textContent = '🚨 EMERGENCY VEHICLE DETECTED IN ' + d.signal.emergency_lane + ' — signal pre-empted';
    } else {
      banner.style.display = 'none';
    }
  }).catch(() => {});
}
setInterval(poll, 1000);
poll();
</script>
</body>
</html>
"""

def start_background_threads():
    """Launches the 1Hz signal-controller loop once, independent of any
    uploaded video's frame rate."""
    t = threading.Thread(target=signal_loop, daemon=True)
    t.start()


if __name__ == "__main__":
    start_background_threads()
    print("=" * 60)
    print(" AI Traffic Management System")
    print(f" SQLite database: {DB_PATH}")
    print(" Open http://127.0.0.1:5000 in your browser")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    