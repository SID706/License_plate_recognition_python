"""Web demo for the licence plate recognition project.

predict.py runs the YOLO detector + EasyOCR pipeline from the command line for files
and videos; number_plate.py needs a local webcam. This module exposes the same
pipeline (detect plates with best.pt, read the characters with EasyOCR) as a small
web page so the project can be exercised from a browser, and stores the readings it
collects so they can be reviewed and managed.
"""

import base64
import os
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone

import cv2
import easyocr
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
DB_PATH = os.environ.get("DB_PATH", "data/records.db")
CONFIDENCE = float(os.environ.get("LPR_CONFIDENCE", "0.25"))
MAX_UPLOAD_MB = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

_lock = threading.Lock()
_model = None
_reader = None


def db():
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return closing(connection)


def init_db():
    with db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                detection_confidence REAL NOT NULL DEFAULT 0,
                ocr_confidence REAL NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'live',
                scanned_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


init_db()


def get_model():
    global _model
    with _lock:
        if _model is None:
            _model = YOLO(MODEL_PATH)
    return _model


def get_reader():
    global _reader
    with _lock:
        if _reader is None:
            _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def read_plate_text(crop):
    """Same rules as ocr_image() in predict.py: one result is taken as is,
    several results only count when they are long enough and confident."""
    if crop.size == 0:
        return "", 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    results = get_reader().readtext(gray)
    text, score = "", 0.0
    for _, detected, confidence in results:
        if len(results) == 1:
            text, score = detected, confidence
        elif len(detected) > 6 and confidence > 0.2:
            text, score = detected, confidence
    return str(text).strip(), float(score)


def annotate(img, plates):
    drawn = img.copy()
    height, width = drawn.shape[:2]
    thickness = max(2, round(min(width, height) / 300))
    for plate in plates:
        x1, y1, x2, y2 = plate["box"]
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 235, 140), thickness + 1)
        label = plate["text"] or "number plate"
        font_scale = max(0.6, min(width, height) / 900)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(0, y1 - th - 12)
        cv2.rectangle(drawn, (x1, top), (x1 + tw + 12, y1), (0, 235, 140), cv2.FILLED)
        cv2.putText(drawn, label, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (10, 18, 14), thickness, cv2.LINE_AA)
    return drawn


def decode_image(upload):
    data = np.frombuffer(upload.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect(img):
    results = get_model().predict(img, conf=CONFIDENCE, verbose=False)
    boxes = results[0].boxes if results else None
    if boxes is None:
        return []

    plates = []
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        text, ocr_score = read_plate_text(img[y1:y2, x1:x2])
        plates.append({
            "text": text,
            "confidence": round(float(box.conf[0]), 4),
            "ocr_confidence": round(ocr_score, 4),
            "box": [x1, y1, x2, y2],
        })
    plates.sort(key=lambda p: p["confidence"], reverse=True)
    return plates


@app.get("/")
def index():
    return render_template("index.html", max_upload_mb=MAX_UPLOAD_MB)


@app.get("/health")
def health():
    return jsonify(status="ok", model=os.path.basename(MODEL_PATH))


@app.post("/api/detect")
def api_detect():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(error="Choose an image of a vehicle first."), 400

    img = decode_image(upload)
    if img is None:
        return jsonify(error="That file could not be read as an image."), 400

    started = time.time()
    try:
        plates = detect(img)
    except Exception as exc:  # surfaced to the page instead of a 500 stack trace
        app.logger.exception("detection failed")
        return jsonify(error=f"Detection failed: {exc}"), 500

    ok, buffer = cv2.imencode(".jpg", annotate(img, plates), [cv2.IMWRITE_JPEG_QUALITY, 88])
    preview = ""
    if ok:
        preview = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()

    return jsonify(
        plates=plates,
        image=preview,
        elapsed_ms=round((time.time() - started) * 1000),
        size=[img.shape[1], img.shape[0]],
    )


@app.get("/api/records")
def list_records():
    with db() as connection:
        rows = connection.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 200").fetchall()
    return jsonify(records=[dict(row) for row in rows])


@app.post("/api/records")
def create_record():
    payload = request.get_json(silent=True) or {}
    plate = str(payload.get("plate") or "").strip()
    if not plate:
        return jsonify(error="There is no plate reading to save."), 400

    source = "live" if payload.get("source") == "live" else "upload"
    scanned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db() as connection:
        cursor = connection.execute(
            "INSERT INTO scans (plate, detection_confidence, ocr_confidence, source, scanned_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                plate[:32],
                float(payload.get("detection_confidence") or 0),
                float(payload.get("ocr_confidence") or 0),
                source,
                scanned_at,
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM scans WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return jsonify(record=dict(row)), 201


@app.delete("/api/records/<int:record_id>")
def delete_record(record_id):
    with db() as connection:
        connection.execute("DELETE FROM scans WHERE id = ?", (record_id,))
        connection.commit()
    return jsonify(deleted=record_id)


@app.delete("/api/records")
def clear_records():
    with db() as connection:
        removed = connection.execute("DELETE FROM scans").rowcount
        connection.commit()
    return jsonify(deleted=removed)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "3000")), debug=True)
