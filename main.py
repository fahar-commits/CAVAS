from ultralytics import YOLO
import cv2
import time
import csv
import threading
import random
import os
import sys
from pathlib import Path

try:
    import serial
except:
    serial = None

try:
    from playsound import playsound
except:
    playsound = None

MODEL_PATH = "yolov8n.pt"
ALERT_CLASSES = {"person", "car", "motorbike", "bicycle", "dog", "cat"}
CONF_THRESHOLD = 0.45
SOUND_PATH = Path("sounds/engine_idle.wav")
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = LOGS_DIR / "detections.csv"
SERIAL_PORTS = ["COM3", "/dev/ttyUSB0", "/dev/ttyACM0"]

def open_serial():
    if serial is None:
        return None
    for p in SERIAL_PORTS:
        try:
            s = serial.Serial(p, 9600, timeout=0.1)
            time.sleep(2)
            return s
        except:
            continue
    return None

def sensor_confirm(ser):
    if ser:
        try:
            ser.write(b"R\n")
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                return line.lower() in ("1", "true", "t", "y", "yes")
        except:
            pass
    return random.random() < 0.7

def play_sound_nonblocking(path):
    if playsound is None:
        return
    def _p():
        try:
            playsound(str(path))
        except:
            pass
    t = threading.Thread(target=_p, daemon=True)
    t.start()

def init_csv():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "detected_object", "confidence", "sensor_confirmed"])

def main():
    model = YOLO(MODEL_PATH)
    ser = open_serial()
    cap = cv2.VideoCapture(0)
    init_csv()
    running = True
    last_play_time = 0
    play_cooldown = 1.0
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, imgsz=640, device=0)
        detections = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            data = getattr(boxes, "data", None)
            if data is None:
                continue
            for row in data.tolist():
                x1, y1, x2, y2, conf, cls = row
                label = model.names[int(cls)]
                if conf < CONF_THRESHOLD:
                    continue
                if label in ALERT_CLASSES:
                    detections.append((label, float(conf), int(x1), int(y1), int(x2), int(y2)))
        detected_flag = False
        for label, conf, x1, y1, x2, y2 in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            confirmed = sensor_confirm(ser)
            with open(CSV_PATH, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([time.time(), label, f"{conf:.2f}", confirmed])
            if confirmed and time.time() - last_play_time > play_cooldown:
                play_sound_nonblocking(SOUND_PATH)
                last_play_time = time.time()
            detected_flag = True
        if not detected_flag:
            cv2.putText(frame, "No relevant object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("CAVAS Simulation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        try:
            ser.close()
        except:
            pass

if __name__ == "__main__":
    main()
