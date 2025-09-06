import threading
import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # 작은 모델로 기본값

latest_frame = None
frame_lock = threading.Lock()

# 카메라 캡처 및 YOLO 감지 스레드
class CameraThread(threading.Thread):
    def __init__(self, device=0):
        super().__init__()
        self.cap = cv2.VideoCapture(device)
        self.running = True

    def run(self):
        global latest_frame
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            # YOLO 감지
            results = model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.03)
        self.cap.release()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is None:
                time.sleep(0.05)
                continue
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    cam_thread = CameraThread(device=0)
    cam_thread.start()
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    finally:
        cam_thread.running = False
        cam_thread.join()
