
import asyncio
import websockets
import threading
import numpy as np
import cv2
import json
import time
from flask import Flask, Response, send_from_directory, jsonify
import torch
import os

# 온습도 센서 라이브러리 (라즈베리파이에서만 동작)
try:
    import board
    import adafruit_dht
    DHT_AVAILABLE = True
except ImportError:
    DHT_AVAILABLE = False

# YOLOv5n 모델 로드 (ultralytics/yolov5 필요)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

app = Flask(__name__)
latestFrame = None
frameLock = threading.Lock()
latestPersonDetected = False
personLock = threading.Lock()
latestFireDetected = False
fireLock = threading.Lock()

# 화재 감지 임계 온도 (Celsius)
FIRE_TEMP_THRESHOLD = 50.0

# 온습도 센서 스레드
def dht_thread():
    global latestFireDetected
    if not DHT_AVAILABLE:
        print('DHT 센서 라이브러리 없음. 화재 감지 비활성화.')
        return
    dhtDevice = adafruit_dht.DHT22(board.D4)
    while True:
        try:
            temperature_c = dhtDevice.temperature
            humidity = dhtDevice.humidity
            print(f"[DHT] Temp: {temperature_c} C, Humidity: {humidity}%")
            fire = temperature_c is not None and temperature_c >= FIRE_TEMP_THRESHOLD
            with fireLock:
                latestFireDetected = fire
        except RuntimeError as error:
            print(f"[DHT] {error.args[0]}")
            time.sleep(2.0)
            continue
        except Exception as error:
            dhtDevice.exit()
            print(f"[DHT] Fatal error: {error}")
            break
        time.sleep(2.0)

# WebSocket 서버: 클라이언트에서 프레임 수신
async def handle_client(websocket):
    global latestFrame, latestPersonDetected
    try:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            imgBytes = bytes(data["img"])
            img = cv2.imdecode(np.frombuffer(imgBytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # 해상도 낮추기
            img = cv2.resize(img, (320, 180))
            # YOLOv5n 감지 (GPU 자동 사용)
            results = model(img)
            person_found = False
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                label = model.names[int(cls)]
                if label == 'person':
                    person_found = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            with frameLock:
                latestFrame = img.copy()
            with personLock:
                latestPersonDetected = person_found
    except websockets.exceptions.ConnectionClosed:
        pass

async def start_ws_server():
    async with websockets.serve(handle_client, "0.0.0.0", 8000):
        print("WebSocket server started on ws://0.0.0.0:8000")
        await asyncio.Future()

def ws_thread():
    asyncio.run(start_ws_server())

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frameLock:
                frame = latestFrame.copy() if latestFrame is not None else None
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

# video.html을 서비스하는 엔드포인트 추가
@app.route('/')
def serve_monitoring():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'video.html')

# 사람 감지 상태를 반환하는 엔드포인트
@app.route('/person_status')
def person_status():
    with personLock:
        detected = latestPersonDetected
    return jsonify({"person": detected})

# 화재 감지 상태를 반환하는 엔드포인트
@app.route('/fire_status')
def fire_status():
    with fireLock:
        fire = latestFireDetected
    return jsonify({"fire": fire})

if __name__ == '__main__':
    t = threading.Thread(target=ws_thread, daemon=True)
    t.start()
    if DHT_AVAILABLE:
        dht_t = threading.Thread(target=dht_thread, daemon=True)
        dht_t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)