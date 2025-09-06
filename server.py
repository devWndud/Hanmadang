
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import asyncio
import websockets
import threading
import numpy as np
import cv2
import json
import time
from flask import Flask, Response, send_from_directory, jsonify, request
import torch
import os

# YOLOv5n 모델 로드 (ultralytics/yolov5 필요)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()  # 평가 모드로 설정하여 최적화

app = Flask(__name__)
latestFrame = None
frameLock = threading.Lock()
latestPersonDetected = False
personLock = threading.Lock()
latestFireDetected = False
fireLock = threading.Lock()
latestTemp = None
latestHumidity = None
envLock = threading.Lock()

# 화재 감지 임계 온도 (Celsius)
FIRE_TEMP_THRESHOLD = 50.0

# WebSocket 서버: 클라이언트에서 프레임+온습도 수신
async def handle_client(websocket):
    global latestFrame, latestPersonDetected, latestTemp, latestHumidity, latestFireDetected
    try:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            imgBytes = bytes(data["img"])
            img = cv2.imdecode(np.frombuffer(imgBytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            # 해상도 낮추기 (더 작게)
            img = cv2.resize(img, (240, 135))
            
            # YOLO 추론 최적화
            with torch.no_grad():  # 그래디언트 계산 비활성화
                results = model(img, verbose=False)
            
            person_found = False
            # 감지 결과 처리 최적화
            if len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                detections = results.xyxy[0].cpu().numpy()
                for *xyxy, conf, cls in detections:
                    if conf < 0.5:  # 신뢰도 0.5 미만은 무시
                        continue
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = model.names[int(cls)]
                    if label == 'person':
                        person_found = True
                    # 박스 그리기 최적화 (더 얇은 선)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(img, f'{label} {conf:.1f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # 프레임 저장 (해상도 복원)
            img_resized = cv2.resize(img, (320, 180))
            with frameLock:
                latestFrame = img_resized.copy()
            with personLock:
                latestPersonDetected = person_found
                
            # 온습도 저장 (클라이언트에서 전송된 경우만)
            if "temperature" in data and "humidity" in data:
                with envLock:
                    latestTemp = data.get("temperature")
                    latestHumidity = data.get("humidity")
                    fire = latestTemp is not None and latestTemp >= FIRE_TEMP_THRESHOLD
                with fireLock:
                    latestFireDetected = fire
    except websockets.exceptions.ConnectionClosed:
        pass

async def start_ws_server():
    async with websockets.serve(handle_client, "0.0.0.0", 8000):
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
                time.sleep(0.1)  # 대기 시간 증가
                continue
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])  # 품질 낮춤
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)  # 프레임 간격 증가
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# video.html을 서비스하는 엔드포인트 추가
@app.route('/')
def serve_monitoring():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'video.html')

# admin.html을 서비스하는 엔드포인트 추가
@app.route('/admin')
def serve_admin():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'admin.html')

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

# 온습도 상태 반환
@app.route('/env_status')
def env_status():
    with envLock:
        temp = latestTemp
        hum = latestHumidity
    return jsonify({"temperature": temp, "humidity": hum})

# 온습도 값 설정 (어드민용)
@app.route('/set_env', methods=['POST'])
def set_env():
    try:
        data = request.get_json()
        temp = data.get('temperature')
        hum = data.get('humidity')
        if temp is None or hum is None:
            return jsonify({"success": False, "error": "온도와 습도가 필요합니다"})
        with envLock:
            global latestTemp, latestHumidity
            latestTemp = temp
            latestHumidity = hum
        # 화재 감지도 업데이트
        fire = temp >= FIRE_TEMP_THRESHOLD
        with fireLock:
            global latestFireDetected
            latestFireDetected = fire
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    t = threading.Thread(target=ws_thread, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)