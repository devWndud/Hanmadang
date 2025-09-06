
import asyncio
import websockets
import threading
import numpy as np
import cv2
import json
import time
from flask import Flask, Response
import torch

# YOLOv5n 모델 로드 (ultralytics/yolov5 필요)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

app = Flask(__name__)
latestFrame = None
frameLock = threading.Lock()

# WebSocket 서버: 클라이언트에서 프레임 수신
async def handle_client(websocket):
    global latestFrame
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
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                label = model.names[int(cls)]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            with frameLock:
                latestFrame = img.copy()
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

if __name__ == '__main__':
    t = threading.Thread(target=ws_thread, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)