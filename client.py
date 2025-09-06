
import websockets
import asyncio
import json
import time
import cv2

# 온습도 센서 라이브러리
try:
    import board
    import adafruit_dht
    DHT_AVAILABLE = True
except ImportError:
    DHT_AVAILABLE = False

ip = '222.231.24.5'  # 서버 IP

class CameraThread:
    def __init__(self, device_path=0):
        self.cap = cv2.VideoCapture(device_path)
        self.ret = False
        self.frame = None
        self.running = True
        if not self.cap.isOpened():
            self.running = False

    def read(self):
        self.ret, self.frame = self.cap.read()
        return self.ret, self.frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

async def send_frames(cam):
    uniqueCode = 'A7X2bQ9cLmY1'
    if DHT_AVAILABLE:
        dhtDevice = adafruit_dht.DHT22(board.D4)
    else:
        dhtDevice = None

    async with websockets.connect(f'ws://{ip}:8000') as ws:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue
            frame = cv2.resize(frame, (640, 360))
            _, img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            # 온습도 읽기
            temperature = None
            humidity = None
            if dhtDevice:
                try:
                    temperature = dhtDevice.temperature
                    humidity = dhtDevice.humidity
                except Exception:
                    pass

            payload = {
                "img": list(img.tobytes()),
                "uniqueCode": uniqueCode,
                "temperature": temperature,
                "humidity": humidity
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(0.5)  # 0.5초마다 전송

def main():
    cam = CameraThread(0)
    if not cam.running:
        print("카메라를 열 수 없습니다.")
        return
    try:
        asyncio.run(send_frames(cam))
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()

if __name__ == "__main__":
    main()