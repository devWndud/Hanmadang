
import websockets
import asyncio
import json
import time
import cv2


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

    async with websockets.connect(f'ws://{ip}:8000') as ws:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                await asyncio.sleep(0.01)
                continue
            frame = cv2.resize(frame, (480, 270))  # 해상도 낮춤
            _, img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # 품질 낮춤

            payload = {
                "img": list(img.tobytes()),
                "uniqueCode": uniqueCode
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(0.3)  # 0.3초마다 전송 (더 빠르게)

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