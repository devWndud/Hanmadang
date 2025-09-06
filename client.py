
import websockets
import asyncio
import json
import time
import cv2
import requests

# 라즈베리파이 5용 GPIO 라이브러리
try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
except ImportError:
    try:
        import RPi.GPIO as GPIO
        GPIO_AVAILABLE = True
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
    except ImportError:
        GPIO_AVAILABLE = False

ip = '222.231.24.5'  # 서버 IP

# GPIO 설정
LED_PIN = 7
if GPIO_AVAILABLE:
    try:
        led = LED(LED_PIN)  # gpiozero 사용
        LED_METHOD = 'gpiozero'
    except:
        GPIO.setup(LED_PIN, GPIO.OUT)  # RPi.GPIO 사용
        LED_METHOD = 'RPi.GPIO'
        led = None
else:
    led = None
    LED_METHOD = None

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
    led_state = False
    last_person_alert = 0
    last_fire_alert = 0

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
            
            # 서버에서 경보 상태 확인
            try:
                # 사람 감지 상태 확인
                person_res = requests.get(f'http://{ip}:5000/person_status', timeout=1)
                person_data = person_res.json()
                
                # 화재 감지 상태 확인
                fire_res = requests.get(f'http://{ip}:5000/fire_status', timeout=1)
                fire_data = fire_res.json()
                
                current_time = time.time()
                
                # 사람 감지 시 LED 깜빡임 (2초간)
                if person_data.get('person', False) and (current_time - last_person_alert) > 2:
                    print("사람 감지! LED 켜짐")
                    if LED_METHOD == 'gpiozero' and led:
                        led.on()
                    elif LED_METHOD == 'RPi.GPIO':
                        GPIO.output(LED_PIN, GPIO.HIGH)
                    led_state = True
                    last_person_alert = current_time
                
                # 화재 감지 시 LED 깜빡임 (1초간)
                elif fire_data.get('fire', False) and (current_time - last_fire_alert) > 1:
                    print("화재 감지! LED 켜짐")
                    if LED_METHOD == 'gpiozero' and led:
                        led.on()
                    elif LED_METHOD == 'RPi.GPIO':
                        GPIO.output(LED_PIN, GPIO.HIGH)
                    led_state = True
                    last_fire_alert = current_time
                
                # LED 자동 끄기 (경보 후 2초 후)
                if led_state and (current_time - last_person_alert) > 2 and (current_time - last_fire_alert) > 2:
                    if LED_METHOD == 'gpiozero' and led:
                        led.off()
                    elif LED_METHOD == 'RPi.GPIO':
                        GPIO.output(LED_PIN, GPIO.LOW)
                    led_state = False
                    print("LED 꺼짐")
                    
            except Exception as e:
                # 네트워크 오류 시 무시
                pass
            
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
        if LED_METHOD == 'RPi.GPIO':
            GPIO.cleanup()  # GPIO 정리
        elif LED_METHOD == 'gpiozero' and led:
            led.close()  # gpiozero 정리

if __name__ == "__main__":
    main()