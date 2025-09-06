# 이 파일은 라즈베리파이에서 카메라 영상을 캡처해 서버로 전송하는 예시입니다.
# minimal_server.py가 라즈베리파이에서 직접 실행된다면 별도 클라이언트는 필요 없습니다.
# 만약 클라이언트-서버 분리 필요시 아래 예시 참고.

import cv2
import requests
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
server_url = 'http://222.231.24.5:5000/upload_frame'  # 서버에 맞게 수정
model = YOLO('yolov8n.pt')  # 필요시 yolov8s.pt 등으로 변경

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    # YOLO 감지 및 박스/라벨 그리기
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(server_url, files={'frame': img_encoded.tobytes()})
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
