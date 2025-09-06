# 이 파일은 라즈베리파이에서 카메라 영상을 캡처해 서버로 전송하는 예시입니다.
# minimal_server.py가 라즈베리파이에서 직접 실행된다면 별도 클라이언트는 필요 없습니다.
# 만약 클라이언트-서버 분리 필요시 아래 예시 참고.

import cv2
import requests

cap = cv2.VideoCapture(0)
server_url = 'http://[서버IP]:5000/upload_frame'  # 서버에 맞게 수정

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(server_url, files={'frame': img_encoded.tobytes()})
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
