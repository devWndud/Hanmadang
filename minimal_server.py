from flask import Flask, Response, request
import threading
import numpy as np
import cv2
import time

app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global latest_frame
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is not None:
        with frame_lock:
            latest_frame = frame
        print("Frame received and updated")
    else:
        print("Frame decode failed")
    return 'OK'

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
    app.run(host='0.0.0.0', port=5000, debug=False)
