from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO

app = Flask(__name__)



# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace with the correct model path

# Capture video from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Error: Webcam not accessible.")

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model.predict(source=frame, conf=0.5)

        # Annotate the frame
        annotated_frame = results[0].plot()

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # HTML page to display the video

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
