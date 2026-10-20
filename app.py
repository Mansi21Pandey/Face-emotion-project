import os
import base64
import time
import traceback
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit


# =====================================
# FLASK + SOCKETIO SETUP
# =====================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "emotion-secret-key"
)

socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =====================================
# LIGHTWEIGHT ONNX EMOTION MODEL
# =====================================

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "emotion-ferplus-8.onnx"
)

emotion_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)


# FER+ model ke original class names
EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

# Frontend ke liye familiar emotion names
EMOTION_DISPLAY_NAMES = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "surprise",
    "sadness": "sad",
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "contempt": "contempt",
}


def predict_emotion(face_bgr):
    """Predict emotions from a cropped face using the ONNX model."""

    # BGR image ko grayscale mein convert karo
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # Model ko 64x64 grayscale input chahiye
    resized = cv2.resize(gray, (64, 64))

    # Expected input shape: (1, 1, 64, 64)
    input_data = resized.astype(np.float32).reshape(1, 1, 64, 64)

    # ONNX inference
    input_name = emotion_session.get_inputs()[0].name
    output = emotion_session.run(None, {input_name: input_data})[0]

    # Stable softmax: scores ko probabilities mein badlo
    logits = output[0].astype(np.float64)
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities = probabilities / np.sum(probabilities)

    # Frontend-friendly emotion scores, percentage mein
    scores = {
        EMOTION_DISPLAY_NAMES[label]: float(probability * 100)
        for label, probability in zip(EMOTION_LABELS, probabilities)
    }

    dominant = max(scores, key=scores.get)
    confidence = scores[dominant]

    return dominant, confidence, scores


# =====================================
# PAGE ROUTES
# =====================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/static-image")
def static_image_page():
    return render_template("static_image.html")


@app.route("/realtime")
def realtime_page():
    return render_template("realtime.html")


# =====================================
# FACE DETECTION WITH ROTATION
# =====================================

def detect_faces_with_rotation(img):
    rotations = [
        ("Original", img),
        ("Rotated clockwise", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("Rotated counterclockwise", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    best_img = img
    best_faces = []
    best_orientation = "Original"

    for orientation, candidate in rotations:
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) > len(best_faces):
            best_img = candidate.copy()
            best_faces = faces
            best_orientation = orientation

    return best_img, best_faces, best_orientation

# =====================================
# STATIC IMAGE PREDICTION
# =====================================

@app.route("/predict_image", methods=["POST"])
def predict_image():
    start_time = time.time()

    try:
        # Validate uploaded file
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        print("PREDICT_IMAGE: request received", flush=True)

        # Decode uploaded image
        file_bytes = np.frombuffer(file.read(), np.uint8)

        if file_bytes.size == 0:
            return jsonify({"error": "Uploaded file is empty"}), 400

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        # Resize large images to reduce memory and CPU usage
        h, w = img.shape[:2]
        max_dimension = 640

        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)

            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            img = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )

        print(
            f"PREDICT_IMAGE: image prepared in "
            f"{time.time() - start_time:.2f}s",
            flush=True,
        )

        # Detect faces
        step_time = time.time()

        img, detections, orientation = detect_faces_with_rotation(img)

        print(
            f"PREDICT_IMAGE: detections={len(detections)}, "
            f"orientation={orientation}, "
            f"detection_time={time.time() - step_time:.2f}s",
            flush=True,
        )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated = img_rgb.copy()

        h, w = img.shape[:2]
        face_results = []

        # Analyze each detected face
        for face_number, (x, y, face_w, face_h) in enumerate(detections, start=1):
            try:
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(w, int(x + face_w))
                y2 = min(h, int(y + face_h))


                if x2 <= x1 or y2 <= y1:
                    continue

                padding = 20

                x1_p = max(0, x1 - padding)
                y1_p = max(0, y1 - padding)
                x2_p = min(w, x2 + padding)
                y2_p = min(h, y2 + padding)

                face_crop = img[y1_p:y2_p, x1_p:x2_p]

                if face_crop.size == 0:
                    continue
                print(
                    f"ONNX: predicting face {face_number}; "
                    f"crop={face_crop.shape[1]}x{face_crop.shape[0]}",
                    flush=True,
                )

                analysis_start = time.time()

                dominant, confidence, emotion_scores = predict_emotion(
                    face_crop
                )

                print(
                    f"ONNX: face {face_number} completed in "
                    f"{time.time() - analysis_start:.2f}s",
                    flush=True,
                )

                face_results.append({
                    "face_number": face_number,
                    "emotion": dominant,
                    "confidence": round(float(confidence), 2),
                    "scores": {
                        key: round(float(value), 2)
                        for key, value in emotion_scores.items()
                    },
                })

                # Draw bounding box and predicted emotion
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = f"{dominant}: {confidence:.1f}%"
                text_y = y1 + 25 if y1 + 25 < h else max(20, y1 - 10)

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            except Exception:
                print(
                    f"ONNX: face {face_number} analysis failed",
                    flush=True,
                )
                traceback.print_exc()

        # Encode the annotated image
        annotated_bgr = cv2.cvtColor(
            annotated, cv2.COLOR_RGB2BGR
        )

        success, buffer = cv2.imencode(".jpg", annotated_bgr)

        if not success:
            return jsonify({
                "error": "Could not encode result image"
            }), 500

        encoded = base64.b64encode(buffer).decode("utf-8")

        print(
            f"PREDICT_IMAGE: completed in "
            f"{time.time() - start_time:.2f}s; "
            f"successful_faces={len(face_results)}",
            flush=True,
        )

        return jsonify({
            "annotated_image": f"data:image/jpeg;base64,{encoded}",
            "orientation": orientation,
            "faces": face_results,
            "total_faces": len(face_results),
        })

    except Exception:
        print("PREDICT_IMAGE: route failed", flush=True)
        traceback.print_exc()

        return jsonify({
            "error": "Image processing failed. Please try another image."
        }), 500


# =====================================
# REAL-TIME WEBSOCKET HANDLER
# =====================================

client_state = {}


@socketio.on("connect")
def handle_connect():
    sid = request.sid

    client_state[sid] = {
        "last_analysis_time": 0,
        "analysis_interval": 1.0,
        "face_emotions": {},
    }

    print(f"Client connected: {sid}", flush=True)


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    client_state.pop(sid, None)

    print(f"Client disconnected: {sid}", flush=True)


def decode_base64_frame(data_url):
    """Convert a base64 data URL into an OpenCV BGR frame."""

    try:
        if not isinstance(data_url, str) or "," not in data_url:
            return None

        _, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded, validate=True)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return frame

    except Exception as exc:
        print("Frame decode error:", exc, flush=True)
        return None


@socketio.on("frame")
def handle_frame(data):
    sid = request.sid
    state = client_state.get(sid)

    if state is None or not isinstance(data, dict):
        return

    frame = decode_base64_frame(data.get("image", ""))

    if frame is None:
        return

    try:
        h, w = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = FACE_CASCADE.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )


        current_time = time.time()

        do_analysis = (
            current_time - state["last_analysis_time"]
            >= state["analysis_interval"]
        )

        # Analyze detected faces at the configured interval
        if do_analysis:
            for i, (x, y, face_w, face_h) in enumerate(detections):
                try:
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(w, int(x + face_w))
                    y2 = min(h, int(y + face_h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue
                    dominant, confidence, scores = predict_emotion(
                        face_crop
                    )
                    state["face_emotions"][i] = (
                        dominant,
                        float(scores[dominant]),
                    )

                except Exception:
                    print(
                        f"Realtime analysis error for face {i}:",
                        flush=True,
                    )
                    traceback.print_exc()

            state["last_analysis_time"] = current_time

        # Draw face boxes and labels
        for i, (x, y, face_w, face_h) in enumerate(detections):
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + face_w))
            y2 = min(h, int(y + face_h))


            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

            if i in state["face_emotions"]:
                emotion, confidence = state["face_emotions"][i]
                label = f"{emotion}: {confidence:.1f}%"
            else:
                label = "Analyzing..."

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        face_count = len(detections)

        cv2.putText(
            frame,
            f"Faces: {face_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

        if not success:
            return

        encoded = base64.b64encode(buffer).decode("utf-8")

        emit("annotated_frame", {
            "image": f"data:image/jpeg;base64,{encoded}",
            "face_count": face_count,
            "emotions": [
                {
                    "face": key,
                    "emotion": value[0],
                    "confidence": value[1],
                }
                for key, value in state["face_emotions"].items()
            ],
        })

    except Exception:
        print("Realtime frame processing failed:", flush=True)
        traceback.print_exc()


# =====================================
# RUN LOCALLY
# =====================================

if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
