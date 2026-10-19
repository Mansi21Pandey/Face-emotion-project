from PIL import Image
import os
import io
import base64
import time
import numpy as np
import cv2
import mediapipe as mp
from deepface import DeepFace
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image

# =====================================
# FLASK + SOCKETIO SETUP
# =====================================

base_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'), static_folder=os.path.join(base_dir, 'static'))
app.config["SECRET_KEY"] = "emotion-secret-key"
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_face = mp.solutions.face_detection

# =====================================
# PRE-WARM DEEPFACE (so first request isn't slow)
# =====================================

print("Pre-loading DeepFace model...")
dummy = np.zeros((48, 48, 3), dtype=np.uint8)
try:
    DeepFace.analyze(
        dummy,
        actions=["emotion"],
        detector_backend="skip",
        enforce_detection=False,
        silent=True,
    )
    print("DeepFace ready.")
except Exception as e:
    print("Warm-up warning:", e)


# =====================================
# ROUTES
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
# STATIC IMAGE ENDPOINT
# =====================================

def detect_faces_with_rotation(img):
    """Try 3 orientations, return best one + detections."""
    rotations = [
        ("Original", img),
        ("Rotated clockwise", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("Rotated counterclockwise", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    best_img = img
    best_detections = []
    best_orientation = "Original"

    for orientation, candidate in rotations:
        rgb_candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)

        with mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.4
        ) as detector:
            result = detector.process(rgb_candidate)
            detections = result.detections or []

        if len(detections) > len(best_detections):
            best_img = candidate.copy()
            best_detections = detections
            best_orientation = orientation

    return best_img, best_detections, best_orientation


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read image from request
    print("PREDICT_IMAGE: request received", flush=True)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Detect faces with rotation logic
    img, detections, orientation = detect_faces_with_rotation(img)
    print(
        f"PREDICT_IMAGE: detections={len(detections)}, "
        f"orientation={orientation}",
        flush=True
    )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated = img_rgb.copy()

    h, w = img.shape[:2]
    face_results = []

    for face_number, detection in enumerate(detections, start=1):
        box = detection.location_data.relative_bounding_box

        x1 = max(0, int(box.xmin * w))
        y1 = max(0, int(box.ymin * h))
        x2 = min(w, int((box.xmin + box.width) * w))
        y2 = min(h, int((box.ymin + box.height) * h))

        if x2 <= x1 or y2 <= y1:
            continue

        padding = 20
        y1_p = max(0, y1 - padding)
        y2_p = min(h, y2 + padding)
        x1_p = max(0, x1 - padding)
        x2_p = min(w, x2 + padding)
        face_crop = img[y1_p:y2_p, x1_p:x2_p]
        if face_crop.size == 0:
            continue

        try:
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            print(f"DeepFace: analyzing face {face_number}", flush=True)
            analysis = DeepFace.analyze(
                img_path=np.array(face_pil),
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
                detector_backend="skip" 
                )

            emotion_scores = analysis[0]["emotion"]
            dominant = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant]

            face_results.append({
                "face_number": face_number,
                "emotion": dominant,
                "confidence": round(float(confidence), 2),
                "scores": {k: round(float(v), 2) for k, v in emotion_scores.items()},
            })

            # Draw rectangle + label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{dominant}: {confidence:.1f}%"
            text_y = y1 + 25 if y1 + 25 < h else y1 - 10
            cv2.putText(
                annotated, label,
                (x1 + 5, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()

    # Encode annotated image to base64
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", annotated_bgr)
    encoded = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "annotated_image": f"data:image/png;base64,{encoded}",
        "orientation": orientation,
        "faces": face_results,
        "total_faces": len(face_results),
    })


# =====================================
# REAL-TIME WEBSOCKET HANDLER
# =====================================

# Per-client state
client_state = {}

@socketio.on("connect")
def handle_connect():
    sid = request.sid
    client_state[sid] = {
        "last_analysis_time": 0,
        "analysis_interval": 1.0,   # seconds
        "face_emotions": {},
    }
    print(f"Client connected: {sid}")


@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    client_state.pop(sid, None)
    print(f"Client disconnected: {sid}")


def decode_base64_frame(data_url):
    """Convert data:image/jpeg;base64,... to numpy BGR frame."""
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print("Frame decode error:", e)
        return None


@socketio.on("frame")
def handle_frame(data):
    sid = request.sid
    state = client_state.get(sid)
    if state is None:
        return

    frame = decode_base64_frame(data.get("image", ""))
    if frame is None:
        return

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as detector:
        results = detector.process(rgb_frame)

    current_time = time.time()
    do_analysis = (current_time - state["last_analysis_time"]) >= state["analysis_interval"]

    if results.detections:
        for i, detection in enumerate(results.detections):
            box = detection.location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            if do_analysis:
                face_crop = frame[y1:y2, x1:x2]
                try:
                    result = DeepFace.analyze(
                        face_crop,
                        actions=["emotion"],
                        detector_backend="skip",
                        enforce_detection=False,
                        silent=True,
                    )
                    state["face_emotions"][i] = (
                        result[0]["dominant_emotion"],
                        float(result[0]["emotion"][result[0]["dominant_emotion"]]),
                    )
                except Exception as e:
                    print("Realtime analysis error:", e)

        if do_analysis:
            state["last_analysis_time"] = current_time

    # Draw boxes + labels
    if results.detections:
        for i, detection in enumerate(results.detections):
            box = detection.location_data.relative_bounding_box
            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if i in state["face_emotions"]:
                emotion, conf = state["face_emotions"][i]
                label = f"{emotion}: {conf:.1f}%"
            else:
                label = "Analyzing..."

            cv2.putText(
                frame, label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

    face_count = len(results.detections) if results.detections else 0
    cv2.putText(
        frame, f"Faces: {face_count}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    # Encode annotated frame and send back
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    encoded = base64.b64encode(buffer).decode("utf-8")

    emit("annotated_frame", {
        "image": f"data:image/jpeg;base64,{encoded}",
        "face_count": face_count,
        "emotions": [
            {"face": k, "emotion": v[0], "confidence": v[1]}
            for k, v in state["face_emotions"].items()
        ],
    })


# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
