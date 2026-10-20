
import os
import base64
import time
import traceback

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
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

mp_face = mp.solutions.face_detection


# =====================================
# PRE-WARM DEEPFACE MODEL
# =====================================

print("Pre-loading DeepFace model...", flush=True)

try:
    dummy = np.zeros((48, 48, 3), dtype=np.uint8)

    DeepFace.analyze(
        img_path=dummy,
        actions=["emotion"],
        detector_backend="skip",
        enforce_detection=False,
        silent=True,
    )

    print("DeepFace ready.", flush=True)

except Exception:
    print("DeepFace warm-up warning:", flush=True)
    traceback.print_exc()


# =====================================
# HELPER: GET FIRST ANALYSIS RESULT
# =====================================

def get_analysis_result(analysis):
    """Handle DeepFace results returned as a list or dictionary."""
    if isinstance(analysis, list):
        if not analysis:
            raise ValueError("DeepFace returned an empty result.")
        return analysis[0]

    if isinstance(analysis, dict):
        return analysis

    raise ValueError("Unexpected DeepFace result format.")


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
    """Try original and two rotated orientations."""

    rotations = [
        ("Original", img),
        (
            "Rotated clockwise",
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        ),
        (
            "Rotated counterclockwise",
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ),
    ]

    best_img = img
    best_detections = []
    best_orientation = "Original"

    for orientation, candidate in rotations:
        rgb_candidate = cv2.cvtColor(
            candidate, cv2.COLOR_BGR2RGB
        )

        with mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.4,
        ) as detector:
            result = detector.process(rgb_candidate)
            detections = result.detections or []

        if len(detections) > len(best_detections):
            best_img = candidate.copy()
            best_detections = detections
            best_orientation = orientation

    return best_img, best_detections, best_orientation


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
        max_dimension = 1280

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
        for face_number, detection in enumerate(detections, start=1):
            try:
                box = detection.location_data.relative_bounding_box

                x1 = max(0, int(box.xmin * w))
                y1 = max(0, int(box.ymin * h))
                x2 = min(w, int((box.xmin + box.width) * w))
                y2 = min(h, int((box.ymin + box.height) * h))

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

                face_rgb = cv2.cvtColor(
                    face_crop, cv2.COLOR_BGR2RGB
                )

                print(
                    f"DeepFace: starting face {face_number}; "
                    f"crop={face_crop.shape[1]}x{face_crop.shape[0]}",
                    flush=True,
                )

                analysis_start = time.time()

                analysis = DeepFace.analyze(
                    img_path=face_rgb,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                    detector_backend="skip",
                )

                print(
                    f"DeepFace: face {face_number} completed in "
                    f"{time.time() - analysis_start:.2f}s",
                    flush=True,
                )

                result = get_analysis_result(analysis)
                emotion_scores = result["emotion"]

                dominant = max(
                    emotion_scores, key=emotion_scores.get
                )
                confidence = float(emotion_scores[dominant])

                face_results.append({
                    "face_number": face_number,
                    "emotion": dominant,
                    "confidence": round(confidence, 2),
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
                    f"DeepFace: face {face_number} analysis failed",
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        ) as detector:
            results = detector.process(rgb_frame)

        detections = results.detections or []

        current_time = time.time()

        do_analysis = (
            current_time - state["last_analysis_time"]
            >= state["analysis_interval"]
        )

        # Analyze detected faces at the configured interval
        if do_analysis:
            for i, detection in enumerate(detections):
                try:
                    box = detection.location_data.relative_bounding_box

                    x1 = max(0, int(box.xmin * w))
                    y1 = max(0, int(box.ymin * h))
                    x2 = min(w, int((box.xmin + box.width) * w))
                    y2 = min(h, int((box.ymin + box.height) * h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    face_rgb = cv2.cvtColor(
                        face_crop, cv2.COLOR_BGR2RGB
                    )

                    result = get_analysis_result(
                        DeepFace.analyze(
                            img_path=face_rgb,
                            actions=["emotion"],
                            detector_backend="skip",
                            enforce_detection=False,
                            silent=True,
                        )
                    )

                    scores = result["emotion"]
                    dominant = max(scores, key=scores.get)

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
        for i, detection in enumerate(detections):
            box = detection.location_data.relative_bounding_box

            x1 = max(0, int(box.xmin * w))
            y1 = max(0, int(box.ymin * h))
            x2 = min(w, int((box.xmin + box.width) * w))
            y2 = min(h, int((box.ymin + box.height) * h))

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
