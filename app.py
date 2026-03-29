import cv2
import os
import tempfile
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from audio_analyzer import analyze_audio

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__, static_folder="static")

# Load trained deepfake model
model = tf.keras.models.load_model("model/deepfake_model_94acc.h5")

# Set max upload size to 50MB to prevent hangs on large files
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 

@app.before_request
def log_request_info():
    if request.path == "/predict":
        print(f"--- INCOMING UPLOAD REQUEST: {request.content_length} bytes ---", flush=True)

# -----------------------------
# CNN score helper
# -----------------------------
def preprocess_image(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (224, 224))
    img = cv2.resize(img_rgb, (224, 224))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img

def cnn_score(img_cv):
    img = preprocess_image(img_cv)
    return float(model.predict(img)[0][0])

# def ai_verification_score(img_cv):
#     img = preprocess_image(img_cv)
#     # Output: 0=fake(ai), 1=real
#     return float(ai_model.predict(img)[0][0])

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("--- NEW REQUEST RECEIVED ---", flush=True)
    file = request.files["file"]
    print(f"File received: {file.filename}, Content-Type: {file.mimetype}", flush=True)

    # -------- VIDEO --------
    if file.mimetype.startswith("video"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        file.save(tmp.name)

        # Video Processing: Extract every 10th frame and average their scores
        cap = cv2.VideoCapture(tmp.name)
        scores = []
        frame_idx = 0
        
        print("--- STARTING VIDEO PROCESSING (Every 10th Frame) ---", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 10th frame
            if frame_idx % 10 == 0:
                score = cnn_score(frame)
                scores.append(score)
                print(f"Frame {frame_idx}: Score {score:.4f}", flush=True)
            
            frame_idx += 1

        cap.release()
        #os.remove(tmp.name)
        print("--- VIDEO PROCESSING COMPLETE ---", flush=True)

        if not scores:
            return jsonify({
                "result": "FAKE",
                "confidence": 0.0,
                "reason": "No readable frames in video to analyze."
            })

        # Average Probability
        avg_prob = sum(scores) / len(scores)
        print(f"Average Probability: {avg_prob:.4f}")

        # Decision Threshold: 0.5
        # >= 0.5 -> REAL
        # < 0.5  -> FAKE
        if avg_prob >= 0.5:
             return jsonify({
                "result": "REAL",
                "confidence": float(avg_prob),
                "reason": f"Video analyzed: Real (Avg Score: {avg_prob:.2f})"
            })
        else:
             return jsonify({
                "result": "FAKE",
                "confidence": float(avg_prob),
                "reason": f"Deepfake detected in video (Avg Score: {avg_prob:.2f})"
            })

    # -------- AUDIO --------
    if file.mimetype.startswith("audio"):
        audio_bytes = file.read()
        result = analyze_audio(audio_bytes, filename=file.filename)
        print(f"AUDIO RESULT: {result}", flush=True)
        return jsonify(result)

    # -------- IMAGE --------
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({
            "result": "ERROR",
            "confidence": 0.0,
            "reason": "Invalid image file"
        })

    df_score = cnn_score(img)
    print(f"MODEL SCORE - Deepfake: {df_score}")

    # POLICY:
    # Score >= 0.5 = REAL, Score < 0.5 = FAKE
    if df_score >= 0.5:
        return jsonify({
            "result": "REAL",
            "confidence": df_score,
            "reason": f"Real image detected (Score: {df_score:.2f})"
        })

    return jsonify({
        "result": "FAKE",
        "confidence": df_score,
        "reason": f"Deepfake / manipulated image detected (Score: {df_score:.2f})"
    })

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("----------------------------------------------------------")
    print(" PROJECT RUNNING SUCCESSFULLY")
    print(" OPEN BROWSER AT: http://127.0.0.1:8080/")
    print("----------------------------------------------------------")
    # Threading disabled to prevent TensorFlow conflicts
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=False)
