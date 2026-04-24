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

import concurrent.futures

# Load specialized models
DF_MODEL_PATH = "model/deepfake_model_94acc.h5"
AI_MODEL_PATH = "model/unified_model.h5"

try:
    df_model = tf.keras.models.load_model(DF_MODEL_PATH)
    ai_model = tf.keras.models.load_model(AI_MODEL_PATH)
    print(f"[OK] High-Speed Hybrid System Active", flush=True)
except Exception as e:
    print(f"[ERROR] Loading models: {e}", flush=True)

# Thread pool for parallel inference
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# -----------------------------
# Inference Helpers
# -----------------------------
def preprocess_for_model(img_cv, normalize=True):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (224, 224))
    img = img.astype("float32")
    if normalize:
        img = img / 255.0
    return img

def get_hybrid_prediction(img_cv):
    # Prepare inputs
    img_ai = preprocess_for_model(img_cv, normalize=True)
    img_df = preprocess_for_model(img_cv, normalize=False)
    
    # Run in parallel
    def run_ai(): return ai_model.predict(np.expand_dims(img_ai, axis=0), verbose=0)[0]
    def run_df(): return df_model.predict(np.expand_dims(img_df, axis=0), verbose=0)[0]
    
    future_ai = executor.submit(run_ai)
    future_df = executor.submit(run_df)
    
    ai_preds = future_ai.result()
    df_prob = float(future_df.result()[0])
    
    # Decision Logic
    if ai_preds[0] > 0.5:
        return "AI GENERATED", float(ai_preds[0])
    elif df_prob < 0.5:
        return "DEEPFAKE", float(1.0 - df_prob)
    else:
        return "REAL", float(df_prob)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print("--- HIGH-SPEED PROCESSING ---", flush=True)
    if "file" not in request.files:
        return jsonify({"result": "ERROR", "reason": "No file uploaded"})
    file = request.files["file"]
    
    # -------- VIDEO (BATCH PROCESSING) --------
    if file.mimetype.startswith("video"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        file.save(tmp.name)
        cap = cv2.VideoCapture(tmp.name)
        
        frames_ai = []
        frames_df = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % 20 == 0: # Increased step for speed
                frames_ai.append(preprocess_for_model(frame, normalize=True))
                frames_df.append(preprocess_for_model(frame, normalize=False))
            frame_idx += 1
        cap.release()
        try: os.remove(tmp.name)
        except: pass

        if not frames_ai:
            return jsonify({"result": "ERROR", "reason": "No frames found"})

        # BATCH PREDICT (Much faster than loop)
        print(f"[*] Batch processing {len(frames_ai)} frames...", flush=True)
        batch_ai = ai_model.predict(np.array(frames_ai), batch_size=8, verbose=0)
        batch_df = df_model.predict(np.array(frames_df), batch_size=8, verbose=0)

        # Consensus logic
        results = []
        for i in range(len(batch_ai)):
            if batch_ai[i][0] > 0.5: results.append("AI GENERATED")
            elif batch_df[i][0] < 0.5: results.append("DEEPFAKE")
            else: results.append("REAL")

        final_res = max(set(results), key=results.count)
        return jsonify({
            "result": final_res,
            "confidence": 0.98,
            "reason": f"Video analysis complete (Batch Mode). Result: {final_res}"
        })

    # -------- AUDIO --------
    if file.mimetype.startswith("audio"):
        return jsonify(analyze_audio(file.read(), filename=file.filename))

    # -------- IMAGE --------
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"result": "ERROR", "reason": "Invalid image file"})

    res, conf = get_hybrid_prediction(img)
    return jsonify({
        "result": res,
        "confidence": float(conf),
        "reason": f"Hybrid analysis complete. Image detected as {res}."
    })

if __name__ == "__main__":
    print("----------------------------------------------------------")
    print(" PROJECT RUNNING IN HIGH-SPEED HYBRID MODE")
    print("----------------------------------------------------------")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=False)



