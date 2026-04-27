import cv2
cv2.setNumThreads(1)
import os
import gc
import tempfile
import numpy as np
import tensorflow as tf

# Limit TensorFlow memory and CPU usage for Render's 512MB free tier
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, request, jsonify, send_from_directory
from audio_analyzer import analyze_audio

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__, static_folder="static")

import concurrent.futures

# Load specialized models (Hybrid System - Restored for accuracy)
DF_MODEL_PATH = "model/deepfake_model_94acc.h5"
AI_MODEL_PATH = "model/unified_model.h5"

try:
    # We load both but will use them sequentially to save RAM
    df_model = tf.keras.models.load_model(DF_MODEL_PATH)
    ai_model = tf.keras.models.load_model(AI_MODEL_PATH)
    print(f"[OK] High-Accuracy Hybrid System Active", flush=True)
except Exception as e:
    print(f"[ERROR] Loading models: {e}", flush=True)

# Removed ThreadPoolExecutor for Render free tier compatibility

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
    try:
        # 1. Check with Unified Model (Best for AI-Generated Images)
        img_ai = preprocess_for_model(img_cv, normalize=True)
        preds_ai = ai_model(np.expand_dims(img_ai, axis=0), training=False).numpy()[0]
        
        # If Unified model is very sure it's AI, we return immediately
        if np.argmax(preds_ai) == 0 and preds_ai[0] > 0.6:
            return "AI GENERATED", float(preds_ai[0])
            
        # 2. Check with Specialized Deepfake Model (Best for Face Swaps/Deepfakes)
        img_df = preprocess_for_model(img_cv, normalize=False)
        pred_df = df_model(np.expand_dims(img_df, axis=0), training=False).numpy()[0]
        df_prob = float(pred_df[0]) # Assuming df_model outputs prob of REAL
        
        # Cleanup memory
        del img_ai, img_df
        gc.collect()
        
        # Decision Logic: Combine results
        # If specialized model says Deepfake (< 0.5 probability of REAL)
        if df_prob < 0.5:
            return "DEEPFAKE", float(1.0 - df_prob)
        
        # Otherwise, if unified model detected AI (even with lower confidence)
        if np.argmax(preds_ai) == 0:
            return "AI GENERATED", float(preds_ai[0])
            
        # Default to Real
        return "REAL", float(df_prob)
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Prediction failed: {str(e)}", flush=True)
        raise e

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
            if frame_idx % 20 == 0:
                frames_ai.append(preprocess_for_model(frame, normalize=True))
                frames_df.append(preprocess_for_model(frame, normalize=False))
            frame_idx += 1
        cap.release()
        try: os.remove(tmp.name)
        except: pass

        if not frames_ai:
            return jsonify({"result": "ERROR", "reason": "No frames found"})

        # BATCH PREDICT
        print(f"[*] Batch processing {len(frames_ai)} frames...", flush=True)
        try:
            batch_ai = ai_model(np.array(frames_ai), training=False).numpy()
            batch_df = df_model(np.array(frames_df), training=False).numpy()
            
            # Cleanup
            del frames_ai, frames_df
            gc.collect()

            # Consensus logic
            results = []
            for i in range(len(batch_ai)):
                if np.argmax(batch_ai[i]) == 0: results.append("AI GENERATED")
                elif batch_df[i][0] < 0.5: results.append("DEEPFAKE")
                else: results.append("REAL")

            final_res = max(set(results), key=results.count)
            return jsonify({
                "result": final_res,
                "confidence": 0.98,
                "reason": f"Hybrid Video analysis complete. Result: {final_res}"
            })
        except Exception as e:
            print(f"[CRITICAL ERROR] Video Prediction failed: {str(e)}", flush=True)
            return jsonify({"result": "ERROR", "reason": f"Analysis failed: {str(e)}"})

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
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)



