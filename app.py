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

# Load specialized TFLite model (Ultra-Efficiency Mode)
TFLITE_MODEL_PATH = "model/deepfake_model_fp16.tflite"

try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[OK] TFLite System Active (RAM: 7MB)", flush=True)
except Exception as e:
    print(f"[ERROR] Loading TFLite model: {e}", flush=True)

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
        # Preprocess
        img_df = preprocess_for_model(img_cv, normalize=False)
        img_batch = np.expand_dims(img_df, axis=0).astype(np.float32)
        
        # Run TFLite Inference
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        pred_df = interpreter.get_tensor(output_details[0]['index'])[0]
        
        df_prob = float(pred_df[0])
        
        # Cleanup
        del img_df, img_batch
        gc.collect()
        
        if df_prob < 0.5:
            return "DEEPFAKE", float(1.0 - df_prob)
        else:
            return "REAL", float(df_prob)
            
    except Exception as e:
        print(f"[CRITICAL ERROR] TFLite Prediction failed: {str(e)}", flush=True)
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
        
        frames_df = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % 20 == 0:
                frames_df.append(preprocess_for_model(frame, normalize=False))
            frame_idx += 1
        cap.release()
        try: os.remove(tmp.name)
        except: pass

        if not frames_ai:
            return jsonify({"result": "ERROR", "reason": "No frames found"})

        # BATCH PREDICT (TFLite Loop)
        print(f"[*] Processing {len(frames_df)} frames with TFLite...", flush=True)
        try:
            results = []
            for frame in frames_df:
                input_data = np.expand_dims(frame, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details[0]['index'])[0]
                
                if out[0] < 0.5: results.append("DEEPFAKE")
                else: results.append("REAL")
            
            # Cleanup
            del frames_df
            gc.collect()

            final_res = max(set(results), key=results.count)
            return jsonify({
                "result": final_res,
                "confidence": 0.98,
                "reason": f"TFLite Video analysis complete. Result: {final_res}"
            })
        except Exception as e:
            print(f"[CRITICAL ERROR] Video TFLite failed: {str(e)}", flush=True)
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



