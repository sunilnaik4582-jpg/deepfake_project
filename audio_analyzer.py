import librosa
import numpy as np
import pickle
import tempfile
import os

# ─── Load Trained Model (if available) ───────────────────────────
MODEL_PATH = "model/audio_model.pkl"
_trained   = None

def _load_trained_model():
    global _trained
    if _trained is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _trained = pickle.load(f)
        print(f"[AudioAnalyzer] Trained model loaded from {MODEL_PATH}")
    return _trained

# ─── Feature Extraction (same as audio_train.py) ─────────────────
def _extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std  = np.std(mfccs, axis=1)

    spec_centroid  = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spec_rolloff   = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spec_flatness  = np.mean(librosa.feature.spectral_flatness(y=y))
    zcr            = np.mean(librosa.feature.zero_crossing_rate(y))
    rms            = np.mean(librosa.feature.rms(y=y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(np.asarray(tempo).flat[0])

    return np.concatenate([
        mfcc_mean, mfcc_std,
        [spec_centroid, spec_bandwidth, spec_rolloff,
         spec_flatness, zcr, rms, tempo_val]
    ])

# ─── Main Analyze Function ────────────────────────────────────────
def analyze_audio(file_bytes, filename="audio.wav"):
    suffix = os.path.splitext(filename)[-1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()

    try:
        y, sr = librosa.load(tmp.name, sr=22050, mono=True, duration=30)
    except Exception as e:
        return {"result": "ERROR", "confidence": 0.0,
                "reason": f"Audio load failed: {str(e)}"}
    finally:
        os.unlink(tmp.name)

    if len(y) < sr * 0.5:
        return {"result": "ERROR", "confidence": 0.0,
                "reason": "Audio too short (min 0.5 seconds)"}

    trained = _load_trained_model()

    # ── PATH A: Trained ML Model ──────────────────────────────────
    if trained:
        features = _extract_features(y, sr).reshape(1, -1)
        features_scaled = trained["scaler"].transform(features)
        proba = trained["model"].predict_proba(features_scaled)[0]
        fake_prob = float(proba[0])   # index 0 = FAKE
        real_prob = float(proba[1])   # index 1 = REAL

        if real_prob >= 0.5:
            return {
                "result": "REAL",
                "confidence": round(real_prob, 3),
                "reason": f"Real human voice detected (confidence: {real_prob*100:.1f}%)"
            }
        else:
            return {
                "result": "FAKE",
                "confidence": round(fake_prob, 3),
                "reason": f"AI-generated audio detected (confidence: {fake_prob*100:.1f}%)"
            }

    # ── PATH B: Heuristic Fallback (no trained model) ─────────────
    mfcc_var      = float(np.mean(np.var(
                        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr           = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    suspicion = 0.0
    reasons   = []

    if mfcc_var < 30:
        suspicion += 0.45
        reasons.append(f"Very uniform voice pattern (MFCC: {mfcc_var:.1f})")
    elif mfcc_var < 60:
        suspicion += 0.20

    if spec_flatness > 0.08:
        suspicion += 0.30
        reasons.append(f"Unnatural spectrum (flatness: {spec_flatness:.4f})")
    elif spec_flatness > 0.05:
        suspicion += 0.15

    if zcr < 0.01 or zcr > 0.20:
        suspicion += 0.20
        reasons.append(f"Abnormal ZCR: {zcr:.4f}")

    suspicion = min(suspicion, 1.0)

    if suspicion >= 0.45:
        return {
            "result": "FAKE",
            "confidence": round(suspicion, 3),
            "reason": "[Heuristic] " + ("; ".join(reasons) or "Suspicious audio patterns")
        }
    return {
        "result": "REAL",
        "confidence": round(1.0 - suspicion, 3),
        "reason": f"[Heuristic] Natural voice (no model trained yet — run audio_train.py for accuracy)"
    }
