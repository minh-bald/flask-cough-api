from flask import Flask, request, render_template_string, jsonify
import numpy as np
import tensorflow as tf
import librosa
import io
import soundfile as sf
import threading
import os
import json

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="tflite-model/cough_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ["Abnormal", "Healthy", "Covid"]

TARGET_SR = 16000
TARGET_DURATION = 3.6
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
N_MELS = 128

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Prediction Result</title>
  <style>
    body { font-family: Arial; background: #0d1b2a; color: #e0e1dd; padding: 20px; }
    h2 { color: #00b4d8; }
    pre { background: #1b263b; color: #e0e1dd; padding: 10px; border-radius: 10px; }
  </style>
</head>
<body>
  <h2>🔍 Cough Detection Result</h2>
  <p><strong>Latest Prediction:</strong> {{ conclusion }}</p>
  <p><strong>Session ID:</strong> {{ session_id }}</p>
  <h3>Raw Model Output:</h3>
  <pre>{{ debug_output }}</pre>
</body>
</html>
"""

latest_html = "<p>No prediction yet.</p>"
session_locks = threading.Lock()


def preprocess_audio(audio, sr):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    else:
        audio = audio[:TARGET_SAMPLES]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=2048,
        hop_length=256,
        n_mels=N_MELS,
        fmin=20,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_resized = librosa.util.fix_length(mel_db, size=224, axis=1)
    if mel_resized.shape[0] < 224:
        mel_resized = np.pad(mel_resized, ((0, 224 - mel_resized.shape[0]), (0, 0)))
    elif mel_resized.shape[0] > 224:
        mel_resized = mel_resized[:224, :]
    mel_norm = (mel_resized - mel_resized.min()) / (mel_resized.max() - mel_resized.min() + 1e-9)
    mel_input = mel_norm.reshape(1, 1, 224, 224).astype(np.float32)
    return mel_input


@app.route('/')
def index():
    return """
    <h3>Flask API is running ✅</h3>
    <p>POST audio to /predict (JSON or WAV). Visit <a href='/latest'>/latest</a> for last result.</p>
    """


@app.route('/predict', methods=['POST'])
def predict():
    global latest_html
    try:
        if request.is_json:
            data = request.get_json()
            if not data or 'audio' not in data:
                return "❌ JSON must contain 'audio'", 400
            session_id = str(data.get("session_id", "default"))
            audio = np.array(data['audio'], dtype=np.uint8).astype(np.float32) / 255.0
            sr = TARGET_SR
        elif "file" in request.files:
            file = request.files["file"]
            wav_bytes = io.BytesIO(file.read())
            audio, sr = sf.read(wav_bytes, dtype="float32")
            session_id = "wav_upload"
        elif request.content_type and "audio/wav" in request.content_type:
            wav_bytes = io.BytesIO(request.data)
            audio, sr = sf.read(wav_bytes, dtype="float32")
            session_id = "wav_stream"
        else:
            return "Request must contain JSON, multipart WAV, or raw audio/wav body", 400

        input_tensor = preprocess_audio(audio, sr)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        prediction = dict(zip(labels, map(float, output)))
        top_label = max(prediction, key=prediction.get)
        conclusion = top_label
        debug_output = json.dumps(prediction, indent=2)

        rendered = render_template_string(
            HTML_TEMPLATE,
            conclusion=conclusion,
            session_id=session_id,
            debug_output=debug_output
        )
        latest_html = rendered

        return jsonify({
            "session_id": session_id,
            "prediction": top_label,
            "raw_output": prediction
        })
    except Exception as e:
        return f"Error: {e}", 500


@app.route('/latest')
def latest():
    return latest_html


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

