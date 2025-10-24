from flask import Flask, request, render_template_string
import numpy as np
import tensorflow as tf
import threading
import io
import soundfile as sf
import os

app = Flask(__name__)

# Load model
interpreter = tf.lite.Interpreter(model_path="tflite-model/cough_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

print(f"✅ Model input details: {input_details}")

# labels
labels = ["Healthy Cough", "Abnormal", "COVID"]

# HTML
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Prediction Result</title></head>
<body>
  <h2> Cough Detection Result </h2>
  <p><strong>Latest Prediction:</strong> {{ conclusion }}</p>
  <p>Session ID: {{ session_id }}</p>
</body>
</html>
"""

latest_html = "<p>No prediction yet.</p>"
session_locks = threading.Lock()

# ---- Helper function: preprocess input ----
def preprocess_audio(audio):
    """Normalize and reshape raw audio to match model input [1,1,224,224]."""
    audio = np.array(audio, dtype=np.float32)

    # normalize between 0 and 1
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # resize/pad to 224*224 = 50176 samples
    target_len = 224 * 224
    if len(audio) < target_len:
        pad = np.zeros(target_len - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad])
    elif len(audio) > target_len:
        audio = audio[:target_len]

    # reshape to [1,1,224,224]
    audio = audio.reshape(1, 1, 224, 224).astype(np.float32)
    return audio

# ---- Routes ----

@app.route('/')
def index():
    return """
    <h3> Flask API is running ✅</h3>
    <p>POST audio to /predict (JSON or WAV). Visit <a href='/latest'>/latest</a> for last result.</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    global latest_html

    try:
        # --- Case 1: Arduino JSON ---
        if request.is_json:
            data = request.get_json()
            if not data or 'audio' not in data:
                return "❌ JSON must contain 'audio'", 400

            session_id = str(data.get("session_id", "default"))
            audio = np.array(data['audio'], dtype=np.uint8).astype(np.float32) / 255.0

        # --- Case 2: File upload (WAV) ---
        elif "file" in request.files:
            file = request.files["file"]
            wav_bytes = io.BytesIO(file.read())
            audio, samplerate = sf.read(wav_bytes, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            session_id = "wav_upload"

        # --- Case 3: Raw WAV stream ---
        elif request.content_type and "audio/wav" in request.content_type:
            wav_bytes = io.BytesIO(request.data)
            audio, samplerate = sf.read(wav_bytes, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            session_id = "wav_stream"

        else:
            return "Request must contain JSON, multipart WAV, or raw audio/wav body", 400

        # preprocess & predict
        input_tensor = preprocess_audio(audio)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # get top label
        prediction = dict(zip(labels, map(float, output)))
        top_label = max(prediction, key=prediction.get)
        conclusion = top_label

        rendered = render_template_string(
            HTML_TEMPLATE,
            conclusion=conclusion,
            session_id=session_id
        )
        latest_html = rendered

        return f" Prediction for session '{session_id}': {conclusion}"

    except Exception as e:
        return f"Error: {e}", 500

@app.route('/latest')
def latest():
    return latest_html

# ---- Run server ----
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
