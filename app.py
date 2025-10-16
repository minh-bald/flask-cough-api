from flask import Flask, request, render_template_string
import numpy as np
import tensorflow as tf
import threading
import io
import soundfile as sf   # for WAV decoding

app = Flask(__name__)
interpreter = tf.lite.Interpreter(model_path="tflite-model/cough_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
expected_length = input_details[0]['shape'][1]

STEP_SIZE = expected_length // 2  
labels = ["Healthy Cough", "Abnormal", "COVID"]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Prediction Result</title></head>
<body>
  <h2> Sliding Window Cough Detection</h2>
  <p><strong>Latest Prediction:</strong> {{ conclusion }}</p>
  <p>Session ID: {{ session_id }} ({{ received_chunks }}/{{ total_chunks }} chunks received)</p>
</body>
</html>
"""

latest_html = "<p>No prediction yet.</p>"
session_audio_buffers = {}
session_locks = threading.Lock()

@app.route('/')
def index():
    return """
    <h3> Flask API is running.</h3>
    <p>Send audio as JSON PCM chunks, multipart WAV file, or raw audio/wav in the body.</p>
    <p>Visit <a href='/latest'>/latest</a> to see the last prediction.</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    global latest_html

    try:
        # Case 1: JSON PCM chunks (Arduino-style)
        if request.is_json:
            data = request.get_json()
            if not data or 'audio' not in data:
                return "❌ JSON must contain 'audio'", 400

            session_id = str(data.get("session_id", "default"))
            chunk_id = int(data.get("chunk_id", 0))
            total_chunks = int(data.get("total_chunks", 100))
            audio = np.array(data['audio'], dtype=np.uint8).astype(np.float32) / 255.0

        # Case 2: WAV file upload (multipart/form-data)
        elif "file" in request.files:
            file = request.files["file"]
            wav_bytes = io.BytesIO(file.read())
            audio, samplerate = sf.read(wav_bytes, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            session_id = "wav_upload"
            chunk_id, total_chunks = 0, 1

        # Case 3: Raw WAV stream (Content-Type: audio/wav)
        elif request.content_type and "audio/wav" in request.content_type:
            wav_bytes = io.BytesIO(request.data)
            audio, samplerate = sf.read(wav_bytes, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            session_id = "wav_stream"
            chunk_id, total_chunks = 0, 1

        else:
            return "Request must contain JSON, multipart WAV, or raw audio/wav body", 400

        # Store into buffer
        with session_locks:
            if session_id not in session_audio_buffers:
                session_audio_buffers[session_id] = {
                    "buffer": np.array([], dtype=np.float32),
                    "received": 0,
                    "total": total_chunks
                }

            sess = session_audio_buffers[session_id]
            sess["buffer"] = np.concatenate([sess["buffer"], audio])
            sess["received"] += 1

            # Sliding window inference
            while len(sess["buffer"]) >= expected_length:
                window = sess["buffer"][:expected_length]
                sess["buffer"] = sess["buffer"][STEP_SIZE:]

                input_tensor = np.expand_dims(window, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]

                prediction = dict(zip(labels, map(float, output)))
                top_label = max(prediction, key=prediction.get)
                conclusion = top_label

                rendered = render_template_string(
                    HTML_TEMPLATE,
                    conclusion=conclusion,
                    session_id=session_id,
                    received_chunks=sess["received"],
                    total_chunks=total_chunks
                )
                latest_html = rendered

            if sess["received"] >= sess["total"]:
                del session_audio_buffers[session_id]

        if conclusion:
            return f"Received audio for session {session_id} — Prediction: {conclusion}"
        else:
            return f"Received audio for session {session_id}, but not enough samples for prediction yet"

    except Exception as e:
        return f"Error: {e}", 500

@app.route('/latest')
def latest():
    return latest_html

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
