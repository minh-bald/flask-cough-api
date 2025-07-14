from flask import Flask, request, render_template_string
import numpy as np
import tensorflow as tf
import threading

app = Flask(__name__)
interpreter = tf.lite.Interpreter(model_path="tflite-model/tflite_learn_6.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
expected_length = input_details[0]['shape'][1]

# Labels for 4 classes
labels = ["Anomaly", "COVID-19", "Healthy", "Non-Cough"]

# HTML template (no percentage display)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Prediction Result</title></head>
<body>
  <h2>🧪 Full Session Cough Detection</h2>
  <p><strong>Prediction:</strong> {{ conclusion }}</p>
  <p>Session ID: {{ session_id }} ({{ total_chunks }} chunks received)</p>
</body>
</html>
"""

latest_html = "<p>No full prediction received yet.</p>"
session_audio_buffers = {}
session_locks = threading.Lock()

@app.route('/')
def index():
    return """
    <h3>✅ Flask API is running.</h3>
    <p>POST JSON with: session_id, chunk_id, total_chunks, and audio array</p>
    <p>Example payload:</p>
    <pre>{
  "session_id": "ABC123",
  "chunk_id": 0,
  "total_chunks": 100,
  "audio": [128, 130, 125, ...]
}</pre>
    <p>Visit <a href='/latest'>/latest</a> for the last full result.</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    global latest_html

    try:
        data = request.get_json()
        if not data or 'audio' not in data:
            return "❌ JSON must contain 'audio'", 400

        session_id = str(data.get("session_id", "default"))
        chunk_id = int(data.get("chunk_id", 0))
        total_chunks = int(data.get("total_chunks", 100))
        audio = np.array(data['audio'], dtype=np.uint8).astype(np.float32) / 255.0

        with session_locks:
            if session_id not in session_audio_buffers:
                session_audio_buffers[session_id] = [None] * total_chunks
            session_audio_buffers[session_id][chunk_id] = audio

            # Check if all chunks are received
            if all(chunk is not None for chunk in session_audio_buffers[session_id]):
                full_signal = np.concatenate(session_audio_buffers[session_id])
                del session_audio_buffers[session_id]  # Free memory

                # Pad or crop to match model input size
                if len(full_signal) > expected_length:
                    full_signal = full_signal[:expected_length]
                elif len(full_signal) < expected_length:
                    full_signal = np.pad(full_signal, (0, expected_length - len(full_signal)), mode='constant')

                # Model inference
                input_tensor = np.expand_dims(full_signal, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]

                prediction = dict(zip(labels, map(float, output)))
                top_label = max(prediction, key=prediction.get)
                conclusion = top_label  # Final result

                rendered = render_template_string(
                    HTML_TEMPLATE,
                    prediction=prediction,
                    conclusion=conclusion,
                    session_id=session_id,
                    total_chunks=total_chunks
                )
                latest_html = rendered
                return rendered

        return f"✅ Chunk {chunk_id + 1}/{total_chunks} received for session {session_id}"

    except Exception as e:
        return f"❌ Error: {e}", 500

@app.route('/latest')
def latest():
    return latest_html

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
