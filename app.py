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

# Sliding window step (hop size in samples)
# e.g. 50% overlap → step = expected_length // 2
STEP_SIZE = expected_length // 2  

# Labels for 3 classes
labels = ["Anomalies", "COVID", "Healthy Cough"]

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Prediction Result</title></head>
<body>
  <h2>🧪 Sliding Window Cough Detection</h2>
  <p><strong>Latest Prediction:</strong> {{ conclusion }}</p>
  <p>Session ID: {{ session_id }} ({{ received_chunks }}/{{ total_chunks }} chunks received)</p>
</body>
</html>
"""

# Shared state
latest_html = "<p>No prediction yet.</p>"
session_audio_buffers = {}
session_locks = threading.Lock()

@app.route('/')
def index():
    return """
    <h3>✅ Flask API is running.</h3>
    <p>Send 100 JSON chunks with: session_id, chunk_id, total_chunks, and audio array</p>
    <p>Visit <a href='/latest'>/latest</a> to see the last prediction.</p>
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
                session_audio_buffers[session_id] = {
                    "buffer": np.array([], dtype=np.float32),
                    "received": 0,
                    "total": total_chunks
                }

            sess = session_audio_buffers[session_id]
            sess["buffer"] = np.concatenate([sess["buffer"], audio])
            sess["received"] += 1

            # Run sliding window inference
            while len(sess["buffer"]) >= expected_length:
                window = sess["buffer"][:expected_length]
                sess["buffer"] = sess["buffer"][STEP_SIZE:]  # slide

                # Model inference
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
                latest_html = rendered  # update global result

            # If session completed → free memory
            if sess["received"] >= sess["total"]:
                del session_audio_buffers[session_id]

        return f"✅ Chunk {chunk_id + 1}/{total_chunks} received for session {session_id}"

    except Exception as e:
        return f"❌ Error: {e}", 500

@app.route('/latest')
def latest():
    return latest_html

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
