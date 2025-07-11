from flask import Flask, request, render_template_string
import numpy as np
import tensorflow as tf

app = Flask(__name__)
interpreter = tf.lite.Interpreter(model_path="tflite-model/tflite_learn_4.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
expected_length = input_shape[1]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Prediction Result</title></head>
<body>
  <h2>🤖 Cough Detection Result</h2>
  <p><strong>{{ conclusion }}</strong></p>
  <ul>
    {% for label, score in prediction.items() %}
      <li>{{ label }}: {{ '%.2f' % (score * 100) }}%</li>
    {% endfor %}
  </ul>
</body>
</html>
"""

# Store latest result for browser viewing
latest_html = "<p>No prediction received yet.</p>"

@app.route('/')
def index():
    return """
    <h3>✅ Flask API is running. Use POST /predict with JSON audio data (array of 0–255 uint8 values).</h3>
    <p>Try sending POST using curl, Arduino WiFi, etc.</p>
    <p>Visit <a href='/latest'>/latest</a> to see the most recent result.</p>
    """

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global latest_html

    if request.method == 'GET':
        return """
        <h3>✅ Use POST /predict with JSON audio data (array of 0–255 uint8 values).</h3>
        <p>Try sending POST using curl or Arduino WiFi</p>
        """

    try:
        data = request.get_json()
        if not data or 'audio' not in data:
            return "❌ JSON must contain 'audio' key.", 400

        audio = np.array(data['audio'], dtype=np.uint8)
        signal = audio.astype(np.float32) / 255.0  # Normalize

        # Adjust signal to match model input
        if len(signal) > expected_length:
            signal = signal[:expected_length]
        elif len(signal) < expected_length:
            signal = np.pad(signal, (0, expected_length - len(signal)), mode='constant')

        input_tensor = np.expand_dims(signal, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        labels = ["Anomaly", "COVID-19", "Non-Cough"]
        prediction = dict(zip(labels, map(float, output)))
        top_label = max(prediction, key=prediction.get)
        conclusion = f"🟢 Likely: {top_label}"

        rendered = render_template_string(HTML_TEMPLATE, prediction=prediction, conclusion=conclusion)
        latest_html = rendered  # Update global latest result
        return rendered

    except Exception as e:
        return f"❌ Error: {e}", 500

@app.route('/latest')
def latest():
    return latest_html

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
