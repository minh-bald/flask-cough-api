from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)
tflite = tf.lite
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="tflite-model/tflite_learn_4.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape (e.g. [1, 4116])
input_shape = input_details[0]['shape']
expected_length = input_shape[1]

@app.route('/')
def index():
    return "✅ Flask API is running. Use POST /predict"
@app.route('/predict', methods=['GET'])
def predict_info():
    return "✅ Use POST /predict with JSON data."
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        audio = np.array(data['audio'], dtype=np.uint8)
        signal = audio.astype(np.float32) / 255.0  # Normalize

        # Crop or pad to expected input length
        if len(signal) > expected_length:
            signal = signal[:expected_length]
        elif len(signal) < expected_length:
            signal = np.pad(signal, (0, expected_length - len(signal)), mode='constant')

        # Prepare input tensor
        input_tensor = np.expand_dims(signal, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        return jsonify({"prediction": output.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
