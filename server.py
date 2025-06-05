from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import os
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and scaler
model = tf.keras.models.load_model("ddos_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Function to block an IP on Windows
def block_ip(ip_address):
    try:
        if not ip_address:
            return "Invalid IP address."

        # Check if the IP is already blocked
        check_command = ["netsh", "advfirewall", "firewall", "show", "rule", f"name=Block_{ip_address}"]
        result = subprocess.run(check_command, capture_output=True, text=True)
        
        if "No rules match the specified criteria" not in result.stdout:
            return f"IP {ip_address} is already blocked."

        # Block the IP
        block_command = ["netsh", "advfirewall", "firewall", "add", "rule",
                         f"name=Block_{ip_address}", "dir=in", "action=block", f"remoteip={ip_address}"]
        subprocess.run(block_command, check=False)

        return f"IP {ip_address} blocked successfully."

    except Exception as e:
        return f"Failed to block IP: {ip_address}. Error: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ensure correct input key format
        ip_address = data.get("ip") or data.get("ip_address")
        source_port = data.get("source_port") or data.get("Source Port")
        protocol = data.get("protocol") or data.get("Protocol")
        flow_duration = data.get("flow_duration") or data.get("Flow Duration")
        total_fwd_packets = data.get("total_fwd_packets") or data.get("Total Fwd Packets")
        total_backward_packets = data.get("total_backward_packets") or data.get("Total Backward Packets")

        # Validate input
        if None in [ip_address, source_port, protocol, flow_duration, total_fwd_packets, total_backward_packets]:
            return jsonify({"error": "Missing required input fields."}), 400

        # Prepare features for prediction
        features = np.array([[source_port, protocol, flow_duration, total_fwd_packets, total_backward_packets]])
        features_scaled = scaler.transform(features).reshape(1, 1, -1)  # Reshape for LSTM

        # Predict
        prediction = model.predict(features_scaled)[0][0]
        label = int(prediction > 0.5)  # Convert probability to binary label (1 = Attack, 0 = Benign)

        # Response
        response = {"ip_address": ip_address, "prediction": label}

        # Block IP if an attack is detected
        if label == 1:
            response["action"] = block_ip(ip_address)
        else:
            response["action"] = "No IP blocked."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
