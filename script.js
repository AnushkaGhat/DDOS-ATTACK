async function predictAttack() {
    const data = {
        "ip_address": document.getElementById("ip_address").value,
        "Source Port": parseInt(document.getElementById("source_port").value),
        "Protocol": parseInt(document.getElementById("protocol").value),
        "Flow Duration": parseInt(document.getElementById("flow_duration").value),
        "Total Fwd Packets": parseInt(document.getElementById("total_fwd_packets").value),
        "Total Backward Packets": parseInt(document.getElementById("total_backward_packets").value)
    };

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", { // Make sure the URL matches your Flask server
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();

        document.getElementById("result").innerHTML = `
            <p><strong>Prediction:</strong> ${result.prediction === 1 ? "🚨 Attack - IP Blocked" : "✅ Benign - No Action Needed"}</p>
            <p><strong>Action:</strong> ${result.action}</p>
        `;

    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerHTML = `
            <p style='color:red;'>Error: ${error.message}</p>
        `;
    }
}
