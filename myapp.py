import gradio as gr
import joblib
import numpy as np
import psutil  # For resource metrics
import time
from prometheus_client import start_http_server, Counter, Histogram
# Load your trained model (update the path to your trained model file)
model = joblib.load("db_model.joblib")

# Define a function for prediction
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, DiabetesPedigreeFunction, Age):
    # Convert inputs to a NumPy array
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, DiabetesPedigreeFunction, Age]])

    # Record the start time for metrics
    start_time = time.time()

    # Predict using the loaded model
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Record the end time and calculate prediction latency
    end_time = time.time()
    latency = end_time - start_time

    # Update Prometheus metrics
    PROMETHEUS_COUNTER.inc()
    PROMETHEUS_LATENCY.observe(latency)

    # Return prediction and probability
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return f"Prediction: {result}, Probability of Diabetes: {probability:.2f}"

# Prometheus metrics setup
from prometheus_client import start_http_server, Counter, Histogram

# Metrics
PROMETHEUS_COUNTER = Counter('gradio_prediction_requests', 'Total number of prediction requests')
PROMETHEUS_LATENCY = Histogram('gradio_prediction_latency_seconds', 'Time taken to process prediction request')

# Create a Gradio interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="No. Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age"),
    ],
    outputs="text",
    title="Diabetes Prediction",
    description="Enter the following health parameters to predict the likelihood of diabetes."
)

# Start Prometheus server to collect metrics
start_http_server(8000)  # Start the Prometheus HTTP server on port 8000

# Launch the Gradio app
interface.launch(server_name="0.0.0.0", server_port=7860)
