import gradio as gr
import joblib  # For loading the trained model
import numpy as np

# Load your trained model (update the path to your trained model file)
model = joblib.load("db_model.joblib")


# Define a function for prediction
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness,BMI, DiabetesPedigreeFunction, Age):
    # Convert inputs to a NumPy array
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,BMI, DiabetesPedigreeFunction, Age]])
    
    # Predict using the loaded model
    prediction = model.predict(input_data)[0]  # Assuming the model has a `predict` method
    probability = model.predict_proba(input_data)[0][1]  # Get the probability for class 1 (diabetes)
    
    # Return prediction and probability
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return f"Prediction: {result}, Probability of Diabetes: {probability:.2f}"

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

# Launch the Gradio app
interface.launch(server_name="0.0.0.0", server_port=7860)
