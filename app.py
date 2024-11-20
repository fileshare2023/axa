from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from prometheus_client import Counter, start_http_server


# Initialize FastAPI
app = FastAPI()
REQUEST_COUNT = Counter('request_count', 'Total number of requests to generate text')
# Load the model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Start the Prometheus HTTP server
start_http_server(8001)
# Define the request body schema using Pydantic
class RequestBody(BaseModel):
    prompt: str

# Inference API endpoint
@app.post("/generate")
def generate_text(request_body: RequestBody):
    prompt = request_body.prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "Model is running"}
