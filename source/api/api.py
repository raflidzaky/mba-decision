# Manage API input and response
from fastapi import FastAPI
import asyncio
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Manage input shape
from pydantic import BaseModel
import numpy as np
import pickle

# Manage environment variables
from dotenv import load_dotenv
import os

# Initialize API app
app = FastAPI()

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
API_URL = os.getenv("API_URL")

# Serve static files (index.html, JS, CSS, etc.)
app.mount("/static", StaticFiles(directory="source/static"), name="static")

# Define request body scheme with Pydantic
class PredictRequest(BaseModel):
    features: list

# Make the model asynchronous (can be handled concurrently)
async def load_model():
    # Load the model asynchronously to avoid blocking other requests (such as GET request for feature or prediction)
     return await asyncio.to_thread(pickle.load, open(MODEL_PATH, 'rb'))

# Initialized model as None
ml_model = None

# Start the server
# It loads the model as the server starts
@app.on_event("startup")
async def startup_event():
    global ml_model
    # Load the model during the app startup
    ml_model = await load_model()

# Say hello for all newcomers in API app
# But in form of HTML 
# So, the app needs to read the HTML file
@app.get("/", response_class=HTMLResponse)
def read_root():
    return open("source/static/index.html").read()

@app.get("/config")
def read_config():
    return {
        "model_path": MODEL_PATH,
        "api_url": API_URL
    }

@app.post("/predict")
def predict(request: PredictRequest):
    # Get the features from the request
    features = np.array(request.features).reshape(1, -1)  # Convert list to numpy array and reshape for prediction
    
    # Make the prediction using the model
    prediction = ml_model.predict(features)
    
    # Return the prediction in a JSON response
    return {"prediction": prediction[0]}