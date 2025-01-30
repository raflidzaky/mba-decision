from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import pickle
import asyncio

# TODO: Update the model (making sure it is correcly formated)

# Initialize API app
app = FastAPI()

# Define request body scheme with Pydantic
class PredictRequest(BaseModel):
    features: list

# Make the model asynchronous (can be handled concurrently)
async def load_model():
    # Load the model asynchronously to avoid blocking other requests (such as GET request for feature or prediction)
     return await asyncio.to_thread(pickle.load, open('C:/Users/Rafli/decision-pursue-mba/model_dtree.pkl', 'rb'))

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
@app.get("/")
def read_root():
    return {"message": "Welcome to the API app!"}

@app.post("/predict")
def predict(request: PredictRequest):
    # Get the features from the request
    features = np.array(request.features).reshape(1, -1)  # Convert list to numpy array and reshape for prediction
    
    # Make the prediction using the model
    prediction = ml_model.predict(features)
    
    # Return the prediction in a JSON response
    return {"prediction": prediction[0]}