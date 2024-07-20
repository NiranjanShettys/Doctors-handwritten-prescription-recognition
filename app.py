# app.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import pandas as pd
from PIL import Image
import io
import torch
from torchvision import transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from models import CRNN  # Replace with your actual CRNN model import

app = FastAPI()

# Load drug names and create a TF-IDF model
drug_names = pd.read_csv('drug_names.csv')['drug_name'].tolist()
vectorizer = TfidfVectorizer()
drug_name_vectors = vectorizer.fit_transform(drug_names)

# Load pre-trained CRNN model
model = CRNN()  # Replace with your CRNN model initialization
model.load_state_dict(torch.load('crnn_model.pth'))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class Prediction(BaseModel):
    predicted_drug_name: str

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
    
    # Convert output to text (assuming you have a decoding function)
    recognized_text = decode_output(output)  # Replace with actual decoding logic
    
    # Predict drug name
    input_vector = vectorizer.transform([recognized_text])
    similarities = cosine_similarity(input_vector, drug_name_vectors)
    best_match_idx = np.argmax(similarities)
    predicted_drug_name = drug_names[best_match_idx]

    return Prediction(predicted_drug_name=predicted_drug_name)

def decode_output(output):
    # Add your decoding logic here
    return "Sample Drug Name"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
