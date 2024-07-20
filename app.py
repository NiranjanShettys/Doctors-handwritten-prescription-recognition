import streamlit as st
import pandas as pd
import numpy as np
import easyocr
import faiss
from transformers import BertTokenizer, BertModel
import torch

# Initialize EasyOCR and BERT model
ocr_reader = easyocr.Reader(['en'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    """Get BERT embedding for a given text."""
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def load_drug_names(drug_names_file):
    """Load drug names from a CSV file."""
    df = pd.read_csv(drug_names_file)
    if 'drug_name' in df.columns:
        return df['drug_name'].dropna().tolist()
    else:
        raise ValueError("CSV file must contain a column named 'drug_name'")

def build_index(drug_names):
    """Build FAISS index for drug names embeddings."""
    embeddings = np.array([get_embedding(name) for name in drug_names])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def predict_drug_name(predicted_text, index, drug_names):
    """Predict the most likely drug name from a list based on predicted text."""
    predicted_embedding = get_embedding(predicted_text)
    _, indices = index.search(predicted_embedding, k=5)
    matches = [(drug_names[i], _) for i in indices[0]]
    return matches

# Streamlit UI
st.title('Doctor\'s Handwritten Prescription Prediction')

# Upload drug names file
st.subheader('Upload Drug Names File')
drug_names_file = st.file_uploader("Choose a CSV file with drug names", type="csv")

drug_names = []
index = None
if drug_names_file is not None:
    try:
        # Load drug names from the uploaded file
        drug_names = load_drug_names(drug_names_file)
        index = build_index(drug_names)
        st.write("Drug names loaded successfully.")
        st.write(drug_names[:10])  # Show a preview of loaded drug names
    except Exception as e:
        st.error(f"Error loading drug names file: {e}")

# Input field for real-time handwriting input
st.subheader('Write Prescription on the Whiteboard')
st.write("Use your stylus to write the prescription below:")

# For demonstration purposes, we will use an uploaded image as a substitute for whiteboard input
uploaded_image = st.file_uploader("Choose an image file of handwritten text", type="png,jpg,jpeg")

if uploaded_image is not None and index is not None:
    try:
        # Use EasyOCR to extract text from the image
        image = uploaded_image.read()
        text = ocr_reader.readtext(image, detail=0, paragraph=False)
        extracted_text = ' '.join(text)
        st.write("Extracted Text:")
        st.write(extracted_text)

        if extracted_text:
            matches = predict_drug_name(extracted_text, index, drug_names)
            st.write("Top Matches:")
            for drug_name, score in matches:
                st.write(f"{drug_name}: {score:.2f}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
