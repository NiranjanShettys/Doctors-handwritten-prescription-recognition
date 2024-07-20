import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from symspellpy.symspellpy import SymSpell, Verbosity
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import time

# Load the dataset
def load_drug_names(file_path):
    try:
        df = pd.read_csv(file_path)
        st.write("CSV Columns:", df.columns.tolist())  # Debugging line to print column names
        if 'drug_names' in df.columns:
            drug_names = df['drug_names'].dropna().tolist()  # Drop NaN values
            return [name.lower() for name in drug_names]
        else:
            st.error("Column 'drug_names' not found in the CSV file. Please check the column names.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

drug_names = load_drug_names('drug_names.csv')

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to get embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Check if precomputed embeddings exist
try:
    with open('drug_embeddings.pkl', 'rb') as f:
        drug_embeddings = pickle.load(f)
except FileNotFoundError:
    # Get embeddings for all drug names
    embeddings = [get_embeddings(name) for name in drug_names]
    drug_embeddings = torch.vstack(embeddings)

    # Save embeddings for future use
    with open('drug_embeddings.pkl', 'wb') as f:
        pickle.dump(drug_embeddings, f)

# Spell correction setup
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.create_dictionary_entry("drug_name", 1)
for name in drug_names:
    sym_spell.create_dictionary_entry(name, 1)

# Prediction function
def predict_drug_name(input_text):
    input_text = input_text.lower().strip()  # Ensure text is stripped of extra spaces
    input_embedding = get_embeddings(input_text)
    
    # Correct spelling if necessary
    suggestions = sym_spell.lookup(input_text, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        corrected_text = suggestions[0].term
        input_embedding = get_embeddings(corrected_text)
    else:
        corrected_text = input_text
    
    # Calculate similarity
    similarities = cosine_similarity(input_embedding.numpy(), drug_embeddings.numpy())
    best_match_index = np.argmax(similarities)
    predicted_drug_name = drug_names[best_match_index]
    
    st.write(f"Input Text: {input_text}")
    st.write(f"Corrected Text: {corrected_text}")
    return predicted_drug_name

# Streamlit app
st.title("Doctor's Handwritten Prescription Prediction")

# Single input prediction
input_text = st.text_input("Enter the partial or misspelled drug name:")
if st.button("Predict"):
    if input_text:
        predicted_drug_name = predict_drug_name(input_text)
        st.write(f"Predicted Drug Name: {predicted_drug_name}")
    else:
        st.write("Please enter a drug name to predict.")
