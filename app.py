import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from symspellpy.symspellpy import SymSpell, Verbosity
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
df = pd.read_csv('drug_names.csv')
drug_names = df['drug_names'].tolist()

# Preprocess the drug names
drug_names = [name.lower() for name in drug_names]

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Get embeddings for all drug names
drug_embeddings = torch.vstack([get_embeddings(name) for name in drug_names])

# Spell correction setup
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.create_dictionary_entry("drug_name", 1)
for name in drug_names:
    sym_spell.create_dictionary_entry(name, 1)

# Prediction function
def predict_drug_name(input_text):
    input_text = input_text.lower()
    input_embedding = get_embeddings(input_text)
    
    # Correct spelling if necessary
    suggestions = sym_spell.lookup(input_text, Verbosity.CLOSEST, max_edit_distance=2)
    if suggestions:
        input_text = suggestions[0].term
        input_embedding = get_embeddings(input_text)
    
    # Calculate similarity
    similarities = cosine_similarity(input_embedding, drug_embeddings)
    best_match_index = np.argmax(similarities)
    return drug_names[best_match_index]

# Batch testing function
def test_model(test_file):
    test_df = pd.read_csv(test_file)
    correct_predictions = 0
    
    for index, row in test_df.iterrows():
        predicted_drug_name = predict_drug_name(row['input_text'])
        if predicted_drug_name == row['correct_drug_name'].lower():  # Ensure case insensitivity
            correct_predictions += 1
    
    accuracy = (correct_predictions / len(test_df)) * 100
    return accuracy

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

# Batch testing
st.header("Batch Testing")
uploaded_file = st.file_uploader("Choose a CSV file for batch testing", type="csv")
if uploaded_file is not None:
    st.write("Uploaded file preview:")
    test_df = pd.read_csv(uploaded_file)
    st.write(test_df.head())
    
    if st.button("Start Batch Testing"):
        accuracy = test_model(uploaded_file)
        st.write(f"Accuracy: {accuracy:.2f}%")

