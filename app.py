import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from symspellpy.symspellpy import SymSpell, Verbosity
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import time
from joblib import Parallel, delayed

# Load the dataset
def load_drug_names(file_path):
    try:
        df = pd.read_csv(file_path)
        st.write("CSV Columns:", df.columns.tolist())  # Debugging line to print column names
        if 'drug_name' in df.columns:
            drug_names = df['drug_name'].dropna().tolist()  # Drop NaN values
            return [name.lower() for name in drug_names]
        else:
            st.error("Column 'drug_name' not found in the CSV file. Please check the column names.")
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

# Batch embedding function
def get_batch_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Check if precomputed embeddings exist
try:
    with open('drug_embeddings.pkl', 'rb') as f:
        drug_embeddings = pickle.load(f)
except FileNotFoundError:
    # Get embeddings for all drug names in parallel
    num_cores = 8  # Adjust this based on your CPU cores
    embeddings = Parallel(n_jobs=num_cores)(delayed(get_embeddings)(name) for name in drug_names)
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

# Batch testing function
def test_model(test_file):
    try:
        test_df = pd.read_csv(test_file)
        st.write("Test CSV Columns:", test_df.columns.tolist())  # Debugging line to print column names
        if 'input_text' not in test_df.columns or 'correct_drug_name' not in test_df.columns:
            st.error("Test file must contain 'input_text' and 'correct_drug_name' columns.")
            return None
    except Exception as e:
        st.error(f"Error reading test CSV file: {e}")
        return None
    
    correct_predictions = 0
    batch_size = 32
    input_texts = test_df['input_text'].dropna().tolist()
    correct_drug_names = test_df['correct_drug_name'].dropna().tolist()
    total_batches = len(input_texts) // batch_size + (1 if len(input_texts) % batch_size != 0 else 0)
    
    start_time = time.time()
    
    results = []
    
    for i in range(total_batches):
        batch_texts = input_texts[i * batch_size:(i + 1) * batch_size]
        batch_correct_names = correct_drug_names[i * batch_size:(i + 1) * batch_size]
        batch_embeddings = get_batch_embeddings(batch_texts)
        
        for j, input_embedding in enumerate(batch_embeddings):
            input_embedding = input_embedding.unsqueeze(0)
            similarities = cosine_similarity(input_embedding.numpy(), drug_embeddings.numpy())
            best_match_index = np.argmax(similarities)
            predicted_drug_name = drug_names[best_match_index]
            
            if predicted_drug_name == batch_correct_names[j].lower():
                correct_predictions += 1
            results.append({
                'input_text': batch_texts[j],
                'predicted_drug_name': predicted_drug_name
            })
    
    accuracy = (correct_predictions / len(test_df)) * 100
    end_time = time.time()
    
    st.write(f"Time taken for batch testing: {end_time - start_time:.2f} seconds")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('predictions.csv', index=False)
    st.write("Batch testing completed. You can download the predictions file below.")
    st.download_button(
        label="Download Predictions",
        data=results_df.to_csv(index=False).encode('utf-8'),
        file_name='predictions.csv',
        mime='text/csv',
    )
    
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
    try:
        test_df = pd.read_csv(uploaded_file)
        st.write(test_df.head())
        st.write("Test CSV Columns:", test_df.columns.tolist())  # Debugging line to print column names
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
    
    if st.button("Start Batch Testing"):
        accuracy = test_model(uploaded_file)
        if accuracy is not None:
            st.write(f"Accuracy: {accuracy:.2f}%")
