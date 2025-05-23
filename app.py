import torch
import streamlit as st
from PIL import Image
import os
from utils import (
    load_trained_model,
    preprocess_image,
    extract_features,
    detect_and_crop
)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model
model = load_trained_model("person_reid_model.pth")

# Streamlit UI
st.title("🧍 Real-Time Person Re-Identification")

uploaded_file = st.file_uploader("📷 Upload a person image", type=["jpg", "png"])

# Function to process query image
def process_query_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = detect_and_crop(img)
    img_tensor = preprocess_image(img)
    features = extract_features(model, img_tensor)
    return features, img

# Load gallery
gallery_folder = "gallery/"
gallery_features = []
gallery_filenames = []

for filename in os.listdir(gallery_folder):
    if filename.lower().endswith((".jpg", ".png")):
        path = os.path.join(gallery_folder, filename)
        img = Image.open(path).convert("RGB")
        tensor = preprocess_image(img)
        feat = extract_features(model, tensor)
        gallery_features.append(feat)
        gallery_filenames.append(filename)

gallery_features = torch.stack(gallery_features).numpy()

# Function to get Top-N matches with confidence scores
def get_top_n_matches(query_feat, gallery_feats, gallery_paths, N=3):
    similarities = cosine_similarity(query_feat.reshape(1, -1), gallery_feats).flatten()
    top_indices = similarities.argsort()[::-1][:N]
    top_matches = [(gallery_paths[i], similarities[i]) for i in top_indices]
    return top_matches

# Function to generate confidence report card
def generate_confidence_report(top_matches):
    report = []
    for i, (fname, score) in enumerate(top_matches):
        confidence = score * 100  # Convert similarity score to percentage
        report.append({
            "Image": fname,
            "Confidence": f"{confidence:.2f}%"
        })
    return report

# On upload
if uploaded_file:
    query_features, cropped_query = process_query_image(uploaded_file)
    query_features_np = query_features.numpy()

    st.subheader("🔍 Query Image")
    st.image(cropped_query, caption="Detected Person", width=300)

# Function to get Top-N matches with confidence threshold
def get_top_n_matches(query_feat, gallery_feats, gallery_paths, N=3, threshold=0.7):
    similarities = cosine_similarity(query_feat.reshape(1, -1), gallery_feats).flatten()
    top_indices = similarities.argsort()[::-1]  # Sort by descending similarity
    top_matches = []

    for idx in top_indices:
        if similarities[idx] >= threshold:
            top_matches.append((gallery_paths[idx], similarities[idx]))
        if len(top_matches) >= N:
            break

    return top_matches  # Ensure to return the matches

# On upload
if uploaded_file:
    query_features, cropped_query = process_query_image(uploaded_file)
    query_features_np = query_features.numpy()

    st.subheader("🔍 Query Image")
    st.image(cropped_query, caption="Detected Person", width=300)

    # Get Top-N matches
    top_n_matches = get_top_n_matches(query_features_np, gallery_features, gallery_filenames, N=3, threshold=0.7)

    if top_n_matches:
        st.subheader("🎯 Top Matches from Gallery")
        
        # Optional: Horizontal separator
        st.markdown("---")

        cols = st.columns(len(top_n_matches))

        for i, (fname, score) in enumerate(top_n_matches):
            name = fname.split('.')[0].replace('_', ' ').title()
            img_path = os.path.join(gallery_folder, fname)

            with cols[i]:
                st.image(img_path, use_container_width=True)
                st.markdown(f"🧾 Name:** {name}", unsafe_allow_html=True)
                st.markdown(f"📊 Similarity:** {score:.4f}", unsafe_allow_html=True)

        # Display Confidence Report Card
        st.subheader("📝 Confidence Report Card")
        report = generate_confidence_report(top_n_matches)
        for item in report:
            st.write(f"{item['Image']}: {item['Confidence']}")

    else:
        st.error("❌ No match found.")
