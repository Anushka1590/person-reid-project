# Person Re-Identification System

A real-time Person Re-Identification system using a Vision Transformer (ViT) backbone, YOLOv8 for person detection, and cosine similarity-based retrieval. Deployed via Streamlit for live querying.

---

## Overview

Person Re-Identification (ReID) aims to recognize the same individual across different camera views. This project leverages a fine-tuned Vision Transformer (TransReID) model trained on the Market-1501 dataset to extract robust 768-dimensional feature embeddings. Embeddings are compared via cosine similarity to retrieve the most similar identities.

A YOLOv8 person detector is used to crop input images for better embedding quality, and the system supports a Streamlit interface for real-time querying.

---

## Key Features

- Vision Transformer (vit_base_patch16_224) backbone for feature extraction

- YOLOv8-based person detection for clean input

- Cosine similarity matching for retrieval

- Real-time querying using Streamlit

- Custom gallery support (images of multiple identities with varying camera angles)

---

## Streamlit UI

<img width="1861" height="717" alt="Screenshot 2025-04-25 022413" src="https://github.com/user-attachments/assets/73fd4e1e-de35-4f1c-95f8-1071786b4dbf" />

Upload Image:
<img width="1904" height="853" alt="Screenshot 2025-04-20 010015" src="https://github.com/user-attachments/assets/059f0b91-47b6-471c-ae65-1ee4e128becf" />

Top matches displayed:
<img width="1891" height="833" alt="Screenshot 2025-04-20 012034" src="https://github.com/user-attachments/assets/2567be0f-2dc4-4f1b-9973-dd9aac8d4ae1" />

Confidence Percentage:
<img width="1863" height="360" alt="Screenshot 2025-04-20 012045" src="https://github.com/user-attachments/assets/b54648a9-c0ef-48d7-b7cd-7054837258aa" />

---

## Pipeline Diagram

Input Image
↓
YOLOv8 → Crop Person
↓
ViT (TransReID) → 768-dim Feature Embedding
↓
Cosine Similarity with Gallery Embeddings
↓
Top-K Similar Persons
↓
Streamlit Display

---

## Installation & Usage

### Clone repo
git clone <repo_url>
cd person-reid-system

### Install dependencies
pip install -r requirements.txt

### Place model weights in root
person_reid_model.pth
yolov8n.pt

### Run Streamlit app
streamlit run app.py

---

## Results & Performance

- Tested on custom gallery of 10 identities, 10-15 images each

- Rank-1 similarity matching: correctly identifies top match in most cases

- Provides confidence scores for retrieved images

#### Note: Moderate accuracy; performance varies with occlusion, lighting, and low-resolution images.

---

## Limitations

- Not production-ready

- Accuracy can drop in complex real-world scenarios

- Currently limited to Market-1501-trained embeddings

---
   
## Future Work

- Improve model accuracy via further fine-tuning and data augmentation

- Enhance YOLO integration for multi-person detection

- Support additional datasets and camera views

- Explore deployment optimizations for speed and scalability

