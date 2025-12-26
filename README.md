# Person Re-Identification System

A Vision Transformer-based Person Re-Identification (ReID) system with a Streamlit interface and cosine similarity matching using the Market-1501 dataset.

---

## Overview

Person Re-Identification is a challenging computer vision task that aims to recognize the same individual across different camera views. This project leverages a fine-tuned Vision Transformer (TransReID) model to address this challenge by extracting robust feature embeddings and performing cosine similarity matching via FAISS for efficient retrieval.

The system includes a user-friendly Streamlit interface for real-time querying and optional integration with YOLO for person detection.

---

## Features

- Person ReID using Vision Transformer (TransReID)  
- Cosine similarity-based feature matching using FAISS  
- Real-time querying via Streamlit UI  
- Tested on Market-1501 dataset  
- Optional YOLO integration for person detection  

---

## Streamlit UI

<img width="1861" height="717" alt="Screenshot 2025-04-25 022413" src="https://github.com/user-attachments/assets/73fd4e1e-de35-4f1c-95f8-1071786b4dbf" />

Upload Image:
<img width="1904" height="853" alt="Screenshot 2025-04-20 010015" src="https://github.com/user-attachments/assets/059f0b91-47b6-471c-ae65-1ee4e128becf" />

Top matches displayed:
<img width="1891" height="833" alt="Screenshot 2025-04-20 012034" src="https://github.com/user-attachments/assets/2567be0f-2dc4-4f1b-9973-dd9aac8d4ae1" />

Confidence Percentage:
<img width="1863" height="360" alt="Screenshot 2025-04-20 012045" src="https://github.com/user-attachments/assets/b54648a9-c0ef-48d7-b7cd-7054837258aa" />





## Current Status & Limitations

> ⚠️ **Note:** This is a research/demo project. The model currently achieves moderate accuracy on Market-1501 and may not perform reliably in all scenarios such as heavy occlusions, varying lighting, or low-resolution images. It is not production-ready and intended for demonstration and experimentation purposes.

---

## How to Run
1. Clone the repository

2. Install dependencies:
   pip install -r requirements.txt
   
3. Model Weights
Download model weights from the links below and place them in the root project directory:

- [person_reid_model.pth](https://drive.google.com/file/d/18W_RWaVadFBlojRKBljXrZtDKp3WKXKt/view?usp=sharing)
- [yolov8n.pt](https://drive.google.com/file/d/1d8Q_Iiyt-UFsTfZXkXsnZ99TW1MyWQ_v/view?usp=sharing)
  
4. Sample Gallery
Include a few images in the `/Gallery` folder to test the app.

5. Run the app:
   streamlit run app.py
   
## Future Work
- Improve model accuracy via further fine-tuning and data augmentation

- Enhance YOLO integration for better real-time detection

- Support additional datasets and camera views

- Explore deployment optimizations for speed and scalability

