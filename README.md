# person-reid-project
A Vision Transformer-based Person Re-Identification system with Streamlit interface and cosine similarity matching using Market-1501 dataset.

# Person Re-Identification System

This project uses a fine-tuned Vision Transformer (TransReID) model for identifying individuals across different camera views. It includes a Streamlit interface and uses cosine similarity for matching.

## Features
- Person ReID with Vision Transformer
- Cosine similarity-based matching using FAISS
- Real-time querying via Streamlit UI
- Tested on Market-1501 dataset
- Optional integration with YOLO for person detection

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run the app:
   streamlit run app.py
   
## Model Weights
Download model weights from the links below and place them in the root project directory:

- [person_reid_model.pth](https://drive.google.com/file/d/18W_RWaVadFBlojRKBljXrZtDKp3WKXKt/view?usp=sharing)
- [yolov8n.pt](https://drive.google.com/file/d/1d8Q_Iiyt-UFsTfZXkXsnZ99TW1MyWQ_v/view?usp=sharing)
  
## Sample Gallery
Include a few images in the `/Gallery` folder to test the app.
