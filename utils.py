# utils.py

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import os

# Model with backbone + classifier (but we’ll only use backbone for ReID)
class TransReIDFullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.backbone.reset_classifier(0)
        self.classifier = nn.Linear(self.backbone.num_features, 751)

    def forward(self, x):
        # Return backbone features only for ReID
        return self.backbone(x)

# Load trained model weights
def load_trained_model(model_path="person_reid_model.pth"):
    model = TransReIDFullModel()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image: Image.Image):
    return transform(image).unsqueeze(0)

# Extract feature embedding (NOT classification output)
def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model(image_tensor)  # output from ViT backbone
    return features.squeeze(0)

# YOLOv8 person detector
yolo_model = YOLO("yolov8n.pt")

def detect_and_crop(image: Image.Image):
    results = yolo_model(image)
    boxes = results[0].boxes
    for box in boxes:
        if int(box.cls) == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            return image.crop((x1, y1, x2, y2))
    return image

# Cosine similarity comparison
def compare_with_gallery(query_features, gallery_features):
    best_match = None
    highest_similarity = -1

    # Normalize the query features once
    query_features = torch.nn.functional.normalize(query_features, dim=0)
    
    for filename, gallery_feat in gallery_features.items():

        # Normalize gallery feature before comparison
        gallery_feat = torch.nn.functional.normalize(gallery_feat, dim=0)


        similarity = torch.cosine_similarity(query_features, gallery_feat, dim=0).item()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = filename
    return best_match if highest_similarity > 0.7 else None
