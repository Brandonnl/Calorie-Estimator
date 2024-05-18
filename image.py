import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os


model_name = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)
clip_model.eval()

IMAGE_SIZE = (224, 224)

def encode_image(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features

def prepare_image(image):
    img = Image.open(image)
    img = img.resize(IMAGE_SIZE)
    return img

def process_single_image(image_path):
    preprocessed_image = prepare_image(image_path)
    image_features = encode_image(preprocessed_image)
    return image_features