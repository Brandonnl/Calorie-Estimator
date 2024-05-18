import torch
import torch.nn as nn
from torchvision import models
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# Define the model architecture
class CLIPToPCA(nn.Module):
    def __init__(self, clip_input_size, pca_output_size, hidden_size):
        super(CLIPToPCA, self).__init__()
        self.clip_input_size = clip_input_size
        self.pca_output_size = pca_output_size
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(in_channels=clip_input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.output_fc = nn.Linear(hidden_size, pca_output_size)
        self.activation = nn.ReLU()
        
    def forward(self, clip_encodings):
        clip_encodings = clip_encodings.permute(0, 2, 1)
        
        conv_out = self.conv1(clip_encodings)
        conv_out = self.activation(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = self.activation(conv_out)
        pooled_out = torch.mean(conv_out, dim=2)
        pca_encodings = self.output_fc(pooled_out)
        return pca_encodings

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Function to load the model
def load_model(path, clip_input_size, pca_output_size, hidden_size):
    model = CLIPToPCA(clip_input_size, pca_output_size, hidden_size)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# CLIP model and processor
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
