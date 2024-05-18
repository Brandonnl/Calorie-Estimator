from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image 
from transformers import CLIPProcessor, CLIPModel
import torch
from image import process_single_image
from pca import pca_ingredients, CLIPToPCA

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'data/images/images/'

if not os.path.exists (UPLOAD_FOLDER):
  os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = CLIPToPCA(512, 900, 706)
model.load_state_dict (torch.load('data/model3.pth'))

def predict_ingredients (img):
  
    new_image_encodings = process_single_image(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(new_image_encodings)
        predicted_encodings = outputs.numpy()

    encodings_list = predicted_encodings.flatten().tolist()
    encodings_sorted = sorted(range(len(encodings_list)), key=lambda k: encodings_list[k], reverse=True)
    top_n = min(20, len(encodings_sorted))
    print (encodings_sorted)
    top_ingredients = [(pca_ingredients[encodings_sorted[i]]) for i in range(top_n)]
    
    return top_ingredients


@app.route('/', methods=['GET', 'POST'])
def upload_file ():

  if request.method == 'POST':
    if 'file' not in request.files:
      return 'Wheres your file goofy'
    file = request.files['file']
    if file.filename == '':
      return 'You didn\'t select a file goofy'
    
    if file:
      filename = os.path.join (app.config ['UPLOAD_FOLDER'], file.filename )
      file.save (filename)
      ingredients = predict_ingredients (filename)
      return render_template('result.html', ingredients=ingredients)
    
  
  return render_template ('upload.html')

@app.route('/images/<filename>')
def display_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

# @app.route('/predict', methods=['POST'])
# def predict_ingredients(img):
#   return "Hi"



if __name__ == '__main__':
  app.run (debug = True )