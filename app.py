from flask import Flask, request, render_template, jsonify 

import os 

import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
import numpy as np 
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

if not os.path.exists (UPLOAD_FOLDER):
  os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file ():

  if request.method == 'POST':
    if 'file' not in request.files:
      return 'Wheres your file goofy'
    file = request.files['file']
    if file.filename == '':
      return 'You didn\'t select a file goofy'
    
    if file:
      filename = os.path.join (app.config ['UPLOAD_FOLDER'], file.filenames )
      file.save (filename)

      return 'Nice your file has been saved :)'
  
  return render_template ('upload.html')
  
df = pd.read_csv('data/food.csv')
print(df.head())

if __name__ == '__main__':
  app.run (debug = True )