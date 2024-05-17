from flask import Flask, request, render_template, jsonify, send_from_directory
import os
# from food_recognition import predict_ingredients
from prediction import predict_ingredients

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'data/images/images'

if not os.path.exists (UPLOAD_FOLDER):
  os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER


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
      result = predict_ingredients (filename)

      return render_template('result.html', predictions=result)
    
  
  return render_template ('upload.html')

@app.route('/images/<filename>')
def display_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

# @app.route('/predict', methods=['POST'])
# def predict_ingredients(img):
#   return "Hi"



if __name__ == '__main__':
  app.run (debug = True )