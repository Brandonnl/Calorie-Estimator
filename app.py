from flask import Flask, request, render_template, jsonify 

import os 

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
      filename = os.path.join (app.config ['UPLOAD_FOLDER'], file.filename )
      file.save (filename)

      return 'Nice your file has been saved :)'
  
  return render_template ('upload.html')
    

@app.route('/predict', methods=['POST'])
def predict ():
  return "Not implemented "


if __name__ == '__main__':
  app.run (debug = True )