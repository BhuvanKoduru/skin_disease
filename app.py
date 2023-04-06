from flask import Flask
from flask_cors import CORS,cross_origin
from flask import request
from flask import jsonify
from werkzeug.utils import secure_filename
import pandas as pd

import base64
import numpy as np
import tensorflow as tf
import keras.models
import cv2
import re
import sys
import os
import glob
from load import *


def init(): 
	loaded_model = keras.models.load_model('/Users/abhilashhathwar/Desktop/Desk/ERSA/backend/skin_disease_classifier.h5')
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),optimizer='adam',metrics=['accuracy'])
	
	return loaded_model

app = Flask(__name__)
CORS(app)

global model

model = init()

UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

softStorey = False

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route("/",methods=['POST','GET'])
def home():
    if request.method == 'POST':
      f = request.files['file']
      return f
    return "hello health partner"


@app.route("/predict",methods=['POST','GET'])
def upload_file():
    #if 'picture' not in request.files:
    #    return 'No image uploaded', 400

    image = request.files['picture']

    if image.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    #image.save('/path/to/save/image')
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
        new_image = cv2.imread('images/'+image.filename)
        imResized = cv2.resize(new_image,(64,64))
        imResized = imResized/255.0
        print(imResized.shape)
        imResized=imResized[None,:,:,:]
        out = model.predict(imResized)
        print(out)
    global pred
    pred=pd.Series(out[0]).idxmax()
    diseases=['Acne and Rosacea','Actinic Keratosis Basal Cell Carcinoma','Atopic Dermatitis','Bullous Disease','Cellulitis Impetigo',
          'Eczema','Exanthems and Drug Eruptions','Alopecia','STDS','Pigmentation','Lupus','Melanoma','Nail Fungus','Contact Dermatitis',
          'Psoriasis','Lyme Disease','Seborrheic Keratoses','Systemic Disease','Ringworm','Urticaria Hives','Vascular Tumors','Vasculitis',
          'Warts Molluscum']
    return jsonify({'disease': diseases[pred]})
    

if __name__ == "__main__":
   app.run(debug=True)
