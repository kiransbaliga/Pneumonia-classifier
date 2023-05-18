import io
import os
import sys
import zipfile

# Flask
from flask import Flask, redirect, send_file, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
from keras.models import load_model
from PIL import Image
import numpy as np
import keras
import tensorflow as tf

from util import base64_to_pil


app = Flask(__name__)





model1 = load_model("./models/model1/model1c1.h5")
model2 = load_model("./models/model2/model2c1.h5")
model3 = load_model("./models/model3/model3c1.h5")

print('Model loaded. Check http://127.0.0.1:5000/')


def makeavg(models):
    globalmodel = models[0]
    
    for i in range(len(models)):
        weights = models[i].get_weights()
        if i == 0:
            avgweights = weights
        else:
            for j in range(len(weights)):
                avgweights[j] = avgweights[j] + weights[j]
    for i in range(len(avgweights)):
        avgweights[i] = avgweights[i]/len(models)

    
    globalmodel.set_weights(avgweights)
    
    return globalmodel


def makeglobalmodel():
    modeldir1 = './models/model1'
    modeldir2 = './models/model2'
    modeldir3 = './models/model3'
    modeldir1paths = os.listdir(modeldir1)
    modeldir2paths = os.listdir(modeldir2)
    modeldir3paths = os.listdir(modeldir3)

    loadedmodels1 = []
    loadedmodels2 = []
    loadedmodels3 = []
    for i in range(len(modeldir1paths)):
        loadedmodels1.append(load_model(modeldir1+'/'+modeldir1paths[i]))
    for i in range(len(modeldir2paths)):
        loadedmodels2.append(load_model(modeldir2+'/'+modeldir2paths[i]))
    for i in range(len(modeldir3paths)):
        loadedmodels3.append(load_model(modeldir3+'/'+modeldir3paths[i]))

    
    globalmodel1 = makeavg(loadedmodels1)
    globalmodel2 = makeavg(loadedmodels2)
    globalmodel3 = makeavg(loadedmodels3)

    globalmodel1.save('./models/model1/globalmodel1.h5')
    globalmodel2.save('./models/model2/globalmodel2.h5')
    globalmodel3.save('./models/model3/globalmodel3.h5')

def predict_pneumonia_ensemble(images,images_new):
    # Make predictions using each model
    preds1 = model1.predict(images)
    preds2 = model2.predict(images)
    preds3 = model3.predict(images_new)
    # Perform ensemble voting
    print(preds1,preds2,preds3)
    ensemble_preds = (preds1 + preds2 + preds3)/3.0
    return ensemble_preds

def makeglobalmodel():
    return


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route("/upload",methods=['GET','POST'])
def indexx():
    if request.method == 'POST':
        f = request.files['file1']
        f2 = request.files['file2']
        f3 = request.files['file3']
        f.save("models/model1"+f.filename)
        f2.save("models/model2"+f2.filename)
        f3.save("models/model3"+f3.filename)
        return "file uploaded successfully"

@app.route("/makeglobal",methods=['GET','POST'])
def makeglobal():
    makeglobalmodel()
    return "global model created successfully"



@app.route("/download",methods=['GET','POST'])
def download():
    file1url="models/model1/globalmodel1.h5"
    file2url="models/model2/globalmodel2.h5"
    file3url="models/model3/globalmodel3.h5"
    # zip the contents into a zip file

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as zf:
        zf.write(file1url)
        zf.write(file2url)
        zf.write(file3url)
    stream.seek(0)
    return send_file(stream,as_attachment=True,download_name='globalmodels.zip')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img.save("./uploads/image.png")
        image=Image.open("./uploads/image.png").convert('RGB')
        resized_image = image.resize((64, 64))
        test_image = np.array(resized_image)
        test_image = np.expand_dims(test_image, axis=0)

        resized_image_new = image.resize((224, 224))
        test_image_new = np.array(resized_image_new)
        test_image_new = np.expand_dims(test_image_new, axis=0)
        # Make prediction
        predictions = predict_pneumonia_ensemble(test_image,test_image_new)
        print(predictions)
        threshold = 0.5
        predicted_class = 'Pneumonia' if predictions[0] >= threshold else 'Normal'
        return jsonify(result=predicted_class)
    return None


if __name__ == '__main__':
    app.run(port=5000, threaded=False)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 80), app)
    # http_server.serve_forever()
