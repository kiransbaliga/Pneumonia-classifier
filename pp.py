import os
import numpy as np
from keras.models import load_model
from PIL import Image
import numpy as np
import keras
import tensorflow as tf

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

makeglobalmodel()