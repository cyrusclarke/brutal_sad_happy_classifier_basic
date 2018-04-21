#get_ipython().magic(u'matplotlib inline')
from __future__ import division,print_function
import theano
import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
from sklearn.externals import joblib

# In[2]:

import utils; reload(utils)
from utils import plots


# Define model and batches:

# In[3]:

path = "mldata/"
batch_size=64
import vgg16
from vgg16 import Vgg16
vgg = Vgg16()
batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)


# Load Trained Model:

# In[4]:

fname = 'vgg2.h5'
vgg.model.load_weights(fname)


# Verify Model is Correct:

# In[5]:

vgg.classes[:3]
batches = vgg.get_batches(path+'train', batch_size=3)


# Print Labels and Images:

# In[6]:

imgs,labels = next(batches)


# Display images and predictions

# In[14]:

plots(imgs, titles=labels)
vgg.predict(imgs, True)


# Test with unknown images via upload and run prediction:

# In[8]:

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from matplotlib.pyplot import imshow

def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


# In[22]:

img,x = get_image('mldata/train/happy/happy.10105.jpg')
imshow(img)

predictions = vgg.predict(x, True)
print(predictions)


# In[23]:

import cPickle as pickle
import resource
import sys


# In[24]:

print(sys.getrecursionlimit())
sys.setrecursionlimit(1000006)




# In[25]:

joblib.dump(vgg, open("data50.pkl", "wb"))