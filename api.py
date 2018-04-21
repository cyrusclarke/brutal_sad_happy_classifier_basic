import flask
from flask import Flask, abort, jsonify, render_template, request, make_response, send_from_directory
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import base64
from io import BytesIO
from skimage import io as skio
from skimage.transform import resize
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':

		# get uploaded image file if it exists
		file = request.files['image']
		if not file: return render_template('index.html', label="No file")
		
		# read in file as raw pixels values
		# (ignore extra alpha channel and reshape as its a single image)
		img = misc.imread(file)
		img = resize(img, (224,224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		predictions = model.predict(x, True)

		#img = img[:,:,:3]
		#img = img.reshape(1, -1)

		# make prediction on new image
		#prediction = model.predict(img)
	
		# squeeze value from 1D array and convert to string for clean return
		label = str(np.squeeze(predictions))

		return render_template('index.html', label=label)


if __name__ == '__main__':
	# load ml model
	model = joblib.load('data50.pkl')
	# start api
	app.run(host='0.0.0.0', port=5000, debug=True)
