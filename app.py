import os, io, base64
from flask import Flask, render_template, request, jsonify

import cv2
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TensorFlow model exported from Azure Cognitive Services Custom Vision

CWD = os.getcwd()

MODELFILE = CWD + '/model/model.pb'
LABELFILE = CWD + '/model/labels.txt'

OUTPUT_LAYER = 'loss:0'
INPUT_NODE = 'Placeholder:0'

GRAPH_DEF = None

with tf.compat.v1.gfile.FastGFile(MODELFILE, 'rb') as f:
    GRAPH_DEF = tf.compat.v1.GraphDef()
    GRAPH_DEF.ParseFromString(f.read())
    tf.compat.v1.import_graph_def(GRAPH_DEF, name='')

LABELS = [line.rstrip() for line in tf.compat.v1.gfile.GFile(LABELFILE)]


# Initialize the web application

app = Flask(__name__)

# The root route, returns the rendered 'home.html' template page
@app.route('/')
def home():
    page_data = {}
    return render_template('home.html', page_data = page_data)

# Our custom API endpoint where we will receive images
@app.route('/process_image', methods=['POST'])
def check_results():

    # Get the JSON passed to the request and extract the image
    body = request.get_json()
    image_bytes = base64.b64decode(body['image_base64'].split(',')[1])
    image = io.BytesIO(image_bytes)

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.import_graph_def(GRAPH_DEF, name='')

    # Convert the image to the appropriate format and size for our tensorflow model
    augmented_image = prepare_image(image)

    with tf.compat.v1.Session() as sess:
        prob_tensor = sess.graph.get_tensor_by_name(OUTPUT_LAYER)
        predictions, = sess.run(prob_tensor, {INPUT_NODE: augmented_image})

        # Get the highest probability label
        highest_probability_index = np.argmax(predictions)
        probability = predictions[highest_probability_index]
    
    return jsonify({'predicted': str(LABELS[highest_probability_index]),
                    'probability': str(probability),
                    'opponent': LABELS[np.random.randint(5)]})


def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def prepare_image(image):
    image = Image.open(image)
    image = convert_to_opencv(image)

    # We next get the largest center square
    h, w = image.shape[:2]
    min_dim = min(w,h)
    max_square_image = crop_center(image, min_dim, min_dim)

    # Resize that square down to 256x256
    augmented_image = resize_to_256_square(max_square_image)

    # Get the input size of the model
    with tf.compat.v1.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name(INPUT_NODE).shape.as_list()
    network_input_size = input_tensor_shape[1]

    # Crop the center for the specified network_input_Size
    augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

    # Need to introduce an additional tensor dimension
    augmented_image = np.expand_dims(augmented_image, axis=0)

    return augmented_image