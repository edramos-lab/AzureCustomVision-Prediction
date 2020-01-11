import os, io, base64
from flask import Flask, render_template, request, jsonify

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


    # return jsonify({'predicted': str(LABELS[highest_probability_index]),
    #                 'probability': str(probability),
    #                 'opponent': LABELS[np.random.randint(5)]})
