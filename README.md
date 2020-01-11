# Azure Custom Vision classification model inferencing with Tensorflow in Python

[Azure Custom Vision](https://docs.microsoft.com/azure/cognitive-services/custom-vision-service/home?WT.mc_id=customvisionclassification-github-beverst) (part of Cognitive Services) enables easy training of image classification machine learning models. It also enables [exporting of sparse models](https://docs.microsoft.com/azure/cognitive-services/custom-vision-service/export-your-model?WT.mc_id=customvisionclassification-github-beverst). This repo features a Flask web application that uses Tensorflow to load an exported model (see the `/model` folder) to enable image classification of images captured via webcam.

This repo contains a Flask web application which does the following:
- Hosts a website that captures images from the webcam.
- Hosts a API endpoint that receives images, loads a tensorflow classification model from file, and predicts the classification.

*Note*: This project is compatible with any CustomVision Multiclass model exported for Tensorflow. Instructions for exporting [can be found here](https://docs.microsoft.com/azure/cognitive-services/custom-vision-service/export-your-model?WT.mc_id=customvisionclassification-github-beverst).

## Running the app

You can run this app like any Python Flask web application. For example via `flask run` command.

## The structure of the code

* `app.py` - this file contains the Flask app. It has two routes:
  * `/` - this loads the `home.html` template, passing in an empty dictionary of data
  * `/process_image` - this receives the image as a JSON document containing the image as Base64 encoded data.
* `model` - this folder contains the contents of the model exported from Azure CustomVision
  * `/model.pb` - the exported tensorflow model
  * `/labels.txt` - the exported classification labels
* `requirements.txt` - the required Python libraries to run this code (install with `pip3 install -r requirements.txt`)
* `templates/home.html` - the template and inline Javascript app to capture images from the webcam
