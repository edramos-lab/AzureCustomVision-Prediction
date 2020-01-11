# Image classification with the Azure Custom Vision Prediction API

[Azure Custom Vision](https://docs.microsoft.com/azure/cognitive-services/custom-vision-service/home?WT.mc_id=customvisionclassification-github-beverst) (part of Cognitive Services) enables easy training of image classification machine learning models.


This repo contains a Flask web application which does the following:
- Hosts a website that captures images from the webcam.
- Hosts a API endpoint that receives images and sends them to the Azure Custom Vision prediction API for classification.

## Running the app

You can run this app like any Python Flask web application. For example via `flask run` command.

## The structure of the code

* `app.py` - this file contains the Flask app. It has two routes:
  * `/` - this loads the `home.html` template, passing in an empty dictionary of data
  * `/process_image` - this receives the image as a JSON document containing the image as Base64 encoded data.
* `requirements.txt` - the required Python libraries to run this code (install with `pip3 install -r requirements.txt`)
* `templates/home.html` - the template and inline Javascript app to capture images from the webcam
