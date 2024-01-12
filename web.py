import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from IPython.display import display, Javascript
from google.colab.output import eval_js
import base64
import requests
import json

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def predict_object(image_array):
    # Preprocess the image array
    img_array = cv2.resize(image_array, (224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get the ResNet50 predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    object_class = decoded_predictions[0][1]

    return object_class

def get_wikipedia_info(object_class):
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'titles': object_class,
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True
    }

    response = requests.get(base_url, params=params)
    data = json.loads(response.text)

    page_id = next(iter(data['query']['pages'].keys()), '-1')

    if page_id != '-1':
        extract = data['query']['pages'][page_id]['extract']
        return extract
    else:
        return "Information not available."

def take_photo_and_predict(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Wait for the video to be loaded before capturing a frame.
            await new Promise((resolve) => setTimeout(resolve, 2000));

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            stream.getVideoTracks()[0].stop();
            div.remove();

            const photo = canvas.toDataURL('image/jpeg', quality);
            const data = photo.split(',')[1];

            return data;
        }
    ''')
    display(js)

    # Get the image data from JavaScript
    data = eval_js('takePhoto({})'.format(quality))

    # Decode the base64-encoded image data
    binary_data = base64.b64decode(data)
    with open(filename, 'wb') as f:
        f.write(binary_data)

    # Read the captured image and predict the object class
    image_array = cv2.imread(filename)
    object_class = predict_object(image_array)

    # Fetch information about the predicted object from Wikipedia
    object_info = get_wikipedia_info(object_class)

    return filename, object_class, object_info

# Call the function to capture a photo and predict the object class
captured_photo, predicted_species, object_info = take_photo_and_predict()

# Display the captured photo, predicted species, and additional information
from IPython.display import Image, display
display(Image(filename=captured_photo))
print(f"Predicted Species: {predicted_species}")
print(f"Information about {predicted_species}: {object_info}")
