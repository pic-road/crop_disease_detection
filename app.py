import os
import requests
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow import keras



image = keras.preprocessing.image

#upload your models to drive, copy file id from link and paste after ("id") in url for each model below respectively

# Mobile_URL = "https://drive.google.com/uc?export=download&id= ..."
# Mobile_PATH = "MobileNet.h5"

# VGG16_URL = "https://drive.google.com/uc?export=download&id= ... "
# VGG16_PATH = "VGG16.h5"
# def download_model(url, filename):
#     if not os.path.exists(filename):
#         print(f"Downloading {filename} from Google Drive...")
#         response = requests.get(url)
#         with open(filename, "wb") as f:
#             f.write(response.content)
#         print("Download complete.")

# # Call before loading model
# download_model(Mobile_URL, Mobile_PATH)
# mobilenet_model = keras.models.load_model(Mobile_PATH)
# download_model(VGG16_URL, VGG16_PATH)
# vgg16_model = keras.models.load_model(Mobile_PATH)


#or else run the jupyter notebooks in developing-models folder and save the models.
#store them locally in models-folder and load them 



app = Flask(__name__)

# Load both models
mobilenet_model = keras.models.load_model('models/MobileNet.h5')
vgg16_model = keras.models.load_model('models/VGG16.h5')


# Define your labels
LABELS = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust_',
    'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x / 255.0

def predict_with_model(model, img_path):
    x = prepare_image(img_path)
    preds = model.predict(x)[0]
    top_index = np.argmax(preds)
    class_name = LABELS[top_index].split('___')
    confidence = preds[top_index]
    return class_name, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/uploads', filename)
        f.save(upload_path)

        # Get predictions from both models
        mob_pred, mob_conf = predict_with_model(mobilenet_model, upload_path)
        vgg_pred, vgg_conf = predict_with_model(vgg16_model, upload_path)

        return render_template('result.html',
                               filename=filename,
                               mob_crop=mob_pred[0],
                               mob_disease=mob_pred[1].replace('_', ' '),
                               mob_conf=f"{mob_conf:.2%}",
                               vgg_crop=vgg_pred[0],
                               vgg_disease=vgg_pred[1].replace('_', ' '),
                               vgg_conf=f"{vgg_conf:.2%}"
                               )

if __name__ == '__main__':
    app.run(debug=True)
