from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import io

app = Flask(__name__)

# specify domain to allow requests once server set up. 
# CORS(app, resources={r"/*": {"origins": "http:// whatever it will be"}})
CORS(app) 

# model = load_model('model.h5')

class_labels_df = pd.read_csv('image_labels.csv')
#map id, label, so model can return label
class_labels = dict(zip(class_labels_df['id'], class_labels_df['label']))


def prepare_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    return image

#FUCNTION to test frontend to backend connection
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'no file uploaded'
    file = request.files['file']    
    if file.filename == '':
        return 'no file selected'
    if file:
        #process
        return file.filename + ' uploaded successfully'


#FUNCTION to be used once model is optimized
# @app.route('/upload', methods=['POST'])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "no file uploaded"})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "no file selected"})
    
#     try:
#         image = Image.open(file.stream)
#         prepared_image = prepare_image(image)
#         prediction = model.predict(prepared_image)
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         predicted_label = class_labels.get(predicted_class, 'unknown')

#         #annotate result on image
#         draw = ImageDraw.Draw(image)
#         draw.text((10, 10), predicted_label, fill='red')

#         img_io = io.BytesIO()
#         image.save(img_io, 'JPEG', quality=70)
#         img_io.seek(0)
        
#         return send_file(img_io, mimetype='image/jpeg')
    
#     except Exception as e:
#         return jsonify({"error": str(e)})

if __name__ == '__main__':  
    app.run(debug=True)



# @app.route('/')
# def hello_world():
#     return 'hello!'




