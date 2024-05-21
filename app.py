import time

from flask import Flask, render_template, request, redirect, jsonify
from pdf2image import convert_from_path
import os
import base64
from io import BytesIO
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import numpy as np
import base64
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload_form.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(pdf_path)
        images = convert_from_path(pdf_path)

        base64_images = []
        for idx, image in enumerate(images):
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'converted_image_{idx}.jpg')
            image.save(image_path)

            img_buffer = BytesIO()
            image.save(img_buffer, format='JPEG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            base64_images.append(img_str)

        first_img = images[0]
        width, height = first_img.size

        return render_template('display_images.html', images=base64_images, width=width, height=height)
    else:
        return redirect(request.url)


@app.route('/rotate/<int:degrees>', methods=['POST'])
def rotate_image(degrees):
    base64_image_data = request.data.decode('utf-8')

    img_data = base64.b64decode(base64_image_data)
    img = Image.open(BytesIO(img_data))
    rotated_img = img.rotate(degrees, expand=True)

    buffered = BytesIO()
    rotated_img.save(buffered, format="JPEG")
    rotated_base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return rotated_base64_image

def is_image_almost_white(image, white_threshold=245):
    np_image = np.array(image)
    avg_pixel_value = np_image.mean()
    return avg_pixel_value >= white_threshold

@app.route('/recognize_text', methods=['POST'])
def recognize_text():
    t1 = int(time.time() * 10)
    data = request.get_json()
    image_base64 = data['image']
    image_data = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(BytesIO(image_data))
    if is_image_almost_white(image):
        return ''
    gray_image = image
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['weights'] = 'model/vgg_seq2seq.pth'
    config['device'] = 'cpu'
    detector = Predictor(config)
    try:
        s = detector.predict(gray_image)
        recognized_text = s
        return jsonify({
            "status_code": 200,  # OK
            "data": recognized_text
        })
    except Exception as e:
        return jsonify({
            "status_code": 500,  # Internal Server Error
            "data": str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0')
