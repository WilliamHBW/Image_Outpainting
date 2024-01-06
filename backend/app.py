# app.py
from flask import Flask, request, send_file, send_from_directory, jsonify
from PIL import Image
from utils.image_processor import resize_image
import io
import base64

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    width = int(request.form['width'])
    height = int(request.form['height'])
    steps = int(request.form['steps'])

    # Open the uploaded image with Pillow
    img = Image.open(file)

    # Generate multiple images
    images = resize_image(img, width, height, steps)

    # Convert images to base64 and save them as JPEG
    base64_images = []
    for img in images:
        byte_io = io.BytesIO()
        img = img.convert('RGB')
        img.save(byte_io, 'JPEG')
        byte_io.seek(0)
        base64_images.append(base64.b64encode(byte_io.getvalue()).decode('utf-8'))

    # Send the images as base64 strings
    return jsonify(base64_images)

if __name__ == '__main__':
    app.run(debug=True)