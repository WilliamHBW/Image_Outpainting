# utils/image_processor.py
from PIL import Image

def resize_image(image, width, height):
    return [image.resize((width, height)), image.resize((width, height))]
