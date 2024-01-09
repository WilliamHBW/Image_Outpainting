# FILEPATH: /home/mmspg/Desktop/Image_Outpainting/backend/test_image_processor.py

import pytest
from PIL import Image
from image_processor import resize_image

def test_resize_image():
    # Create a new image with a size of 200x200 pixels
    img = Image.new('RGB', (200, 200), color = 'red')

    # Define the new widths and heights
    new_sizes = [(100, 100), (50, 50), (150, 150)]

    for new_width, new_height in new_sizes:
        steps = 10

        # Call the resize_image function
        resized_images = resize_image(img, new_width, new_height, steps)

        # Check if the function returns a list
        assert isinstance(resized_images, list)

        # Check if the function returns at least one image
        assert len(resized_images) > 0

        # Check if the first image in the list has the correct size
        assert resized_images[0].size == (new_width, new_height)

if __name__ == '__main__':
    pytest.main([__file__])