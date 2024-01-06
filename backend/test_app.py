# test_app.py
import io
import pytest
from PIL import Image
from flask_testing import TestCase
from app import app

class MyTest(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    def test_upload_image(self):
        with open('/Users/willamhuang/Desktop/1.png', 'rb') as img:
            img_io = io.BytesIO(img.read())
        response = self.client.post('/upload', data={'image': (img_io, 'test_image.png'), 'width': 100, 'height': 100}, content_type='multipart/form-data')
        assert response.status_code == 200
        assert response.mimetype == 'image/png'

if __name__ == '__main__':
    pytest.main()