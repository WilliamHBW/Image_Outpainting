# Image Outpainting

## Description

Combining LLMs for image outpainting automatically. Serves as the backend software for image processing.

## Environment

- see requirements.py

## File Structure
The project has the following structure:

- `__pycache__/`
- `log_file`
- `main.py`
- `requirements.txt`
- `README.md`
- `models/`
  - `ESRGAN/`
  - `blip-image-captioning-large/`
  - `stable-diffusion-inpainting/`
- `results/`
- `test/`
- `utils/`
  - `image_processor.py`
  
## Models

Put all models in models/.

- Prompt Generation: blip-image-captioning-large
- Image Outpainting: stable-diffusion-inpainting
- Image Super-resolution:ESRGAN

## Test
```shell script
python main.py
