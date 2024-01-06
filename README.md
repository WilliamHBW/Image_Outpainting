# Image Outpainting

## Description

Combining LLMs for image outpainting automatically. Serves as the backend software for image processing.

## Environment

- see requirements.py

## File Structure

## Models

Put all models in models/.

- Prompt Generation: blip-image-captioning-large
- Image Outpainting: stable-diffusion-inpainting
- Image Super-resolution:pillow resize

## Test

## TODO
- solve upscale factor bug when input resolution is not in the range.
- use unconditional image-to-text model instead of conditioned.
- allow users to choose which LLM model they want to use for outpainting and image-to-text.
