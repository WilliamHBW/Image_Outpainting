import numpy as np
from perlin_noise import PerlinNoise
import click
from sklearn.preprocessing import minmax_scale
from PIL import Image
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

def add_noise_to_img(img, dist_from_center=50, amount_of_noise=50):
  img_arr = np.array(img)
  mask = np.zeros_like(img_arr)

  dimx = img_arr.shape[0]
  dimy = img_arr.shape[1]
  prob_noise = 1
  dist_noise = 0.5
  x,y = dimx,dimy
  rx,ry = dimx,dimy


  tr_size = dist_from_center/200.0  # [0,100]
  tr_strt = amount_of_noise  #[0,90]


  for i in range(rx):
    for j in range(int(ry*tr_size)):
      if 100 > tr_strt+(j/(ry*tr_size))*(100-tr_strt):
        mask[i][j] = [255,255,255]
        img_arr[i][j] = [0,0,0] 

  for i in range(int(rx*tr_size)):
    for j in range(ry):
      if 100 > tr_strt+(i/(rx*tr_size))*(100-tr_strt):
        mask[i][j] = [255,255,255]
        img_arr[i][j] = [0,0,0] 

  for i in range(rx-1, -1, -1):
    for j in range(ry-1,int(ry-(ry*tr_size)), -1):
      if 100 > tr_strt+((ry-j)/(ry*tr_size))*(100-tr_strt):
        img_arr[i][j] = [0,0,0] 
        mask[i][j] = [255,255,255]

  for i in range(rx-1,int(rx-(rx*tr_size)), -1):
    for j in range(ry-1,-1, -1):
      if 100 > tr_strt+((rx-i)/(rx*tr_size))*(100-tr_strt):
        img_arr[i][j] = [0,0,0] 
        mask[i][j] = [255,255,255]



  return img_arr, mask

noise11 = PerlinNoise(octaves=10)
noise12 = PerlinNoise(octaves=5)

noise21 = PerlinNoise(octaves=10)
noise22 = PerlinNoise(octaves=5)

noise31 = PerlinNoise(octaves=10)
noise32 = PerlinNoise(octaves=5)


def noise_mult_1(i,j, hpix=512,wpix=512):
  return noise11([i/hpix, j/wpix]) + 0.5 * noise12([i/hpix, j/wpix]) #+ 0.25 * noise3([i/hpix, j/wpix]) + 1.125 * noise4([i/hpix, j/wpix])

def noise_mult_2(i,j, hpix=512,wpix=512):
  return noise21([i/hpix, j/wpix]) + 0.5 * noise22([i/hpix, j/wpix]) #+ 0.25 * noise3([i/hpix, j/wpix]) + 1.125 * noise4([i/hpix, j/wpix])

def noise_mult_3(i,j, hpix=512,wpix=512):
  return noise31([i/hpix, j/wpix]) + 0.5 * noise32([i/hpix, j/wpix]) #+ 0.25 * noise3([i/hpix, j/wpix]) + 1.125 * noise4([i/hpix, j/wpix])

def get_mask_image(img, downscale_factor=4, noise_distance=20, noise_prob=0):
  img_downscaled = img.resize((int(img.size[0] / downscale_factor), int(img.size[1] / downscale_factor)))
  noised_img, mask = add_noise_to_img(img_downscaled, 20, 0)
  img_arr = np.array(img)
  dimx = img_arr.shape[0]
  dimy = img_arr.shape[1]
  y_up = int((dimx - (dimx / downscale_factor)) / 2)
  y_down = y_up + int(dimx / downscale_factor)
  x_left = int((dimy - (dimy / downscale_factor)) / 2)
  x_right = x_left + int(dimy / downscale_factor)

  full_mask = np.zeros_like(np.array(img))
  full_mask.fill(255)
  for i in range(dimx):
    for j in range(dimy):
      if i >= y_up and j >= x_left and i < y_down and j < x_right:
        full_mask[i][j] = mask[i-y_up][j-x_left]

  return full_mask, noised_img



def get_init_image(noised_img, full_mask, downscale_factor, hpix=512,wpix=512):
  click.echo('Generating noise ...')
  pic = [[[noise_mult_1(i,j), noise_mult_2(i,j), noise_mult_3(i,j) ] for j in range(wpix)] for i in range(hpix)]
  click.echo('Noise generated !')
  scaled_noise = minmax_scale(np.array(pic).flatten(), (0,255)).reshape((hpix,wpix, 3))
  scaled_noise = scaled_noise.astype(np.uint8)

  y_up = int((hpix - (hpix / downscale_factor)) / 2)
  y_down = y_up + int(hpix / downscale_factor)
  x_left = int((wpix - (wpix / downscale_factor)) / 2)
  x_right = x_left + int(wpix / downscale_factor)

  init_image = scaled_noise.copy()
  noised_img_arr = np.array(noised_img)
  #print(noised_img_arr.shape)

  for i in range(hpix):
    for j in range(wpix):
      if i >= y_up and j >= x_left and i < y_down and j < x_right and list(full_mask[i][j]) != [255,255,255]:
        init_image[i][j] = noised_img_arr[i-y_up][j-x_left]
  return init_image

def get_init_mask_image(img, downscale_factor=4, noise_distance=20, noise_prob=0):
  full_mask, noised_image = get_mask_image(img, downscale_factor,noise_distance, noise_prob)
  hpix = full_mask.shape[0]
  wpix = full_mask.shape[1]
  init_image = get_init_image(noised_image, full_mask, downscale_factor, hpix, wpix)
  return init_image, full_mask


def get_init_image(noised_img, full_mask, hpix=512,wpix=512):
  click.echo('Generating noise ...')
  pic = [[[noise_mult_1(i,j), noise_mult_2(i,j), noise_mult_3(i,j) ] for j in range(wpix)] for i in range(hpix)]
  click.echo('Noise generated !')
  scaled_noise = minmax_scale(np.array(pic).flatten(), (0,255)).reshape((hpix,wpix, 3))
  scaled_noise = scaled_noise.astype(np.uint8)

  noised_img_arr = np.array(noised_img)
  dimH, dimW, _ = noised_img_arr.shape
  y_up = int((hpix - dimH) / 2)
  y_down = y_up + dimH
  x_left = int((wpix - dimW) / 2)
  x_right = x_left + dimW

  init_image = scaled_noise.copy()

  for i in range(hpix):
    for j in range(wpix):
      if list(full_mask[i][j]) != [255,255,255]:
        init_image[i][j] = noised_img_arr[i-y_up][j-x_left]
  return init_image

def get_init_mask_image(img, res=(512,512)):
  full_mask = np.zeros((res[0],res[1],3),dtype=np.uint8)
  full_mask.fill(255)
  img_arr = np.array(img)
  dimH = img_arr.shape[0]
  dimW = img_arr.shape[1]
  downscale_factor = min(res[0]/dimH, res[1]/dimW)
  down_img = img.resize((int(dimW*downscale_factor),int(dimH*downscale_factor)), Image.LANCZOS)
  down_mask = np.zeros_like(down_img)
                         
  h_top = int((res[0] - dimH*downscale_factor)/2)
  h_down = h_top + int(dimH*downscale_factor)
  w_left = int((res[1] - dimW*downscale_factor)/2)
  w_right = w_left + int(dimW*downscale_factor)
  
  for i in range(res[0]):
    for j in range(res[1]):
      if i>h_top+1 and i<h_down-1 and j>w_left+1 and j<w_right-1:
        full_mask[i][j] = down_mask[i-h_top][j-w_left]

  init_image = get_init_image(down_img, full_mask, res[0], res[1])
  return init_image, full_mask

def downscale_image(img):
    # Load the image
    # Compute the downscaling factor
    max_dimension = max(img.size)
    scale_factor = 512 / max_dimension if max_dimension > 512 else 1

    # Compute new dimensions
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    return resized_img

def upscale(prompt, img):
  width, height = img.size
  out_img = img.resize((height*2, width*2), Image.Resampling.LANCZOS)
  return out_img

if __name__=='__main__':
  input_image = "./640.jpg"
  target_res = (720,1080)#[H,W]
  raw_image = Image.open(input_image).convert('RGB')
  print("Input image resolution: ",raw_image.size)#W,H
  down_image = downscale_image(raw_image)
  down_factor = 512/max(target_res[0], target_res[1])
  down_res = (int(target_res[0]*down_factor), int(target_res[1]*down_factor))
  down_res = (down_res[0] - down_res[0]%8, down_res[1] - down_res[1]%8)
  init_image, mask_image = get_init_mask_image(raw_image, down_res)
  init_image_ = Image.fromarray(init_image)
  mask_image_ = Image.fromarray(mask_image)
  init_image_.save("init_image.png")
  mask_image_.save("mask_image.png")