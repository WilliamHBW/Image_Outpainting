import torch
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import BlipProcessor, BlipForConditionalGeneration
import img_utils as iu

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
'''
def resize_image(image, width, height):
    return [image.resize((width, height)), image.resize((width, height))]
'''

def resize_image(img, width, height, steps=100):
    # ================================= define parameters ========================================
    # image resolution parameters
    target_res = (height, width)#[H,W]
    upscale_factor = 2

    # image-to-text parameters

    # diffusion model parameters
    noise_distance = 20
    noise_prob = 0
    lora_model = "../../models/eddiemauroLora2.safetensors"

    # ================================= initialization ========================================
    generated_images = []

    # get raw image
    raw_image = img.convert('RGB')
    print("Input image resolution: ",raw_image.size)

    # ================================= prompt generation ========================================
    # use image-to-text model to generate prompt
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    prompt_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # unconditional image to text 
    inputs = processor(raw_image, return_tensors = "pt")
    out = prompt_model.generate(**inputs)
    prompt = processor.decode(out[0], skip_special_tokens=True)
    print(prompt)

    # ================================= mask generation ========================================
    # generate init image and mask for diffusion inpainting
    down_factor = 512/max(target_res[0], target_res[1])
    down_res = (int(target_res[0]*down_factor), int(target_res[1]*down_factor))
    down_res = (down_res[0] - down_res[0]%8, down_res[1] - down_res[1]%8)
    init_image, mask_image = iu.get_init_mask_image(raw_image, down_res)
    init_image_ = Image.fromarray(init_image)
    mask_image_ = Image.fromarray(mask_image)
    # init_image_.save("./results/init_image.png")
    # mask_image_.save("./results/mask_image.png")

    # ================================= diffusion inpainting ========================================
    # build stable diffusion inpaint pipeline
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
    #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora_model)
    pipe.to(device)

    cur_images = pipe(
        prompt=prompt, 
        image=init_image_, 
        mask_image=mask_image_, 
        height=down_res[0], 
        width=down_res[1], 
        num_inference_steps=steps, 
        num_images_per_prompt=5).images
    
    # ================================= upscaling ========================================
    # calculate upscale factors and steps, each time do 2x upscale to ensure performance
    upscale_size = max(max(int(target_res[0]/down_res[0]), int(target_res[1]/down_res[1])),1)
    nstep = int(upscale_size/upscale_factor)

    for img in cur_images:
        low_res_img = img
        low_res = down_res
        print("raw resolution is :", img.size)
        # img.save("./results/"+str(idx)+"-"+str(low_res[0])+"-"+str(low_res[1])+".png")
        up_res_img = low_res_img
        for i in range(nstep):
            high_res = (low_res[0]*upscale_factor, low_res[1]*upscale_factor)
            low_res_img = low_res_img.resize((high_res[1], high_res[0]))
            up_res_img = iu.upscale(prompt, low_res_img)
            low_res = high_res
            low_res_img = up_res_img
        up_res_img = up_res_img.resize((target_res[1], target_res[0]), Image.LANCZOS)
        generated_images.append(up_res_img)
    
    return generated_images

if __name__=="__main__":
    test_path = "../../data/test1.jpg"
    images = resize_image(Image.open(test_path), 1080, 1920)

    for i, img in enumerate(images):
        img.save("../../data/result_"+str(i)+".png")