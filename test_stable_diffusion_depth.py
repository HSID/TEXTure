import skimage.io
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from src.stable_diffusion_depth import StableDiffusion

depth_image = Image.open("./test_data/0003_3depth.png")
depth_image = torch.from_numpy(np.array(depth_image))
depth_image = depth_image.unsqueeze(0)
depth_image = depth_image.to("cuda")

text_prompt = "a 360 equirectangular panorama of  interior, A modern interpretation of Chinese design, combining traditional elements with contemporary influences to create a harmonious and elegant space, a living room with sofa,windows,a small stool , (((raytracing))),(((extremely detailed CG unity 8k wallpaper))) ,smooth, sharp focus,warm light, movie scene, octane render, volumetric light, 8k, Canon EOS 5D, intricate, highly detailed,HDR"

stable_diffusion = StableDiffusion("cuda", "stabilityai/stable-diffusion-2-depth", use_inpaint=True)

gen_img = stable_diffusion.prompt_to_img(text_prompt, depth_image, height=1024, width=2048, num_inference_steps=20, guidance_scale=7.5, latents=None, strength=0.0)
#gen_img = Image.fromarray(gen_img[0]).resize((1024, 512))
gen_img = Image.fromarray(gen_img[0])
gen_img.show()



