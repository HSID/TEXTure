from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet")
  
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-depth", subfolder="unet")
