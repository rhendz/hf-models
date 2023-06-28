import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pipe = StableDiffusionPipeline.from_pretrained(
    "hakurei/waifu-diffusion", torch_dtype=torch.float32
)

pipe = pipe.to(device)

prompt = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"

image = pipe(prompt).images[0]

image.save("test.png")
