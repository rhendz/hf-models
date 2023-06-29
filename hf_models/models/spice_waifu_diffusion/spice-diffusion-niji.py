import torch
from diffusers import StableDiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionPipeline.from_pretrained(
    "hakurei/waifu-diffusion", torch_dtype=torch.float32, safety_checker=None
)

pipe = pipe.to(device)

prompt = "masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, watercolor, night, turtleneck"
negative_prompt = ""

# save prompts for each image generation

# create feedback loop on frontend for users to select accurate prompt-image pairs for free generations

num_images_per_prompt = 6
pipe_out = pipe(
    prompt, num_images_per_prompt=num_images_per_prompt, negative_prompt=negative_prompt
)

for i in range(num_images_per_prompt):
    image = pipe_out.images[i]
    image.save(f"test/test-{i}.png")
