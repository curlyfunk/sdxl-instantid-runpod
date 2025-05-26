import runpod
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from io import BytesIO
import base64

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

def encode_base64_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def handler(event):
    prompt = event["input"].get("prompt", "a photo of a person")
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        added_cond_kwargs={}  # това оправя проблема
    ).images[0]
    return {"output_image": encode_base64_image(image)}

runpod.serverless.start({"handler": handler})
