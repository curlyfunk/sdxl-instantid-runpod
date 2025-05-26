import runpod
import base64
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline

# Зареждане на модела
model_path = "/workspace/models/sdxl-turbo.safetensors"
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

def decode_base64_image(b64_str):
    image_data = base64.b64decode(b64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

def encode_base64_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def handler(event):
    input_data = event["input"]
    prompt = input_data.get("prompt", "a photo of a person")
    face_b64 = input_data.get("face_image")

    # Декодиране на лицето (може да се подаде на InstantID по-късно)
    face_image = decode_base64_image(face_b64) if face_b64 else None

    # TODO: интеграция с InstantID (embedding injection)
    # Засега просто генерираме изображение от prompt
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

    return {"output_image": encode_base64_image(image)}

runpod.serverless.start({"handler": handler})
