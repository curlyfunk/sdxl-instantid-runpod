import runpod
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from io import BytesIO
import base64
import os
import sys

sys.path.append("/workspace/InstantID")  # достъп до local imports от InstantID

# InstantID зависимости
from pipelines.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
from insightface.app import FaceAnalysis
from utils.insightface_helpers import get_image_embedding

# Зареждане на модела
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Зареждане на InstantID модули
face_encoder_path = "/workspace/InstantID/IP-Adapter/models/ip-adapter-faceid_sdxl.bin"
pipe.load_ip_adapter_instantid(face_encoder_path)

app = FaceAnalysis(name='buffalo_l', root="/workspace/InstantID/checkpoints", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))

# Кодиране на изображение в base64
def encode_base64_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Декодиране от base64 към PIL
def decode_base64_image(base64_str):
    return Image.open(BytesIO(base64.b64decode(base64_str))).convert("RGB")

def handler(event):
    prompt = event["input"].get("prompt", "portrait photo")
    face_b64 = event["input"].get("face_image")

    added_cond_kwargs = {}

    if face_b64:
        face_image = decode_base64_image(face_b64)
        image_embed = get_image_embedding(face_image, app, pipe.image_encoder)
        added_cond_kwargs["image_embeds"] = image_embed

    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        added_cond_kwargs=added_cond_kwargs
    ).images[0]

    return {"output_image": encode_base64_image(image)}

runpod.serverless.start({"handler": handler})
