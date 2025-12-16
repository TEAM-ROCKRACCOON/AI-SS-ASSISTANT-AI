# main.py
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import io

from img_utils import make_mask, expand_mask
from app_remove import MVRemoverController

# FastAPI
app = FastAPI(title="Image Cleaning Service")

# SmartEraser
weight_dtype = torch.float16
checkpoint_dir = "./ckpts/smarteraser-weights"
clip_dir = "openai/clip-vit-large-patch14"

controller = MVRemoverController(
    weight_dtype=weight_dtype,
    checkpoint_dir=checkpoint_dir,
    clip_dir=clip_dir
)

def run_smarteaser(
    img,
    mask,
    expand_px=20,
    guidance_scale=1.4,
    steps=50,
    seed=1767861908
):
    big_mask = expand_mask(mask, expand_px)
    input_dict = {"image": img, "mask": big_mask}

    result_list, _ = controller.infer(
        input_image=input_dict,
        ddim_steps=steps,
        scale=guidance_scale,
        seed=seed,
    )

    return result_list[0]


# API Endpoint
@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 임시 저장 (GroundingDINO 입력용)
    temp_path = "/tmp/input.jpg"
    img.save(temp_path)

    mask = make_mask(temp_path)
    result = run_smarteaser(img, mask)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    return {
        "elapsed": elapsed,
        "image_bytes": buf.getvalue()
    }

# Local Test 
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    image_path = "test/image/room3.jpg"
    img = Image.open(image_path).convert("RGB")

    start_time = time.time()
    mask = make_mask(image_path)
    result = run_smarteaser(img, mask)
    elapsed = time.time() - start_time

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(result)
    axes[1].set_title(f"Result\nelapsed: {elapsed:.2f}s")
    axes[1].axis("off")

    plt.show()
