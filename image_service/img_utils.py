# img_utils.py
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

# ===== Path 설정 (Grounded-SAM) =====
BASE_PATH = os.getcwd()
sys.path.append(os.path.join(BASE_PATH, "Grounded-Segment-Anything"))
sys.path.append(os.path.join(BASE_PATH, "Grounded-Segment-Anything", "GroundingDINO"))
sys.path.append(os.path.join(BASE_PATH, "Grounded-Segment-Anything", "segment_anything"))

from segment_anything import sam_model_registry, SamPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ===== Device =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== GroundingDINO =====
processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to(DEVICE)

# ===== SAM =====
SAM_CKPT = "models/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
sam.to(DEVICE)
sam_predictor = SamPredictor(sam)

# Utils
def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=1)


def generate_sam_mask(predictor, image, boxes):
    predictor.set_image(image)
    masks_out = []
    for box in boxes:
        masks, _, _ = predictor.predict(
            box=np.array(box),
            multimask_output=False
        )
        masks_out.append(masks[0])
    return masks_out


def make_mask(image_path, threshold=0.17):
    image_cv = cv2.imread(image_path)[:, :, ::-1]

    box_prompt = ["trash", "clothes", "garbage bags", "small objects", "box"]

    inputs = processor(
        images=image_cv,
        text=box_prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    logits = outputs.logits.sigmoid()[0]
    boxes = outputs.pred_boxes[0]

    scores = logits.max(dim=1)[0]
    keep = scores > threshold
    boxes = boxes[keep]

    if len(boxes) == 0:
        raise ValueError("No object detected")

    boxes_xyxy = box_cxcywh_to_xyxy(boxes)

    H, W = image_cv.shape[:2]
    boxes_xyxy[:, [0, 2]] *= W
    boxes_xyxy[:, [1, 3]] *= H
    boxes_filt = boxes_xyxy.cpu().numpy().astype(np.int32)

    masks = generate_sam_mask(sam_predictor, image_cv, boxes_filt)

    if len(masks) > 1:
        mask = np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8)
    else:
        mask = masks[0]

    mask_uint8 = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8).convert("RGB")


def expand_mask(mask_pil, expand_px=20):
    if expand_px == 0:
        return mask_pil

    mask = np.array(mask_pil.convert("L"))
    mask = np.where(mask > 128, 255, 0).astype(np.uint8)
    kernel = np.ones((expand_px, expand_px), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    return Image.fromarray(dilated_mask)