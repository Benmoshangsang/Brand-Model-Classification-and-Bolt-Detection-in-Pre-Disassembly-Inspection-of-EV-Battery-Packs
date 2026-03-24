# tools/ocr_utils.py
import paddleocr
import numpy as np
import torch
import hashlib

ocr_model = paddleocr.OCR(use_angle_cls=True, lang='ch')  # 已在你的环境中安装成功

def extract_text_from_image(image, bbox):
    x, y, w, h = [int(v) for v in bbox]
    crop = image[y:y+h, x:x+w]
    result = ocr_model.ocr(crop, cls=True)
    if result and len(result[0]) > 0:
        text = result[0][0][1][0]
        return text
    return ""

def encode_text_to_vector(text, max_len=16, dim=128):
    if not text:
        return torch.zeros(dim)
    hash_val = hashlib.md5(text.encode()).hexdigest()
    vec = np.array([int(hash_val[i:i+2], 16) for i in range(0, min(len(hash_val), dim * 2), 2)])
    vec = vec.astype(np.float32)
    vec = vec / 255.0  # normalize
    if len(vec) < dim:
        vec = np.pad(vec, (0, dim - len(vec)), mode='constant')
    return torch.tensor(vec[:dim])
