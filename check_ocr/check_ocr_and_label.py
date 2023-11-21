import os.path
import random
from typing import List, Dict

import requests
from datasets import load_dataset

import base64
import io
from PIL import Image, ImageDraw

raw_dataset = load_dataset('../docvqa_zh')


def get_sample():
    data = raw_dataset['train'].select(range(3))

    for i in range(3):
        sample = data[i]
        raw_img = base64.b64decode(sample['image'])
        with open(f'img-{i}.png', 'wb') as f:
            f.write(raw_img)
        pil_img = Image.open(io.BytesIO(raw_img))
        draw = ImageDraw.ImageDraw(pil_img)
        for b in sample['bbox']:
            draw.rectangle(b, outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        with open(f'img-{i}-bbox.png', 'wb') as f:
            pil_img.save(f)


def do_ocr(filepath) -> List[Dict]:
    with open(filepath, 'rb') as f:
        result = requests.post('http://localhost:5000', files={'file': (os.path.basename(filepath), f)}).json()
        return result


def ocr_draw(filepath, i):
    ocr_res = do_ocr(filepath)
    pil_img = Image.open(filepath)
    draw = ImageDraw.ImageDraw(pil_img)
    for item in ocr_res:
        draw.rectangle(item['bbox'], outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    with open(f'img-{i}-ocr.png', 'wb') as f:
        pil_img.save(f)


if __name__ == '__main__':
    ocr_draw('/check_ocr/img-0.png', 0)
