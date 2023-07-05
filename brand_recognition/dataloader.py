
from PIL import Image
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler
import io
from lxml import html
import re
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from tqdm import tqdm
from paddleocr import PaddleOCR
import math
from collections import Counter
from selection_model.dataloader import ShotDataset
from tldextract import tldextract

def question_template(html_text):
    return \
        {
            "role": "user",
            "content": f"Given the HTML webpage text: <start>{html_text}<end>, Question: What is the brand's domain? Answer: "
        }

def construct_prompt(dataset, few_shot_k = 4, use_ocr=False):
    prompt = [
        {
            "role": "system",
            "content": "Given the webpage HTML, your task is to decide the brand of the webpage. Just give short answer of the brand's domain."
        },
    ]

    for it in range(few_shot_k):
        url, _, html_text = dataset.__getitem__(it, use_ocr)
        domain = tldextract.extract(url).domain+'.'+tldextract.extract(url).suffix
        prompt.append(question_template(html_text))
        prompt.append({
            "role": "assistant",
            "content": f"{domain}."
        })

    return prompt

if __name__ == '__main__':

    dataset = ShotDataset(annot_path='./datasets/alexa_screenshots.txt')
    print(len(dataset))
    print(Counter(dataset.labels))

    prompt = construct_prompt(dataset, 3, True)
    with open('./brand_recognition/prompt.json', 'w', encoding='utf-8') as f:
        json.dump(prompt, f)

    # url, label, ocr_text = dataset.__getitem__(743, use_ocr=False)
    # print(ocr_text)
    # print()


