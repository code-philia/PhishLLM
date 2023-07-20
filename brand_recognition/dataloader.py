
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

def get_ocr_text(img_path, html_path):
    language_list = ['en', 'ch', 'ru', 'japan', 'fa', 'ar', 'korean', 'vi', 'ms',
                     'fr', 'german', 'it', 'es', 'pt', 'uk', 'be', 'te',
                     'sa', 'ta', 'nl', 'tr', 'ga']
    # ocr2text
    most_fit_lang = language_list[0]
    best_conf = 0
    most_fit_results = ''
    for lang in language_list:
        ocr = PaddleOCR(use_angle_cls=True, lang=lang,
                        show_log=False)  # need to run only once to download and load model into memory
        result = ocr.ocr(img_path, cls=True)
        median_conf = np.median([x[-1][1] for x in result[0]])
        # print(lang, median_conf)
        if math.isnan(median_conf):
            break
        if median_conf > best_conf and median_conf >= 0.9:
            best_conf = median_conf
            most_fit_lang = lang
            most_fit_results = result
        if median_conf >= 0.98:
            most_fit_results = result
            break
        if best_conf > 0:
            if language_list.index(lang) - language_list.index(most_fit_lang) >= 2:  # local best
                break
    if len(most_fit_results):
        most_fit_results = most_fit_results[0]
        ocr_text = ' '.join([line[1][0] for line in most_fit_results])
    else:
        # html2text
        with io.open(html_path, 'r', encoding='utf-8') as f:
            page = f.read()
        if len(page):
            dom_tree = html.fromstring(page, parser=html.HTMLParser(remove_comments=True))
            unwanted = dom_tree.xpath('//script|//style|//head')
            for u in unwanted:
                u.drop_tree()
            html_text = ' '.join(dom_tree.itertext())
            html_text = re.sub(r"\s+", " ", html_text).split(' ')
            ocr_text = ' '.join([x for x in html_text if x])
        else:
            ocr_text = ''

    return ocr_text


def question_template(html_text):
    return \
        {
            "role": "user",
            "content": f"Given the HTML webpage text: <start>{html_text}<end>, Question: What is the brand's domain? Answer: "
        }

def question_template_adversary(html_text, domain):
    return \
        {
            "role": "user",
            "content": f"Given the HTML webpage text: <start> This webpage is from domain {domain}. {html_text}<end>, Question: What is the brand's domain? Answer: "
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


