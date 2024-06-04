
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
from selection_model.dataloader import ShotDataset, get_ocr_text
from tldextract import tldextract
from model_chain.PhishIntentionWrapper import PhishIntentionWrapper
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import base64
from PIL import Image, ImageDraw, ImageFont

def get_caption(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_image = img.convert("RGB")

    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption",
                                                         model_type="base_coco",
                                                         is_eval=True,
                                                         device=device)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    result = model.generate({"image": image})
    del model, vis_processors
    return ' '.join(result)

def transparent_text_injection(image, text, margin=(10, 10)):

    width, height = image.size
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Load a font
    font = ImageFont.truetype("./fonts/simfang.ttf", size=int(min(width, height)/10))  # Change this if you have a specific font in mind
    # Get text size
    text_width, text_height = draw.textsize(text, font=font)

    # Calculate position for the text to be in the bottom right corner with specified margin
    position = (image.width - text_width - margin[0], image.height - text_height - margin[1])

    # Ensure text does not exceed image boundaries
    position = (max(0, position[0]), max(0, position[1]))

    # Draw the text on the image with a thin edge
    draw.text(position, text, fill='black', font=font)

    return image

def get_ocr_text_coord(img_path):
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
        ocr_text = [line[1][0] for line in most_fit_results]
        ocr_coord = [line[0][0] + line[0][2] for line in most_fit_results]
    else:
        ocr_text = []
        ocr_coord = []

    return ocr_text, ocr_coord


def compute_overlap_areas_between_lists(bboxes1, bboxes2):
    # Convert bboxes lists to 3D arrays
    bboxes1 = np.array(bboxes1)[:, np.newaxis, :]
    bboxes2 = np.array(bboxes2)

    # Compute overlap for x and y axes separately
    overlap_x = np.maximum(0, np.minimum(bboxes1[:, :, 2], bboxes2[:, 2]) - np.maximum(bboxes1[:, :, 0], bboxes2[:, 0]))
    overlap_y = np.maximum(0, np.minimum(bboxes1[:, :, 3], bboxes2[:, 3]) - np.maximum(bboxes1[:, :, 1], bboxes2[:, 1]))

    # Compute overlapping areas for each pair
    overlap_areas = overlap_x * overlap_y
    return overlap_areas


def expand_bbox(bbox, image_width, image_height, expand_ratio):
    # Extract the coordinates
    x1, y1, x2, y2 = bbox

    # Calculate the center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate new width and height
    new_width = (x2 - x1) * expand_ratio
    new_height = (y2 - y1) * expand_ratio

    # Determine new coordinates
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # Ensure coordinates are legitimate
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]

class ShotDataset_Caption(Dataset):
    def __init__(self, annot_path):

        self.urls = []
        self.shot_paths = []
        self.labels = []
        self.phishintention_cls = PhishIntentionWrapper()

        for line in tqdm(open(annot_path).readlines()[::-1]):
            url, save_path, label = line.strip().split('\t')
            if os.path.exists(save_path):
                self.urls.append(url)
                self.shot_paths.append(save_path)
                self.labels.append(label) # A, B

        assert len(self.urls)==len(self.shot_paths)
        assert len(self.labels)==len(self.shot_paths)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx, adv=False):
        img_path = self.shot_paths[idx]
        url = self.urls[idx]
        label = self.labels[idx]
        # report logo
        screenshot_img = Image.open(img_path)
        screenshot_img = screenshot_img.convert("RGB")
        with open(img_path, "rb") as image_file:
            screenshot_encoding = base64.b64encode(image_file.read())
        logo_boxes = self.phishintention_cls.predict_all_uis4type(screenshot_encoding, 'logo')
        caption = ''
        extra_description = ''
        ocr_text = []
        reference_logo = None

        if (logo_boxes is not None) and len(logo_boxes):
            logo_box = logo_boxes[0] # get coordinate for logo
            x1, y1, x2, y2 = logo_box
            reference_logo = screenshot_img.crop((x1, y1, x2, y2)) # crop logo out
            if adv: # adversarial text injection attack
                injected_logo = transparent_text_injection(reference_logo.convert('RGB'), 'abc.com')
                caption = get_caption(injected_logo)
                screenshot_img.paste(injected_logo, (int(x1), int(y1)))
                img_path = img_path.replace('shot', 'shot_adv')
                screenshot_img.save(img_path)
            else:
                # generation caption for logo
                caption = get_caption(reference_logo)

            # get extra description on the webpage
            ocr_text, ocr_coord = get_ocr_text_coord(img_path)
            # expand the logo bbox a bit to see the surrounding region
            expand_logo_box = expand_bbox(logo_box, image_width=screenshot_img.size[0],
                                          image_height=screenshot_img.size[1], expand_ratio=1.5)

            if len(ocr_coord):
                # get the OCR text description surrounding the logo
                overlap_areas = compute_overlap_areas_between_lists([expand_logo_box], ocr_coord)
                extra_description = np.array(ocr_text)[overlap_areas[0] > 0].tolist()
                extra_description = ' '.join(extra_description)

        return url, label, caption, extra_description, ' '.join(ocr_text), reference_logo

def question_template(html_text):
    return \
        {
            "role": "user",
            "content": f"Given the HTML webpage text: <start>{html_text}<end>, Question: What is the brand's domain? Answer: "
        }

def question_template_caption(logo_caption, logo_ocr):
    return \
        {
            "role": "user",
            "content": f"Given the following description on the brand's logo: '{logo_caption}', and the logo's OCR text: '{logo_ocr}', Question: What is the brand's domain? Answer: "
        }

def question_template_industry(html_text):
    return \
        [
            {
                "role": "system",
                "content": f"Your task is to predict the industry sector given webpage content. Only give the industry sector, do not output any explanation."
            },
            {
                "role": "user",
                "content": f"Given the following webpage text: '{html_text}', Question: What is the webpage's industry sector? Answer: "
            }
        ]

def question_template_caption_industry(logo_caption, logo_ocr, industry):
    return \
        {
            "role": "user",
            "content": f"Given the following description on the brand's logo: '{logo_caption}', the logo's OCR text: '{logo_ocr}', and the industry sector: '{industry}' Question: What is the brand's domain? Answer: "
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
    with open('./brand_recognition/simple_prompt.json', 'w', encoding='utf-8') as f:
        json.dump(prompt, f)

    # url, label, ocr_text = dataset.__getitem__(743, use_ocr=False)
    # print(ocr_text)
    # print()


