
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
from selection_model.draw_utils import draw_ocr

class ShotDataset(Dataset):
    def __init__(self, annot_path):

        self.urls = []
        self.shot_paths = []
        self.labels = []
        self.language_list = ['en', 'ch', 'ru', 'japan', 'fa', 'ar', 'korean', 'vi', 'ms',
                             'fr', 'german', 'it', 'es', 'pt', 'uk', 'be', 'te',
                             'sa', 'ta', 'nl', 'tr', 'ga']

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

    def __getitem__(self, idx, use_ocr=True):
        img_path = self.shot_paths[idx]
        html_path = img_path.replace('shot.png', 'index.html')
        url = self.urls[idx]
        label = self.labels[idx]

        if use_ocr:
            # ocr2text
            most_fit_lang = self.language_list[0]
            best_conf = 0
            most_fit_results = ''
            for lang in self.language_list:
                ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log = False)  # need to run only once to download and load model into memory
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
                    if self.language_list.index(lang) - self.language_list.index(most_fit_lang) >= 2:  # local best
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

            return url, label, ocr_text

        else:
            with io.open(html_path, 'r', encoding='utf-8') as f:
                page = f.read()
            if len(page):
                dom_tree = html.fromstring(page, parser=html.HTMLParser(remove_comments=True))
                unwanted = dom_tree.xpath('//script|//style|//head')
                for u in unwanted:
                    u.drop_tree()
                html_text = html.tostring(dom_tree, encoding='unicode')
                html_text = html_text.replace('"', " ")
                html_text = (
                    html_text.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
                )
                html_text = re.sub(r"<text>(.*?)</text>", r"\1", html_text)
                html_escape_table = [
                    ("&quot;", '"'),
                    ("&amp;", "&"),
                    ("&lt;", "<"),
                    ("&gt;", ">"),
                    ("&nbsp;", " "),
                    ("&ndash;", "-"),
                    ("&rsquo;", "'"),
                    ("&lsquo;", "'"),
                    ("&ldquo;", '"'),
                    ("&rdquo;", '"'),
                    ("&#39;", "'"),
                    ("&#40;", "("),
                    ("&#41;", ")"),
                ]
                for k, v in html_escape_table:
                    html_text = html_text.replace(k, v)
                html_text = re.sub(r"\s+", " ", html_text).strip()
            else:
                html_text = ''
            return url, label, html_text


def question_template(html_text):
    return \
        {
            "role": "user",
            "content": f"Given the HTML webpage text: <start>{html_text}<end>, \n Question: A. This is a credential-requiring page. B. This is not a credential-requiring page. \n Answer: "
        }

def construct_prompt(dataset, few_shot_k = 4, use_ocr=False):
    prompt = [
            {
                "role": "system",
                "content": "Given the webpage HTML, your task is to decide the status of the webpage."
                           "A credential-requiring page is where the users are asked to fill-in their sensitive information, including usernames, passwords, birth date; contact details such as addresses, phone numbers, emails, and financial information such as credit card numbers, social security numbers etc. "
            },
    ]

    labels_set = list(set(np.asarray(dataset.labels)))
    label_to_indices = {label: np.where(np.asarray(dataset.labels) == label)[0] for label in labels_set}
    print(label_to_indices)
    few_shot_per_cls = few_shot_k // len(labels_set) # number of samples per class
    for lbl in labels_set:
        for it in range(few_shot_per_cls):
            ind = label_to_indices[lbl][it]
            url, _, html_text = dataset.__getitem__(ind, use_ocr)
            prompt.append(question_template(html_text))
            prompt.append({
                "role": "assistant",
                "content": f"{lbl}."
            })

    return prompt

if __name__ == '__main__':

    # python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    # pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+

    dataset = ShotDataset(annot_path='./datasets/alexa_screenshots.txt')
    print(len(dataset))
    print(Counter(dataset.labels))

    # draw result
    # img_path = dataset.shot_paths[0]
    # most_fit_lang = dataset.language_list[0]
    # best_conf = 0
    # most_fit_results = ''
    # for lang in dataset.language_list:
    #     ocr = PaddleOCR(use_angle_cls=True, lang=lang,
    #                     show_log=False)  # need to run only once to download and load model into memory
    #     result = ocr.ocr(img_path, cls=True)
    #     median_conf = np.median([x[-1][1] for x in result[0]])
    #     # print(lang, median_conf)
    #     if math.isnan(median_conf):
    #         break
    #     if median_conf > best_conf and median_conf >= 0.9:
    #         best_conf = median_conf
    #         most_fit_lang = lang
    #         most_fit_results = result
    #     if median_conf >= 0.98:
    #         most_fit_results = result
    #         break
    #     if best_conf > 0:
    #         if dataset.language_list.index(lang) - dataset.language_list.index(most_fit_lang) >= 2:  # local best
    #             break
    # most_fit_results = most_fit_results[0]
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in most_fit_results]
    # txts = [line[1][0] for line in most_fit_results]
    # scores = [line[1][1] for line in most_fit_results]
    # im_show = draw_ocr(image, boxes, txts, scores, font_path='./selection_model/fonts/simfang.ttf')
    # im_show = Image.fromarray(im_show)
    # im_show.save('./debug.png')
    # exit()

    # prompt = construct_prompt(dataset, 2, True)
    # with open('./selection_model/prompt.json', 'w', encoding='utf-8') as f:
    #     json.dump(prompt, f)

    # url, label, ocr_text = dataset.__getitem__(743, use_ocr=False)
    # print(ocr_text)
    # print()


