from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, BatchSampler
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from tqdm import tqdm
from paddleocr import PaddleOCR
import math
from scripts.utils.PhishIntentionWrapper import PhishIntentionWrapper
import torch
from lavis.models import load_model_and_preprocess
import base64
from PIL import Image, ImageDraw, ImageFont
import io
from lxml import html
import re
from scripts.data.dom_utils import prune_tree

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

class ShotDataset(Dataset):
    def __init__(self, annot_path):

        self.urls = []
        self.shot_paths = []
        self.labels = []

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
            ocr_text = get_ocr_text(img_path, html_path)
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

def format_input_generation(
    html_path, dom_path, keep_html_brackets=False
):
    with io.open(html_path, 'r', encoding='utf-8') as f:
        page = f.read()
    if len(page) == 0:
        return ''
    dom_tree = html.fromstring(page)
    tree_repr = prune_tree(dom_tree, dom_path)
    if tree_repr is None:
        return ''
    tree_repr = tree_repr.replace('"', " ")
    tree_repr = (
        tree_repr.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
    )
    tree_repr = re.sub(r"<text>(.*?)</text>", r"\1", tree_repr)
    if not keep_html_brackets:
        tree_repr = tree_repr.replace("/>", "$/$>")
        tree_repr = re.sub(r"</(.+?)>", r")", tree_repr)
        tree_repr = re.sub(r"<(.+?)>", r"(\1", tree_repr)
        tree_repr = tree_repr.replace("$/$", ")")

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
        tree_repr = tree_repr.replace(k, v)
    tree_repr = re.sub(r"\s+", " ", tree_repr).strip()

    return tree_repr

class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size):
        """
            BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
            Returns batches of size n_classes * n_samples
        """
        self.labels = labels
        self.labels_set = list(set(np.asarray(self.labels)))
        self.label_to_indices = {label: np.where(np.asarray(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}

        self.count = 0
        self.n_classes = len(self.labels_set)
        self.batch_size = batch_size
        self.n_samples = batch_size // self.n_classes
        self.n_dataset = len(self.labels)
        # self.n_dataset = min([len(x) for x in list(self.label_to_indices.values())]) * self.n_classes # fixme: imbalanced dataset, downsample the majority class to align with minority class

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False) # randomly choose n_classes from all classes
            indices = []
            for class_ in classes:
                start_index = self.used_label_indices_count[class_]
                end_index = start_index + self.n_samples
                indices.extend(self.label_to_indices[class_][start_index:end_index])
                self.used_label_indices_count[class_] += self.n_samples # have been visited before
                # if the next end_index will exceed the length, shuffle and select again
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

class ButtonDataset(Dataset):
    def __init__(self, annot_path, root, preprocess):

        self.img_paths = []
        self.urls = []
        self.dom_paths = []
        self.labels = []  # todo
        self.tree_reprs = []
        self.root = root
        self.preprocess = preprocess

        path_set = set()

        for line in tqdm(open(annot_path).readlines()[::-1]):
            url, dom, save_path, label = line.strip().split('\t')
            if (url, dom) in path_set:
                continue
            else:
                path_set.add((url, dom))
                html_path = '{}/{}/index.html'.format(root, url.split('https://')[1])
                if not os.path.exists(html_path):
                    continue

                self.img_paths.append(save_path)
                self.urls.append(url)
                self.dom_paths.append(dom.lower())
                # self.tree_reprs.append(tree_repr)
                self.labels.append(int(label))

        assert len(self.img_paths) == len(self.urls)
        assert len(self.img_paths) == len(self.dom_paths)
        assert len(self.img_paths) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.preprocess(Image.open(img_path))
        label = self.labels[idx]
        url = self.urls[idx]
        dom_path = self.dom_paths[idx]
        html_path = '{}/{}/index.html'.format(self.root, url.split('https://')[1])
        tree_repr = format_input_generation(html_path, dom_path)

        return image, label, dom_path, tree_repr, img_path, url
