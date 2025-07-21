import numpy as np
from phishintention.src.AWL_detector import find_element_type, element_config
from phishintention.src.OCR_aided_siamese import pred_siamese_OCR, phishpedia_config_OCR_easy
import os
from PIL import Image
import io
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as transform
import cv2
from contextlib import contextmanager
import time
import pickle
from typing import Union
import yaml
import subprocess
import importlib_resources  # backport for Python 3.8

@contextmanager
def time_block(label, store):
    start = time.time()
    yield
    store[label] = time.time() - start

# Helper function to perform image transformations
def pil2tensor(image: Image):
    transformation = transform.Compose([transform.Resize((256, 512)),
                                        transform.ToTensor()])
    return transformation(image)

def encoding2array(screenshot_encoding):
    screenshot_img = encoding2pil(screenshot_encoding)
    screenshot_img_arr = np.asarray(screenshot_img)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1).astype(np.uint8) # rgb2bgr
    return screenshot_img_arr

def encoding2pil(screenshot_encoding):
    screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
    return screenshot_img.convert("RGB")

def path2array(screenshot_path):
    screenshot_img = path2pil(screenshot_path)
    screenshot_img_arr = np.asarray(screenshot_img)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1).astype(np.uint8)
    return screenshot_img_arr

def path2encoding(screenshot_path):
    with open(screenshot_path, "rb") as image_file:
        screenshot_encoding = base64.b64encode(image_file.read())
    return screenshot_encoding

def path2pil(screenshot_path):
    screenshot_img = Image.open(screenshot_path).convert("RGB")
    return screenshot_img

# Helper function to run model inference
def run_object_detection(img_arr: np.ndarray, model):
    pred = model(img_arr)
    pred_i = pred["instances"].to('cpu')
    pred_classes = pred_i.pred_classes  # Boxes types
    pred_boxes = pred_i.pred_boxes.tensor  # Boxes coords
    pred_scores = pred_i.scores  # Boxes prediction scores

    pred_classes = pred_classes.detach().cpu()
    pred_boxes = pred_boxes.detach().cpu()
    pred_scores = pred_scores.detach().cpu()
    return pred_classes, pred_boxes, pred_scores

def run_classifier(image: Image, model):
    with torch.no_grad():
        pred_features = model.features(image[None, ...].to(PhishIntentionWrapper._DEVICE, dtype=torch.float))
        pred_orig = model(image[None, ...].to(PhishIntentionWrapper._DEVICE, dtype=torch.float))
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item()
        conf = F.softmax(pred_orig, dim=-1).detach().cpu()
    return pred, conf, pred_features

def load_config(cfg_path: Union[str, None] = None, reload_targetlist=False, device='cuda'):

    #################### '''Default''' ####################
    if cfg_path is None:
        with importlib_resources.open_text("phishintention", "configs.yaml") as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
    else:
        with open(cfg_path) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

    # element recognition model
    AWL_CFG_PATH = configs['AWL_MODEL']['CFG_PATH']
    AWL_WEIGHTS_PATH = configs['AWL_MODEL']['WEIGHTS_PATH']
    AWL_CONFIG, AWL_MODEL = element_config(rcnn_weights_path=AWL_WEIGHTS_PATH,
                                           rcnn_cfg_path=AWL_CFG_PATH, device=device)
    # siamese model
    print('Load protected logo list')
    if configs['SIAMESE_MODEL']['TARGETLIST_PATH'].endswith('.zip') \
            and not os.path.isdir('{}'.format(configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0])):
        subprocess.run('cd {} && unzip expand_targetlist.zip -d .'.format(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH'])), shell=True)

    SIAMESE_MODEL, OCR_MODEL = phishpedia_config_OCR_easy(
        num_classes=configs['SIAMESE_MODEL']['NUM_CLASSES'],
        weights_path=configs['SIAMESE_MODEL']['WEIGHTS_PATH'],
        ocr_weights_path=configs['SIAMESE_MODEL']['OCR_WEIGHTS_PATH'],
        )

    return AWL_MODEL, SIAMESE_MODEL, OCR_MODEL

class PhishIntentionWrapper:
    _caller_prefix = "PhishIntentionWrapper"
    SIAMESE_THRE_RELAX = 0.83
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    _RETRIES = 3

    def __init__(self, reload_targetlist=False):
        self._load_config(reload_targetlist)
        self._to_device()

    def _load_config(self, reload_targetlist):
        self.AWL_MODEL, self.SIAMESE_MODEL, self.OCR_MODEL = load_config(device=self._DEVICE, reload_targetlist=reload_targetlist)

    def _to_device(self):
        self.SIAMESE_MODEL.to(self._DEVICE)
        self.OCR_MODEL.to(self._DEVICE)

    def reset_model(self, config_path=None, reload_target=True):
        if not config_path:
            self._load_config(reload_targetlist=reload_target)
            self._to_device()
        else:
            self.AWL_MODEL, self.SIAMESE_MODEL, self.OCR_MODEL = load_config(cfg_path=config_path, device=self._DEVICE, reload_targetlist=reload_target)

    def _load_domain_map(self, domain_map_path):
        with open(domain_map_path, 'rb') as handle:
            return pickle.load(handle)

    def return_logo_feat(self, logo: Image):
        return pred_siamese_OCR(img=logo, model=self.SIAMESE_MODEL, ocr_model=self.OCR_MODEL)

    '''Draw utils'''
    def layout_vis(self, screenshot_path, pred_boxes, pred_classes):
        class_dict = {0: 'logo', 1: 'input', 2: 'button', 3: 'label', 4: 'block'}
        screenshot_img_arr = path2array(screenshot_path)
        if pred_boxes is None or len(pred_boxes) == 0:
            return screenshot_img_arr

        pred_boxes = pred_boxes.detach().cpu().numpy()
        pred_classes = pred_classes.detach().cpu().numpy()

        # draw rectangles
        for j, box in enumerate(pred_boxes):
            if class_dict[pred_classes[j].item()] != 'block':
                cv2.rectangle(screenshot_img_arr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (69,139,0), 2)
                cv2.putText(screenshot_img_arr, class_dict[pred_classes[j].item()], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        return screenshot_img_arr

    '''Layout detection'''
    def predict_all_uis(self, screenshot_encoding):
        screenshot_img_arr = encoding2array(screenshot_encoding)  # RGB2BGR
        pred_classes, pred_boxes, _ = run_object_detection(img_arr=screenshot_img_arr, model=self.AWL_MODEL)
        if pred_boxes is None or len(pred_boxes) == 0:
            return None, None
        return pred_boxes, pred_classes

    def predict_all_uis4type(self, screenshot_encoding, type):
        assert type in ['label', 'button', 'input', 'logo', 'block']
        pred_boxes, pred_classes = self.predict_all_uis(screenshot_encoding)

        if pred_boxes is None:
            return None

        pred_boxes, pred_classes = find_element_type(pred_boxes, pred_classes, bbox_type=type)
        if pred_boxes is None:
            return None

        return pred_boxes.detach().cpu().numpy()

    def predict_logos(self, screenshot_encoding):
        screenshot_img = encoding2pil(screenshot_encoding)
        pred_boxes = self.predict_all_uis4type(screenshot_encoding, 'logo')

        if pred_boxes is None:
            return None

        return [screenshot_img.crop((x1, y1, x2, y2)) for x1, y1, x2, y2 in pred_boxes]
