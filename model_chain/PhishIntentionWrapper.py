import numpy as np
from phishintention.phishintention_config import load_config
from phishintention.src.AWL_detector import find_element_type
from PIL import Image
import io
import base64
import torch

class PhishIntentionWrapper:
    _caller_prefix = "PhishIntentionWrapper"
    SIAMESE_THRE_RELAX = 0.85
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, reload_targetlist=False):
        self._load_config(reload_targetlist)
        self._to_device()

    def _load_config(self, reload_targetlist):
        self.AWL_MODEL, self.CRP_CLASSIFIER, self.CRP_LOCATOR_MODEL, self.SIAMESE_MODEL, self.OCR_MODEL, \
        self.SIAMESE_THRE, self.LOGO_FEATS, self.LOGO_FILES, self.DOMAIN_MAP_PATH = load_config(device=self._DEVICE, reload_targetlist=reload_targetlist)

    def _to_device(self):
        self.CRP_CLASSIFIER.to(self._DEVICE)
        self.OCR_MODEL.to(self._DEVICE)

    def reset_model(self, config_path):
        self._load_config(reload_targetlist=True)
        self._to_device()
        print(f'Length of reference list = {len(self.LOGO_FEATS)}')

    @staticmethod
    def element_recognition_reimplement(img_arr: np.ndarray, model):
        pred = model(img_arr)
        pred_i = pred["instances"].to('cpu')
        pred_classes = pred_i.pred_classes  # Boxes types
        pred_boxes = pred_i.pred_boxes.tensor  # Boxes coords
        pred_scores = pred_i.scores  # Boxes prediction scores

        pred_classes = pred_classes.detach().cpu()
        pred_boxes = pred_boxes.detach().cpu()
        pred_scores = pred_scores.detach().cpu()

        return pred_classes, pred_boxes, pred_scores

    def _decode_and_convert_image(self, screenshot_encoding):
        screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
        return screenshot_img.convert("RGB")

    def return_all_bboxes(self, screenshot_encoding):
        screenshot_img = self._decode_and_convert_image(screenshot_encoding)
        screenshot_img_arr = np.flip(np.asarray(screenshot_img), -1)  # RGB2BGR
        pred_classes, pred_boxes, pred_scores = self.element_recognition_reimplement(img_arr=screenshot_img_arr,
                                                                                     model=self.AWL_MODEL)
        if pred_boxes is None or len(pred_boxes) == 0:
            return None, None
        return pred_boxes, pred_classes

    def return_all_bboxes4type(self, screenshot_encoding, type):
        assert type in ['label', 'button', 'input', 'logo', 'block']
        pred_boxes, pred_classes = self.return_all_bboxes(screenshot_encoding)

        if pred_boxes is None:
            return None

        pred_boxes, pred_classes = find_element_type(pred_boxes, pred_classes, bbox_type=type)
        if not pred_boxes:
            return None

        return pred_boxes.detach().cpu().numpy()

    def return_all_logos(self, screenshot_encoding):
        screenshot_img = self._decode_and_convert_image(screenshot_encoding)
        pred_boxes = self.return_all_bboxes4type(screenshot_encoding, 'logo')

        if pred_boxes is None:
            return None

        return [screenshot_img.crop((x1, y1, x2, y2)) for x1, y1, x2, y2 in pred_boxes]
