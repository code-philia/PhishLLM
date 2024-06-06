import numpy as np
from phishintention.phishintention_config import load_config
from phishintention.src.AWL_detector import find_element_type
from phishintention.src.OCR_aided_siamese import pred_siamese_OCR
from phishintention.src.OCR_siamese_utils.utils import brand_converter, resolution_alignment
from phishintention.src.crp_classifier_utils.bit_pytorch.grid_divider import coord2pixel_reverse
from phishintention.src.crp_classifier import html_heuristic, credential_classifier_mixed_al
from scripts.utils.utils import Regexes
import re
import os
from PIL import Image
import io
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as transform
import time
import cv2
from contextlib import contextmanager
import time
import pickle
import tldextract
import selenium.common.exceptions

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


class PhishIntentionWrapper:
    _caller_prefix = "PhishIntentionWrapper"
    SIAMESE_THRE_RELAX = 0.83
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    _RETRIES = 3

    def __init__(self, reload_targetlist=False):
        self._load_config(reload_targetlist)
        self._to_device()

    def _load_config(self, reload_targetlist):
        self.AWL_MODEL, self.CRP_CLASSIFIER, self.CRP_LOCATOR_MODEL, self.SIAMESE_MODEL, self.OCR_MODEL, \
        self.SIAMESE_THRE, self.LOGO_FEATS, self.LOGO_FILES, self.DOMAIN_MAP_PATH = load_config(device=self._DEVICE, reload_targetlist=reload_targetlist)
        print(f'Length of reference list = {len(self.LOGO_FEATS)}')

    def _to_device(self):
        self.CRP_CLASSIFIER.to(self._DEVICE)
        self.OCR_MODEL.to(self._DEVICE)

    def reset_model(self, config_path=None, reload_target=True):
        if not config_path:
            self._load_config(reload_targetlist=reload_target)
            self._to_device()
            print(f'Length of reference list = {len(self.LOGO_FEATS)}')
        else:
            self.AWL_MODEL, self.CRP_CLASSIFIER, self.CRP_LOCATOR_MODEL, self.SIAMESE_MODEL, self.OCR_MODEL, \
            self.SIAMESE_THRE, self.LOGO_FEATS, self.LOGO_FILES, self.DOMAIN_MAP_PATH = load_config(cfg_path=config_path,
                                                                                                    device=self._DEVICE,
                                                                                                    reload_targetlist=reload_target)
            print(f'Length of reference list = {len(self.LOGO_FEATS)}')

    def _load_domain_map(self, domain_map_path):
        with open(domain_map_path, 'rb') as handle:
            return pickle.load(handle)

    def return_logo_feat(self, logo: Image):
        return pred_siamese_OCR(img=logo, model=self.SIAMESE_MODEL, ocr_model=self.OCR_MODEL)

    def has_logo(self, screenshot_path: str):

        try:
            with open(screenshot_path, "rb") as image_file:
                screenshot_encoding = base64.b64encode(image_file.read())
        except:
            return False, False

        cropped_logos = self.predict_logos(screenshot_encoding=screenshot_encoding)
        if cropped_logos is None or len(cropped_logos) == 0:
            return False, False

        cropped = cropped_logos[0]
        img_feat = self.return_logo_feat(cropped)
        sim_list = self.LOGO_FEATS @ img_feat.T  # take dot product for every pair of embeddings (Cosine Similarity)

        if np.sum(sim_list >= self.SIAMESE_THRE_RELAX) == 0: # not exceed siamese relaxed threshold, not in targetlist
            return True, False
        else:
            return True, True

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

    '''Deep Siamese Model for Logo Comparison'''
    def run_siamese_inference(self, domain_map, reference_logo):
        img_feat = pred_siamese_OCR(img=reference_logo,
                                    model=self.SIAMESE_MODEL,
                                    ocr_model=self.OCR_MODEL)
        sim_list = np.matmul(self.LOGO_FEATS, img_feat.T)

        # Sort and take the top 3 brands
        idx = np.argsort(sim_list)[::-1][:3]
        top3_brands = np.array(self.LOGO_FILES)[idx]
        top3_simlist = np.array(sim_list)[idx]

        top3_logolist = [Image.open(x) for x in top3_brands]
        top3_brandlist = [brand_converter(os.path.basename(os.path.dirname(x))) for x in top3_brands] # get the top3 matched brands
        top3_domainlist = [domain_map[x] for x in top3_brandlist]

        for j, (brand, logo, domain, sim) in enumerate(zip(top3_brandlist, top3_logolist, top3_domainlist, top3_simlist)):
            if j > 0 and brand != top3_brandlist[0]:
                continue

            final_sim = sim if sim >= self.SIAMESE_THRE else None

            # Try resolution alignment if the largest similarity doesn't exceed the threshold
            if final_sim is None:
                cropped, candidate_logo = resolution_alignment(reference_logo, logo)
                img_feat = pred_siamese_OCR(img=cropped,
                                    model=self.SIAMESE_MODEL,
                                    ocr_model=self.OCR_MODEL)
                logo_feat = pred_siamese_OCR(img=candidate_logo,
                                    model=self.SIAMESE_MODEL,
                                    ocr_model=self.OCR_MODEL)
                final_sim = logo_feat.dot(img_feat) if logo_feat.dot(img_feat) >= self.SIAMESE_THRE else None

            if final_sim is None:
                break  # Stop if no match found

            # Check aspect ratio
            ratio_crop = reference_logo.size[0] / reference_logo.size[1]
            ratio_logo = logo.size[0] / logo.size[1]
            if max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) <= 2.5:
                return brand, domain, final_sim

        return None, None, top3_simlist[0]

    def perform_brand_identification(self, reference_logo, extracted_domain):
        domain_map = self._load_domain_map(self.DOMAIN_MAP_PATH)
        target_this, domain_this, this_conf = self.run_siamese_inference(domain_map, reference_logo)

        if (target_this is None) or (extracted_domain in domain_this):
            return None, None

        return target_this, this_conf

    '''CRP Classification MOdel'''
    @staticmethod
    def run_crp_classifier_cv(image: Image, coords, types, model):
        image = pil2tensor(image)
        img_arr = np.asarray(image)
        grid_tensor = coord2pixel_reverse(img_path=img_arr, coords=coords, types=types, reshaped_size=(256, 512))
        image = torch.cat((image.double(), grid_tensor), dim=0) # overlay the original image with layout segmentation maps
        assert image.shape == (8, 256, 512)

        return run_classifier(image, model)

    def run_crp_classifier(self, num_username, num_password, screenshot_encoding):
        cre_pred = 0 if (num_username or num_password) else 1 # heuristic
        if cre_pred == 1:
            screenshot_img_arr = encoding2array(screenshot_encoding)
            pred_classes_crp, pred_boxes_crp, _ = run_object_detection(img_arr=screenshot_img_arr, model=self.AWL_MODEL)
            cre_pred, _, _ = self.run_crp_classifier_cv(
                                                    image=encoding2pil(screenshot_encoding),
                                                    coords=pred_boxes_crp,
                                                    types=pred_classes_crp,
                                                    model=self.CRP_CLASSIFIER)
        return cre_pred


    # Helper function to run the CRP classifier and update the reach_crp flag
    def perform_crp_classification(self, driver):
        new_screenshot_encoding = driver.get_screenshot_encoding()
        ret_password, ret_username = driver.get_all_visible_username_password_inputs()
        num_username, num_password = len(ret_username), len(ret_password)
        cre_pred = self.run_crp_classifier(num_username, num_password, new_screenshot_encoding)
        return cre_pred == 0

    def perform_crp_classification_static(self, screenshot_path, html_path, pred_boxes, pred_classes):
        cre_pred = html_heuristic(html_path)
        if cre_pred == 1:  # if HTML heuristic report as nonCRP
            cre_pred, cred_conf, _ = credential_classifier_mixed_al(img=screenshot_path, coords=pred_boxes,
                                                                    types=pred_classes, model=self.CRP_CLASSIFIER)

        return cre_pred == 0


    '''CRP Locator Model'''
    # Helper function to handle post-processing after CRP search
    def post_process_crp_locator(self, driver, reach_crp, orig_url, current_url, obfuscate):
        if not reach_crp:
            try:
                driver.get(orig_url)
                if obfuscate:
                    driver.obfuscate_page()
            except:
                print("Cannot go back to the original URL, Exit ...")
        return reach_crp, orig_url, current_url

    def run_crp_locator_heuristic(self, driver, obfuscate=False):
        ct = 0
        reach_crp = False
        orig_url = driver.current_url()
        current_url = orig_url
        page_text = driver.get_page_text().split('\n') if driver.get_page_text() else []

        for line in page_text:
            if len(line.replace(' ', '')) > 300:
                continue
            keyword_finder = re.findall(Regexes.CREDENTIAL_TAKING_KEYWORDS, " ".join(line.split()), re.IGNORECASE)
            if len(keyword_finder) > 0:
                ct += 1
                elements = driver.get_clickable_elements_contains(line)
                if len(elements):
                    success_clicked = driver.click(elements[0])
                    time.sleep(0.5)
                    if success_clicked:
                        reach_crp = self.perform_crp_classification(driver)
                        if reach_crp:
                            break
                if ct >= PhishIntentionWrapper._RETRIES:
                    break

        return self.post_process_crp_locator(driver, reach_crp, orig_url, current_url, obfuscate)

    def run_crp_locator_cv(self, driver, obfuscate=False):
        reach_crp = False
        orig_url = driver.current_url()
        current_url = orig_url

        screenshot_encoding = driver.get_screenshot_encoding()
        old_screenshot_img_arr = encoding2array(screenshot_encoding)
        _, login_buttons, _ = run_object_detection(img_arr=old_screenshot_img_arr,
                                                   model=self.CRP_LOCATOR_MODEL)

        if login_buttons is None or len(login_buttons) == 0:
            return self.post_process_crp_locator(driver, reach_crp, orig_url, current_url, obfuscate)

        login_buttons = login_buttons.numpy()

        for bbox in login_buttons[:min(self._RETRIES, len(login_buttons))]:  # only for top3 boxes
            x1, y1, x2, y2 = bbox
            element = driver.find_element_by_location((x1 + x2) // 2, (y1 + y2) // 2)

            try:
                success_clicked = driver.click(element)
                time.sleep(0.5)
            except selenium.common.exceptions.StaleElementReferenceException:
                continue

            if success_clicked:
                reach_crp = self.perform_crp_classification(driver)
                if reach_crp:
                    break

        return self.post_process_crp_locator(driver, reach_crp, orig_url, current_url, obfuscate)

    # Helper function to perform CRP analysis
    def perform_crp_transition(self, driver, obfuscate=False):
        # HTML heuristic based login finder
        successful, orig_url, current_url = self.run_crp_locator_heuristic(driver=driver, obfuscate=obfuscate)
        if not successful:
            # If HTML login finder did not find CRP, call CV-based login finder
            successful, orig_url, current_url = self.run_crp_locator_cv(driver=driver, obfuscate=obfuscate)

        if not successful:
            try:
                driver.get(orig_url)
            except:
                print("Cannot go back to the original URL, Exit ...")
        return successful, orig_url, current_url

    def dynamic_analysis_reimplement(self, driver):
        successful, orig_url, current_url = self.perform_crp_transition(driver)
        return successful, orig_url, current_url

    def dynamic_analysis_and_save_reimplement(self, orig_url, screenshot_path, driver, obfuscate=False):
        new_screenshot_path = screenshot_path.replace('shot.png', 'new_shot.png')
        new_info_path = new_screenshot_path.replace('new_shot.png', 'new_info.txt')
        process_time = 0.

        try:
            driver.get(orig_url)
            if obfuscate:
                driver.obfuscate_page()
        except:
            return orig_url, screenshot_path, False, process_time

        start_time = time.time()
        successful, orig_url, current_url = self.perform_crp_transition(driver, orig_url)
        process_time += time.time() - start_time

        if not successful:
            return orig_url, screenshot_path, successful, process_time

        try:
            driver.save_screenshot(new_screenshot_path)
        except Exception as e:
            return orig_url, screenshot_path, successful, process_time

        with open(new_info_path, 'w', encoding='ISO-8859-1') as f:
            f.write(current_url)

        return current_url, new_screenshot_path, successful, process_time

    '''PhishIntention'''

    def test_orig_phishintention(self, url, screenshot_path, driver,
                                 include_crp_locator=True,
                                 obfuscate=False):
        timings = {}
        waive_crp_classifier = False
        dynamic = False
        phish_category = 0  # 0 for benign, 1 for phish
        pred_target = None
        plotvis = self.layout_vis(screenshot_path, None, None)

        while True:
            ####################### Step1: layout detector ##############################################
            screenshot_img = path2pil(screenshot_path)
            screenshot_encoding = path2encoding(screenshot_path)

            with time_block('ele_detector_time', timings):
                pred_boxes, pred_classes = self.predict_all_uis(screenshot_encoding)
            if not waive_crp_classifier:  # first time entering the loop, update the plot
                plotvis = self.layout_vis(screenshot_path, pred_boxes, pred_classes)

            if pred_boxes is None or len(pred_boxes) == 0:
                print('No element is detected, report as benign')
                return 0, None, plotvis, dynamic, timings, pred_boxes, pred_classes

            logo_pred_boxes, logo_pred_classes = find_element_type(pred_boxes=pred_boxes,
                                                                   pred_classes=pred_classes,
                                                                   bbox_type='logo')

            if logo_pred_boxes is None or len(logo_pred_boxes) == 0:
                print('No logo is detected')
                return 0, None, plotvis, dynamic, timings, pred_boxes, pred_classes

            x1, y1, x2, y2 = logo_pred_boxes.numpy()[0]
            reference_logo = screenshot_img.crop((x1, y1, x2, y2))

            ######################## Step2: Siamese (logo matcher) ########################################
            extracted_domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix

            with time_block('siamese_time', timings):
                pred_target, siamese_conf = self.perform_brand_identification(reference_logo=reference_logo,
                                                                              extracted_domain=extracted_domain)
            if pred_target is None:
                print('Did not match to any brand, report as benign')
                return 0, None, plotvis, dynamic, timings, pred_boxes, pred_classes

            # first time entering the loop
            if not waive_crp_classifier:
                pred_target_initial = pred_target
                url_orig = url
            else:  # second time entering the loop, the page before and after transition are matched to different target
                if pred_target_initial != pred_target:
                    print('After CRP transition, the logo\'s brand has changed, report as benign')
                    return 0, None, plotvis, dynamic, timings, pred_boxes, pred_classes

            ######################## Step3: CRP checker (if a target is reported) #################################
            print('A target is reported by siamese, enter CRP classifier')
            if waive_crp_classifier:  # only run dynamic analysis ONCE
                break

            with time_block('crp_time', timings):
                html_path = screenshot_path.replace("shot.png", "html.txt")
                cre_pred = self.perform_crp_classification_static(screenshot_path=screenshot_path, html_path=html_path,
                                                                  pred_boxes=pred_boxes, pred_classes=pred_classes)

            if not cre_pred:
                ######################## Step4: Dynamic analysis #################################
                waive_crp_classifier = True  # only run dynamic analysis ONCE
                with time_block('dynamic_time', timings):
                    try:
                        url, screenshot_path, successful, process_time = self.dynamic_analysis_and_save_reimplement(orig_url=url,
                                                                                                               screenshot_path=screenshot_path,
                                                                                                               driver=driver,
                                                                                                               obfuscate=obfuscate)
                    except selenium.common.exceptions.TimeoutException:
                        successful = False

                # If dynamic analysis did not reach a CRP, or jump to a third-party domain
                if (successful == False) or (tldextract.extract(url).domain != tldextract.extract(url_orig).domain):
                    print('Dynamic analysis cannot find any link redirected to a CRP page, report as benign')
                    return 0, None, plotvis, dynamic, timings, pred_boxes, pred_classes
                else:  # dynamic analysis successfully found a CRP
                    dynamic = True
                    print('Dynamic analysis found a CRP, go back to layout detector')
            else:
                print('Already a CRP, continue')
                break

        ######################## Step5: Return #################################
        if pred_target is not None:
            print('Phishing is found!')
            phish_category = 1
            # Visualize, add annotations
            cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
                        (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return phish_category, pred_target, plotvis, dynamic, timings, pred_boxes, pred_classes


    '''Phishpedia'''
    def test_orig_phishpedia(self, url, screenshot_path):
        timings = {}
        extracted_domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
        screenshot_img = path2pil(screenshot_path)
        screenshot_encoding = path2encoding(screenshot_path)


        with time_block("ele_detector_time", timings):
            pred_boxes, pred_classes = self.predict_all_uis(screenshot_encoding)
        plotvis = self.layout_vis(screenshot_path, pred_boxes, pred_classes)

        if (pred_boxes is None) or (len(pred_boxes) == 0):
            return 0, None, plotvis, timings, pred_boxes, pred_classes

        logo_pred_boxes, _ = find_element_type(pred_boxes, pred_classes, bbox_type='logo')
        if (logo_pred_boxes is None) or (len(logo_pred_boxes) == 0):
            return 0, None, plotvis, timings, pred_boxes, pred_classes

        x1, y1, x2, y2 = logo_pred_boxes.numpy()[0]
        reference_logo = screenshot_img.crop((x1, y1, x2, y2))

        with time_block("siamese_time", timings):
            pred_target, siamese_conf = self.perform_brand_identification(reference_logo, extracted_domain)

        if not pred_target:
            return 0, None, plotvis, timings, pred_boxes, pred_classes
        else:
            cv2.putText(plotvis, f"Target: {pred_target} with confidence {siamese_conf:.4f}", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return 1, pred_target, plotvis, timings, pred_boxes, pred_classes

