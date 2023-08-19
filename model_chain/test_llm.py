from tldextract import tldextract
import openai
import time
import json
from PIL import Image
import io
import base64
import torch
import clip
import re
from xdriver.XDriver import XDriver
from selenium.webdriver.remote.webelement import WebElement
from phishintention.src.AWL_detector import find_element_type
from phishintention.src.OCR_aided_siamese import pred_siamese_OCR
from model_chain.utils import *
from paddleocr import PaddleOCR
import math
import os
from lxml import html
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
from tqdm import tqdm
import cv2
from model_chain.web_utils import WebUtil
from xdriver.xutils.Logger import Logger
import shutil
import requests
from field_study.draw_utils import draw_annotated_image_box
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Set, Dict, Optional, Union
from lavis.models import load_model_and_preprocess
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key2.txt').read()

class TestLLM():

    def __init__(self, phishintention_cls):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        state_dict = torch.load("./checkpoints/epoch{}_model.pt".format(4))
        self.clip_model.load_state_dict(state_dict)
        self.LLM_model = "gpt-3.5-turbo-16k"
        self.prediction_prompt = './selection_model/prompt3.json'
        # self.brand_prompt = './brand_recognition/prompt_field.json'
        self.brand_prompt = './brand_recognition/prompt_caption.json'
        self.phishintention_cls = phishintention_cls
        self.language_list = ['en', 'ch', 'ru', 'japan', 'fa', 'ar', 'korean', 'vi', 'ms',
                             'fr', 'german', 'it', 'es', 'pt', 'uk', 'be', 'te',
                             'sa', 'ta', 'nl', 'tr', 'ga']
        self.proxies = {
                    "http": "http://127.0.0.1:7890",
                    "https": "http://127.0.0.1:7890",
                }

    def query2image(self, query: str, SEARCH_ENGINE_API: str, SEARCH_ENGINE_ID: str, num: int=10) -> Tuple[List[str], List[str]]:
        '''
            Retrieve the images from Google image search
            :param query:
            :param SEARCH_ENGINE_API:
            :param SEARCH_ENGINE_ID:
            :param num:
            :return:
        '''
        returned_urls = []
        context_links = []
        if len(query) == 0:
            return returned_urls, context_links

        URL = f"https://www.googleapis.com/customsearch/v1?key={SEARCH_ENGINE_API}&cx={SEARCH_ENGINE_ID}&q={query}&searchType=image&num={num}"
        while True:
            try:
                data = requests.get(URL, proxies=self.proxies).json()
                break
            except requests.exceptions.SSLError as e:
                print(e)
                time.sleep(2)
        if 'error' in list(data.keys()):
            if data['error']['code'] == 429:
                raise RuntimeError("Google search exceeds quota limit")
        search_items = data.get("items")
        if search_items is None:
            return returned_urls, context_links

        # iterate over results found
        for i, search_item in enumerate(search_items, start=1):
            link = search_item.get("image")["thumbnailLink"]
            context_link = search_item.get("image")['contextLink']
            returned_urls.append(link)
            context_links.append(context_link)

        return returned_urls, context_links

    def download_image(self, url: str) -> Optional[Image.Image]:
        '''
            Download images from given url (Google image context links)
            :param url:
            :return:
        '''
        try:
            response = requests.get(url, proxies=self.proxies)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                return img
            else:
                print(f"Failed to download image: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while downloading image: {e}")

        return None

    def get_images(self, image_urls: List[str]) -> List[Image.Image]:
        '''
            Run download_image in multiple threads
            :param image_urls:
            :return:
        '''
        images = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.download_image, url) for url in image_urls]
            for future in futures:
                img = future.result()
                if img:
                    images.append(img)

        return images

    def is_valid_domain(self, domain: str) -> bool:
        '''
            Check if the provided string is a valid domain
            :param domain:
            :return:
        '''
        # Regular expression to check if the string is a valid domain without spaces
        pattern = re.compile(
            r'^(?!-)'  # Cannot start with a hyphen
            r'(?!.*--)'  # Cannot have two consecutive hyphens
            r'(?!.*\.\.)'  # Cannot have two consecutive periods
            r'(?!.*\s)'  # Cannot contain any spaces
            r'[a-zA-Z0-9-]{1,63}'  # Valid characters are alphanumeric and hyphen
            r'(?:\.[a-zA-Z]{2,})+$'  # Ends with a valid top-level domain
        )
        it_is_a_domain = bool(pattern.fullmatch(domain))
        if it_is_a_domain:
            ct_limit = 1
            while True:
                if ct_limit == 3:
                    break
                try:
                    response = requests.get('https://'+domain, timeout=60, proxies=self.proxies)
                    if response.status_code >= 200 and response.status_code < 400: # it is alive
                        return True
                    break
                except Exception as err:
                    print(f'Error {err} when checking the aliveness of domain {domain}')
                    ct_limit += 1
        return False

    def detect_webpage_ocr(self, shot_path: str, html_path: str) -> Tuple[List[str], List[List[float]], str]:
        '''
            Run OCR
            :param shot_path:
            :param html_path:
            :return:
        '''
        detected_text = ''
        ocr_text = []
        ocr_coord = []
        most_fit_lang = self.language_list[0]
        best_conf = 0
        most_fit_results = ''
        sure_thre = 0.98
        unsure_thre = 0.9

        for lang in self.language_list:
            try:
                ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)  # need to run only once to download and load model into memory
                result = ocr.ocr(shot_path, cls=True)
            except MemoryError:
                ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, use_gpu=False)  # need to run only once to download and load model into memory
                result = ocr.ocr(shot_path, cls=True)
            median_conf = np.median([x[-1][1] for x in result[0]])
            if math.isnan(median_conf): # no text is detected
                break
            if median_conf >= sure_thre: # confidence is so high
                most_fit_results = result
                break
            if median_conf > best_conf and median_conf >= unsure_thre:
                best_conf = median_conf
                most_fit_lang = lang
                most_fit_results = result
            if best_conf > 0 and self.language_list.index(lang) - self.language_list.index(most_fit_lang) >= 2:  # local best language
                break
        # OCR can return results
        if len(most_fit_results):
            most_fit_results = most_fit_results[0]
            ocr_text = [line[1][0] for line in most_fit_results]
            ocr_coord = [line[0][0] + line[0][2] for line in most_fit_results]
            detected_text = ' '.join(ocr_text)

        # if OCR does not work, use the raw HTML
        elif os.path.exists(html_path):
            with io.open(html_path, 'r', encoding='utf-8') as f:
                page = f.read()
            if len(page):
                dom_tree = html.fromstring(page, parser=html.HTMLParser(remove_comments=True))
                unwanted = dom_tree.xpath('//script|//style|//head')
                for u in unwanted:
                    u.drop_tree()
                html_text = ' '.join(dom_tree.itertext())
                html_text = re.sub(r"\s+", " ", html_text).split(' ')
                detected_text = ' '.join([x for x in html_text if x])

        return ocr_text, ocr_coord, detected_text

    def generate_logo_caption(self, img: Image.Image) -> str:

        raw_image = img.convert("RGB")

        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption",
                                                             model_type="base_coco",
                                                             is_eval=True,
                                                             device=self.device)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        result = model.generate({"image": image})
        del model, vis_processors
        return ' '.join(result)

    def detect_logo(self, save_shot_path):
        # Logo detection
        screenshot_img = Image.open(save_shot_path)
        screenshot_img = screenshot_img.convert("RGB")
        with open(save_shot_path, "rb") as image_file:
            screenshot_encoding = base64.b64encode(image_file.read())
        logo_boxes = self.phishintention_cls.return_all_bboxes4type(screenshot_encoding, 'logo')

        if (logo_boxes is not None) and len(logo_boxes):
            logo_box = logo_boxes[0]  # get coordinate for logo
            x1, y1, x2, y2 = logo_box
            reference_logo = screenshot_img.crop((x1, y1, x2, y2))  # crop logo out
        else:
            reference_logo = None
            logo_box = None
        return logo_box, reference_logo


    def brand_recognition_llm(self, reference_logo: Optional[Image.Image],
                              logo_box: Optional[List[float]],
                              ocr_text: List[str], ocr_coord: List[List[float]],
                              image_width: int, image_height: int) -> Tuple[Optional[str], Optional[Image.Image]]:
        '''
            Brand Recognition Model
            :param reference_logo:
            :param logo_box:
            :param ocr_text:
            :param ocr_coord:
            :param image_width:
            :param image_height:
            :return:
        '''
        company_domain, company_logo = None, None
        if reference_logo:
            # generation caption for logo
            logo_caption = self.generate_logo_caption(reference_logo)
            logo_ocr = ''
            if len(ocr_coord):
                # get the OCR text description surrounding the logo
                expand_logo_box = expand_bbox(logo_box, image_width=image_width, image_height=image_height, expand_ratio=1.5)
                overlap_areas = compute_overlap_areas_between_lists([expand_logo_box], ocr_coord)
                logo_ocr = np.array(ocr_text)[overlap_areas[0] > 0].tolist()
                logo_ocr = ' '.join(logo_ocr)

            question = question_template_caption(logo_caption, logo_ocr)

            with open(self.brand_prompt, 'rb') as f:
                prompt = json.load(f)
            new_prompt = prompt
            new_prompt.append(question)

            inference_done = False
            while not inference_done:
                try:
                    start_time = time.time()
                    response = openai.ChatCompletion.create(
                        model=self.LLM_model,
                        messages=new_prompt,
                        temperature=0,
                        max_tokens=10,
                    )
                    inference_done = True
                except Exception as e:
                    Logger.spit('LLM Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                    time.sleep(10) # retry

            answer = ''.join([choice["message"]["content"] for choice in response['choices']])
            print('LLM prediction time:', time.time() - start_time)
            print(f'Detected brand {answer}')

            # check the validity of the returned domain, i.e. liveness
            if len(answer) > 0 and self.is_valid_domain(answer):
                company_logo = reference_logo
                company_domain = answer

        return company_domain, company_logo

    def crp_prediction_llm(self, html_text: str) -> int:
        '''
            Use LLM to classify credential-requiring page v.s. non-credential-requiring page
            :param html_text:
            :return:
        '''
        question = question_template_prediction(html_text)
        with open(self.prediction_prompt, 'rb') as f:
            prompt = json.load(f)
        new_prompt = prompt
        new_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                response = openai.ChatCompletion.create(
                    model=self.LLM_model,
                    messages=new_prompt,
                    temperature=0,
                    max_tokens=100,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                inference_done = True
            except Exception as e:
                Logger.spit('LLM Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
                time.sleep(10)

        answer = ''.join([choice["message"]["content"] for choice in response['choices']])
        print(f'CRP prediction {answer}')
        if 'A.' in answer:
            return 0 # CRP
        else:
            return 1


    def page_transition(self, driver: XDriver, dom: str, save_html_path: str, save_shot_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        '''
            Click an element and save the updated screenshot and HTML
            :param driver:
            :param dom:
            :param save_html_path:
            :param save_shot_path:
            :return:
        '''
        try:
            element = driver.find_elements_by_xpath(dom)
            if element:
                try:
                    driver.execute_script("arguments[0].style.border='3px solid red'", element[0]) # hightlight the element to click
                    time.sleep(0.5)
                except:
                    pass
                driver.move_to_element(element[0])
                driver.click(element[0])
                time.sleep(7)  # fixme: must allow some loading time here, dynapd is slow
            current_url = driver.current_url()
        except Exception as e:
            print(e)
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return None, None, None

        try:
            driver.save_screenshot(save_shot_path)
            print('new screenshot saved')
            with open(save_html_path, "w", encoding='utf-8') as f:
                f.write(driver.page_source())
            return current_url, save_html_path, save_shot_path
        except Exception as e:
            print(e)
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return None, None, None

    def ranking_model(self, url: str, driver: XDriver, ranking_model_refresh_page: bool) -> \
                                Tuple[Union[List, str], List[torch.Tensor], XDriver]:
        '''
            Use CLIP to rank the UI elements to find the most probable login button
            :param url:
            :param driver:
            :param ranking_model_refresh_page:
            :return:
        '''
        if ranking_model_refresh_page:
            try:
                driver.get(url)
                time.sleep(5)
            except Exception as e:
                print(e)
                Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                driver.quit()
                XDriver.set_headless()
                driver = XDriver.boot(chrome=True)
                driver.set_script_timeout(30)
                driver.set_page_load_timeout(60)
                time.sleep(3)
                return [], [], driver

        try:
            (btns, btns_dom),  \
                (links, links_dom), \
                (images, images_dom), \
                (others, others_dom) = driver.get_all_clickable_elements()
        except Exception as e:
            print(e)
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return [], [], driver

        all_clickable = btns + links + images + others
        all_clickable_dom = btns_dom + links_dom + images_dom + others_dom

        # element screenshot
        candidate_uis = []
        candidate_uis_imgs = []
        for it in range(min(300, len(all_clickable))):
            try:
                driver.scroll_to_top()
                x1, y1, x2, y2 = driver.get_location(all_clickable[it])
            except Exception as e:
                print(e)
                Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                continue

            if x2 - x1 <= 0 or y2 - y1 <= 0 or y2 >= driver.get_window_size()['height']//2: # invisible or at the bottom
                continue

            try:
                ele_screenshot_img = Image.open(io.BytesIO(base64.b64decode(all_clickable[it].screenshot_as_base64)))
                candidate_uis_imgs.append(self.clip_preprocess(ele_screenshot_img))
                candidate_uis.append(all_clickable_dom[it])
            except Exception as e:
                print(e)
                Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                continue

        # rank them
        if len(candidate_uis_imgs):
            print(f'Find {len(candidate_uis_imgs)} candidate UIs')
            final_probs = torch.tensor([], device='cpu')
            batch_size = 64
            texts = clip.tokenize(["not a login button", "a login button"]).to(self.device)

            for batch in range(math.ceil(len(candidate_uis)/batch_size)):
                images = torch.stack(candidate_uis_imgs[batch*batch_size : min(len(candidate_uis), (batch+1)*batch_size)]).to(self.device)
                logits_per_image, logits_per_text = self.clip_model(images, texts)
                probs = logits_per_image.softmax(dim=-1)  # (N, C)
                final_probs = torch.cat([final_probs, probs.detach().cpu()], dim=0)
                del images

            conf = final_probs[torch.arange(final_probs.shape[0]), 1]  # take the confidence (N, 1)
            _, ind = torch.topk(conf, 1)  # top1 index

            return candidate_uis[ind], candidate_uis_imgs[ind], driver
        else:
            print('No candidate login button to click')
            return [], [], driver


    def test(self, url: str, reference_logo: Optional[Image.Image],
             logo_box: Optional[List[float]],
             shot_path: str, html_path: str, driver: XDriver, limit: int=2,
             brand_recog_time: float=0, crp_prediction_time: float=0, crp_transition_time: float=0,
             ranking_model_refresh_page: bool=True,
             skip_brand_recognition: bool=False,
             brand_recognition_do_validation: bool=False,
             company_domain: Optional[str]=None, company_logo: Optional[Image.Image]=None,
             ):
        '''
            PhishLLM
            :param url:
            :param reference_logo:
            :param shot_path:
            :param html_path:
            :param driver:
            :param limit:
            :param brand_recog_time:
            :param crp_prediction_time:
            :param crp_transition_time:
            :param ranking_model_refresh_page:
            :param skip_brand_recognition:
            :param brand_recognition_do_validation:
            :param company_domain:
            :param company_logo:
            :return:
        '''

        ## Run OCR to extract text
        ocr_text, ocr_coord, detected_text = self.detect_webpage_ocr(shot_path, html_path)
        plotvis = Image.open(shot_path)
        image_width, image_height = plotvis.size

        ## Brand recognition model
        if not skip_brand_recognition:
            start_time = time.time()
            company_domain, company_logo = self.brand_recognition_llm(reference_logo, logo_box, ocr_text, ocr_coord, image_width, image_height)
            brand_recog_time += time.time() - start_time
            time.sleep(1) # fixme: allow the openai api to rest, not sure whether this help
        # check domain-brand inconsistency
        phish_condition = company_domain and (tldextract.extract(company_domain).domain != tldextract.extract(url).domain or
                                              tldextract.extract(company_domain).suffix != tldextract.extract(url).suffix)

        if phish_condition and brand_recognition_do_validation:
            ## Brand recognition model : result validation
            validation_success = False
            start_time = time.time()
            API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]
            returned_urls, _ = self.query2image(query=company_domain + ' logo',
                                                SEARCH_ENGINE_ID=SEARCH_ENGINE_ID, SEARCH_ENGINE_API=API_KEY,
                                                num=5)
            logos = self.get_images(returned_urls)
            print('Crop the logo time:', time.time() - start_time)

            if reference_logo and len(logos)>0:
                reference_logo_feat = pred_siamese_OCR(img=reference_logo,
                                                       model=self.phishintention_cls.SIAMESE_MODEL,
                                                       ocr_model=self.phishintention_cls.OCR_MODEL)
                start_time = time.time()

                for logo in logos:
                    logo_feat = pred_siamese_OCR(img=logo,
                                                 model=self.phishintention_cls.SIAMESE_MODEL,
                                                 ocr_model=self.phishintention_cls.OCR_MODEL)
                    matched_sim = reference_logo_feat @ logo_feat
                    if matched_sim >= 0.7:  # logo similarity exceeds a threshold
                        validation_success = True
                        break

                print('Logo matching time:', time.time() - start_time)
            if not validation_success:
                phish_condition = False

        if phish_condition:
            # CRP prediction model
            start_time = time.time()
            crp_cls = self.crp_prediction_llm(detected_text)
            crp_prediction_time += time.time() - start_time
            time.sleep(1) # fixme: allow the openai api to rest, not sure whether this help

            if crp_cls == 0: # CRP page is detected
                plotvis = draw_annotated_image_box(plotvis, company_domain, logo_box)
                return 'phish', company_domain, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis
            else:
                # CRP transition
                if limit == 0:  # reach interaction limit -> just return
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time, plotvis

                # Ranking model
                start_time = time.time()
                candidate_ele, candidate_img, driver = self.ranking_model(url, driver, ranking_model_refresh_page)
                crp_transition_time += time.time() - start_time

                if len(candidate_ele):
                    save_html_path = re.sub("index[0-9]?.html", f"index{limit}.html", html_path)
                    save_shot_path = re.sub("shot[0-9]?.png", f"shot{limit}.png", shot_path)
                    print("Click login button")
                    current_url, *_ = self.page_transition(driver, candidate_ele, save_html_path, save_shot_path)
                    if current_url: # click success
                        ranking_model_refresh_page = current_url != url
                        # logo detection
                        logo_box, reference_logo = self.detect_logo(save_shot_path)
                        return self.test(current_url, reference_logo, logo_box,
                                         save_shot_path, save_html_path, driver, limit-1,
                                         brand_recog_time, crp_prediction_time, crp_transition_time,
                                         ranking_model_refresh_page=ranking_model_refresh_page,
                                         skip_brand_recognition=True, brand_recognition_do_validation=brand_recognition_do_validation,
                                         company_domain=company_domain, company_logo=company_logo)

        return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time, plotvis



if __name__ == '__main__':

    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    XDriver.set_headless()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you
    Logger.set_debug_on()

    driver.get('http://phishing.localhost')
    time.sleep(5)
    all_links = [x.strip().split(',')[-2] for x in open('./datasets/Brand_Labelled_130323.csv').readlines()[1:]]
    # all_links = driver.get_all_links_orig()

    root_folder = './datasets/dynapd'
    result = './datasets/dynapd_wo_validation.txt'
    os.makedirs(root_folder, exist_ok=True)


    for ct, target in enumerate(all_links):
        if ct <= 5470:
            continue
        hash = target.split('/')[3]
        target_folder = os.path.join(root_folder, hash)
        os.makedirs(target_folder, exist_ok=True)
        if os.path.exists(result) and hash in open(result).read():
            continue

        try:
            driver.get(target, click_popup=True, allow_redirections=False)
            time.sleep(5)
            Logger.spit(f'Target URL = {target}', caller_prefix=XDriver._caller_prefix, debug=True)
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            shutil.rmtree(target_folder)
            continue

        try:
            page_text = driver.get_page_text()
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            shutil.rmtree(target_folder)
            continue

        try:
            error_free = web_func.page_error_checking(driver)
            if not error_free:
                Logger.spit('Error page or White page', caller_prefix=XDriver._caller_prefix, debug=True)
                shutil.rmtree(target_folder)
                continue
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            shutil.rmtree(target_folder)
            continue

        if "Index of" in page_text:
            try:
                # skip error URLs
                error_free = web_func.page_interaction_checking(driver)
                white_page = web_func.page_white_screen(driver, 1)
                if (error_free == False) or white_page:
                    Logger.spit('Error page or White page', caller_prefix=XDriver._caller_prefix, debug=True)
                    shutil.rmtree(target_folder)
                    continue
                target = driver.current_url()
            except Exception as e:
                Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                shutil.rmtree(target_folder)
                continue

        if target.endswith('https/') or target.endswith('genWeb/'):
            shutil.rmtree(target_folder)
            continue

        try:
            shot_path = os.path.join(target_folder, 'shot.png')
            html_path = os.path.join(target_folder, 'index.html')
            # take screenshots
            screenshot_encoding = driver.get_screenshot_encoding()
            screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
            screenshot_img.save(shot_path)
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, warning=True)
            shutil.rmtree(target_folder)
            continue

        try:
            # record HTML
            with open(html_path, 'w+', encoding='utf-8') as f:
                f.write(driver.page_source())
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            pass

        if os.path.exists(shot_path):
            pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, _ = llm_cls.test(target, None, shot_path, html_path, driver)
            with open(result, 'a+') as f:
                f.write(hash+'\t'+pred+'\t'+brand+'\t'+str(brand_recog_time)+'\t'+str(crp_prediction_time)+'\t'+str(crp_transition_time)+'\n')

    driver.quit()






