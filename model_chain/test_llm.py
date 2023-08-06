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
from field_study.draw_utils import draw_annotated_image, draw_annotated_image_nobox
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

class TestLLM():

    def __init__(self, phishintention_cls):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        state_dict = torch.load("./checkpoints/epoch{}_model.pt".format(4))
        self.clip_model.load_state_dict(state_dict)
        self.LLM_model = "gpt-3.5-turbo-16k"
        self.prediction_prompt = './selection_model/prompt3.json'
        self.brand_prompt = './brand_recognition/prompt_field.json'
        self.phishintention_cls = phishintention_cls
        self.language_list = ['en', 'ch', 'ru', 'japan', 'fa', 'ar', 'korean', 'vi', 'ms',
                             'fr', 'german', 'it', 'es', 'pt', 'uk', 'be', 'te',
                             'sa', 'ta', 'nl', 'tr', 'ga']

    @staticmethod
    def is_valid_domain(domain):
        """Check if the provided string is a valid domain name without any spaces."""
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
            try:
                proxies = {
                    "http": "http://127.0.0.1:7890",
                    "https": "http://127.0.0.1:7890",
                }
                response = requests.get('https://'+domain, timeout=30, proxies=proxies)
                if response.status_code >= 200 and response.status_code < 400: # it is alive
                    return True
            except Exception as err:
                print(f'Error {err} when checking the aliveness of domain {domain}')
        return False

    def check_webhosting_domain(self, domain):
        prompt = [{"role": "system",
                   "content": "You are a helpful assistant who is knowledgable about brands."},
                    {"role": "user",
                    "content": f"Question: Is this domain {domain} a web hosting (e.g. webmail, cpanel, shopify, afrihost, zimbra), a cloud service (e.g. zabbix, okta), a VPN service (e.g. firezone), a web development framework (e.g. dolibarr, laravel, 1jabber, stdesk, firezone), a domain hosting (e.g. godaddy), a domain parking, or an online betting domain? Answer Yes or No. Answer:"
                  }]
        inference_done = False
        while not inference_done:
            try:
                response = openai.ChatCompletion.create(
                    model=self.LLM_model,
                    messages=prompt,
                    temperature=0,
                    max_tokens=50,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                inference_done = True
            except Exception as e:
                Logger.spit('LLM Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                prompt[-1]['content'] = prompt[-1]['content'][:len(prompt[-1]['content']) // 2]
                time.sleep(10)
        answer = ''.join([choice["message"]["content"] for choice in response['choices']])
        print(answer)
        if 'yes' in answer.lower():
            return True
        return False

    def detect_text(self, shot_path, html_path):
        '''
            Run OCR
            Args:
                shot_path
                html_path
            Returns:
                ocr_text
        '''
        ocr_text = ''
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

        if len(most_fit_results):
            most_fit_results = most_fit_results[0]
            ocr_text = ' '.join([line[1][0] for line in most_fit_results])

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
                ocr_text = ' '.join([x for x in html_text if x])

        return ocr_text

    def url2logo(self, driver, URL):
        '''
            URL2logo
            Args:
                driver: selenium driver
                URL
            Returns:
                logo
                exception
        '''
        try:
            driver.get(URL, allow_redirections=False)
            time.sleep(3)  # fixme: must allow some loading time here
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return None, str(e)

        # the URL is for a webpage not the logo image
        try:
            screenshot_encoding = driver.get_screenshot_encoding()
            screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
            screenshot_img = screenshot_img.convert("RGB")
            screenshot_img_arr = np.asarray(screenshot_img)
            screenshot_img_arr = np.flip(screenshot_img_arr, -1)  # RGB2BGR
            pred = self.phishintention_cls.AWL_MODEL(screenshot_img_arr)
            pred_i = pred["instances"].to('cpu')
            pred_classes = pred_i.pred_classes.detach().cpu()  # Boxes types
            pred_boxes = pred_i.pred_boxes.tensor.detach().cpu()  # Boxes coords

            if pred_boxes is None or len(pred_boxes) == 0:
                all_logos_coords = None
            else:
                all_logos_coords, _ = find_element_type(pred_boxes=pred_boxes,
                                                       pred_classes=pred_classes,
                                                       bbox_type='logo')
            if all_logos_coords is None:
                return None, None
            else:
                logo_coord = all_logos_coords[0]
                logo = screenshot_img.crop((int(logo_coord[0]), int(logo_coord[1]), int(logo_coord[2]), int(logo_coord[3])))
        except Exception as e:
            Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
            return None, None

        return logo, None

    def click_and_save(self, driver, dom, save_html_path, save_shot_path):
        '''
            Click an element and save the updated screenshot and HTML
            Args:
                driver: selenium driver
                dom: dom path for the interested element
                save_html_path: path to save HTML
                save_shot_path: path to save screenshot
            Returns:
                current_url
                save_html_path
                save_shot_path
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



    def brand_recognition_llm(self,
                              reference_logo, html_text,
                              ):
        '''
            Use LLM to report targeted brand
            Args:
                reference_logo: the logo on the original webpage (for validation purpose)
                html_text
                driver
                do_validation: do we validate the returned brand?
            Returns:
                company_domain
                company_logo
        '''
        company_domain, company_logo = None, None
        question = question_template_brand(html_text)

        with open(self.brand_prompt, 'rb') as f:
            prompt = json.load(f)
        new_prompt = prompt
        new_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                start_time = time.time()
                response = openai.ChatCompletion.create(
                    model=self.LLM_model,
                    messages=new_prompt,
                    temperature=0,
                    max_tokens=50,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                inference_done = True
            except Exception as e:
                Logger.spit('LLM Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
                time.sleep(10)

        answer = ''.join([choice["message"]["content"] for choice in response['choices']])
        print('LLM prediction time:', time.time() - start_time)
        print(f'Detected brand {answer}')

        if len(answer) > 0 and self.is_valid_domain(answer):
            # if not self.check_webhosting_domain(answer):
            company_logo = reference_logo
            company_domain = answer

        return company_domain, company_logo

    def crp_prediction_llm(self, html_text):
        '''
            Use LLM to classify credential-requiring page v.s. non-credential-requiring page
            Args:
                html_text
            Returns:
                0 for CRP, 1 for non-CRP
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

    def ranking_model(self, url, driver, ranking_model_refresh_page):
        '''
            Use CLIP to rank the UI elements to find the most probable login button
            Args:
                url
                driver
                ranking_model_refresh_page: do we need to refresh the webpage before running the model?
            Returns:
                candidate_uis: DOM paths for candidate uis
                candidate_uis_imgs: screenshots for candidate uis
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

            for batch in range(math.ceil(len(candidate_uis)/100)):
                images = torch.stack(candidate_uis_imgs[batch*100 : min(len(candidate_uis), (batch+1)*100)]).to(self.device)
                texts = clip.tokenize(["not a login button", "a login button"]).to(self.device)

                logits_per_image, logits_per_text = self.clip_model(images, texts)
                probs = logits_per_image.softmax(dim=-1)  # (N, C)
                final_probs = torch.cat([final_probs, probs.detach().cpu()], dim=0)

            conf = final_probs[torch.arange(final_probs.shape[0]), 1]  # take the confidence (N, 1)
            _, ind = torch.topk(conf, 1)  # top1 index

            return candidate_uis[ind], candidate_uis_imgs[ind], driver
        else:
            print('No candidate login button to click')
            return [], [], driver


    def test(self, url, reference_logo,
             shot_path, html_path, driver, limit=2,
             brand_recog_time=0, crp_prediction_time=0, crp_transition_time=0,
             ranking_model_refresh_page=True,
             skip_brand_recognition=False,
             brand_recognition_do_validation=False,
             company_domain=None, company_logo=None,
             ):
        '''
            PhishLLM
            Args:
                url
                shot_path
                html_path
                driver
                limit: depth limit to run CRP transition model (ranking model)
                brand_recog_time
                crp_prediction_time
                crp_transition_time
                ranking_model_refresh_page
                skip_brand_recognition: whether to skip the brand recognition after CRP transition?
                company_domain: the reported targeted brand domain
                company_logo
            Returns:
                pred: 'benign' or 'phish'
                target: company_domain or 'None'
                brand_recog_time
                crp_prediction_time
                crp_transition_time
        '''

        html_text = self.detect_text(shot_path, html_path)
        plotvis = Image.open(shot_path)

        if not skip_brand_recognition:
            start_time = time.time()
            company_domain, company_logo = self.brand_recognition_llm(reference_logo, html_text)
            brand_recog_time += time.time() - start_time
            time.sleep(1) # fixme: allow the openai api to rest, not sure whether this help
        # domain-brand inconsistency
        phish_condition = company_domain and (tldextract.extract(company_domain).domain != tldextract.extract(url).domain or
                                              tldextract.extract(company_domain).suffix != tldextract.extract(url).suffix)

        if phish_condition and brand_recognition_do_validation:
            validation_success = False
            start_time = time.time()
            logo, exception = self.url2logo(driver=driver, URL='https://'+company_domain)
            if exception:
                driver.quit()
                XDriver.set_headless()
                driver = XDriver.boot(chrome=True)
                driver.set_script_timeout(30)
                driver.set_page_load_timeout(60)
                time.sleep(3)
            print('Crop the logo time:', time.time() - start_time)

            if logo and reference_logo:
                # Domain matching OR Logo matching
                start_time = time.time()
                reference_logo_feat = pred_siamese_OCR(img=reference_logo,
                                                       model=self.phishintention_cls.SIAMESE_MODEL,
                                                       ocr_model=self.phishintention_cls.OCR_MODEL)
                logo_feat = pred_siamese_OCR(img=logo,
                                             model=self.phishintention_cls.SIAMESE_MODEL,
                                             ocr_model=self.phishintention_cls.OCR_MODEL)

                matched_sim = reference_logo_feat @ logo_feat
                if matched_sim >= self.phishintention_cls.SIAMESE_THRE_RELAX:  # logo similarity exceeds a threshold
                    validation_success = True
                print('Logo matching time:', time.time() - start_time)
            if not validation_success:
                phish_condition = False

        if phish_condition:
            start_time = time.time()
            crp_cls = self.crp_prediction_llm(html_text)
            crp_prediction_time += time.time() - start_time
            time.sleep(1) # fixme: allow the openai api to rest, not sure whether this help

            if crp_cls == 0: # CRP
                plotvis = draw_annotated_image_nobox(plotvis, company_domain)
                return 'phish', company_domain, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis
            else: # do CRP transition
                if limit == 0:  # reach interaction limit -> just return
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time, plotvis

                start_time = time.time()
                candidate_dom, candidate_img, driver = self.ranking_model(url, driver, ranking_model_refresh_page)
                crp_transition_time += time.time() - start_time

                if len(candidate_dom):
                    save_html_path = re.sub("index[0-9]?.html", f"index{limit}.html", html_path)
                    save_shot_path = re.sub("shot[0-9]?.png", f"shot{limit}.png", shot_path)
                    print("Click login button")
                    current_url, *_ = self.click_and_save(driver, candidate_dom, save_html_path, save_shot_path)
                    if current_url: # click success
                        ranking_model_refresh_page = current_url != url
                        return self.test(current_url, reference_logo, save_shot_path, save_html_path, driver, limit-1,
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






