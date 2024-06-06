from openai import OpenAI
import torch
import clip
from phishintention.src.OCR_aided_siamese import pred_siamese_OCR
from scripts.utils.utils import *
from scripts.utils.web_utils.web_utils import *
from scripts.utils.logger_utils import *
from server.announcer import Announcer, AnnouncerEvent, AnnouncerPrompt
from paddleocr import PaddleOCR
import math
import os
from lxml import html
from experiments.field_study.draw_utils import draw_annotated_image_box
from typing import List, Tuple, Optional, Union
from lavis.models import load_model_and_preprocess
from scripts.utils.PhishIntentionWrapper import PhishIntentionWrapper
import yaml
import PIL
from tldextract import tldextract
import urllib3
from urllib3.exceptions import MaxRetryError
urllib3.disable_warnings()
http = urllib3.PoolManager(maxsize=10)  # Increase the maxsize to a larger value, e.g., 10

os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()
os.environ['CURL_CA_BUNDLE'] = ''


class TestLLM():

    def __init__(self, phishintention_cls, param_dict, proxies=None, frontend_api=False):

        self.frontend_api = frontend_api # whether the phishllm is deployed with server/server.py
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proxies = proxies

        ## Ranking model
        self.clip_model, self.clip_preprocess = clip.load(param_dict['rank']['model_name'], device=self.device)
        if self.device == "cpu": # https://github.com/openai/CLIP/issues/57
            self.clip_model.float()
        state_dict = torch.load(param_dict['rank']['checkpoint_path'], map_location=self.device)
        self.clip_model.load_state_dict(state_dict)

        ## Image Captioning model
        self.caption_model, self.caption_preprocess, _ = load_model_and_preprocess(name="blip_caption",
                                                                                   model_type="base_coco",
                                                                                   is_eval=True,
                                                                                   device=self.device)

        ## LLM
        self.LLM_model = param_dict["LLM_model"]
        self.brand_prompt = param_dict['brand_recog']['prompt_path']
        self.crp_prompt = param_dict['crp_pred']['prompt_path']
        self.phishintention_cls = phishintention_cls
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # OCR model
        try:
            self.default_ocr_model = PaddleOCR(use_angle_cls=False, lang="en", show_log=False, use_gpu=self.device == 'cuda')
        except MemoryError:
            self.default_ocr_model = PaddleOCR(use_angle_cls=False, lang="en", show_log=False, use_gpu= False)
        self.ocr_language_list = param_dict['ocr']['supported_langs']

        # Load the Google API key and SEARCH_ENGINE_ID once during initialization
        self.API_KEY, self.SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]

        ## Load hyperparameters
        self.ocr_sure_thre, self.ocr_unsure_thre, self.ocr_local_best_window = param_dict['ocr']['sure_thre'], param_dict['ocr']['unsure_thre'], param_dict['ocr']['local_best_window']
        self.logo_expansion_ratio = param_dict['logo_caption']['expand_ratio']

        self.brand_recog_temperature, self.brand_recog_max_tokens = param_dict['brand_recog']['temperature'], param_dict['brand_recog']['max_tokens']
        self.brand_recog_sleep = param_dict['brand_recog']['sleep_time']
        self.do_brand_validation = param_dict['brand_valid']['activate']
        self.brand_valid_k, self.brand_valid_siamese_thre = param_dict['brand_valid']['k'], param_dict['brand_valid']['siamese_thre']
        self.get_industry = param_dict['brand_recog']['ask_industry']
        self.industry_temperature, self.industry_max_tokens = param_dict['brand_recog']['industry']['temperature'], param_dict['brand_recog']['industry']['max_tokens']

        self.crp_temperature, self.crp_max_tokens = param_dict['crp_pred']['temperature'], param_dict['crp_pred']['max_tokens']
        self.crp_sleep = param_dict['crp_pred']['sleep_time']

        self.rank_max_uis, self.rank_batch_size = param_dict['rank']['max_uis_process'], param_dict['rank']['batch_size']
        self.rank_driver_sleep = param_dict['rank']['driver_sleep_time']
        self.rank_driver_script_timeout = param_dict['rank']['script_timeout']
        self.rank_driver_page_load_timeout = param_dict['rank']['page_load_timeout']
        self.interaction_limit = param_dict['rank']['depth_limit']

        # webhosting domains as blacklist
        self.webhosting_domains = [x.strip() for x in open('./datasets/hosting_blacklists.txt').readlines()]

    def update_params(self, param_dict):
        '''
            Update hyperparameters
            :param param_dict:
            :return:
        '''
        self.brand_recog_temperature = param_dict['brand_recog']['temperature']
        self.brand_valid_k, self.brand_valid_siamese_thre = param_dict['brand_valid']['k'], param_dict['brand_valid']['siamese_thre']
        self.industry_temperature = param_dict['brand_recog']['industry']['temperature']

        self.crp_temperature = param_dict['crp_pred']['temperature']

        self.interaction_limit = param_dict['rank']['depth_limit']
        self.do_brand_validation = param_dict['brand_valid']['activate']
        print(param_dict)

    def detect_logo(self, save_shot_path: Union[str, bytes]) -> Tuple[Optional[List[float]], Optional[Image.Image]]:
        '''
            Logo detection
            :param save_shot_path:
            :return:
        '''
        reference_logo = None
        logo_box = None

        try:
            if isinstance(save_shot_path, str):
                screenshot_img = Image.open(save_shot_path).convert("RGB")
                with open(save_shot_path, "rb") as image_file:
                    screenshot_encoding = base64.b64encode(image_file.read())
            else:
                screenshot_encoding = save_shot_path
                image_data = base64.b64decode(screenshot_encoding)
                image_stream = io.BytesIO(image_data)
                screenshot_img = Image.open(image_stream)

            logo_boxes = self.phishintention_cls.predict_all_uis4type(screenshot_encoding, 'logo')

            if (logo_boxes is not None) and len(logo_boxes)>0:
                logo_box = logo_boxes[0]  # get coordinate for logo
                x1, y1, x2, y2 = logo_box
                reference_logo = screenshot_img.crop((x1, y1, x2, y2))  # crop logo out
        except PIL.UnidentifiedImageError:
            pass

        return logo_box, reference_logo

    def generate_webpage_ocr(self, shot_path: str, html_path: str) -> Tuple[List[str], List[List[float]], str, float]:
        '''
            Get OCR results for the whole webpage screenshot
            :param shot_path:
            :param html_path:
            :return:
        '''
        detected_text = ''
        ocr_text = []
        ocr_coord = []
        most_fit_lang = self.ocr_language_list[0]
        best_conf = 0
        ocr_processing_time = 0
        most_fit_results = ''
        ocr = self.default_ocr_model

        for lang in self.ocr_language_list:
            if lang != 'en':
                try:
                    ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False, use_gpu=self.device == 'cuda')  # need to run only once to download and load model into memory
                except MemoryError:
                    ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False, use_gpu=False)  # need to run only once to download and load model into memory

            with Image.open(shot_path) as img:
                # Resize the image
                width, height = img.size
                new_width = int(width)
                new_height = int(height)
                resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

                # Save the resized image to a temporary path
                temp_path = "/tmp/resized_image.png"
                resized_img.save(temp_path)

            start_time = time.time()
            result = ocr.ocr(temp_path, cls=False)
            ocr_processing_time = time.time() - start_time

            if result is None:
                break
            if result[0] is None:
                break
            median_conf = np.median([x[-1][1] for x in result[0]])

            if math.isnan(median_conf): # no text is detected
                break
            if median_conf >= self.ocr_sure_thre: # confidence is so high
                most_fit_results = result
                break
            elif median_conf > best_conf and median_conf >= self.ocr_unsure_thre: # confidence is moderately high, need further checking
                best_conf = median_conf
                most_fit_lang = lang
                most_fit_results = result
            if best_conf > 0 and self.ocr_language_list.index(lang) - self.ocr_language_list.index(most_fit_lang) >= self.ocr_local_best_window:  # local best language
                break

        # OCR can return results
        if len(most_fit_results):
            most_fit_results = most_fit_results[0]
            ocr_text = [line[1][0] for line in most_fit_results]
            ocr_coord = [line[0][0] + line[0][2] for line in most_fit_results] # [x1, y1, x2, y2]
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

        return ocr_text, ocr_coord, detected_text, ocr_processing_time

    def generate_logo_caption(self, img: Image.Image) -> str:
        '''
            Get the image captioning result for the logo
            :param img:
            :return:
        '''
        raw_image = img.convert("RGB")
        image = self.caption_preprocess["eval"](raw_image).unsqueeze(0).to(self.device)
        result = self.caption_model.generate({"image": image})
        return ' '.join(result)

    def preprocessing(self, shot_path:str, html_path:str,
                      reference_logo:Image.Image,
                      logo_box: Optional[List[float]],
                      image_width: int, image_height: int,
                      announcer: Tuple[Announcer, float]=None
                      ) -> Tuple[Tuple[str, str, str], Tuple[float, float]]:
        '''
            Preprocessing the webpage (OCR + Image Captioning)
            :param shot_path:
            :param html_path:
            :param reference_logo:
            :param logo_box:
            :param image_width:
            :param image_height:
            :param announcer:
            :return:
        '''
        image_caption_processing_time = 0
        webpage_text_list, text_coords_list, webpage_text, ocr_processing_time = self.generate_webpage_ocr(shot_path, html_path)
        if reference_logo:
            # generation image caption for logo
            start_time = time.time()
            logo_caption = self.generate_logo_caption(reference_logo)
            image_caption_processing_time = time.time() - start_time
            logo_ocr = ''
            if len(text_coords_list):
                # get the OCR text description surrounding the logo
                expand_logo_box = expand_bbox(logo_box, image_width=image_width, image_height=image_height, expand_ratio=self.logo_expansion_ratio)
                overlap_areas = pairwise_intersect_area([expand_logo_box], text_coords_list)
                logo_ocr = np.array(webpage_text_list)[overlap_areas[0] > 0].tolist()
                logo_ocr = ' '.join(logo_ocr)
        else:
            logo_caption = ''
            logo_ocr = ' '.join(webpage_text_list) # if not logo is detected, simply return the whole webpage ocr result as the logo ocr result

        msg = f"Detected Logo caption: {logo_caption}<br>Logo OCR results: {logo_ocr}"
        if self.frontend_api:
            announcer.spit(msg, AnnouncerEvent.RESPONSE)
            time.sleep(0.5)
        PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        return (webpage_text, logo_caption, logo_ocr), (ocr_processing_time, image_caption_processing_time)

    def ask_industry(self, html_text, announcer):
        '''
            Ask gpt to predict the industry sector given the webpage
            :param html_text:
            :param announcer:
            :return:
        '''

        industry = ''
        if self.get_industry and len(html_text):
            prompt = question_template_industry(html_text)
            if self.frontend_api:
                announcer.spit(AnnouncerPrompt.question_template_industry(html_text), AnnouncerEvent.PROMPT)
                time.sleep(0.5)
            inference_done = False
            while not inference_done:
                try:
                    response = self.client.chat.completions.create(
                        model=self.LLM_model,
                        messages=prompt,
                        temperature=self.industry_temperature,
                        max_tokens=self.industry_max_tokens,  # we're only counting input tokens here, so let's not waste tokens on the output
                    )
                    inference_done = True
                except Exception as e:
                    PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                    prompt[-1]['content'] = prompt[-1]['content'][:len(prompt[-1]['content']) // 2]
                    time.sleep(self.brand_recog_sleep)

            industry = ''.join([choice.message.content for choice in response.choices])
            if len(industry) > 30:
                industry = ''

        return industry

    def brand_recognition_llm(self, reference_logo: Optional[Image.Image],
                              webpage_text: str, logo_caption: str,
                              logo_ocr: str,
                              announcer: Optional[Announcer]) -> Tuple[Optional[str], Optional[Image.Image], float]:
        '''
            Brand Recognition Model
            :param reference_logo:
            :param webpage_text:
            :param logo_caption:
            :param logo_ocr:
            :param announcer:
            :return:
        '''
        company_domain, company_logo = None, None
        brand_llm_pred_time = 0

        industry = ''
        if len(webpage_text) and self.get_industry:
            industry = self.ask_industry(webpage_text, announcer)

        if self.frontend_api:
            announcer.spit(f'Industry: {industry}', AnnouncerEvent.RESPONSE)
            time.sleep(0.5)
        PhishLLMLogger.spit(f'Logo caption: {logo_caption}<br>Logo OCR: {logo_ocr}<br>Industry: {industry}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

        if len(logo_caption) > 0 or len(logo_ocr) > 0:
            input_ocr_text = logo_ocr if len(logo_ocr) > 0 else webpage_text
            if self.get_industry:
                question = question_template_brand_industry(logo_caption, input_ocr_text, industry)
                if self.frontend_api:
                    announcer.spit(AnnouncerPrompt.question_template_brand_industry(logo_caption, input_ocr_text, industry), AnnouncerEvent.PROMPT)
            else:
                question = question_template_brand(logo_caption, input_ocr_text)
                if self.frontend_api:
                    announcer.spit(AnnouncerPrompt.question_template_brand(logo_caption, input_ocr_text), AnnouncerEvent.PROMPT)

            with open(self.brand_prompt, 'rb') as f:
                prompt = json.load(f)
            new_prompt = prompt
            new_prompt.append(question)

            inference_done = False
            while not inference_done:
                try:
                    start_time = time.time()
                    response = self.client.chat.completions.create(
                        model=self.LLM_model,
                        messages=new_prompt,
                        temperature=self.brand_recog_temperature,
                        max_tokens=self.brand_recog_max_tokens,
                    )
                    brand_llm_pred_time = time.time() - start_time
                    inference_done = True
                except Exception as e:
                    PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                    new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]  # maybe the prompt is too long, cut by half
                    time.sleep(self.brand_recog_sleep) # retry

            answer = ''.join([choice.message.content for choice in response.choices])

            if self.frontend_api:
                announcer.spit(f"Time taken for LLM brand prediction: {brand_llm_pred_time}<br>Detected brand: {answer}", AnnouncerEvent.RESPONSE)
                time.sleep(0.5)
            PhishLLMLogger.spit(f"Time taken for LLM brand prediction: {brand_llm_pred_time}<br>Detected brand: {answer}", debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

            # check the validity of the returned domain, i.e. liveness
            if len(answer) > 0 and is_valid_domain(answer):
                company_logo = reference_logo
                company_domain = answer

        else:
            msg = 'No logo description'
            PhishLLMLogger.spit(msg, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
            if self.frontend_api:
                announcer.spit(msg, AnnouncerEvent.RESPONSE)

        return company_domain, company_logo, brand_llm_pred_time

    def popularity_validation(self, company_domain: str) -> Tuple[bool, float]:
        '''
            Brand recognition model : result validation
            :param company_domain:
            :return:
        '''
        validation_success = False

        start_time = time.time()
        returned_urls = query2url(query=company_domain,
                                  SEARCH_ENGINE_ID=self.SEARCH_ENGINE_ID,
                                  SEARCH_ENGINE_API=self.API_KEY,
                                  num=self.brand_valid_k,
                                  proxies=self.proxies)
        searching_time = time.time() - start_time

        returned_domains = ['.'.join(part for part in tldextract.extract(url) if part).split('www.')[-1] for url in returned_urls]
        if company_domain in returned_domains:
            validation_success = True

        return validation_success, searching_time

    def brand_validation(self, company_domain: str, reference_logo: Image.Image) -> Tuple[bool, float, float]:
        '''
            Brand recognition model : result validation
            :param company_domain:
            :param reference_logo:
            :return:
        '''
        logo_searching_time, logo_matching_time = 0, 0
        validation_success = False

        if not reference_logo:
            return True, logo_searching_time, logo_matching_time

        start_time = time.time()
        returned_urls = query2image(query=company_domain + ' logo',
                                    SEARCH_ENGINE_ID=self.SEARCH_ENGINE_ID, SEARCH_ENGINE_API=self.API_KEY,
                                    num=self.brand_valid_k,
                                    proxies=self.proxies)
        logo_searching_time = time.time() - start_time
        logos = get_images(returned_urls, proxies=self.proxies)
        msg = f'Number of logos found on google images {len(logos)}'
        print(msg)
        PhishLLMLogger.spit(msg, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

        if len(logos) > 0:
            reference_logo_feat = pred_siamese_OCR(img=reference_logo,
                                                   model=self.phishintention_cls.SIAMESE_MODEL,
                                                   ocr_model=self.phishintention_cls.OCR_MODEL)
            start_time = time.time()
            sim_list = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(pred_siamese_OCR, logo,
                                           self.phishintention_cls.SIAMESE_MODEL,
                                           self.phishintention_cls.OCR_MODEL) for logo in logos]
                for future in futures:
                    logo_feat = future.result()
                    matched_sim = reference_logo_feat @ logo_feat
                    sim_list.append(matched_sim)

            if any([x > self.brand_valid_siamese_thre for x in sim_list]):
                validation_success = True

            logo_matching_time = time.time() - start_time

        return validation_success, logo_searching_time, logo_matching_time

    def crp_prediction_llm(self, html_text: str, announcer: Optional[Announcer]) -> Tuple[bool, float]:
        '''
            Use LLM to classify credential-requiring page v.s. non-credential-requiring page
            :param html_text:
            :param announcer:
            :return:
        '''
        crp_llm_pred_time = 0

        question = question_template_prediction(html_text)
        if self.frontend_api:
            announcer.spit(AnnouncerPrompt.question_template_prediction(html_text), AnnouncerEvent.PROMPT)
            time.sleep(0.5)
        with open(self.crp_prompt, 'rb') as f:
            prompt = json.load(f)
        new_prompt = prompt
        new_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.LLM_model,
                    messages=new_prompt,
                    temperature=self.crp_temperature,
                    max_tokens=self.crp_max_tokens,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                crp_llm_pred_time = time.time() - start_time
                inference_done = True
            except Exception as e:
                PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2] # maybe the prompt is too long, cut by half
                time.sleep(self.crp_sleep)

        answer = ''.join([choice.message.content for choice in response.choices])
        msg = f'Time taken for LLM CRP classification: {crp_llm_pred_time}<br>CRP prediction: {answer}'
        PhishLLMLogger.spit(msg, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
        if self.frontend_api:
            announcer.spit(msg, AnnouncerEvent.RESPONSE)
            time.sleep(0.5)
        if 'A.' in answer:
            return True, crp_llm_pred_time # CRP
        else:
            return False, crp_llm_pred_time

    def ranking_model(self, url: str, driver: CustomWebDriver, ranking_model_refresh_page: bool, announcer: Optional[Announcer]) -> \
                                Tuple[Union[List, str], List[torch.Tensor], CustomWebDriver, float]:
        '''
            Use CLIP to rank the UI elements to find the most probable login button
            :param url:
            :param driver:
            :param ranking_model_refresh_page:
            :param announcer:
            :return:
        '''
        clip_pred_time = 0
        if ranking_model_refresh_page:
            try:
                driver.get(url)
                time.sleep(self.rank_driver_sleep)
            except Exception as e:
                print(e)
                driver.quit()
                driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")
                driver.set_script_timeout(self.rank_driver_script_timeout)
                driver.set_page_load_timeout(self.rank_driver_page_load_timeout)
                return [], [], driver, clip_pred_time

        try:
            (btns, btns_dom),  \
                (links, links_dom), \
                (images, images_dom), \
                (others, others_dom) = driver.get_all_clickable_elements()
        except Exception as e:
            print(e)
            return [], [], driver, clip_pred_time

        all_clickable = btns + links + images + others
        all_clickable_dom = btns_dom + links_dom + images_dom + others_dom

        # element screenshot
        candidate_uis = []
        candidate_uis_imgs = []
        candidate_uis_text = []

        for it in range(min(self.rank_max_uis, len(all_clickable))):
            try:
                candidate_ui, candidate_ui_img, candidate_ui_text = screenshot_element(all_clickable[it],
                                                                                       all_clickable_dom[it],
                                                                                       driver,
                                                                                       self.clip_preprocess)
            except (MaxRetryError, WebDriverException, TimeoutException) as e:
                PhishLLMLogger.spit(e, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                driver.quit()
                driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")
                driver.set_script_timeout(self.rank_driver_script_timeout)
                driver.set_page_load_timeout(self.rank_driver_page_load_timeout)
                driver.get(url)
                time.sleep(self.rank_driver_sleep)
                continue

            if (candidate_ui is not None) and (candidate_ui_img is not None) and (candidate_ui_text is not None):
                candidate_uis.append(candidate_ui)
                candidate_uis_imgs.append(candidate_ui_img)
                candidate_uis_text.append(candidate_ui_text)

        # rank them
        if len(candidate_uis_imgs):
            msg = f'Find {len(candidate_uis_imgs)} candidate UIs'
            PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            if self.frontend_api:
                announcer.spit(msg, AnnouncerEvent.RESPONSE)
                time.sleep(0.5)
            final_probs = torch.tensor([], device='cpu')
            batch_size = self.rank_batch_size
            texts = clip.tokenize(["not a login button", "a login button"]).to(self.device)

            start_time = time.time()
            for batch in range(math.ceil(len(candidate_uis)/batch_size)):
                chunked_images = candidate_uis_imgs[batch*batch_size : min(len(candidate_uis), (batch+1)*batch_size)]
                images = torch.stack(chunked_images).to(self.device)
                logits_per_image, _ = self.clip_model(images, texts)
                probs = logits_per_image.softmax(dim=-1)  # (N, C)
                final_probs = torch.cat([final_probs, probs.detach().cpu()], dim=0)
                del images
            clip_pred_time = time.time() - start_time

            conf = final_probs[torch.arange(final_probs.shape[0]), 1]  # take the confidence (N, 1)

            # if the element text matches to any obvious credential-taking words, just shift it to the top
            regex_match = [bool(re.search(Regexes.CREDENTIAL_TAKING_KEYWORDS, text, re.IGNORECASE | re.VERBOSE)) if text else False for text in
                           candidate_uis_text]
            for i, is_match in enumerate(regex_match):
                if is_match:
                    conf[i] = 1.0 - (1e-3)*i
            _, indices = torch.topk(conf, min(5, len(candidate_uis_imgs)))  # top5 elements
            candidate_uis_selected = [candidate_uis[ind] for ind in indices]
            candidate_imgs_selected = [candidate_uis_imgs[ind] for ind in indices]
            return candidate_uis_selected, candidate_imgs_selected, driver, clip_pred_time
        else:
            msg = 'No candidate login button to click'
            PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            if self.frontend_api:
                announcer.spit(msg, AnnouncerEvent.RESPONSE)
                time.sleep(0.5)
            return [], [], driver, clip_pred_time

    def estimate_cost_phishllm(self, url: str, reference_logo: Optional[Image.Image], logo_box: Optional[List[float]],
                               shot_path: str, html_path: str, driver: Union[CustomWebDriver, float]=None,
                               brand_llm_cost=0, crp_llm_cost=0, brand_validation_cost=0, popularity_validation_cost=0,
                               ocr_processing_time=0, image_caption_processing_time=0,
                               brand_recog_time=0, crp_prediction_time=0, clip_prediction_time=0,
                               brand_validation_searching_time=0, brand_validation_matching_time=0,
                               popularity_validation_time=0):

        ## compute num tokens
        def count_tokens(text, model_name):
            import tiktoken
            encoder = tiktoken.encoding_for_model(model_name)
            return len(encoder.encode(text))

        def append_prompt_and_count(file_path, new_text, model_name):
            with open(file_path, 'rb') as file:
                predefined_prompt = json.load(file)
            predefined_prompt.append(new_text)
            content = ''.join([x['content'] for x in predefined_prompt])
            return count_tokens(content, model_name)

        def update_cost(current_cost, num_tokens):
            return current_cost + (0.0015 / 1000) * num_tokens # https://openai.com/api/pricing/

        plotvis = Image.open(shot_path)
        image_width, image_height = plotvis.size
        preprocessing_results = self.preprocessing(shot_path=shot_path, html_path=html_path,
                                                   reference_logo=reference_logo, logo_box=logo_box,
                                                   image_width=image_width, image_height=image_height)
        ocr_processing_time += preprocessing_results[1][0]
        image_caption_processing_time += preprocessing_results[1][1]
        webpage_text, logo_caption, logo_ocr = preprocessing_results[0][0], preprocessing_results[0][1], preprocessing_results[0][2]

        # Token calculation and cost update for brand LLM
        fill_in_prompt = question_template_brand(logo_caption, logo_ocr) if len(logo_ocr) else question_template_brand(logo_caption, webpage_text)
        brand_prompt_num_tokens = append_prompt_and_count(self.brand_prompt, fill_in_prompt, "gpt-3.5-turbo")
        brand_llm_cost = update_cost(brand_llm_cost, brand_prompt_num_tokens)

        company_domain, company_logo, brand_recog_time_sub = self.brand_recognition_llm(reference_logo=reference_logo,
                                                                                        webpage_text=webpage_text,
                                                                                        logo_caption=logo_caption,
                                                                                        logo_ocr=logo_ocr,
                                                                                        announcer=None)
        brand_recog_time += brand_recog_time_sub

        if is_valid_domain(company_domain):
            validation_success, brand_validation_searching_time_sub, brand_validation_matching_time_sub = self.brand_validation(company_domain=company_domain,
                                                                                                                                reference_logo=reference_logo)
            brand_validation_searching_time += brand_validation_searching_time_sub
            brand_validation_matching_time += brand_validation_matching_time_sub
            brand_validation_cost += 5 / 1000 # https://developers.google.com/custom-search/v1/overview#pricing

            if validation_success:
                fill_in_prompt = question_template_prediction(webpage_text)
                crp_prompt_num_tokens = append_prompt_and_count(self.crp_prompt, fill_in_prompt, "gpt-3.5-turbo")
                crp_llm_cost = update_cost(crp_llm_cost, crp_prompt_num_tokens)

                crp_cls, crp_prediction_time_sub = self.crp_prediction_llm(html_text=webpage_text,
                                                                       announcer=None)
                crp_prediction_time += crp_prediction_time_sub

                if not crp_cls:
                    candidate_elements, _, driver, clip_prediction_time_sub = self.ranking_model(url=url, driver=driver,
                                                                                         ranking_model_refresh_page=True,
                                                                                         announcer=None)
                    clip_prediction_time += clip_prediction_time_sub
                else:
                    popularity_validation_success, popularity_validation_time_sub = self.popularity_validation(company_domain='.'.join(part for part in tldextract.extract(url) if part))
                    popularity_validation_time += popularity_validation_time_sub
                    popularity_validation_cost += 5 / 1000  # https://developers.google.com/custom-search/v1/overview#pricing

        return (brand_llm_cost, crp_llm_cost, brand_validation_cost, popularity_validation_cost), \
               (ocr_processing_time, image_caption_processing_time,
                brand_recog_time, crp_prediction_time, clip_prediction_time,
                brand_validation_searching_time, brand_validation_matching_time, popularity_validation_time)

    def test(self, url: str,
             reference_logo: Optional[Image.Image],
             logo_box: Optional[List[float]],
             shot_path: str,
             html_path: str,
             driver: Union[CustomWebDriver, float]=None,
             limit: int=0,
             brand_recog_time: float=0, crp_prediction_time: float=0, clip_prediction_time: float=0,
             ranking_model_refresh_page: bool=True,
             skip_brand_recognition: bool=False,
             company_domain: Optional[str]=None, company_logo: Optional[Image.Image]=None,
             announcer: Optional[Announcer]=None
             ):
        '''
            PhishLLM
            :param url:
            :param reference_logo:
            :param logo_box:
            :param shot_path:
            :param html_path:
            :param driver:
            :param limit:
            :param brand_recog_time:
            :param crp_prediction_time:
            :param clip_prediction_time:
            :param ranking_model_refresh_page:
            :param skip_brand_recognition:
            :param company_domain:
            :param company_logo:
            :param announcer:
            :return:
        '''
        ## Run OCR to extract text
        plotvis = Image.open(shot_path)

        image_width, image_height = plotvis.size
        (webpage_text, logo_caption, logo_ocr), (ocr_processing_time, image_caption_processing_time)  = self.preprocessing(shot_path=shot_path, html_path=html_path,
                                                                   reference_logo=reference_logo, logo_box=logo_box,
                                                                   image_width=image_width, image_height=image_height,
                                                                   announcer=announcer)

        ## Brand recognition model
        if not skip_brand_recognition:
            company_domain, company_logo, brand_recog_time = self.brand_recognition_llm(reference_logo=reference_logo, webpage_text=webpage_text,
                                                                      logo_caption=logo_caption, logo_ocr=logo_ocr, announcer=announcer)
            time.sleep(self.brand_recog_sleep) # fixme: allow the openai api to rest, not sure whether this help
        # check domain-brand inconsistency
        domain_brand_inconsistent = False
        if company_domain:
            if company_domain in self.webhosting_domains:
                msg = '[\U00002705] Benign, since it is a brand providing cloud services'
                if self.frontend_api:
                    announcer.spit(msg, AnnouncerEvent.SUCCESS)
                PhishLLMLogger.spit(msg)
                return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis

            domain4pred, suffix4pred = tldextract.extract(company_domain).domain, tldextract.extract(company_domain).suffix
            domain4url, suffix4url = tldextract.extract(url).domain, tldextract.extract(url).suffix 
            domain_brand_inconsistent = (domain4pred != domain4url) or (suffix4pred != suffix4url)

        phish_condition = domain_brand_inconsistent

        # Brand prediction results validation
        if phish_condition and (not skip_brand_recognition):
            if self.do_brand_validation: # we can check the validity by comparing the logo on the webpage with the logos for the predicted brand
                validation_success, logo_searching_time, logo_matching_time = self.brand_validation(company_domain=company_domain,
                                                                                                   reference_logo=reference_logo)
                brand_recog_time += logo_searching_time
                brand_recog_time += logo_matching_time
                phish_condition = validation_success
                msg = f"Time taken for brand validation (logo matching with Google Image search results): {logo_searching_time+logo_matching_time}<br>Domain {company_domain} is relevant and valid? {validation_success}"
                if self.frontend_api:
                    announcer.spit(msg, AnnouncerEvent.RESPONSE)
                    time.sleep(0.5)
                PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            # else: # alternatively, we can check the aliveness of the predicted brand
                # validation_success = is_alive_domain(domain4pred + '.' + suffix4pred , self.proxies) # fixme
                # phish_condition = validation_success
                # msg = f"Brand Validation: Domain {company_domain} is alive? {validation_success}"
                # if self.frontend_api:
                #     announcer.spit(msg, AnnouncerEvent.RESPONSE)
                #     time.sleep(0.5)
                # PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

        if phish_condition:
            # CRP prediction model
            crp_cls, crp_prediction_time = self.crp_prediction_llm(html_text=webpage_text, announcer=announcer)
            time.sleep(self.crp_sleep)

            if crp_cls: # CRP page is detected
                plotvis = draw_annotated_image_box(plotvis, company_domain, logo_box)
                msg = f'[\u2757\uFE0F] Phishing discovered, phishing target is {company_domain}'
                if self.frontend_api:
                    announcer.spit(msg, AnnouncerEvent.SUCCESS)
                PhishLLMLogger.spit(msg)
                return 'phish', company_domain, brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis
            else:
                # Not a CRP page => CRP transition
                if limit >= self.interaction_limit:  # reach interaction limit -> just return
                    msg = '[\U00002705] Benign, reached interaction limit ...'
                    if self.frontend_api:
                        announcer.spit(msg, AnnouncerEvent.SUCCESS)
                    PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis

                # Ranking model
                candidate_elements, _, driver, clip_prediction_time = self.ranking_model(url=url, driver=driver, ranking_model_refresh_page=ranking_model_refresh_page, announcer=announcer)

                if len(candidate_elements):
                    save_html_path = re.sub("index[0-9]?.html", f"index{limit}.html", html_path)
                    save_shot_path = re.sub("shot[0-9]?.png", f"shot{limit}.png", shot_path)

                    if not ranking_model_refresh_page: # if previous click didnt refresh the page select the lower ranked element to click
                        msg = f"Since previously the URL has not changed, trying to click the Top-{min(len(candidate_elements), limit+1)} login button instead: "
                        candidate_ele = candidate_elements[min(len(candidate_elements)-1, limit)]
                    else: # else, just click the top-1 element
                        msg = "Trying to click the Top-1 login button: "
                        candidate_ele = candidate_elements[0]

                    # record the webpage elements before clicking the button
                    prev_screenshot_elements = get_screenshot_elements(phishintention_cls=self.phishintention_cls, driver=driver)
                    element_text, current_url, *_ = page_transition(driver=driver, dom=candidate_ele, save_html_path=save_html_path, save_shot_path=save_shot_path)
                    msg += f'{element_text}'
                    if self.frontend_api:
                        announcer.spit(msg, AnnouncerEvent.RESPONSE)
                        time.sleep(0.5)
                    PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    if current_url: # click success
                        ranking_model_refresh_page = has_page_content_changed(phishintention_cls=self.phishintention_cls, driver=driver, prev_screenshot_elements=prev_screenshot_elements)
                        msg = f"Has the webpage changed? {ranking_model_refresh_page}"
                        if self.frontend_api:
                            announcer.spit(msg, AnnouncerEvent.RESPONSE)
                            time.sleep(0.5)
                        PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                        # logo detection on new webpage
                        logo_box, reference_logo = self.detect_logo(save_shot_path)
                        return self.test(current_url, reference_logo, logo_box,
                                         save_shot_path, save_html_path, driver, limit + 1,
                                         brand_recog_time, crp_prediction_time, clip_prediction_time,
                                         announcer=announcer,
                                         ranking_model_refresh_page=ranking_model_refresh_page,
                                         skip_brand_recognition=True,
                                         company_domain=company_domain, company_logo=company_logo)
                else:
                    msg = '[\U00002705] Benign'
                    if self.frontend_api:
                        announcer.spit(msg, AnnouncerEvent.SUCCESS)
                    PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis

        msg = '[\U00002705] Benign'
        if self.frontend_api:
            announcer.spit(msg, AnnouncerEvent.SUCCESS)
        PhishLLMLogger.spit(msg)
        return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis


class TestVLM(TestLLM):

    def __init__(self, phishintention_cls, param_dict, proxies=None, frontend_api=False):
        super().__init__(phishintention_cls=phishintention_cls, param_dict=param_dict, proxies=proxies, frontend_api=frontend_api)

    def brand_recognition_llm(self, reference_logo: Optional[Image.Image], announcer: Optional[Announcer]) -> Tuple[Optional[str], Optional[Image.Image], float]:
        '''
            Brand Recognition Model
            :param reference_logo:
            :param announcer:
            :return:
        '''
        company_domain, company_logo = None, None
        brand_llm_pred_time = 0

        if not reference_logo:
            return company_domain, company_logo, brand_llm_pred_time

        few_shot_logo_image = Image.open('./brand_recognition/test_case/img.png')
        new_prompt = [
              {"role": "system",
                "content": "You are knowledgeable about brands and their associated logos. "
                           "Given the logo image, your task is to decide the brand of the logo. "
                           "Just give the brand's domain in English, do not output any explanation. "
                           "If there are multiple possible brands, output the most likely domain."},

              {"role": "user",
                "content": [
                           {"type": "text", "text": "Given the brand's logo, Question: What is the brand's domain? Answer: "},
                           {
                               "type": "image_url",
                               "image_url": {
                                   "url": f"data:image/jpeg;base64,{image2base64(few_shot_logo_image)}"
                               },
                           },
                       ]
               },

              {"role": "assistant",
                "content": "usenix.org"},
        ]
        question = vlm_question_template_brand(reference_logo)
        new_prompt.append(question)

        inference_done = False
        while not inference_done:
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=new_prompt,
                    temperature=self.brand_recog_temperature,
                    max_tokens=self.brand_recog_max_tokens,
                )
                brand_llm_pred_time = time.time() - start_time
                inference_done = True
            except Exception as e:
                PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True,
                                    caller_prefix=PhishLLMLogger._caller_prefix)
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(
                    new_prompt[-1]['content']) // 2]  # maybe the prompt is too long, cut by half
                time.sleep(self.brand_recog_sleep)  # retry

        answer = ''.join([choice.message.content for choice in response.choices])

        if self.frontend_api:
            announcer.spit(
                f"Time taken for LLM brand prediction: {brand_llm_pred_time}<br>Detected brand: {answer}",
                AnnouncerEvent.RESPONSE)
            time.sleep(0.5)
        PhishLLMLogger.spit(
            f"Time taken for LLM brand prediction: {brand_llm_pred_time}<br>Detected brand: {answer}", debug=True,
            caller_prefix=PhishLLMLogger._caller_prefix)

        # check the validity of the returned domain, i.e. liveness
        if len(answer) > 0 and is_valid_domain(answer):
            company_logo = reference_logo
            company_domain = answer

        return company_domain, company_logo, brand_llm_pred_time

    def crp_prediction_llm(self, webpage_screenshot: Image.Image, announcer: Optional[Announcer]) -> Tuple[bool, float]:
        '''
            Use LLM to classify credential-requiring page v.s. non-credential-requiring page
            :param html_text:
            :param announcer:
            :return:
        '''
        crp_llm_pred_time = 0

        few_shot_screenshot_image = Image.open('./datasets/maxbounty_screenshot.png')
        new_prompt = [
            {"role": "system",
             "content": "Given the webpage screenshot, your task is to decide the status of the webpage. "
                        "A credential-requiring page is where the users are asked to fill-in their sensitive information."
                        "Webpages that ask the users to download suspicious apps, connect token wallet and scan QR code are also considered as credential-taking."},

            {"role": "user",
             "content": [
                 {"type": "text",
                  "text": "Given the HTML webpage screenshot, Question: A. This is a credential-requiring page. B. This is not a credential-requiring page. \n Answer: "},
                 {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{image2base64(few_shot_screenshot_image)}"
                     },
                 },
             ]
             },

            {"role": "assistant",
             "content": "First we find the keywords that are related to sensitive information: Email address, Password. "
                        "After that we find the keywords that are related to login: Sign in, Login."
                        "Therefore the answer would be A."},
        ]
        question = vlm_question_template_prediction(webpage_screenshot)
        new_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=new_prompt,
                    temperature=self.crp_temperature,
                    max_tokens=self.crp_max_tokens,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                crp_llm_pred_time = time.time() - start_time
                inference_done = True
            except Exception as e:
                PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2] # maybe the prompt is too long, cut by half
                time.sleep(self.crp_sleep)

        answer = ''.join([choice.message.content for choice in response.choices])
        msg = f'Time taken for LLM CRP classification: {crp_llm_pred_time}<br>CRP prediction: {answer}'
        PhishLLMLogger.spit(msg, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
        if self.frontend_api:
            announcer.spit(msg, AnnouncerEvent.RESPONSE)
            time.sleep(0.5)
        if 'A.' in answer:
            return True, crp_llm_pred_time # CRP
        else:
            return False, crp_llm_pred_time

    def test(self, url: str,
             reference_logo: Optional[Image.Image],
             logo_box: Optional[List[float]],
             shot_path: str,
             html_path: str,
             driver: Union[CustomWebDriver, float]=None,
             limit: int=0,
             brand_recog_time: float=0, crp_prediction_time: float=0, clip_prediction_time: float=0,
             ranking_model_refresh_page: bool=True,
             skip_brand_recognition: bool=False,
             company_domain: Optional[str]=None, company_logo: Optional[Image.Image]=None,
             announcer: Optional[Announcer]=None
             ):
        '''
            PhishLLM
            :param url:
            :param reference_logo:
            :param logo_box:
            :param shot_path:
            :param html_path:
            :param driver:
            :param limit:
            :param brand_recog_time:
            :param crp_prediction_time:
            :param clip_prediction_time:
            :param ranking_model_refresh_page:
            :param skip_brand_recognition:
            :param company_domain:
            :param company_logo:
            :param announcer:
            :return:
        '''
        ## Run OCR to extract text
        plotvis = Image.open(shot_path)

        ## Brand recognition model
        if not skip_brand_recognition:
            company_domain, company_logo, brand_recog_time = self.brand_recognition_llm(reference_logo=reference_logo, announcer=announcer)
            time.sleep(self.brand_recog_sleep) # fixme: allow the openai api to rest, not sure whether this help
        # check domain-brand inconsistency
        domain_brand_inconsistent = False
        if company_domain:
            if company_domain in self.webhosting_domains:
                msg = '[\U00002705] Benign, since it is a brand providing cloud services'
                if self.frontend_api:
                    announcer.spit(msg, AnnouncerEvent.SUCCESS)
                PhishLLMLogger.spit(msg)
                return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis

            domain4pred, suffix4pred = tldextract.extract(company_domain).domain, tldextract.extract(company_domain).suffix
            domain4url, suffix4url = tldextract.extract(url).domain, tldextract.extract(url).suffix
            domain_brand_inconsistent = (domain4pred != domain4url) or (suffix4pred != suffix4url)

        phish_condition = domain_brand_inconsistent

        # Brand prediction results validation
        if phish_condition and (not skip_brand_recognition):
            if self.do_brand_validation: # we can check the validity by comparing the logo on the webpage with the logos for the predicted brand
                validation_success, logo_searching_time, logo_matching_time = self.brand_validation(company_domain=company_domain,
                                                                                                   reference_logo=reference_logo)
                brand_recog_time += logo_searching_time
                brand_recog_time += logo_matching_time
                phish_condition = validation_success
                msg = f"Time taken for brand validation (logo matching with Google Image search results): {logo_searching_time+logo_matching_time}<br>Domain {company_domain} is relevant and valid? {validation_success}"
                if self.frontend_api:
                    announcer.spit(msg, AnnouncerEvent.RESPONSE)
                    time.sleep(0.5)
                PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

        if phish_condition:
            # CRP prediction model
            crp_cls, crp_prediction_time = self.crp_prediction_llm(webpage_screenshot=plotvis, announcer=announcer)
            time.sleep(self.crp_sleep)

            if crp_cls: # CRP page is detected
                plotvis = draw_annotated_image_box(plotvis, company_domain, logo_box)
                msg = f'[\u2757\uFE0F] Phishing discovered, phishing target is {company_domain}'
                if self.frontend_api:
                    announcer.spit(msg, AnnouncerEvent.SUCCESS)
                PhishLLMLogger.spit(msg)
                return 'phish', company_domain, brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis
            else:
                # Not a CRP page => CRP transition
                if limit >= self.interaction_limit:  # reach interaction limit -> just return
                    msg = '[\U00002705] Benign, reached interaction limit ...'
                    if self.frontend_api:
                        announcer.spit(msg, AnnouncerEvent.SUCCESS)
                    PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis

                # Ranking model
                candidate_elements, _, driver, clip_prediction_time = self.ranking_model(url=url, driver=driver, ranking_model_refresh_page=ranking_model_refresh_page, announcer=announcer)

                if len(candidate_elements):
                    save_html_path = re.sub("index[0-9]?.html", f"index{limit}.html", html_path)
                    save_shot_path = re.sub("shot[0-9]?.png", f"shot{limit}.png", shot_path)

                    if not ranking_model_refresh_page: # if previous click didnt refresh the page select the lower ranked element to click
                        msg = f"Since previously the URL has not changed, trying to click the Top-{min(len(candidate_elements), limit+1)} login button instead: "
                        candidate_ele = candidate_elements[min(len(candidate_elements)-1, limit)]
                    else: # else, just click the top-1 element
                        msg = "Trying to click the Top-1 login button: "
                        candidate_ele = candidate_elements[0]

                    # record the webpage elements before clicking the button
                    prev_screenshot_elements = get_screenshot_elements(phishintention_cls=self.phishintention_cls, driver=driver)
                    element_text, current_url, *_ = page_transition(driver=driver, dom=candidate_ele, save_html_path=save_html_path, save_shot_path=save_shot_path)
                    msg += f'{element_text}'
                    if self.frontend_api:
                        announcer.spit(msg, AnnouncerEvent.RESPONSE)
                        time.sleep(0.5)
                    PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    if current_url: # click success
                        ranking_model_refresh_page = has_page_content_changed(phishintention_cls=self.phishintention_cls, driver=driver, prev_screenshot_elements=prev_screenshot_elements)
                        msg = f"Has the webpage changed? {ranking_model_refresh_page}"
                        if self.frontend_api:
                            announcer.spit(msg, AnnouncerEvent.RESPONSE)
                            time.sleep(0.5)
                        PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                        # logo detection on new webpage
                        logo_box, reference_logo = self.detect_logo(save_shot_path)
                        return self.test(current_url, reference_logo, logo_box,
                                         save_shot_path, save_html_path, driver, limit + 1,
                                         brand_recog_time, crp_prediction_time, clip_prediction_time,
                                         announcer=announcer,
                                         ranking_model_refresh_page=ranking_model_refresh_page,
                                         skip_brand_recognition=True,
                                         company_domain=company_domain, company_logo=company_logo)
                else:
                    msg = '[\U00002705] Benign'
                    if self.frontend_api:
                        announcer.spit(msg, AnnouncerEvent.SUCCESS)
                    PhishLLMLogger.spit(msg, caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                    return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis

        msg = '[\U00002705] Benign'
        if self.frontend_api:
            announcer.spit(msg, AnnouncerEvent.SUCCESS)
        PhishLLMLogger.spit(msg)
        return 'benign', 'None', brand_recog_time, crp_prediction_time, clip_prediction_time, plotvis



if __name__ == '__main__':

    # load hyperparameters
    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    PhishLLMLogger.set_debug_on()
    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls,
                      param_dict=param_dict,
                      proxies={"http": "http://127.0.0.1:7890",
                               "https": "http://127.0.0.1:7890",
                               }
                      )
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")  # Using the proxy_url variable
    driver.set_script_timeout(timeout_time / 2)
    driver.set_page_load_timeout(timeout_time)
#
    all_links = [x.strip().split(',')[-2] for x in open('./datasets/Brand_Labelled_130323.csv').readlines()[1:]]

    root_folder = './datasets/dynapd'
    result = './datasets/dynapd_llm.txt'
    os.makedirs(root_folder, exist_ok=True)

    for ct, target in enumerate(all_links):
        # if ct <= 5470:
        #     continue
        hash = target.split('/')[3]
        target_folder = os.path.join(root_folder, hash)
        os.makedirs(target_folder, exist_ok=True)
        if os.path.exists(result) and hash in open(result).read():
            continue
        if hash != 'b1baeaa9460713e9edf59076a80774ed':
            continue
        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        URL = f'http://127.0.0.5/{hash}'

        if os.path.exists(shot_path):
            # try:
            #     driver.get(URL)
            #     time.sleep(2)
            #     print(f'Target URL = {URL}')
            #     page_text = driver.get_page_text()
            #     error_free = web_func.page_error_checking(driver)
            #     if not error_free:
            #         print('Error page or White page')
            #         continue
            #
            #     if "Index of" in page_text:
            #         # skip error URLs
            #         error_free = web_func.page_interaction_checking(driver)
            #         if not error_free:
            #             print('Error page or White page')
            #             continue
            #
            # except Exception as e:
            #     print('Exception {}'.format(e))
            #     continue

            target = driver.current_url()
            logo_box, reference_logo = llm_cls.detect_logo(shot_path)
            pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, _ = llm_cls.test(target,
                                                                                                    reference_logo,
                                                                                                    logo_box,
                                                                                                    shot_path,
                                                                                                    html_path,
                                                                                                    driver,
                                                                                                    )
            with open(result, 'a+') as f:
                f.write(hash+'\t'+str(pred)+'\t'+str(brand)+'\t'+str(brand_recog_time)+'\t'+str(crp_prediction_time)+'\t'+str(crp_transition_time)+'\n')

    driver.quit()

    # 3.236595869064331
    # 0.3363449573516845
    # 0.3751556873321533
    # Total = 6075, LLM recall = 0.7501234567901235,
    # Phishpedia recall = 0.4388477366255144, PhishIntention recall = 0.33925925925925926
    # LLM precision = 1.0, Phishpedia precision = 0.9077289751447055, PhishIntention precision = 0.9795627376425855,






