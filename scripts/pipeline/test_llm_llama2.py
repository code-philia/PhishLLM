from scripts.utils.utils import *
from scripts.utils.web_utils.web_utils import *
from scripts.utils.logger_utils import *
from server.announcer import Announcer, AnnouncerEvent, AnnouncerPrompt
from experiments.field_study.draw_utils import draw_annotated_image_box
from typing import List, Tuple, Optional
from scripts.utils.PhishIntentionWrapper import PhishIntentionWrapper
import yaml
from tldextract import tldextract
from llama import Llama
from pipeline.test_llm import TestLLM
import contextlib
import os
import PIL

@contextlib.contextmanager
def set_env_vars(vars):
    original_vars = {var: os.environ.get(var) for var in vars}
    os.environ.update(vars)
    yield
    for var, value in original_vars.items():
        if value is None:
            del os.environ[var]
        else:
            os.environ[var] = value

def initialize_llama2():
    with set_env_vars({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12356', 'RANK': '0', 'WORLD_SIZE': '1'}):
        return Llama.build(
            ckpt_dir='../llama/llama-2-7b-chat',
            tokenizer_path='../llama/tokenizer.model',
            max_seq_len=2048,
            max_batch_size=2,
        )

class TestLlama2(TestLLM):

    def __init__(self, phishintention_cls, param_dict, proxies=None, frontend_api=False):
        self.llama2_model = initialize_llama2()
        super(TestLlama2, self).__init__(phishintention_cls, param_dict, proxies, frontend_api)
        self.top_p = param_dict['brand_recog']['top_p']

    @staticmethod
    def extract_domain(text):
        # This regex pattern looks for a generic domain structure: subdomain.domain.tld
        pattern = r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+\b"
        match = re.search(pattern, text)
        if match:
            return match.group(0)
        else:
            return ''

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
            inference_done = False
            while not inference_done:
                try:
                    response = self.llama2_model.chat_completion(
                        [prompt],  # type: ignore
                        max_gen_len=5,
                        temperature=self.brand_recog_temperature,
                        top_p=self.top_p,
                    )
                    inference_done = True
                except Exception as e:
                    print(f"Error was: {e}")
                    prompt[-1]['content'] = prompt[-1]['content'][:int(2*len(prompt[-1]['content'])/3)]

            industry = response[0]['generation']['content'].strip().lower()
            if len(industry) > 30:
                industry = ''

        return industry

    def brand_recognition_llm(self, reference_logo: Optional[Image.Image],
                              webpage_text: str, logo_caption: str, logo_ocr: str,
                              announcer: Optional[Announcer]) -> Tuple[Optional[str], Optional[Image.Image]]:
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

        industry = ''
        if len(webpage_text) and self.get_industry:
            industry = self.ask_industry(webpage_text, announcer)

        if self.frontend_api:
            announcer.spit(f'Industry: {industry}', AnnouncerEvent.RESPONSE)
        PhishLLMLogger.spit(f'Logo caption: {logo_caption}<br>Logo OCR: {logo_ocr}<br>Industry: {industry}', debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

        if len(logo_caption) > 0 or len(logo_ocr) > 0:
            # ask gpt to predict brand
            if self.get_industry:
                question = question_template_brand_industry(logo_caption, logo_ocr, industry)
                if self.frontend_api:
                    announcer.spit(AnnouncerPrompt.question_template_brand_industry(logo_caption, logo_ocr, industry), AnnouncerEvent.PROMPT)
            else:
                question = question_template_brand(logo_caption, logo_ocr)
                if self.frontend_api:
                    announcer.spit(AnnouncerPrompt.question_template_brand(logo_caption, logo_ocr), AnnouncerEvent.PROMPT)
	   
            with open(self.brand_prompt, 'rb') as f:
                prompt = json.load(f)
            new_prompt = prompt
            new_prompt.append(question)

            inference_done = False
            while not inference_done:
                try:
                    start_time = time.time()
                    response = self.llama2_model.chat_completion(
                        [new_prompt],  # type: ignore
                        max_gen_len=10,
                        temperature=self.brand_recog_temperature,
                        top_p=self.top_p,
                    )
                    inference_done = True
                except Exception as e:
                    PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

            answer = response[0]['generation']['content'].strip().lower()
            answer = self.extract_domain(answer)

            if self.frontend_api:
                announcer.spit(f"Time taken for LLM brand prediction: {time.time() - start_time}<br>Detected brand: {answer}", AnnouncerEvent.RESPONSE)
            PhishLLMLogger.spit(f"LLM prediction time: {time.time() - start_time}<br>Detected brand: {answer}", debug=True, caller_prefix=PhishLLMLogger._caller_prefix)

            # check the validity of the returned domain, i.e. liveness
            if len(answer) > 0 and is_valid_domain(answer):
                company_logo = reference_logo
                company_domain = answer
        else:
            msg = 'No logo description'
            PhishLLMLogger.spit(msg, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
            if self.frontend_api:
                announcer.spit(msg, AnnouncerEvent.RESPONSE)

        return company_domain, company_logo


    def crp_prediction_llm(self, html_text: str, announcer: Optional[Announcer]) -> bool:
        '''
            Use LLM to classify credential-requiring page v.s. non-credential-requiring page
            :param html_text:
            :param announcer:
            :return:
        '''
        question = question_template_prediction(html_text)
        if self.frontend_api:
            announcer.spit(AnnouncerPrompt.question_template_prediction(html_text), AnnouncerEvent.PROMPT)
        with open(self.crp_prompt, 'rb') as f:
            prompt = json.load(f)
        new_prompt = prompt
        new_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                start_time = time.time()
                response = self.llama2_model.chat_completion(
                    [new_prompt],  # type: ignore
                    max_gen_len=100,
                    temperature=self.brand_recog_temperature,
                    top_p=self.top_p,
                )
                inference_done = True
            except Exception as e:
                PhishLLMLogger.spit('LLM Exception {}'.format(e), debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2] # maybe the prompt is too long, cut by half

        answer = response[0]['generation']['content'].strip().lower()
        msg = f'Time taken for LLM CRP classification: {time.time() - start_time}<br>CRP prediction: {answer}'
        PhishLLMLogger.spit(msg, debug=True, caller_prefix=PhishLLMLogger._caller_prefix)
        if self.frontend_api:
            announcer.spit(msg, AnnouncerEvent.RESPONSE)
        if ('not a credential-requiring' not in answer) and (' b.' not in answer) and ('(b)' not in answer):
            return True # CRP
        else:
            return False

    def test(self, url: str, reference_logo: Optional[Image.Image],
             logo_box: Optional[List[float]],
             shot_path: str, html_path: str, driver: CustomWebDriver, limit: int=0,
             brand_recog_time: float=0, crp_prediction_time: float=0, crp_transition_time: float=0,
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
            :param crp_transition_time:
            :param ranking_model_refresh_page:
            :param skip_brand_recognition:
            :param company_domain:
            :param company_logo:
            :param announcer:
            :return:
        '''
        ## Run OCR to extract text
        try:
            plotvis = Image.open(shot_path)
        except PIL.UnidentifiedImageError:
            return 'benign', 'None', 0, 0, 0, None

        image_width, image_height = plotvis.size
        (webpage_text, logo_caption, logo_ocr), (ocr_processing_time, image_caption_processing_time)  = self.preprocessing(shot_path=shot_path, html_path=html_path,
                                                                                                                           reference_logo=reference_logo, logo_box=logo_box,
                                                                                                                           image_width=image_width, image_height=image_height,
                                                                                                                           announcer=announcer)

        ## Brand recognition model
        if not skip_brand_recognition:
            start_time = time.time()
            company_domain, company_logo = self.brand_recognition_llm(reference_logo=reference_logo, webpage_text=webpage_text,
                                                                      logo_caption=logo_caption, logo_ocr=logo_ocr, announcer=announcer)
            brand_recog_time += time.time() - start_time

        # check domain-brand inconsistency
        domain_brand_inconsistent = False
        if company_domain:
            if company_domain in self.webhosting_domains:
                msg = '[\U00002705] Benign, since it is a brand providing cloud services'
                if self.frontend_api:
                    announcer.spit(msg, AnnouncerEvent.SUCCESS)
                PhishLLMLogger.spit(msg)
                return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time, plotvis

            domain4pred, suffix4pred = tldextract.extract(company_domain).domain, tldextract.extract(company_domain).suffix
            domain4url, suffix4url = tldextract.extract(url).domain, tldextract.extract(url).suffix 
            domain_brand_inconsistent = (domain4pred != domain4url) or (suffix4pred != suffix4url)

        phish_condition = domain_brand_inconsistent

        # Brand prediction results validation
        if phish_condition and (not skip_brand_recognition):
            if self.do_brand_validation and (reference_logo is not None): # we can check the validity by comparing the logo on the webpage with the logos for the predicted brand
                validation_success, logo_cropping_time, logo_matching_time = self.brand_validation(company_domain=company_domain, reference_logo=reference_logo)
                brand_recog_time += logo_cropping_time
                brand_recog_time += logo_matching_time
                phish_condition = validation_success
                msg = f"Time taken for brand validation (logo matching with Google Image search results): {logo_cropping_time+logo_matching_time}<br>Domain {company_domain} is relevant and valid? {validation_success}"
                if self.frontend_api:
                    announcer.spit(msg, AnnouncerEvent.RESPONSE)
            # else: # alternatively, we can check the aliveness of the predicted brand
            #     validation_success = is_alive_domain(company_domain, self.proxies)
            #     phish_condition = validation_success
            #     msg = f"Brand Validation: Domain {company_domain} is alive? {validation_success}"
            #     announcer.spit(msg, AnnouncerEvent.RESPONSE)

        if phish_condition:
            # CRP prediction model
            start_time = time.time()
            crp_cls = self.crp_prediction_llm(html_text=webpage_text, announcer=announcer)
            crp_prediction_time += time.time() - start_time

            if crp_cls: # CRP page is detected
                plotvis = draw_annotated_image_box(plotvis, company_domain, logo_box)
                msg = f'[\u2757\uFE0F] Phishing discovered, phishing target is {company_domain}'
                if self.frontend_api:
                    announcer.spit(msg, AnnouncerEvent.SUCCESS)
                PhishLLMLogger.spit(msg)
                return 'phish', company_domain, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis
            else:
                return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time, plotvis

        msg = '[\U00002705] Benign'
        if self.frontend_api:
            announcer.spit(msg, AnnouncerEvent.SUCCESS)
        PhishLLMLogger.spit(msg)
        return 'benign', 'None', brand_recog_time, crp_prediction_time, crp_transition_time, plotvis



if __name__ == '__main__':

    # load hyperparameters
    with open('./param_dict_llama2.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    PhishLLMLogger.set_debug_on()
    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLlama2(phishintention_cls,
                      param_dict=param_dict,
                      proxies={"http": "http://127.0.0.1:7890",
                               "https": "http://127.0.0.1:7890",
                               }
                      )
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")  # Using the proxy_url variable
    driver.set_script_timeout(timeout_time / 2)
    driver.set_page_load_timeout(timeout_time)

    all_links = [x.strip().split(',')[-2] for x in open('./datasets/Brand_Labelled_130323.csv').readlines()[1:]]

    root_folder = './datasets/dynapd'
    result = './datasets/dynapd_llama2.txt'
    os.makedirs(root_folder, exist_ok=True)

    for ct, target in enumerate(all_links):
        # if ct <= 5470:
        #     continue
        hash = target.split('/')[3]
        target_folder = os.path.join(root_folder, hash)
        os.makedirs(target_folder, exist_ok=True)
        if os.path.exists(result) and hash in open(result).read():
            continue
        # if hash != 'b1baeaa9460713e9edf59076a80774ed':
        #     continue
        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        URL = f'http://127.0.0.5/{hash}'

        if os.path.exists(shot_path):
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







