import openai
from phishintention.src.OCR_aided_siamese import pred_siamese_OCR
from model_chain.web_utils import *
from model_chain.logger_utils import *
from model_chain.dynaphish.brand_knowledge_utils import BrandKnowledgeConstruction
import os
from model_chain.PhishIntentionWrapper import PhishIntentionWrapper
import yaml
from tldextract import tldextract
import pickle
from mmocr.apis import MMOCRInferencer
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./datasets/google_cloud.json"
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['CURL_CA_BUNDLE'] = ''

from PIL import Image
import io
import base64
import numpy as np
from phishintention.src.AWL_detector import element_config
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


class SubmissionButtonLocator():

    def __init__(self, button_locator_weights_path, button_locator_config):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _, self.BUTTON_SUBMISSION_MODEL = element_config(rcnn_weights_path=button_locator_weights_path,
                                                         rcnn_cfg_path=button_locator_config,
                                                         device=device)


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

    def return_submit_button(self, screenshot_encoding):
        screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
        screenshot_img = screenshot_img.convert("RGB")
        screenshot_img_arr = np.asarray(screenshot_img)
        screenshot_img_arr = np.flip(screenshot_img_arr, -1)  # RGB2BGR

        pred_classes, pred_boxes, pred_scores = self.element_recognition_reimplement(img_arr=screenshot_img_arr,
                                                                                     model=self.BUTTON_SUBMISSION_MODEL)
        if pred_boxes is None or len(pred_boxes) == 0:
            return None
        pred_boxes = pred_boxes.detach().cpu().numpy()
        return pred_boxes

class DynaPhish():

    def __init__(self, phishintention_cls, phishintention_config_path,
                 interaction_model, brandknowledge,
                 standard_sleeping_time=7, timeout=60):

        self.Phishintention = phishintention_cls
        self.phishintention_config_path = phishintention_config_path
        self.KnowledgeExpansion = brandknowledge
        self.Interaction = interaction_model
        self.standard_sleeping_time = standard_sleeping_time
        self.timeout = timeout
        self.webhosting_domains = [x.strip() for x in open('./datasets/hosting_blacklists.txt').readlines()]

    def domain_already_in_targetlist(self, domain_map_path, new_brand):
        with open(domain_map_path, 'rb') as handle:
            domain_map = pickle.load(handle)
        existing_brands = domain_map.keys()

        if new_brand in existing_brands:
            return domain_map, True
        return domain_map, False

    def plot_layout(self, screenshot_path):
        screenshot_img = Image.open(screenshot_path)
        screenshot_img = screenshot_img.convert("RGB")
        screenshot_img_arr = np.asarray(screenshot_img)
        screenshot_img_arr = np.flip(screenshot_img_arr, -1)  # RGB2BGR
        pred_classes, pred_boxes, pred_scores = self.Phishintention.element_recognition_reimplement(
                                                                    img_arr=screenshot_img_arr,
                                                                    model=self.Phishintention.AWL_MODEL)
        plotvis = self.Phishintention.layout_vis(screenshot_path, pred_boxes, pred_classes)
        return plotvis

    def expand_targetlist(self, config_path, new_brand, new_domains, new_logos):

        # Load configurations
        with open(config_path) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)

        # Update domain map
        domain_map, domain_in_target = self.domain_already_in_targetlist(
                                            domain_map_path=configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH'],
                                            new_brand=new_brand)
        if not domain_in_target:
            domain_map[new_brand] = list(set(new_domains))
            with open(configs['SIAMESE_MODEL']['DOMAIN_MAP_PATH'], 'wb') as handle:
                pickle.dump(domain_map, handle)

        # Process logos if valid
        valid_logo = [a for a in new_logos if a is not None]
        if not valid_logo:
            return

        # Setup logo save directory
        targetlist_path = configs['SIAMESE_MODEL']['TARGETLIST_PATH'].split('.zip')[0]
        new_logo_save_folder = os.path.join(targetlist_path, new_brand)
        os.makedirs(new_logo_save_folder, exist_ok=True)

        # Save logos in parallel
        def save_logo(logo, idx):
            save_path = os.path.join(new_logo_save_folder, f'{idx}.png')
            if os.path.exists(save_path):
                save_path = os.path.join(new_logo_save_folder, f'{idx}_expand.png')
            logo.save(save_path)
            return save_path

        def process_logo(logo_path, model, ocr_model):
            logo_features = pred_siamese_OCR(img=logo_path, model=model, ocr_model=ocr_model, grayscale=False)
            return logo_features, str(logo_path)

        with ThreadPoolExecutor(max_workers=10) as executor:
            start_idx = len(os.listdir(new_logo_save_folder))
            tasks = [executor.submit(save_logo, logo, start_idx + ct) for ct, logo in enumerate(valid_logo)]
            new_logo_save_paths = [task.result() for task in as_completed(tasks)]

        # Initialize lists to hold the results
        new_logo_feats = []
        new_file_name_list = []

        # Use ThreadPoolExecutor to parallelize the processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Create a list of future tasks
            future_tasks = [executor.submit(process_logo, logo_path, self.Phishintention.SIAMESE_MODEL, self.Phishintention.OCR_MODEL) for logo_path in new_logo_save_paths]

            # As each future task completes, extract the results
            for future in as_completed(future_tasks):
                logo_features, logo_path_str = future.result()
                new_logo_feats.append(logo_features)
                new_file_name_list.append(logo_path_str)

        # Update logo features and file lists
        prev_logo_feats_path = os.path.join(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH']),
                                            'LOGO_FEATS.npy')
        prev_file_name_list_path = os.path.join(os.path.dirname(configs['SIAMESE_MODEL']['TARGETLIST_PATH']),
                                                'LOGO_FILES.npy')

        if os.path.exists(prev_logo_feats_path) and os.path.exists(prev_file_name_list_path):
            prev_logo_feats = np.load(prev_logo_feats_path).tolist()
            prev_file_name_list = np.load(prev_file_name_list_path).tolist()
        else:
            prev_logo_feats, prev_file_name_list = [], []

        agg_logo_feats = prev_logo_feats + new_logo_feats
        agg_file_name_list = prev_file_name_list + new_file_name_list

        np.save(prev_logo_feats_path, np.asarray(agg_logo_feats))
        np.save(prev_file_name_list_path, np.asarray(agg_file_name_list))

        # Update reference list
        print(f'Length of references before: {len(self.Phishintention.LOGO_FEATS)}')
        self.Phishintention.LOGO_FEATS = np.asarray(agg_logo_feats)
        self.Phishintention.LOGO_FILES = np.asarray(agg_file_name_list)
        print(f'Length of references after: {len(self.Phishintention.LOGO_FEATS)}')

    def knowledge_expansion(self, driver, URL, screenshot_path, branch):

        query_domain = tldextract.extract(URL).domain
        query_tld = tldextract.extract(URL).suffix

        _, new_brand_domains, new_brand_name, new_brand_logos, knowledge_discovery_runtime, comment = \
            self.KnowledgeExpansion.runit_simplified(driver=driver,
                                                     shot_path=screenshot_path,
                                                     query_domain=query_domain,
                                                     query_tld=query_tld,
                                                     type=branch)

        '''If the found knowledge is not inside targetlist -> expand targetlist'''
        if len(new_brand_domains) and np.sum([x is not None for x in new_brand_logos]) > 0:
            self.expand_targetlist(config_path=self.phishintention_config_path,
                                   new_brand=new_brand_name,
                                   new_domains=new_brand_domains,
                                   new_logos=new_brand_logos)

        return new_brand_domains, new_brand_name, new_brand_logos, \
               knowledge_discovery_runtime, comment

    def test_dynaphish(self, URL, screenshot_path, kb_driver,
                       base_model, knowledge_expansion_branch,
                       kb_enabled=True):

        phishpedia_runtime = 0
        knowledge_discovery_runtime = 0
        web_interaction_algo_time = 0
        web_interaction_total_time = 0
        brand_in_targetlist = False
        phish_category = 0
        phish_target = None
        knowledge_discovery_branch = None
        found_knowledge = False
        interaction_success = True
        redirection_evasion, no_verification = False, False
        plotvis = None

        # Has a logo or not?
        has_logo, in_target_list = self.Phishintention.has_logo(screenshot_path=screenshot_path)
        print('Has logo? {} Is in targetlist? {}'.format(has_logo, in_target_list))

        # domain(w)
        query_domain, query_tld = tldextract.extract(URL).domain, tldextract.extract(URL).suffix

        if query_domain+'.'+query_tld in self.webhosting_domains:
            return phish_category, phish_target, plotvis, \
                   has_logo, brand_in_targetlist, \
                   found_knowledge, knowledge_discovery_branch, \
                   str(phishpedia_runtime) + '|' + str(knowledge_discovery_runtime) + '|' + str(
                       web_interaction_algo_time) + '|' + str(web_interaction_total_time), \
                   str(interaction_success) + '|' + str(redirection_evasion) + '|' + str(no_verification)

        # If rep(w)!=null, i.e. has logo
        if has_logo:
            if in_target_list:
                '''Report as phishing'''
                if base_model == 'phishpedia':
                    start_time = time.time()
                    phish_category, phish_target, plotvis, siamese_conf, time_breakdown, pred_boxes, pred_classes = \
                        self.Phishintention.test_orig_phishpedia(URL, screenshot_path)
                    phishpedia_runtime = time.time() - start_time

                elif base_model == 'phishintention':
                    ph_driver = CustomWebDriver.boot(chrome=True)
                    time.sleep(self.standard_sleeping_time)
                    ph_driver.set_page_load_timeout(self.timeout)
                    ph_driver.set_script_timeout(self.timeout)
                    start_time = time.time()
                    phish_category, phish_target, plotvis, dynamic, time_breakdown, pred_boxes, pred_classes = \
                        self.Phishintention.test_orig_phishintention(URL, screenshot_path, ph_driver)
                    phishpedia_runtime = time.time() - start_time
                    ph_driver.quit()
                else:
                    raise NotImplementedError

            elif (not in_target_list) and kb_enabled:
                _, new_brand_domains, new_brand_name, new_brand_logos, knowledge_discovery_runtime, knowledge_discovery_branch = \
                    self.KnowledgeExpansion.runit_simplified(kb_driver, screenshot_path, query_domain, query_tld, knowledge_expansion_branch)
                print('Comment: ', knowledge_discovery_branch)

                # Ref* <- Ref* + <domain(w_target), rep(w_target)>
                if len(new_brand_domains) > 0 and np.sum([x is not None for x in new_brand_logos]) > 0:
                    print('==== Adding new brand... ====')
                    self.expand_targetlist(config_path=self.phishintention_config_path,
                                           new_brand=new_brand_name,
                                           new_domains=new_brand_domains,
                                           new_logos=new_brand_logos)

                    print('==== New brand added ====')

                    '''Report as phishing'''
                    if base_model == 'phishpedia':
                        start_time = time.time()
                        phish_category, phish_target, plotvis, siamese_conf, time_breakdown, pred_boxes, pred_classes = \
                            self.Phishintention.test_orig_phishpedia(URL, screenshot_path)
                        phishpedia_runtime = time.time() - start_time
                    elif base_model == 'phishintention':
                        ph_driver = CustomWebDriver.boot(chrome=True)
                        time.sleep(self.standard_sleeping_time); ph_driver.set_page_load_timeout(self.timeout); ph_driver.set_script_timeout(self.timeout)
                        start_time = time.time()
                        phish_category, phish_target, plotvis, dynamic, time_breakdown, pred_boxes, pred_classes = \
                            self.Phishintention.test_orig_phishintention(URL, screenshot_path, ph_driver)
                        phishpedia_runtime = time.time() - start_time
                        ph_driver.quit()
                    else:
                        raise NotImplementedError

        return phish_category, phish_target, plotvis, \
               has_logo, brand_in_targetlist, \
               found_knowledge, knowledge_discovery_branch, \
               str(phishpedia_runtime) + '|' + str(knowledge_discovery_runtime) + '|' + str(
                   web_interaction_algo_time) + '|' + str(web_interaction_total_time), \
               str(interaction_success) + '|' + str(redirection_evasion) + '|' + str(no_verification)


if __name__ == '__main__':

    # with open('/home/ruofan/git_space/ScamDet/model_chain/dynaphish/domain_map.pkl', "rb") as handle:
    #     domain_map = pickle.load(handle)
    # #
    # domain_map['Delta Air Lines'] = ['delta']
    # #
    # with open('/home/ruofan/git_space/ScamDet/model_chain/dynaphish/domain_map.pkl', "wb") as handle:
    #     pickle.dump(domain_map, handle)
    # exit()

    PhishLLMLogger.set_debug_on()
    phishintention_config_path = '/home/ruofan/git_space/ScamDet/model_chain/dynaphish/configs.yaml'
    PhishIntention = PhishIntentionWrapper()
    PhishIntention.reset_model(phishintention_config_path, False)

    API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]
    KnowledgeExpansionModule = BrandKnowledgeConstruction(API_KEY, SEARCH_ENGINE_ID, PhishIntention)

    mmocr_model = MMOCRInferencer(det=None,
                                    rec='ABINet',
                                    device='cuda' if torch.cuda.is_available() else 'cpu')
    button_locator_model = SubmissionButtonLocator(
        button_locator_config="/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/config.yaml",
        button_locator_weights_path="/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/model_final.pth")

    dynaphish_cls = DynaPhish(PhishIntention,
                              phishintention_config_path,
                              None,
                              KnowledgeExpansionModule)

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
    result_txt = './datasets/dynapd_dynaphish.txt'
    os.makedirs(root_folder, exist_ok=True)

    for ct, target in enumerate(all_links):
        # if ct <= 5470:
        #     continue
        hash = target.split('/')[3]
        target_folder = os.path.join(root_folder, hash)
        os.makedirs(target_folder, exist_ok=True)
        if os.path.exists(result_txt) and hash in open(result_txt).read():
            continue
        # if hash != 'b1baeaa9460713e9edf59076a80774ed':
        #     continue
        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        URL = f'http://127.0.0.5/{hash}'

        if os.path.exists(shot_path):

            phish_category, phish_target, plotvis, has_logo, brand_in_targetlist, \
            found_knowledge, knowledge_discovery_branch, runtime_breakdown, interaction_success = \
                dynaphish_cls.test_dynaphish(URL=URL, screenshot_path=shot_path, kb_driver=driver,
                                            base_model='phishintention',
                                            knowledge_expansion_branch='logo2brand',
                                            kb_enabled=True)

            # write results as well as predicted image
            try:
                with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                    f.write(hash + "\t")
                    f.write(str(phish_category) + "\t")
                    f.write(str(phish_target) + "\t")  # write top1 prediction only
                    f.write(runtime_breakdown + "\n")

            except UnicodeEncodeError:
                continue


    driver.quit()

    #






