import shutil

import openai
from scripts.utils.web_utils.web_utils import *
from scripts.utils.logger_utils import *
import os
from scripts.utils.PhishIntentionWrapper import PhishIntentionWrapper
import yaml
from scripts.pipeline.test_llm import TestVLM
from tqdm import tqdm

os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if __name__ == '__main__':

    # load hyperparameters
    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    # debug_list = './datasets/cryptocurrency_phishing/'
    # root_folder = './datasets/public_phishing_feeds'
    # result = './datasets/cryptocurrency_phishing.txt'

    debug_list = './datasets/LLM has insufficient information to determine its brand/'
    root_folder = './datasets/public_phishing_feeds'
    result = './datasets/insufficient_description_phishing_vlm.txt'

    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestVLM(phishintention_cls,
                      param_dict=param_dict,
                      proxies={"http": "http://127.0.0.1:7890",
                               "https": "http://127.0.0.1:7890",
                               }
                      )
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy

    sleep_time = 3; timeout_time = 60
    driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")  # Using the proxy_url variable
    driver.set_script_timeout(timeout_time)
    driver.set_page_load_timeout(timeout_time)

    PhishLLMLogger.set_debug_on()
    PhishLLMLogger.set_verbose(True)
    # PhishLLMLogger.set_logfile("./datasets/cryptocurrency_phishing.log")

    sampled_crypto_phishing = os.listdir(debug_list)
    for filename in tqdm(sampled_crypto_phishing):

        pattern = r"(\d{4}-\d{2}-\d{2})_(.+)"
        match = re.match(pattern, filename)
        date = match.group(1)
        file = match.group(2).split('.png')[0]

        target_folder = os.path.join(root_folder, date, file)

        if os.path.exists(result) and target_folder in [x.strip().split('\t')[0] for x in open(result).readlines()]:
             continue

        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        info_path = os.path.join(target_folder, 'info.txt')
        if os.path.exists(info_path):
            URL = open(info_path, encoding='utf-8').read()
        else:
            URL = f'http://{file}'

        if os.path.exists(shot_path):
            logo_box, reference_logo = llm_cls.detect_logo(shot_path)
            PhishLLMLogger.spit(URL)
            pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, _ = llm_cls.test(URL,
                                                                                                        reference_logo,
                                                                                                        logo_box,
                                                                                                        shot_path,
                                                                                                        html_path,
                                                                                                        driver,
                                                                                                      )
            with open(result, 'a+') as f:
                f.write(target_folder+'\t'+str(pred)+'\t'+str(brand)+'\t'+str(brand_recog_time)+'\t'+str(crp_prediction_time)+'\t'+str(crp_transition_time)+'\n')
        else:
            shutil.rmtree(target_folder)

        driver.delete_all_cookies()

    driver.quit()

    ct = 0
    total_ct = 0
    for line in open(result).readlines()[:50]:
        if "\tphish\t" in line:
            ct += 1
        total_ct += 1

    print(total_ct, ct, ct/total_ct)
    # Croptocurrency phishing VLM: 36 out of 50
    # Insufficient textual information: 11 out of 19, the left FNs switch to the 'non-standard credential' FN reason
