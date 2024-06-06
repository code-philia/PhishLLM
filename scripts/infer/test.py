import os
import time

from scripts.pipeline.test_llm import *
import argparse
from tqdm import tqdm
import yaml
import openai
from datetime import datetime, date, timedelta
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./datasets/field_study/2023-09-02/")
    parser.add_argument("--config", default='./param_dict.yaml', help="Config .yaml path")
    args = parser.parse_args()

    PhishLLMLogger.set_debug_on()
    PhishLLMLogger.set_verbose(True)

    # load hyperparameters
    with open(args.config) as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    # PhishLLM
    proxy_url = os.environ.get('proxy_url', None)
    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls,
                      param_dict=param_dict,
                      proxies={"http": proxy_url,
                               "https": proxy_url,
                               })
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = proxy_url # set openai proxy

    # boot driver
    driver = CustomWebDriver.boot(proxy_server=proxy_url)  # Using the proxy_url variable
    driver.set_script_timeout(param_dict['rank']['script_timeout'])
    driver.set_page_load_timeout(param_dict['rank']['page_load_timeout'])

    day = date.today().strftime("%Y-%m-%d")
    result_txt = '{}_phishllm.txt'.format(day)

    if not os.path.exists(result_txt):
        with open(result_txt, "w+") as f:
            f.write("folder" + "\t")
            f.write("phish_prediction" + "\t")
            f.write("target_prediction" + "\t")  # write top1 prediction only
            f.write("brand_recog_time" + "\t")
            f.write("crp_prediction_time" + "\t")
            f.write("crp_transition_time" + "\n")

    for ct, folder in tqdm(enumerate(os.listdir(args.folder))):
        if folder in [x.split('\t')[0] for x in open(result_txt, encoding='ISO-8859-1').readlines()]:
            continue

        info_path = os.path.join(args.folder, folder, 'info.txt')
        html_path = os.path.join(args.folder, folder, 'html.txt')
        shot_path = os.path.join(args.folder, folder, 'shot.png')
        predict_path = os.path.join(args.folder, folder, 'predict.png')
        if not os.path.exists(shot_path):
            continue

        try:
            if len(open(info_path, encoding='ISO-8859-1').read()) > 0:
                url = open(info_path, encoding='ISO-8859-1').read()
            else:
                url = 'https://' + folder
        except FileNotFoundError:
            url = 'https://' + folder

        logo_box, reference_logo = llm_cls.detect_logo(shot_path)
        while True:
            try:
                pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis = llm_cls.test(url=url,
                                                                                                                reference_logo=reference_logo,
                                                                                                                logo_box=logo_box,
                                                                                                                shot_path=shot_path,
                                                                                                                html_path=html_path,
                                                                                                                driver=driver,
                                                                                                                )
                driver.delete_all_cookies()
                break

            except (WebDriverException) as e:
                print(f"Driver crashed or encountered an error: {e}. Restarting driver.")
                driver.quit()
                time.sleep(1)
                driver = CustomWebDriver.boot(proxy_server=proxy_url)  # Using the proxy_url variable
                driver.set_script_timeout(param_dict['rank']['script_timeout'])
                driver.set_page_load_timeout(param_dict['rank']['page_load_timeout'])
                continue

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(f"{folder}\t{pred}\t{brand}\t{brand_recog_time}\t{crp_prediction_time}\t{crp_transition_time}\n")
            if pred == 'phish':
                plotvis.save(predict_path)

        except UnicodeEncodeError:
            continue


    driver.quit()

