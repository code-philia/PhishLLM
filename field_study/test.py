from model_chain.test_llm import *
import argparse
from tqdm import tqdm
import yaml

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./datasets/field_study/2024-01-26/")
    parser.add_argument("--date", default="2024-01-26", help="%Y-%m-%d")
    parser.add_argument("--validate", action='store_true', help="Whether or not to activate the results validation for brand recognition model")
    parser.add_argument("--config", default='./param_dict.yaml', help="Config .yaml path")
    args = parser.parse_args()

    PhishLLMLogger.set_debug_on()
    PhishLLMLogger.set_logfile('./field_study/results/{}_phishllm.log'.format(args.date))

    # load hyperparameters
    with open(args.config) as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    # PhishLLM
    proxy_url = os.environ.get('proxy_url', None)
    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls,
                      param_dict=param_dict,
                      proxies={"http": "http://127.0.0.1:7890",
                               "https": "http://127.0.0.1:7890",
                               })
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = proxy_url # set openai proxy

    # boot driver
    sleep_time = 3; timeout_time = 5
    driver = CustomWebDriver.boot(proxy_server=proxy_url)  # Using the proxy_url variable
    driver.set_script_timeout(timeout_time)
    driver.set_page_load_timeout(timeout_time)

    os.makedirs('./field_study/results/', exist_ok=True)

    result_txt = './field_study/results/{}_phishllm.txt'.format(args.date)

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

        # if folder != 'login.umbrella.com':
        #     continue
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
        except:
            url = 'https://' + folder

        if url.startswith('cpanel'): # skip cpanel hosting
            continue

        # url = 'file:///home/ruofan/git_space/ScamDet/datasets/field_study/2023-09-05/msrgrptranstaging.trafficmanager.net/copy.html'
        # candidate_elements, _, driver = llm_cls.ranking_model(url=url, driver=driver,
        #                                                    ranking_model_refresh_page=True,
        #                                                    announcer=None)

        PhishLLMLogger.spit(f"Folder {os.path.join(args.folder, folder)}", caller_prefix=PhishLLMLogger._caller_prefix, debug=True)

        logo_box, reference_logo = llm_cls.detect_logo(shot_path)
        pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis = llm_cls.test(url, reference_logo, logo_box,
                                                                                                        shot_path, html_path, driver,
                                                                                                        )
        #
        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(folder + "\t")
                f.write(str(pred) + "\t")
                f.write(str(brand) + "\t")  # write top1 prediction only
                f.write(str(brand_recog_time) + "\t")
                f.write(str(crp_prediction_time) + "\t")
                f.write(str(crp_transition_time) + "\n")
            if pred == 'phish':
                plotvis.save(predict_path)

        except UnicodeEncodeError:
            continue

        if (ct + 501) % 500 == 0:
            driver.quit()
            driver = CustomWebDriver.boot(proxy_server=proxy_url)  # Using the proxy_url variable
            driver.set_script_timeout(timeout_time)
            driver.set_page_load_timeout(timeout_time)

    driver.quit()

