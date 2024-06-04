from model_chain.test_baseline import *
from model_chain.web_utils import CustomWebDriver
from model_chain.dynaphish.brand_knowledge_utils import BrandKnowledgeConstruction
from model_chain.test_dynaphish import DynaPhish, SubmissionButtonLocator
import argparse
import cv2
from mmocr.apis import MMOCRInferencer
import torch
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./datasets/field_study/2024-01-29/")
    parser.add_argument("--date", default="2024-01-29", help="%Y-%m-%d")
    parser.add_argument("--method", default='dynaphish', choices=['phishpedia', 'phishintention', 'dynaphish'])
    args = parser.parse_args()

    # PhishLLM
    phishintention_cls = PhishIntentionWrapper()
    base_cls = TestBaseline(phishintention_cls)

    # Xdriver
    if args.method != 'phishpedia':
        sleep_time = 3; timeout_time = 60
        driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")  # Using the proxy_url variable
        driver.set_script_timeout(timeout_time / 2)
        driver.set_page_load_timeout(timeout_time)

    if args.method == 'dynaphish':
        phishintention_config_path = '/home/ruofan/git_space/ScamDet/model_chain/dynaphish/configs.yaml' # todo: copy a new one
        phishintention_cls.reset_model(phishintention_config_path, False)

        API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]
        KnowledgeExpansionModule = BrandKnowledgeConstruction(API_KEY, SEARCH_ENGINE_ID, phishintention_cls,
                                                              proxies={"http": "http://127.0.0.1:7890",
                                                                       "https": "http://127.0.0.1:7890",
                                                                       })

        mmocr_model = MMOCRInferencer(det=None,
                                      rec='ABINet',
                                      device='cuda')
        button_locator_model = SubmissionButtonLocator(
            button_locator_config="/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/config.yaml",
            button_locator_weights_path="/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/model_final.pth")

        dynaphish_cls = DynaPhish(phishintention_cls,
                                  phishintention_config_path,
                                  None,
                                  KnowledgeExpansionModule)

    os.makedirs('./field_study/results/', exist_ok=True)
    result_txt = './field_study/results/{}_{}.txt'.format(args.date, args.method)

    if not os.path.exists(result_txt):
        with open(result_txt, "w+") as f:
            f.write("folder" + "\t")
            f.write("phish_prediction" + "\t")
            f.write("target_prediction" + "\t")  # write top1 prediction only
            f.write("runtime" + "\n")

    for ct, folder in tqdm(enumerate(os.listdir(args.folder))):
        if folder in [x.split('\t')[0] for x in open(result_txt, encoding='ISO-8859-1').readlines()]:
            continue

        info_path = os.path.join(args.folder, folder, 'info.txt')
        html_path = os.path.join(args.folder, folder, 'html.txt')
        shot_path = os.path.join(args.folder, folder, 'shot.png')
        predict_path = os.path.join(args.folder, folder, 'predict_{}.png'.format(args.method))
        if not os.path.exists(shot_path):
            continue

        try:
            if len(open(info_path, encoding='ISO-8859-1').read()) > 0:
                url = open(info_path, encoding='ISO-8859-1').read()
            else:
                url = 'https://' + folder
        except:
            url = 'https://' + folder

        try:
            if args.method == 'phishpedia':
                pred, brand, runtime, plotvis = base_cls.test_phishpedia(url, shot_path)
            elif args.method == 'phishintention':
                pred, brand, runtime, plotvis = base_cls.test_phishintention(url, shot_path, driver)
            elif args.method == 'dynaphish':
                pred, brand, plotvis, _, _, \
                _, _, runtime_breakdown, _ = \
                    dynaphish_cls.test_dynaphish(URL=url,
                                                 screenshot_path=shot_path,
                                                 kb_driver=driver,
                                                 base_model='phishintention',
                                                 knowledge_expansion_branch='logo2brand',
                                                 kb_enabled=True)
                parts = runtime_breakdown.split('|')
                float_parts = [float(part) for part in parts]
                runtime = sum(float_parts)

        except KeyError:
            continue

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(folder + "\t")
                f.write(str(pred) + "\t")
                f.write(str(brand) + "\t")  # write top1 prediction only
                f.write(str(runtime) + "\n")
            if pred == 1:
                cv2.imwrite(predict_path, plotvis)

        except UnicodeEncodeError:
            continue

        if (ct + 501) % 500 == 0:
            if args.method != 'phishpedia':
                driver.quit()
                driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")  # Using the proxy_url variable
                driver.set_script_timeout(timeout_time / 2)
                driver.set_page_load_timeout(timeout_time)

    if args.method != 'phishpedia':
        driver.quit()