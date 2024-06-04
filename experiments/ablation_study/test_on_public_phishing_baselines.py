from model_chain.test_baseline import TestBaseline
from model_chain.PhishIntentionWrapper import PhishIntentionWrapper
from model_chain.web_utils import CustomWebDriver
import argparse
import yaml
import cv2
import os
from tqdm import tqdm
from datetime import date
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='phishintention', choices=['phishpedia', 'phishintention'])
    args = parser.parse_args()

    # load hyperparameters
    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    # PhishLLM
    phishintention_cls = PhishIntentionWrapper()
    base_cls = TestBaseline(phishintention_cls)

    import pickle
    with open(f"{os.getenv('ANACONDA_ENV_PATH', '')}/lib/python3.8/site-packages/phishintention/src/phishpedia_siamese/domain_map.pkl", 'rb') as handle:
        domain_map = pickle.load(handle)
    domain_map['MWeb'] = ['mweb.co.za']
    domain_map['Uber'] = ['uber.com']
    domain_map['Caixa Geral de Depositos'] = ['cgd.pt']
    domain_map['Comcast Corporation'] = ['comcast.com']
    domain_map['cetelem'] = ['cetelem.fr']
    domain_map['Juno Online Services'] = ['juno.com']
    domain_map['Delta Air Lines'] = ['delta.com']
    domain_map['Credit du Nord'] = ['credit-du-nord.fr']
    domain_map['RBC Royal Bank'] = ['rbc.com', 'rbcroyalbank.com']
    with open(f"{os.getenv('ANACONDA_ENV_PATH', '')}/lib/python3.8/site-packages/phishintention/src/phishpedia_siamese/domain_map.pkl", 'wb') as handle:
        pickle.dump(domain_map, handle)

    # Xdriver
    if args.method != 'phishpedia':
        sleep_time = 3; timeout_time = 60
        driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")  # Using the proxy_url variable
        driver.set_script_timeout(timeout_time / 2)
        driver.set_page_load_timeout(timeout_time)

    root_folder = './datasets/public_phishing_feeds'

    datetime = '2024-05-21'
    result = f'./datasets/public_phishing/{datetime}_{args.method}.txt'
    os.makedirs("./datasets/public_phishing", exist_ok=True)

    for it, folder in tqdm(enumerate(os.listdir(os.path.join(root_folder, datetime)))):
        target_folder = os.path.join(root_folder, datetime, folder)

        if os.path.exists(result) and folder in [x.strip().split('\t')[0] for x in open(result).readlines()]:
             continue

        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        info_path = os.path.join(target_folder, 'info.txt')
        if os.path.exists(info_path):
            URL = open(info_path, encoding='utf-8').read()
        else:
            URL = f'http://{folder}'

        if os.path.exists(shot_path):
            if args.method == 'phishpedia':
                pred, brand, runtime, plotvis = base_cls.test_phishpedia(URL, shot_path)
            elif args.method == 'phishintention':
                pred, brand, runtime, plotvis = base_cls.test_phishintention(URL, shot_path, driver)

            try:
                with open(result, "a+", encoding='ISO-8859-1') as f:
                    f.write(folder + "\t")
                    f.write(str(pred) + "\t")
                    f.write(str(brand) + "\t")  # write top1 prediction only
                    f.write(str(runtime) + "\n")

            except UnicodeEncodeError:
                continue


    if args.method != 'phishpedia':
        driver.quit()

