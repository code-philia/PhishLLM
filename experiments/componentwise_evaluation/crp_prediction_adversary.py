import time
import openai
import pandas as pd
import os
from tqdm import tqdm
import yaml
from scripts.pipeline.test_llm import TestLLM
from scripts.utils.PhishIntentionWrapper import PhishIntentionWrapper
import numpy as np
from PIL import Image

def test(result_file):
    ct = 0
    total = 0
    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        hash, gt, pred, time = data
        if 'A.' in pred:
            ct += 1
        total += 1

        pbar.set_description( f"After attack with defense recall: {ct / total} ", refresh=True)

    print(f"After attack with defense recall: {ct / total} ")


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    proxy_url = "http://127.0.0.1:7890"
    openai.proxy = proxy_url  # proxy

    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls,
                      param_dict=param_dict,
                      proxies={"http": proxy_url,
                               "https": proxy_url,
                               }
                      )

    root_folder = './datasets/dynapd'
    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_wo_validation.txt').readlines()]
    df = pd.read_csv('./datasets/Brand_Labelled_130323.csv')
    crp_list = [x.strip() for x in open('./datasets/dynapd_crp_list.txt').readlines()]

    result = './datasets/dynapd_llm_adv_selection_defense.txt'

    for hash in tqdm(all_folders):
        target_folder = os.path.join(root_folder, hash)
        if os.path.exists(result) and hash in open(result).read():
            continue
        if hash not in crp_list:
            continue

        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        if not os.path.exists(shot_path):
            continue

        pk_info = df.loc[df['HASH'] == hash]
        try:
            URL = list(pk_info['URL'])[0]
        except IndexError:
            URL = f'http://127.0.0.5/{hash}'

        logo_box, reference_logo = llm_cls.detect_logo(shot_path)
        plotvis = Image.open(shot_path)
        image_width, image_height = plotvis.size
        (webpage_text, logo_caption, logo_ocr), (ocr_processing_time, image_caption_processing_time) = \
            llm_cls.preprocessing(shot_path=shot_path,
                                  html_path=html_path,
                                  reference_logo=reference_logo,
                                  logo_box=logo_box,
                                  image_width=image_width,
                                  image_height=image_height,
                                  announcer=None)

        crp_cls, crp_prediction_time = llm_cls.crp_prediction_llm(html_text=webpage_text,
                                                                  announcer=None)

        with open(result, 'a+') as f:
            f.write(hash+'\t'+'A'+'\t'+str(crp_cls)+'\t'+str(crp_prediction_time)+'\n')

    test(result)

