
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
from xdriver.XDriver import XDriver
import time
import os
import pandas as pd
from tqdm import tqdm
from model_chain.test_baseline import TestBaseline
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':

    phishintention_cls = PhishIntentionWrapper()
    base_cls = TestBaseline(phishintention_cls)

    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_llm_adv.txt').readlines()]
    df = pd.read_csv('./datasets/Brand_Labelled_130323.csv')
    root_folder = './datasets/dynapd_adv'
    # result = './datasets/dynapd_phishpedia_adv.txt'
    result = './datasets/dynapd_phishintention_adv.txt'


    for hash in tqdm(all_folders):
        target_folder = os.path.join(root_folder, hash)
        if os.path.exists(result) and hash in open(result).read():
            continue

        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        if os.path.exists(shot_path):
            pk_info = df.loc[df['HASH'] == hash]
            try:
                URL = list(pk_info['URL'])[0]
            except IndexError:
                URL = f'http://127.0.0.5/{hash}'
            try:
                # pred, brand, runtime = base_cls.test_phishpedia(URL, shot_path)
                pred, brand, runtime = base_cls.test_phishintention(URL, shot_path, obfuscate=True)
            except Exception as e:
                print(e)
                continue

            with open(result, 'a+') as f:
                f.write(hash + '\t' + str(pred) + '\t' + str(brand) + '\t' + str(runtime) + '\n')
    print(len(all_folders))


