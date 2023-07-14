
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
from xdriver.XDriver import XDriver
import time
import os
import pandas as pd
from tqdm import tqdm
class TestBaseline():

    def __init__(self, phishintention_cls):

        self.phishintention_cls = phishintention_cls

    def test_phishpedia(self, URL, screenshot_path):
        start_time = time.time()
        phish_category, phish_target, plotvis, siamese_conf, time_breakdown, pred_boxes, pred_classes = \
            self.phishintention_cls.test_orig_phishpedia(URL, screenshot_path)
        phishpedia_runtime = time.time() - start_time

        return phish_category, phish_target, str(phishpedia_runtime)

    def test_phishintention(self, URL, screenshot_path):

        XDriver.set_headless()
        ph_driver = XDriver.boot(chrome=True)
        time.sleep(5)
        ph_driver.set_page_load_timeout(30)
        ph_driver.set_script_timeout(60)
        start_time = time.time()
        phish_category, phish_target, plotvis, siamese_conf, dynamic, time_breakdown, pred_boxes, pred_classes = \
            self.phishintention_cls.test_orig_phishintention(URL, screenshot_path, ph_driver)
        phishintention_runtime = time.time() - start_time
        ph_driver.quit()

        return phish_category, phish_target, str(phishintention_runtime)


if __name__ == '__main__':

    phishintention_cls = PhishIntentionWrapper()
    base_cls = TestBaseline(phishintention_cls)

    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_wo_validation.txt').readlines()]
    df = pd.read_csv('./datasets/Brand_Labelled_130323.csv')
    root_folder = './datasets/dynapd'
    # result = './datasets/dynapd_phishpedia.txt'
    result = './datasets/dynapd_phishintention.txt'

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
            # pred, brand, runtime = base_cls.test_phishpedia(URL, shot_path)
            pred, brand, runtime = base_cls.test_phishintention(URL, shot_path)

            with open(result, 'a+') as f:
                f.write(hash + '\t' + str(pred) + '\t' + str(brand) + '\t' + str(runtime) + '\n')
    print(len(all_folders))


