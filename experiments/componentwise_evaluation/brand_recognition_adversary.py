import pandas as pd
from scripts.utils.web_utils.web_utils import is_valid_domain
import os
import re
from tqdm import tqdm
import openai
import yaml
from scripts.utils.PhishIntentionWrapper import PhishIntentionWrapper
from scripts.pipeline.test_llm import TestLLM
from scripts.data.data_utils import transparent_text_injection
from PIL import Image


def test(result_file):

    orig_ct = 0
    adv_ct = 0
    total = 0

    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        url, orig_pred, adv_pred, time = data
        total += 1

        pattern = re.compile(
            r'^(?!-)'  # Cannot start with a hyphen
            r'(?!.*--)'  # Cannot have two consecutive hyphens
            r'(?!.*\.\.)'  # Cannot have two consecutive periods
            r'(?!.*\s)'  # Cannot contain any spaces
            r'[a-zA-Z0-9-]{1,63}'  # Valid characters are alphanumeric and hyphen
            r'(?:\.[a-zA-Z]{2,})+$'  # Ends with a valid top-level domain
        )
        it_is_a_domain_orig = bool(pattern.fullmatch(orig_pred))
        it_is_a_domain_adv = bool(pattern.fullmatch(adv_pred))

        if it_is_a_domain_orig:
            orig_ct += 1
        if it_is_a_domain_adv:
            adv_ct += 1

        pbar.set_description(f"Original Recall (% brand recognized) = {orig_ct/total} \n"
                             f"After adv Recall (% brand recognized) = {adv_ct/total}", refresh=True)

    print(f"Original Recall (% brand recognized) = {orig_ct/total} \n"
          f"After adv Recall (% brand recognized) = {adv_ct/total}")


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
    result_file = './datasets/dynapd_llm_caption_adv_brand.txt'

    for hash in tqdm(all_folders):
        target_folder = os.path.join(root_folder, hash)
        if os.path.exists(result_file) and hash in open(result_file).read():
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

        # report logo
        logo_box, reference_logo = llm_cls.detect_logo(shot_path)
        plotvis = Image.open(shot_path)
        image_width, image_height = plotvis.size

        orig_answer = 'no prediction'
        answer = 'no prediction'
        total_time = 0

        # has logo
        if (reference_logo is not None):
            # preprocessing
            (webpage_text, logo_caption, logo_ocr), (ocr_processing_time, image_caption_processing_time) = \
                                                                                    llm_cls.preprocessing(shot_path=shot_path,
                                                                                                          html_path=html_path,
                                                                                                          reference_logo=reference_logo,
                                                                                                          logo_box=logo_box,
                                                                                                          image_width=image_width,
                                                                                                          image_height=image_height,
                                                                                                          announcer=None)

            orig_answer, _, total_time = llm_cls.brand_recognition_llm(reference_logo=reference_logo,
                                                                    logo_ocr=logo_ocr,
                                                                    logo_caption=logo_caption,
                                                                    webpage_text=webpage_text,
                                                                       announcer=None)
            print('Original answer: ', orig_answer)

            if is_valid_domain(orig_answer):
                '''perform adversarial attack '''
                print('Adversarial attack')
                injected_logo = transparent_text_injection(reference_logo.convert('RGB'), 'abc.com')

                # get image caption for injected logo
                x1, y1, x2, y2 = logo_box
                plotvis.paste(injected_logo, (int(x1), int(y1)))
                adv_shot_path = shot_path.replace('shot.png', 'shot_adv.png')
                plotvis.save(adv_shot_path)

                # get extra description on the webpage
                (adv_webpage_text, adv_logo_caption, adv_logo_ocr), (_, _) = \
                                                            llm_cls.preprocessing(shot_path=adv_shot_path,
                                                                                  html_path=html_path,
                                                                                  reference_logo=injected_logo,
                                                                                  logo_box=logo_box,
                                                                                  image_width=image_width,
                                                                                  image_height=image_height,
                                                                                  announcer=None)
                answer, _, total_time = llm_cls.brand_recognition_llm(reference_logo=injected_logo,
                                                                    logo_ocr=adv_logo_ocr,
                                                                    logo_caption=adv_logo_caption,
                                                                    webpage_text=adv_webpage_text,
                                                                    announcer=None)
                print('After attack answer: ', answer)

        with open(result_file, 'a+') as f:
            f.write(hash+'\t'+orig_answer+'\t'+answer+'\t'+str(total_time)+'\n')

    test(result_file)