import shutil
import time
import os
import openai
from models.brand_recognition.dataloader import *
import idna
from models.utils.web_utils import is_valid_domain
from pipeline.test_llm import TestLLM
import yaml
import Levenshtein as lev
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key2.txt').read().strip()
os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"

def test(result_file):

    success = 0
    strict_success = 0
    total = 0
    runtime = []

    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.split('\t')
        brand, URL, answer, failure_reason, brand_recog_time, brand_validation_time  = data
        total += 1

        if is_valid_domain(answer):
            success += 1
            if len(failure_reason) == 0:
                strict_success += 1
            elif lev.ratio(answer, brand) >= 0.8:
                strict_success += 1

        pbar.set_description(f"Completeness (% brand recognized) = {success/total} ", refresh=True)

    print(f"Completeness (% brand recognized) = {success/total} \n "
          f"Completeness after brand validation (% brand recognized) = {strict_success/total}")

if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy


    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    model = "gpt-3.5-turbo-16k"
    result_file = './datasets/alexa_middle_5k.txt'

    # phishintention_cls = PhishIntentionWrapper()
    # llm_cls = TestLLM(phishintention_cls,
    #                   param_dict=param_dict,
    #                   proxies={"http": "http://127.0.0.1:7890",
    #                            "https": "http://127.0.0.1:7890",
    #                            }
    #                   )


    # for brand in tqdm(os.listdir('./datasets/alexa_middle_5k')):
    #     if brand.startswith('.'):
    #         continue
    #
    #     folder = os.path.join('./datasets/alexa_middle_5k', brand)
    #     if os.path.exists(result_file) and brand in [x.strip().split('\t')[0] for x in open(result_file).readlines()]:
    #         continue
    #
    #     # if subdomain != 'mymhs.nmfn.com':
    #     #     continue
    #
    #     shot_path = os.path.join(folder, 'shot.png')
    #     html_path = os.path.join(folder, 'index.html')
    #     info_path = os.path.join(folder, 'info.txt')
    #     if os.path.exists(info_path):
    #         URL = open(info_path, encoding='utf-8').read()
    #     else:
    #         URL = f'http://{brand}'
    #     DOMAIN = brand
    #
    #     if not os.path.exists(shot_path):
    #         shutil.rmtree(folder)
    #         continue
    #
    #     logo_cropping_time, logo_matching_time, google_search_time, brand_recog_time = 0, 0, 0, 0
    #
    #     plotvis = Image.open(shot_path)
    #     logo_box, reference_logo = llm_cls.detect_logo(shot_path)
    #     answer = ''
    #     failure_reason = ''
    #     if reference_logo is None:
    #         failure_reason = 'cannot detect logo'
    #     else:
    #         image_width, image_height = plotvis.size
    #         webpage_text, logo_caption, logo_ocr = llm_cls.preprocessing(shot_path=shot_path,
    #                                                                      html_path=html_path,
    #                                                                       reference_logo=reference_logo,
    #                                                                       logo_box=logo_box,
    #                                                                       image_width=image_width,
    #                                                                       image_height=image_height,
    #                                                                       announcer=None)
    #         if len(logo_caption) == 0 and len(logo_ocr) == 0:
    #             failure_reason = 'no text in logo'
    #         else:
    #             ## Brand recognition model
    #             start_time = time.time()
    #             predicted_domain, _ = llm_cls.brand_recognition_llm(reference_logo=reference_logo,
    #                                                                            webpage_text=webpage_text,
    #                                                                            logo_caption=logo_caption,
    #                                                                            logo_ocr=logo_ocr,
    #                                                                            announcer=None)
    #             brand_recog_time = time.time() - start_time
    #
    #             ## Domain validation
    #             if predicted_domain and len(predicted_domain) > 0 and is_valid_domain(predicted_domain):
    #                 answer = predicted_domain
    #                 validation_success, logo_cropping_time, logo_matching_time = llm_cls.brand_validation(
    #                                                                                     company_domain=predicted_domain,
    #                                                                                     reference_logo=reference_logo)
    #                 if not validation_success:
    #                     failure_reason = 'failure in logo matching'
    #
    #             else:
    #                 failure_reason = 'no prediction'
    #
    #     print(brand, answer, failure_reason)
    #     with open(result_file, 'a+', encoding='utf-8') as f:
    #         f.write(brand + '\t' + URL + '\t' + answer + '\t' + failure_reason + '\t' + str(brand_recog_time) + "\t" + str(logo_cropping_time+logo_matching_time) + '\n')

    test(result_file)
    # Completeness (% brand recognized) = 0.8674089068825911
    # Completeness after brand validation (% brand recognized) = 0.7064777327935222