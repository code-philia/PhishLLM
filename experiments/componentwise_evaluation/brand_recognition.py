import os
import openai
from scripts.data.data_utils import ShotDataset_Caption
from scripts.pipeline.test_llm import TestLLM
import yaml
from scripts.utils.PhishIntentionWrapper import PhishIntentionWrapper
from tldextract import tldextract
from tqdm import tqdm
import numpy as np
from scripts.utils.web_utils.web_utils import is_valid_domain


def list_correct(result_file):
    correct = []
    result_lines = open(result_file).readlines()
    for line in result_lines:
        data = line.strip().split('\t')
        url, gt, pred, time = data
        if is_valid_domain(pred):  # has prediction
            correct.append(url)
    return correct


def test(result_file):
    ct = 0
    total = 0
    runtime = []

    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        url, gt, pred, time = data
        total += 1
        runtime.append(float(time))

        if is_valid_domain(pred):
            ct += 1
        else:
            print(url, gt, pred)

    print(f"Completeness (% brand recognized) = {ct / total} "
          f"Median runtime {np.median(runtime)}, Mean runtime {np.mean(runtime)}, "
          f"Min runtime {min(runtime)}, Max runtime {max(runtime)}, "
          f"Std runtime {np.std(runtime)}")


def test_check_precision(result_file):
    ct_correct = 0
    ct = 0
    total = 0
    runtime = []

    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        url, gt, pred, time = data
        total += 1
        runtime.append(float(time))

        if is_valid_domain(pred):
            ct += 1
            if tldextract.extract(pred).domain in gt:
                ct_correct += 1
        else:
            print(url, gt, pred)

    print(f"Completeness (% brand recognized) = {ct / total} "
          f"Precision = {ct_correct / ct} "
          f"Median runtime {np.median(runtime)}, Mean runtime {np.mean(runtime)}, "
          f"Min runtime {min(runtime)}, Max runtime {max(runtime)}, "
          f"Std runtime {np.std(runtime)}")


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

    dataset = ShotDataset_Caption(annot_path='./datasets/alexa_screenshots_orig.txt')
    result_file = './datasets/alexa_brand_testllm_caption.txt'

    for it in tqdm(range(len(dataset))):

        if os.path.exists(result_file) and dataset.urls[it] in open(result_file).read():
            continue

        url, _, logo_caption, logo_ocr, webpage_text, reference_logo = dataset.__getitem__(it)
        domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
        print('Logo caption: ', logo_caption)
        print('Logo OCR: ', logo_ocr)
        total_time = 0

        if len(logo_caption) or len(logo_ocr):
            predicted_domain, _, brand_llm_pred_time = llm_cls.brand_recognition_llm(reference_logo=reference_logo,
                                                                 webpage_text=webpage_text,
                                                                 logo_caption=logo_caption,
                                                                 logo_ocr=logo_ocr,
                                                                 announcer=None)
            total_time += brand_llm_pred_time

            if predicted_domain and len(predicted_domain) > 0 and is_valid_domain(predicted_domain):
                validation_success, logo_cropping_time, logo_matching_time = llm_cls.brand_validation(
                                                                                    company_domain=predicted_domain,
                                                                                    reference_logo=reference_logo)

                total_time += logo_matching_time

                if not validation_success:
                    answer = 'failure in logo matching'
            else:
                answer = 'no prediction'

        else:
            answer = 'no text in logo'

        with open(result_file, 'a+') as f:
            f.write(url + '\t' + domain + '\t' + answer + '\t' + str(total_time) + '\n')

    test(result_file)
    test_check_precision(result_file)
