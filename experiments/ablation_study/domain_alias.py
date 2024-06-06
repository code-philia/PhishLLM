import shutil
import os
import openai
from scripts.brand_recognition.dataloader import *
from scripts.utils.web_utils.web_utils import is_valid_domain
from scripts.pipeline.test_llm import TestLLM
import yaml
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()
os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"
# def list_correct(result_file):
#     correct = []
#     result_lines = open(result_file).readlines()
#     for line in result_lines:
#         data = line.strip().split('\t')
#         url, gt, pred, time = data
#         if is_valid_domain(pred):  # has prediction
#             correct.append(url)
#     return correct
#
# def test(result_file):
#
#     ct = 0
#     total = 0
#     runtime = []
#
#     result_lines = open(result_file).readlines()
#     pbar = tqdm(result_lines, leave=False)
#     for line in pbar:
#         data = line.strip().split('\t')
#         url, gt, pred, time = data
#         total += 1
#         runtime.append(float(time))
#
#         if is_valid_domain(pred):
#             ct += 1
#         else:
#             print(url, gt, pred)
#
#         # pbar.set_description(f"Completeness (% brand recognized) = {ct/total} ", refresh=True)
#
#     print(f"Completeness (% brand recognized) = {ct/total} "
#           f"Median runtime {np.median(runtime)}, Mean runtime {np.mean(runtime)}, "
#           f"Min runtime {min(runtime)}, Max runtime {max(runtime)}, "
#           f"Std runtime {np.std(runtime)}")
#


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy


    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    model = "gpt-3.5-turbo-16k"
    result_file = './datasets/domain_alias100.txt'

    for brand in os.listdir('./datasets/domain_alias_100'):
        if brand.startswith('.'):
            continue

        ct = 0
        for subdomain in os.listdir(os.path.join('./datasets/domain_alias_100', brand)):
            if subdomain.startswith('.'):
                continue
            subdomain_folder = os.path.join('./datasets/domain_alias_100', brand, subdomain)
            shot_path = os.path.join(subdomain_folder, 'shot.png')
            if os.path.exists(shot_path):
                ct += 1
        if ct < 2:
            shutil.rmtree(os.path.join('./datasets/domain_alias_100', brand))


    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls,
                      param_dict=param_dict,
                      proxies={"http": "http://127.0.0.1:7890",
                               "https": "http://127.0.0.1:7890",
                               }
                      )

    for brand in os.listdir('./datasets/domain_alias_100'):
        if brand.startswith('.'):
            continue
        for subdomain in os.listdir(os.path.join('./datasets/domain_alias_100', brand)):
            if subdomain.startswith('.'):
                continue
            subdomain_folder = os.path.join('./datasets/domain_alias_100', brand, subdomain)
            if os.path.exists(result_file) and subdomain in [x.strip().split('\t')[0] for x in open(result_file).readlines()]:
                continue

            # if subdomain != 'mymhs.nmfn.com':
            #     continue

            shot_path = os.path.join(subdomain_folder, 'shot.png')
            html_path = os.path.join(subdomain_folder, 'index.html')
            info_path = os.path.join(subdomain_folder, 'info.txt')
            if os.path.exists(info_path):
                URL = open(info_path, encoding='utf-8').read()
            else:
                URL = f'http://{subdomain}'
            DOMAIN = '.'.join(part for part in tldextract.extract(URL) if part)

            if not os.path.exists(shot_path):
                shutil.rmtree(subdomain_folder)
                continue

            logo_cropping_time, logo_matching_time, google_search_time, brand_recog_time = 0, 0, 0, 0
            popularity_validation_success = False

            plotvis = Image.open(shot_path)
            logo_box, reference_logo = llm_cls.detect_logo(shot_path)
            if reference_logo is None:
                answer = 'cannot detect logo'
            else:
                image_width, image_height = plotvis.size
                (webpage_text, logo_caption, logo_ocr), (ocr_processing_time, image_caption_processing_time) = llm_cls.preprocessing(shot_path=shot_path,
                                                                             html_path=html_path,
                                                                              reference_logo=reference_logo,
                                                                              logo_box=logo_box,
                                                                              image_width=image_width,
                                                                              image_height=image_height,
                                                                              announcer=None)
                if len(logo_caption) == 0 and len(logo_ocr) == 0:
                    answer = 'no text in logo'
                else:
                    ## Brand recognition model
                    predicted_domain, _, brand_recog_time = llm_cls.brand_recognition_llm(reference_logo=reference_logo,
                                                                                   webpage_text=webpage_text,
                                                                                   logo_caption=logo_caption,
                                                                                   logo_ocr=logo_ocr,
                                                                                   announcer=None)

                    ## Domain validation
                    if predicted_domain and len(predicted_domain) > 0 and is_valid_domain(predicted_domain):
                        validation_success, logo_cropping_time, logo_matching_time = llm_cls.brand_validation(
                                                                                            company_domain=predicted_domain,
                                                                                            reference_logo=reference_logo)
                        if not validation_success:
                            answer = 'failure in logo matching'
                        else:
                            answer = predicted_domain
                    else:
                        answer = 'no prediction'

                    ## Popularity validation
                    if is_valid_domain(answer):
                        popularity_validation_success, google_search_time = llm_cls.popularity_validation(company_domain=DOMAIN)

            print(subdomain, answer)
            with open(result_file, 'a+', encoding='utf-8') as f:
                f.write(subdomain + '\t' + URL + '\t' + answer + '\t' + str(popularity_validation_success) + '\t' + str(brand_recog_time) + "\t" + str(logo_cropping_time+logo_matching_time) + "\t" + str(google_search_time) + '\n')


    total = 0
    brand_recog_ct = 0
    fp_ct = 0
    for line in open(result_file).readlines():
        subdomain, URL, answer, popularity_validation_success, brand_recog_time, domain_validation_time, popularity_validation_time = line.split('\t')
        if is_valid_domain(answer):
            brand_recog_ct += 1
        if is_valid_domain(answer) and (not eval(popularity_validation_success)):
            print(line)
            fp_ct += 1
        total += 1

    print(brand_recog_ct, brand_recog_ct/total)
    print(fp_ct, fp_ct/total)
