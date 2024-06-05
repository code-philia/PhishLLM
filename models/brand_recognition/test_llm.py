import time
import os
import openai
from models.brand_recognition.dataloader import *
import idna
from models.utils.web_utils import is_valid_domain, query2image, get_images, url2logo, query2url
from phishintention.src.OCR_aided_siamese import pred_siamese_OCR
from concurrent.futures import ThreadPoolExecutor
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key3.txt').read()

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

    print(f"Completeness (% brand recognized) = {ct/total} "
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

    print(f"Completeness (% brand recognized) = {ct/total} "
          f"Precision = {ct_correct/ct} "
          f"Median runtime {np.median(runtime)}, Mean runtime {np.mean(runtime)}, "
          f"Min runtime {min(runtime)}, Max runtime {max(runtime)}, "
          f"Std runtime {np.std(runtime)}")


def ask_industry(model, html_text):
    industry = ''
    if len(html_text):
        prompt = question_template_industry(html_text)
        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    temperature=0,
                    max_tokens=50,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                inference_done = True
            except Exception as e:
                print(f"Error was: {e}")
                prompt[-1]['content'] = prompt[-1]['content'][:len(prompt[-1]['content']) // 2]
                time.sleep(10)
        industry = ''.join([choice["message"]["content"] for choice in response['choices']])
        print('Industry: ', industry)
        if len(industry) > 30:
            industry = ''
    return industry

if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }

    dataset = ShotDataset_Caption(annot_path='./datasets/alexa_screenshots_orig.txt')
    print(len(dataset))
    model = "gpt-3.5-turbo-16k"
    result_file = './datasets/alexa_brand_testllm_caption.txt'

    phishintention_cls = PhishIntentionWrapper()
    API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]

    for it in tqdm(range(len(dataset))):

        if os.path.exists(result_file) and dataset.urls[it] in open(result_file).read():
            continue
        url, _, logo_caption, logo_ocr, html_text, reference_logo = dataset.__getitem__(it)
        domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
        print('Logo caption: ', logo_caption)
        print('Logo OCR: ', logo_ocr)
        total_time = 0

        if len(logo_caption) or len(logo_ocr):

            start_time = time.time()
            if len(logo_ocr) > 0:
                top1_retrieved_url = query2url(query=logo_ocr,
                                           SEARCH_ENGINE_ID=SEARCH_ENGINE_ID, SEARCH_ENGINE_API=API_KEY,
                                           proxies=proxies)
            else:
                top1_retrieved_url = query2url(query=logo_caption,
                                               SEARCH_ENGINE_ID=SEARCH_ENGINE_ID, SEARCH_ENGINE_API=API_KEY,
                                               proxies=proxies)
            total_time = time.time() - start_time

            if top1_retrieved_url:

                validation_success = False
                start_time = time.time()
                returned_urls = query2image(query=tldextract.extract(top1_retrieved_url).domain + '.' + tldextract.extract(top1_retrieved_url).suffix + ' logo',
                                            SEARCH_ENGINE_ID=SEARCH_ENGINE_ID, SEARCH_ENGINE_API=API_KEY,
                                            num=5, proxies=proxies
                                               )
                logos = get_images(returned_urls, proxies=proxies)
                d_logo = url2logo(top1_retrieved_url, phishintention_cls)
                if d_logo:
                    logos.append(d_logo)
                print('Crop the logo time:', time.time() - start_time)

                if reference_logo and len(logos) > 0:
                    reference_logo_feat = pred_siamese_OCR(img=reference_logo,
                                                           model=phishintention_cls.SIAMESE_MODEL,
                                                           ocr_model=phishintention_cls.OCR_MODEL)
                    start_time = time.time()
                    sim_list = []
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(pred_siamese_OCR, logo,
                                                   phishintention_cls.SIAMESE_MODEL,
                                                   phishintention_cls.OCR_MODEL) for logo in logos]
                        for future in futures:
                            logo_feat = future.result()
                            matched_sim = reference_logo_feat @ logo_feat
                            sim_list.append(matched_sim)
                    if any([x > 0.7 for x in sim_list]):
                        validation_success = True
                        answer = tldextract.extract(top1_retrieved_url).domain + '.' + tldextract.extract(top1_retrieved_url).suffix

                if not validation_success:
                    answer = 'failure in logo matching'

            else:
                answer = 'failure in google search'

        else:
            answer = 'no prediction'

        with open(result_file, 'a+') as f:
            f.write(url+'\t'+domain+'\t'+answer+'\t'+str(total_time)+'\n')

    test(result_file)
    test_check_precision(result_file)
