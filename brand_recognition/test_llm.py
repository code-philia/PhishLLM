import time
import os
import openai
from brand_recognition.dataloader import *
import idna
from model_chain.web_utils import is_valid_domain, query2image, get_images, url2logo
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

        pbar.set_description(f"Completeness (% brand recognized) = {ct/total} ", refresh=True)

    print(f"Completeness (% brand recognized) = {ct/total} "
          f"Median runtime {np.median(runtime)}, Mean runtime {np.mean(runtime)}")


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
    # result_file = './datasets/alexa_brand_testllm_caption.txt'
    # result_file = './datasets/alexa_brand_testllm_caption_caponly.txt'
    result_file = './datasets/alexa_brand_testllm_caption_ocronly.txt'
    phishintention_cls = PhishIntentionWrapper()

    for it in tqdm(range(len(dataset))):

        if os.path.exists(result_file) and dataset.urls[it] in open(result_file).read():
            continue
        # if not any([x in dataset.urls[it] for x in ['https://aliyundrive.com']]):
        #     continue
        url, _, logo_caption, logo_ocr, html_text, reference_logo = dataset.__getitem__(it)
        print('Logo caption: ', logo_caption)
        print('Logo OCR: ', logo_ocr)
        domain = tldextract.extract(url).domain+'.'+tldextract.extract(url).suffix

        if len(logo_caption) or len(logo_ocr):
            industry = ask_industry(model, html_text)

            # question = question_template_caption_industry(logo_caption, logo_ocr, industry)
            # ablation study
            question = question_template_caption('', logo_ocr)
            # question = question_template_caption(logo_caption, '')

            with open('./brand_recognition/prompt.json', 'rb') as f:
                prompt = json.load(f)
            new_prompt = prompt
            new_prompt.append(question)

            # example token count from the OpenAI API
            inference_done = False
            while not inference_done:
                try:
                    start_time = time.time()
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=new_prompt,
                        temperature=0,
                        max_tokens=50,  # we're only counting input tokens here, so let's not waste tokens on the output
                    )
                    inference_done = True
                except Exception as e:
                    print(f"Error was: {e}")
                    time.sleep(10)
            total_time = time.time() - start_time

            answer = ''.join([choice["message"]["content"] for choice in response['choices']])
            print(answer)
            if len(answer) > 0 and is_valid_domain(answer):

                validation_success = False
                start_time = time.time()
                API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]
                returned_urls = query2image(query=answer + ' logo',
                                            SEARCH_ENGINE_ID=SEARCH_ENGINE_ID, SEARCH_ENGINE_API=API_KEY,
                                            num=5, proxies=proxies
                                               )
                logos = get_images(returned_urls, proxies=proxies)
                d_logo = url2logo(f'https://{answer}', phishintention_cls)
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

                if not validation_success:
                    answer = 'failure in logo matching'
        else:
            answer = 'no prediction'

        with open(result_file, 'a+') as f:
            f.write(url+'\t'+domain+'\t'+answer+'\t'+str(total_time)+'\n')

    # (.*failure.*\n)|(.*unable.*\n)|(.*not.*\n)|(.*no prediction.*\n)

    test(result_file)
    # Completeness (% brand recognized) = 0.6548672566371682 Median runtime 0.8912811279296875, Mean runtime 1.384584855650906
    # Caption only Completeness (% brand recognized) = 0.3827433628318584 Median runtime 1.0176382064819336, Mean runtime 1.5962941735587288
    # OCR only Completeness (% brand recognized) = 0.5176991150442478 Median runtime 1.101432204246521, Mean runtime 1.804533622433654


    '''Dont have enough information'''
    # liepin.com Logo caption:  an orange and white sign with asian writing
    # Logo OCR:  猫猎聘 ●全国、
    # Industry:  Real Estate
    # There is not enough information provided to determine the brand's domain.

    '''fail logo validation'''
    # qoo10.jp
    # Logo caption:  a qqqq logo with the word qqqqq on it
    # Logo OCR:  Coo10 COMING n eBay compan!
    # Industry:  e-commerce
    # predict as ebay.com, fail logo validation

    # Logo caption:  a blue and white sign with chinese characters on it
    # Logo OCR:  阿里云盘
    # Industry:  Technology
    # aliyun.com