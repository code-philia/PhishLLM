import time
import openai
import backoff
from openai.error import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
)
from transformers import GPT2TokenizerFast
import os
import openai
from brand_recognition.dataloader import *
import idna
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()



def test(result_file):

    ct = 0
    total = 0
    reported = 0
    runtime = []

    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        url, gt, pred, time = data
        total += 1
        runtime.append(float(time))

        if len(pred) < 30 and len(pred) > 0 and ('N/A' not in pred): # has prediction
            reported += 1
            try:
                translated_domain = idna.encode(pred).decode('utf-8')
            except idna.core.InvalidCodepoint:
                translated_domain = pred
            if gt in pred or tldextract.extract(gt).domain in pred:
                ct += 1
            elif gt in translated_domain:
                ct += 1
            elif tldextract.extract(pred).domain in gt:
                ct += 1
            elif 'google' in pred or 'microsoft' in pred or 'home.barclays' in pred \
                    or 'telekom' in pred or 'steamcommunity' in pred or\
                'steampowered' in pred or 'barclays.com' in pred\
                :
                ct += 1
            else:
                print(data)

        pbar.set_description(f"Recall (% brand recognized) = {ct/total} "
                             f"Precision (brand reported correct) = {ct/reported} ", refresh=True)

    print(f"Recall, i.e. % brand recognized = {ct/total} "
          f"Precision, i.e. % brand reported correct = {ct/reported} "
          f"Median runtime {np.median(runtime)}, Mean runtime {np.mean(runtime)}")


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    #
    dataset = ShotDataset(annot_path='./datasets/alexa_screenshots_orig.txt')
    print(len(dataset))
    model = "gpt-3.5-turbo-16k"
    result_file = './datasets/alexa_brand_testllm_u2.txt'

    # for it in tqdm(range(len(dataset))):
    #     start_time = time.time()
    # #
    #     if os.path.exists(result_file) and dataset.urls[it] in open(result_file).read():
    #         continue
    #     url, _, html_text = dataset.__getitem__(it, True)
    #     domain = tldextract.extract(url).domain+'.'+tldextract.extract(url).suffix
    #     question = question_template(html_text)
    #
    #     with open('./brand_recognition/prompt.json', 'rb') as f:
    #         prompt = json.load(f)
    #     new_prompt = prompt
    #     new_prompt.append(question)
    #
    #     # example token count from the OpenAI API
    #     inference_done = False
    #     while not inference_done:
    #         try:
    #             response = openai.ChatCompletion.create(
    #                 model=model,
    #                 messages=new_prompt,
    #                 temperature=0,
    #                 max_tokens=50,  # we're only counting input tokens here, so let's not waste tokens on the output
    #             )
    #             inference_done = True
    #         except Exception as e:
    #             print(f"Error was: {e}")
    #             new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
    #             time.sleep(10)
    #     total_time = time.time() - start_time
    #
    #     answer = ''.join([choice["message"]["content"] for choice in response['choices']])
    #     print(answer)
    #     with open(result_file, 'a+') as f:
    #         f.write(url+'\t'+domain+'\t'+answer+'\t'+str(total_time)+'\n')

    test(result_file) # Recall, i.e. % brand recognized = 0.7283540802213001 Precision, i.e. % brand reported correct = 0.8409453848610667

    # result_lines = open(result_file).readlines()
    # pbar = tqdm(result_lines, leave=False)
    # correct = []
    # for line in pbar:
    #     data = line.strip().split('\t')
    #     url, gt, pred, time = data
    #     if gt in pred:
    #         pass
    #         correct.append(line)
    #     elif tldextract.extract(gt).domain in pred:
    #         pass
    #         correct.append(line)

        # else:
        #     print(line)

    # for line in correct:
    #     with open('./datasets/alexa_brand_testllm_u2.txt', 'a+') as f:
    #         f.write(line)
