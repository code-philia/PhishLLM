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
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()



if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    #
    dataset = ShotDataset(annot_path='./datasets/alexa_screenshots_orig.txt')
    print(len(dataset))
    model = "gpt-3.5-turbo-16k"
    result_file = './datasets/alexa_brand_testllm_u.txt'
    #
    for it in tqdm(range(len(dataset))):
        start_time = time.time()
    #
        if os.path.exists(result_file) and dataset.urls[it] in open(result_file).read():
            continue
        url, _, html_text = dataset.__getitem__(it, True)
        domain = tldextract.extract(url).domain+'.'+tldextract.extract(url).suffix
        question = question_template(html_text)

        with open('./brand_recognition/prompt.json', 'rb') as f:
            prompt = json.load(f)
        new_prompt = prompt
        new_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=new_prompt,
                    temperature=0,
                    max_tokens=50,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                inference_done = True
            except Exception as e:
                print(f"Error was: {e}")
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
                time.sleep(10)
        total_time = time.time() - start_time

        answer = ''.join([choice["message"]["content"] for choice in response['choices']])
        print(answer)
        with open(result_file, 'a+') as f:
            f.write(url+'\t'+domain+'\t'+answer+'\t'+str(total_time)+'\n')

    # correct_lines = []
    # ct = 0
    # result_file_v2 = './datasets/alexa_brand_testllm_u.txt'
    # result_lines = open(result_file).readlines()
    # pbar = tqdm(result_lines, leave=False)
    # for line in pbar:
    #     data = line.strip().split('\t')
    #     url, gt, pred, time = data
    #     if gt in pred:
    #         ct += 1
    #         with open(result_file_v2, 'a+') as f:
    #             f.write(line)
    #
    # print(ct)
