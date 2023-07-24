import time
import openai
from brand_recognition.dataloader import *
import pandas as pd
import idna
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key3.txt').read()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def test(result_file):

    ct = 0
    total = 0

    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        if len(data) == 3:
            url, pred, time = data
        elif len(data) == 2:
            url, pred = data
        else:
            print(data)
            raise
        total += 1

        if len(pred) < 30 and len(pred) > 0 and ('N/A' not in pred) and ('abc.com' not in pred) and ('No Prediction' not in pred): # has prediction
            ct += 1

        pbar.set_description(f"Recall (% brand recognized) = {ct/total} ", refresh=True)

    print(f"Recall, i.e. % brand recognized = {ct/total} ")


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy

    model = "gpt-3.5-turbo-16k"
    root_folder = './datasets/dynapd'
    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_wo_validation.txt').readlines()]
    df = pd.read_csv('./datasets/Brand_Labelled_130323.csv')

    # result = './datasets/dynapd_llm_adv_brand.txt'
    # result = './datasets/dynapd_llm_brand.txt'
    result = './datasets/dynapd_llm_adv_brand_defense.txt'

    # for hash in tqdm(all_folders):
    #     target_folder = os.path.join(root_folder, hash)
    #     if os.path.exists(result) and hash in open(result).read():
    #         continue
    #
    #     shot_path = os.path.join(target_folder, 'shot.png')
    #     html_path = os.path.join(target_folder, 'index.html')
    #     if os.path.exists(shot_path):
    #         pk_info = df.loc[df['HASH'] == hash]
    #         try:
    #             URL = list(pk_info['URL'])[0]
    #         except IndexError:
    #             URL = f'http://127.0.0.5/{hash}'
    #
    #         html_text = get_ocr_text(shot_path, html_path)
    #         if len(html_text):
    #             question = question_template_adversary(html_text, 'abc.com')
    #             # question = question_template(html_text)
    #
    #             with open('./brand_recognition/prompt.json', 'rb') as f:
    #                 prompt = json.load(f)
    #             new_prompt = prompt
    #             new_prompt.append(question)
    #
    #             # example token count from the OpenAI API
    #             start_time = time.time()
    #             inference_done = False
    #             while not inference_done:
    #                 try:
    #                     response = openai.ChatCompletion.create(
    #                         model=model,
    #                         messages=new_prompt,
    #                         temperature=0,
    #                         max_tokens=50,  # we're only counting input tokens here, so let's not waste tokens on the output
    #                     )
    #                     inference_done = True
    #                 except Exception as e:
    #                     print(f"Error was: {e}")
    #                     new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
    #                     time.sleep(43.2)
    #             total_time = time.time() - start_time
    #
    #             answer = ''.join([choice["message"]["content"] for choice in response['choices']])
    #         else:
    #             answer = ''
    #         print(answer)
    #
    #         with open(result, 'a+') as f:
    #             f.write(hash+'\t'+answer+'\t'+str(total_time)+'\n')

    test(result) # before attack: 0.8280707293969056, after attack: 0.14761604041679824, after attack with defense: 0.7665246884366619


