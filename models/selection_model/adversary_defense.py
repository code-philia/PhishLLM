import time
import openai
from models.selection_model.dataloader import *
import pandas as pd
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key3.txt').read()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def test(result_file):
    ct = 0
    total = 0
    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        hash, gt, pred, time = data
        if 'A.' in pred:
            ct += 1
        total += 1

        pbar.set_description( f"After attack with defense recall: {ct / total} ", refresh=True)

    print(f"After attack with defense recall: {ct / total} ")


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy

    model = "gpt-3.5-turbo-16k"
    root_folder = './datasets/dynapd'
    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_wo_validation.txt').readlines()]
    df = pd.read_csv('./datasets/Brand_Labelled_130323.csv')
    crp_list = [x.strip() for x in open('./datasets/dynapd_crp_list.txt').readlines()]

    result = './datasets/dynapd_llm_adv_selection_defense.txt'

    # for hash in tqdm(all_folders):
    #     target_folder = os.path.join(root_folder, hash)
    #     if os.path.exists(result) and hash in open(result).read():
    #         continue
    #     if hash not in crp_list:
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
    #         html_text = get_ocr_text(shot_path, html_path)
    #         if len(html_text):
    #             question = question_template_adversary(html_text)
    #
    #             with open('./selection_model/prompt.json', 'rb') as f:
    #                 prompt = json.load(f)
    #             new_prompt = prompt
    #             new_prompt.append(question)
    #
    #             # original answer
    #             inference_done = False
    #             start_time = time.time()
    #             while not inference_done:
    #                 try:
    #                     response = openai.ChatCompletion.create(
    #                         model=model,
    #                         messages=new_prompt,
    #                         temperature=0,
    #                         max_tokens=100,  # we're only counting input tokens here, so let's not waste tokens on the output
    #                     )
    #                     inference_done = True
    #                 except Exception as e:
    #                     print(f"Error was: {e}")
    #                     # new_prompt = new_prompt[:65540]
    #                     new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
    #                     time.sleep(10)
    #             total_time = time.time() - start_time
    #             answer = ''.join([choice["message"]["content"] for choice in response['choices']])
    #         else:
    #             answer = ''
    #         print(answer)
    #
    #         with open(result, 'a+') as f:
    #             f.write(hash+'\t'+'A'+'\t'+answer+'\t'+str(total_time)+'\n')

    test(result)

