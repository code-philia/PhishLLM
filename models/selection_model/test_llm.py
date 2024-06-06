import time
import openai
import backoff
from transformers import GPT2TokenizerFast
import os
import openai
from models.selection_model.dataloader import *
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

def test(result_file):
    correct = 0
    total = 0
    pred_pos = 0
    true_pos = 0
    pred_pos_and_true_pos = 0
    runtime = []
    false_positives = []
    false_negatives = []

    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        url, gt, pred, time = data

        pred_pos += float(' A.' in pred)
        true_pos += float(gt == 'A')
        pred_pos_and_true_pos += float(' A.' in pred) * float(gt == 'A')
        if (gt == 'A') and (' A.' in pred):
            correct += 1
        elif (gt == 'B') and (' A.' not in pred): # if the gt is non-crp, then no prediction is also correct prediction
            correct += 1
        elif gt == 'A':
            false_negatives.append(url)
        else:
            false_positives.append(url)

        total += 1
        runtime.append(float(time))


        # pbar.set_description(f"test classification acc: {correct / total}, "
        #                      f"test precision: {pred_pos_and_true_pos / (pred_pos + 1e-8)} "
        #                      f"test recall: {pred_pos_and_true_pos / (true_pos + 1e-8)} ", refresh=True)

    print(f"test classification acc: {correct / total}, "
          f"test precision: {pred_pos_and_true_pos / (pred_pos + 1e-8)} "
          f"test recall: {pred_pos_and_true_pos / (true_pos + 1e-8)} "
          f"Median runtime {np.median(runtime)}, Mean runtime {np.mean(runtime)} "
          f"Min runtime {min(runtime)}, Max runtime {max(runtime)}, "
          f"Std runtime {np.std(runtime)}"
          )

    return false_positives, false_negatives


if __name__ == '__main__':

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy


    dataset = ShotDataset(annot_path='./datasets/alexa_screenshots.txt')
    print(len(dataset))
    model = "gpt-3.5-turbo-16k"
    # result_file = './datasets/alexa_shot_testllm2.txt'
    result_file = './datasets/alexa_shot_testllm_wo_cot.txt'

    # for it in tqdm(range(len(dataset))):
    #     # if it <= 1000:
    #     #     continue
    #     start_time = time.time()
    #
    #     # if os.path.exists(result_file) and dataset.urls[it] in open(result_file).read():
    #     #     continue
    #     if 'douban.com' not in dataset.urls[it]:
    #         continue
    #     url, gt, html_text = dataset.__getitem__(it, True)
    #     question = question_template(html_text)
    #
    #     with open('./selection_model/prompt.json', 'rb') as f:
    #         prompt = json.load(f)
    #     # with open('./selection_model/prompt_wo_cot.json', 'rb') as f:
    #     #     prompt = json.load(f)
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
    #                 max_tokens=100,  # we're only counting input tokens here, so let's not waste tokens on the output
    #             )
    #             inference_done = True
    #         except Exception as e:
    #             print(f"Error was: {e}")
    #             new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
    #             time.sleep(10)
    #     total_time = time.time() - start_time
    #
    #     answer = ''.join([choice["message"]["content"] for choice in response['choices']])
    #
    #     print(answer)
    #     # with open(result_file, 'a+') as f:
    #     #     f.write(url+'\t'+gt+'\t'+answer+'\t'+str(total_time)+'\n')

    false_positives, false_negatives = test(result_file)
