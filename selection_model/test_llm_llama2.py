import time
from llama import Llama, Dialog
from selection_model.dataloader import *
import contextlib
import json
from tqdm import tqdm
import tldextract
import time
from concurrent.futures import ThreadPoolExecutor

@contextlib.contextmanager
def set_env_vars(vars):
    original_vars = {var: os.environ.get(var) for var in vars}
    os.environ.update(vars)
    yield
    for var, value in original_vars.items():
        if value is None:
            del os.environ[var]
        else:
            os.environ[var] = value

def initialize_llama2():
    with set_env_vars({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12356', 'RANK': '0', 'WORLD_SIZE': '1'}):
        return Llama.build(
            ckpt_dir='../llama/llama-2-7b-chat',
            tokenizer_path='../llama/tokenizer.model',
            max_seq_len=2048,
            max_batch_size=8,
        )


def parse_data(result_file):
    groups = []
    file = open(result_file).readlines()
    current_record = []
    for line in file:
        line = line.strip()
        if line.startswith('http') and current_record:
            full_record = ' '.join(current_record).split('\t')
            if len(full_record) == 4:
                group = {
                    'URL': full_record[0],
                    'Ground Truth Class': full_record[1],
                    'Predicted Class': full_record[2],
                    'Runtime': full_record[3]
                }
                groups.append(group)
            current_record = [line]
        else:
            current_record.append(line)

    if current_record:
        full_record = ' '.join(current_record).split('\t')
        if len(full_record) == 4:
            group = {
                'URL': full_record[0],
                'Ground Truth Class': full_record[1],
                'Predicted Class': full_record[2],
                'Runtime': full_record[3]
            }
            groups.append(group)

    return groups

def test(parsed_data):
    correct = 0
    total = 0
    pred_pos = 0
    true_pos = 0
    pred_pos_and_true_pos = 0
    runtime = []

    for data in tqdm(parsed_data, leave=False):
        gt = data['Ground Truth Class']
        pred = data['Predicted Class'].lower()
        time = data['Runtime']

        pred_as_pos = float('not a credential-requiring' not in pred)*float(' b.' not in pred)*float('(b)' not in pred)
        pred_pos += pred_as_pos
        true_pos += float(gt == 'A')
        pred_pos_and_true_pos += pred_as_pos * float(gt == 'A')

        if (gt == 'A') and pred_as_pos:
            correct += 1
        elif (gt == 'B') and (not pred_as_pos):  # if the gt is non-crp, then no prediction is also correct prediction
            correct += 1
        total += 1
        runtime.append(float(time))

    print(f"Test Classification Accuracy: {correct / total}, "
          f"Precision: {pred_pos_and_true_pos / (pred_pos + 1e-8)} "
          f"Recall: {pred_pos_and_true_pos / (true_pos + 1e-8)} "
          f"Median Runtime: {np.median(runtime)}, Mean Runtime: {np.mean(runtime)} "
          f"Min runtime {min(runtime)}, Max runtime {max(runtime)}, "
          f"Std runtime {np.std(runtime)}"
          )


if __name__ == '__main__':

    dataset = ShotDataset(annot_path='./datasets/alexa_screenshots.txt')
    result_file = './datasets/alexa_shot_llama2.txt'

    # llama2_model = initialize_llama2()
    # for it in tqdm(range(len(dataset))):
    #
    #     if os.path.exists(result_file) and dataset.urls[it] in open(result_file).read():
    #         continue
    #     url, gt, html_text = dataset.__getitem__(it, True)
    #     question = question_template(html_text)
    #
    #     with open('./selection_model/prompt.json', 'rb') as f:
    #         prompt = json.load(f)
    #     new_prompt = prompt
    #     new_prompt.append(question)
    #
    #     # example token count from the OpenAI API
    #     inference_done = False
    #     while not inference_done:
    #         try:
    #             start_time = time.time()
    #             results = llama2_model.chat_completion(
    #                 [new_prompt],  # type: ignore
    #                 max_gen_len=None,
    #                 temperature=0,
    #                 top_p=0.9,
    #             )
    #             inference_done = True
    #         except Exception as e:
    #             print(f"Error was: {e}")
    #             new_prompt[-1]['content'] = new_prompt[-1]['content'][:int(2*len(new_prompt[-1]['content'])/3)]
    #     total_time = time.time() - start_time
    #
    #     answer = results[0]['generation']['content'].strip().lower()
    #
    #     print(answer)
    #     with open(result_file, 'a+') as f:
    #         f.write(url+'\t'+gt+'\t'+answer+'\t'+str(total_time)+'\n')

    parsed_groups = parse_data(result_file)
    test(parsed_groups)
    # LLAMA2 Test Classification Accuracy: 0.803026880505986, Precision: 0.5989263803635052 Recall: 0.6911504424717598 Median Runtime: 2.229814052581787, Mean Runtime: 2.765490301150298 Min runtime 0.7339944839477539, Max runtime 28.440632104873657, Std runtime 1.8549931390627965
    # LLAMA2 has restricted token limit of 2048