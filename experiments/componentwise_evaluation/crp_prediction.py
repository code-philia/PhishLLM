import os
import openai
from tqdm import tqdm
from scripts.data.data_utils import ShotDataset
import yaml
from scripts.pipeline.test_llm import TestLLM
from scripts.utils.PhishIntentionWrapper import PhishIntentionWrapper
import numpy as np

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

    dataset = ShotDataset(annot_path='./datasets/alexa_screenshots.txt')
    print(len(dataset))
    result_file = './datasets/alexa_shot_testllm2.txt'

    for it in tqdm(range(len(dataset))):

        if os.path.exists(result_file) and dataset.urls[it] in open(result_file).read():
            continue
        url, gt, webpage_text = dataset.__getitem__(it, True)

        crp_cls, crp_prediction_time = llm_cls.crp_prediction_llm(html_text=webpage_text,
                                                                  announcer=None)

        with open(result_file, 'a+') as f:
            f.write(url+'\t'+gt+'\t'+str(crp_cls)+'\t'+str(crp_prediction_time)+'\n')

    false_positives, false_negatives = test(result_file)
