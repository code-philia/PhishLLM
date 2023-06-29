from torch import nn, optim
from ranking_model.dataloader import *
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import logging
from sklearn.metrics import average_precision_score
from typing import List
import numpy as np
import os
import csv
from sentence_transformers.evaluation import BinaryClassificationEvaluator
logger = logging.getLogger(__name__)

class CEBinaryClassificationEvaluator:
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and binary labels (0 and 1),
    it compute the average precision and the best possible f1 score
    """
    def __init__(self, sentence_pairs: List[List[str]], labels: List[int],
                 name: str='', show_progress_bar: bool = False, write_csv: bool = True):
        assert len(sentence_pairs) == len(labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.name = name

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "CEBinaryClassificationEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy", "Accuracy_Threshold", "F1", "F1_Threshold", "Precision", "Recall", "Average_Precision"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        print("CEBinaryClassificationEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=self.show_progress_bar)

        acc, acc_threshold = BinaryClassificationEvaluator.find_best_acc_and_threshold(pred_scores, self.labels, True)
        f1, precision, recall, f1_threshold = BinaryClassificationEvaluator.find_best_f1_and_threshold(pred_scores, self.labels, True)
        ap = average_precision_score(self.labels, pred_scores)

        print("Accuracy:           {:.2f}\t(Threshold: {:.4f})".format(acc * 100, acc_threshold))
        print("F1:                 {:.2f}\t(Threshold: {:.4f})".format(f1 * 100, f1_threshold))
        print("Precision:          {:.2f}".format(precision * 100))
        print("Recall:             {:.2f}".format(recall * 100))
        print("Average Precision:  {:.2f}\n".format(ap * 100))


@torch.no_grad()
def tester_rank(model, urls, texts, labels):

    model.eval()
    correct = 0
    total = 0

    df = pd.DataFrame({'url': urls,
                       'text':  texts,
                       'label': labels})
    grp = df.groupby('url')
    grp = dict(list(grp), keys=lambda x: x[0])  # {url: List[dom_path, save_path]}

    for url, data in tqdm(grp.items()):
        try:
            text = data.texts
            labels = data.labels
        except:
            continue

        labels = torch.tensor(np.asarray(labels))
        pred_scores = model.predict(text, convert_to_numpy=True, show_progress_bar=True)
        probs = pred_scores # (N, C)
        conf = probs[torch.arange(probs.shape[0]), 1] # take the confidence (N, 1)
        _, ind = torch.topk(conf, min(1, len(conf))) # top1 index

        if (labels == 1).sum().item(): # has login button
            if (labels[ind] == 1).sum().item(): # has login button and it is reported
                correct += 1
            # else:
                # visualize
                # os.makedirs('./datasets/debug', exist_ok=True)
                # f, axarr = plt.subplots(6, 1)
                # for it in range(min(5, len(conf))):
                #     img_path_sorted = np.asarray(img_paths)[ind.cpu()]
                #     axarr[it].imshow(Image.open(img_path_sorted[it]))
                #     axarr[it].set_title(str(conf[ind][it].item()))
                #
                # gt_ind = torch.where(labels == 1)[0]
                # if len(gt_ind) > 1:
                #     gt_ind = gt_ind[0]
                # axarr[5].imshow(Image.open(np.asarray(img_paths)[gt_ind.cpu()]))
                # axarr[5].set_title('ground_truth'+str(conf[gt_ind].item()))
                #
                # plt.savefig(
                #     f"./datasets/debug/{url.split('https://')[1]}.png")
                # plt.close()
            total += 1

    print(correct, total)

if __name__ == '__main__':
    model = CrossEncoder('./checkpoints/text_model')

    # train_samples = []
    # for line in tqdm(open('./datasets/alexa_login_train.txt').readlines()[::-1]): # fixme
    #     url, dom, save_path, label = line.strip().split('\t')
    #     html_path = './{}/{}/index.html'.format('./datasets/alexa_login', url.split('https://')[1])
    #     if not os.path.exists(html_path):
    #         continue
    #     with io.open(html_path, 'r', encoding='utf-8') as f:
    #         page = f.read()
    #     if len(page) == 0:
    #         continue
    #     tree_repr = format_input_generation(html_path, dom)
    #     dom_repr = dom + ' ' + tree_repr
    #     train_samples.append(InputExample(texts=[dom_repr, "a login button"],
    #                                     label=int(label)))
    #
    test_samples = []
    urls = []
    texts = []
    labels = []
    for line in tqdm(open('./datasets/alexa_login_test.txt').readlines()[::-1]): # fixme
        url, dom, save_path, label = line.strip().split('\t')
        html_path = './{}/{}/index.html'.format('./datasets/alexa_login', url.split('https://')[1])
        if not os.path.exists(html_path):
            continue
        with io.open(html_path, 'r', encoding='utf-8') as f:
            page = f.read()
        if len(page) == 0:
            continue
        tree_repr = format_input_generation(html_path, dom)
        dom_repr = dom + ' ' + tree_repr
        test_samples.append(InputExample(texts=[dom_repr, "a login button"],
                                         label=int(label)))
        urls.append(url)
        texts.append(dom_repr + "; a login button")
        labels.append(int(label))

    evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name='test')
    evaluator(model) # Precision:          61.77, Recall:             49.89