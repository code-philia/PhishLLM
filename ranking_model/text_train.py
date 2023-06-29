from torch import nn, optim
from ranking_model.dataloader import *
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def trainer(EPOCH, model, evaluator, train_dataloader):
    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * EPOCH * 0.1)  # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_dataloader=train_dataloader,
              epochs=EPOCH,
              warmup_steps=warmup_steps,
              evaluator=evaluator,
              evaluation_steps=5000,
              output_path='./checkpoints/text_model')


if __name__ == '__main__':


    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)

    # Define our Cross-Encoder
    BATCH_SIZE = 32
    EPOCH = 10

    # We use distilroberta-base as base model and set num_labels=1, which predicts a continous score between 0 and 1
    model = CrossEncoder('distilroberta-base', num_labels=1)

    train_samples = []
    labels = []
    for line in tqdm(open('./datasets/alexa_login_train.txt').readlines()[::-1]): # fixme
        url, dom, save_path, label = line.strip().split('\t')
        html_path = './{}/{}/index.html'.format('./datasets/alexa_login', url.split('https://')[1])
        if not os.path.exists(html_path):
            continue
        with io.open(html_path, 'r', encoding='utf-8') as f:
            page = f.read()
        if len(page) == 0:
            continue
        tree_repr = format_input_generation(html_path, dom)
        if len(tree_repr) == 0:
            continue
        train_samples.append(InputExample(texts=[tree_repr, "a login button"],
                                          label=int(label)))
        labels.append(int(label))

    # downsample the 0 class
    print(np.sum(np.asarray(labels) == 1),
          np.sum(np.asarray(labels) == 0))

    minority_length = np.sum(np.asarray(labels) == 1)
    aug_train_samples = []
    aug_labels = []
    for ind in range(len(train_samples)):
        if labels[ind] == 1:
            aug_train_samples.append(train_samples[ind])
        else:
            if np.sum(np.asarray(aug_labels) == 0) < minority_length:
                aug_train_samples.append(train_samples[ind])
            else:
                continue
        aug_labels.append(labels[ind])

    print(len(aug_train_samples))
    print(np.sum(np.asarray(aug_labels) == 1),
          np.sum(np.asarray(aug_labels) == 0))

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(aug_train_samples, batch_size=BATCH_SIZE, shuffle=True)

    # During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(train_samples,
                                                                    name='train')

    trainer(EPOCH, model, evaluator, train_dataloader)
