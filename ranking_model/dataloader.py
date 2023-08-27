
import os
from PIL import Image
import glob
import os
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torch
import clip
import lxml
from ranking_model.dom_utils import *
import io
from lxml import html
import re
import pandas as pd
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import matplotlib.pyplot as plt
from tqdm import tqdm

def format_input_generation(
    html_path, dom_path, keep_html_brackets=False
):
    with io.open(html_path, 'r', encoding='utf-8') as f:
        page = f.read()
    if len(page) == 0:
        return ''
    dom_tree = html.fromstring(page)
    tree_repr = prune_tree(dom_tree, dom_path)
    if tree_repr is None:
        return ''
    tree_repr = tree_repr.replace('"', " ")
    tree_repr = (
        tree_repr.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
    )
    tree_repr = re.sub(r"<text>(.*?)</text>", r"\1", tree_repr)
    if not keep_html_brackets:
        tree_repr = tree_repr.replace("/>", "$/$>")
        tree_repr = re.sub(r"</(.+?)>", r")", tree_repr)
        tree_repr = re.sub(r"<(.+?)>", r"(\1", tree_repr)
        tree_repr = tree_repr.replace("$/$", ")")

    html_escape_table = [
        ("&quot;", '"'),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&ndash;", "-"),
        ("&rsquo;", "'"),
        ("&lsquo;", "'"),
        ("&ldquo;", '"'),
        ("&rdquo;", '"'),
        ("&#39;", "'"),
        ("&#40;", "("),
        ("&#41;", ")"),
    ]
    for k, v in html_escape_table:
        tree_repr = tree_repr.replace(k, v)
    tree_repr = re.sub(r"\s+", " ", tree_repr).strip()

    return tree_repr

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform():
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class ButtonDataset(Dataset):
    def __init__(self, annot_path, root, preprocess):

        self.img_paths = []
        self.urls = []
        self.dom_paths = []
        self.labels = []  # todo
        self.tree_reprs = []
        self.root = root
        self.preprocess = preprocess

        path_set = set()

        for line in tqdm(open(annot_path).readlines()[::-1]):
            url, dom, save_path, label = line.strip().split('\t')
            if (url, dom) in path_set:
                continue
            else:
                path_set.add((url, dom))
                html_path = '{}/{}/index.html'.format(root, url.split('https://')[1])
                if not os.path.exists(html_path):
                    continue

                self.img_paths.append(save_path)
                self.urls.append(url)
                self.dom_paths.append(dom.lower())
                # self.tree_reprs.append(tree_repr)
                self.labels.append(int(label))

        assert len(self.img_paths) == len(self.urls)
        assert len(self.img_paths) == len(self.dom_paths)
        assert len(self.img_paths) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.preprocess(Image.open(img_path))
        label = self.labels[idx]
        url = self.urls[idx]
        dom_path = self.dom_paths[idx]
        html_path = '{}/{}/index.html'.format(self.root, url.split('https://')[1])
        tree_repr = format_input_generation(html_path, dom_path)

        return image, label, dom_path, tree_repr, img_path, url


# https://github.com/pytorch/pytorch/blob/e5742494f6080c8e6f43c37689fc18a7c4b39dfd/torch/utils/data/dataloader.py#L145
class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size):
        """
            BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
            Return batches of size n_classes * n_samples
        """
        self.labels = labels
        self.labels_set = list(set(np.asarray(self.labels)))
        self.label_to_indices = {label: np.where(np.asarray(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}

        self.count = 0
        self.n_classes = len(self.labels_set)
        self.batch_size = batch_size
        self.n_samples = batch_size // self.n_classes
        self.n_dataset = len(self.labels)
        # self.n_dataset = min([len(x) for x in list(self.label_to_indices.values())]) * self.n_classes # fixme: imbalanced dataset, downsample the majority class to align with minority class

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False) # randomly choose n_classes from all classes
            indices = []
            for class_ in classes:
                start_index = self.used_label_indices_count[class_]
                end_index = start_index + self.n_samples
                indices.extend(self.label_to_indices[class_][start_index:end_index])
                self.used_label_indices_count[class_] += self.n_samples # have been visited before
                # if the next end_index will exceed the length, shuffle and select again
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

if __name__ == '__main__':
    BATCH_SIZE = 128
    # model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


    train_dataset = ButtonDataset(annot_path='./datasets/alexa_login_train.txt',
                                  root='./datasets/alexa_login',
                                  preprocess=_transform())

    # f, axarr = plt.subplots(3, 3)
    # for it in range(9):
    #     image, caption, dom_path, tree_repr, img_path, url = dataset.__getitem__(it+21700)
    #     axarr[it // 3, it % 3].imshow(Image.open(img_path))
    #     axarr[it // 3, it % 3].set_title(os.path.dirname(img_path)+'\n'+dom_path+'\n'+caption)
    #     print(dom_path, caption)
    # plt.savefig('debug.png')
    # print()

    train_sampler = BalancedBatchSampler(train_dataset.labels, BATCH_SIZE)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    print(len(train_dataloader))

    for batch in train_dataloader:
        image, label, dom_path, tree_repr, img_path, url = batch
        print(image.shape)
        print(label)
        print(label.shape)
        break