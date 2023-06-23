
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
from ranking_model.dom_utils import prune_tree, get_tree_repr
import io
from lxml import html

def format_input_generation(
    html_path, dom_path_list,  keep_html_brackets=False
):
    with io.open(html_path, 'r', encoding='ISO-8859-1') as f:
        page = f.read()
    dom_tree = html.fromstring(page)
    dom_tree = prune_tree(dom_tree, dom_path_list)
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
d

candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choices.append(
            [
                node.attrib["backend_node_id"],
                " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                ),
            ]
        )
    return choices

class ButtonDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []
        for img_path, captions in data.items():
            for cap in captions:
                self.img_paths.append(img_path)
                self.captions.append(cap)
        self.processed_cache = {}
        for img_path in data:
            self.processed_cache[img_path] = self.preprocess(Image.open(img_path))
        self.img_paths_set = list(data.keys())
        self.path2label = {path: self.img_paths_set.index(path) for path in self.img_paths_set}

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.processed_cache[img_path]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return image, caption, label


# https://github.com/pytorch/pytorch/blob/e5742494f6080c8e6f43c37689fc18a7c4b39dfd/torch/utils/data/dataloader.py#L145
class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        """
            BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
            Returns batches of size n_classes * n_samples
        """
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

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
    # IMG_ROOT = "./datasets/alexa_login"
    # JSON_ROOT = "../input/meme-project-clean-json"
    # img_paths = glob.glob(os.path.join(IMG_ROOT, "*", "*.png"))

    # print(len(img_paths))
    # train test split
    # train_img_paths, test_img_paths = train_test_split(img_paths, test_size=0.2, random_state=42)
    # d_train = {k: d[k] for k in train_img_paths}
    # d_test = {k: d[k] for k in test_img_paths}
    # len(d_train), len(d_test)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    # train_dataset = MemeDataset(d_train, preprocess)
    # test_dataset = MemeDataset(d_test, preprocess)
    # len(train_dataset), len(test_dataset), train_dataset[0]
    format_input_generation(html_path='./datasets/alexa_login/huxiu.com/index.html',
                            dom_path_list=['//html/BODY/DIV[3]/DIV[1]/HEADER[1]/DIV[1]/DIV[2]/NAV[1]/DIV[1]/DIV[3]/BUTTON[1]'])