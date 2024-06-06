from scripts.data.data_utils import ButtonDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import clip
import numpy as np
from PIL import Image
import pandas as pd

@torch.no_grad()
def tester(model, test_dataloader, device):

    model.eval()
    correct = 0
    total = 0
    pred_pos = 0
    true_pos = 0
    pred_pos_and_true_pos = 0

    pbar = tqdm(test_dataloader, leave=False)
    for batch in pbar:
        images, ground_truth, _, _, img_path, url = batch
        images = images.to(device)
        texts = clip.tokenize(["not a login button", "a login button"]).to(device)

        # align
        logits_per_image, logits_per_text = model(images, texts)
        pred = logits_per_image.argmax(dim=-1).cpu()

        correct += torch.eq(ground_truth, pred).sum().item()
        total += images.shape[0]

        pred_pos += pred.item()
        true_pos += ground_truth.item()
        pred_pos_and_true_pos += (pred.item()) * (ground_truth.item())

        pbar.set_description(f"test classification acc: {correct/total}, "
                             f"test precision: {pred_pos_and_true_pos/(pred_pos+1e-8)} "
                             f"test recall: {pred_pos_and_true_pos/(true_pos+1e-8)} ", refresh=True)

    print(f"test classification acc: {correct/total}, "
         f"test precision: {pred_pos_and_true_pos/(pred_pos+1e-8)} "
         f"test recall: {pred_pos_and_true_pos/(true_pos+1e-8)} ")
    return correct, total


@torch.no_grad()
def tester_rank(model, test_dataset, preprocess, device):
    model.eval()
    correct = 0
    total = 0

    df = pd.DataFrame({'url': test_dataset.urls,
                       'path':  test_dataset.img_paths,
                       'label': test_dataset.labels})
    grp = df.groupby('url')
    grp = dict(list(grp), keys=lambda x: x[0])  # {url: List[dom_path, save_path]}

    for url, data in tqdm(grp.items()):
        try:
            img_paths = data.path
            labels = data.label
        except:
            continue
        labels = torch.tensor(np.asarray(labels))
        images = []
        for path in img_paths:
            img_process = preprocess(Image.open(path))
            images.append(img_process)

        images = torch.stack(images).to(device)
        texts = clip.tokenize(["not a login button", "a login button"]).to(device)
        logits_per_image, logits_per_text = model(images, texts)
        probs = logits_per_image.softmax(dim=-1) # (N, C)
        conf = probs[torch.arange(probs.shape[0]), 1] # take the confidence (N, 1)
        _, ind = torch.topk(conf, min(10, len(conf))) # top1 index

        if (labels == 1).sum().item(): # has login button
            if (labels[ind] == 1).sum().item(): # has login button and it is reported
                correct += 1

            total += 1

    print(correct, total)



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    if device == "cpu":
        model.float() # https://github.com/openai/CLIP/issues/57

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    train_dataset = ButtonDataset(annot_path='./datasets/alexa_login_train.txt',
                                 root='./datasets/alexa_login',
                                 preprocess=preprocess)

    test_dataset = ButtonDataset(annot_path='./datasets/alexa_login_test.txt',
                                 root='./datasets/alexa_login',
                                 preprocess=preprocess)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    print(len(train_dataloader))
    print(len(test_dataloader))

    state_dict = torch.load("./checkpoints/epoch{}_model.pt".format(4))
    model.load_state_dict(state_dict)
    # tester(model, test_dataloader, device)
    # overall image test classification acc: 0.9960878724044538, test precision: 0.9518987341531164 test recall: 0.8392857142669802

    tester_rank(model, test_dataset, preprocess, device)
    # top1 recall: 0.9128, top3 recall: 0.9283, top5 recall: 0.9470, top10 recall: 0.9720




