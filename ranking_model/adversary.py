from ranking_model.train import *
from ranking_model.torchattacks.attacks.fgsm import FGSM
from ranking_model.torchattacks.attacks.bim import BIM
from ranking_model.torchattacks.attacks.deepfool import DeepFool
from ranking_model.torchattacks.attacks.pgd import PGD
import math
import torch
import torch.nn.functional as F
from torch import nn

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class QuantizeGelu(nn.Module):
    def __init__(self, step_size = 0.01):
        super().__init__()
        self.step_size = step_size

    def forward(self, x):
        out = torch.mul(torch.floor(x / self.step_size), self.step_size) # quantize by step_size
        return out * torch.sigmoid(1.702 * out)

def protect(model):
    for name, module in model.named_modules():
        if module != model:
            if name == 'gelu':
                setattr(model, name, QuantizeGelu())
            elif isinstance(module, nn.Module) or isinstance(module, nn.Sequential):
                protect(module)

def print_model(model):
    for name, module in model.named_modules():
        print(module)

def tester_rank(model, test_dataset, device):
    total = 0
    perturb_correct = 0

    df = pd.DataFrame({'url': test_dataset.urls,
                       'path': test_dataset.img_paths,
                       'label': test_dataset.labels})
    grp = df.groupby('url')
    grp = dict(list(grp), keys=lambda x: x[0])  # {url: List[dom_path, save_path]}

    for url, data in tqdm(grp.items()):
        torch.cuda.empty_cache()
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

        if (labels == 1).sum().item():  # has login button
            total += 1
        else:
            continue

        # adversary attack
        images = torch.stack(images).to(device)
        target_labels = torch.zeros_like(labels)
        target_labels = target_labels.long().to(device)

        attack_cls = DeepFool(model, device=device)
        adv_images = attack_cls(images, labels, target_labels)
        images.detach()
        del attack_cls

        # perturbed prediction
        model.load_state_dict(torch.load("./checkpoints/epoch4_model.pt"))
        model.eval()
        protect(model.visual)
        model.token_embedding.eval()
        model.transformer.eval()
        model.ln_final.eval()
        model.positional_embedding.requires_grad = False
        model.text_projection.requires_grad = False
        model.logit_scale.requires_grad = False

        texts = clip.tokenize(["not a login button", "a login button"]).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(adv_images, texts)
        probs = logits_per_image.softmax(dim=-1)  # (N, C)
        conf = probs[torch.arange(probs.shape[0]), 1]  # take the confidence (N, 1)
        _, ind = torch.topk(conf, min(1, len(conf)))  # top1 index
        del adv_images, images, texts

        if (labels[ind] == 1).sum().item():  # has login button and it is reported
            perturb_correct += 1

        print(f"After attack correct count = {perturb_correct}, Total = {total}")

    print(f"After attack correct count = {perturb_correct}, Total = {total}, Recall@K = {perturb_correct/total}")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # https://github.com/openai/CLIP/issues/57
    if device == "cpu":
        model.float()

    test_dataset = ButtonDataset(annot_path='./datasets/alexa_login_test.txt',
                                 root='./datasets/alexa_login',
                                 preprocess=preprocess)

    state_dict = torch.load("./checkpoints/epoch{}_model.pt".format(4))
    model.load_state_dict(state_dict)

    protect(model.visual)
    model.token_embedding.eval()
    model.transformer.eval()
    model.ln_final.eval()
    model.positional_embedding.requires_grad = False
    model.text_projection.requires_grad = False
    model.logit_scale.requires_grad = False

    tester_rank(model, test_dataset, device)
    # FGSM: After attack correct count = 210, Total = 321, Recall@K = 0.6542056074766355
    # BIM (iterative FGSM, but gradually increasing the perturbation magnitude) After attack correct count = 224, Total = 321, Recall@K = 0.6978193146417445
    # DeepFool After attack correct count = 27, Total = 321, Recall@K = 0.08411214953271028
    # PGD: Iterative FGSM, but the first step is random direction After attack correct count = 183, Total = 321, Recall@K = 0.5700934579439252

    ### With step-relu
    # FGSM After attack correct count = 291, Total = 321, Recall@K = 0.9065420560747663
    # BIM After attack correct count = 282, Total = 321, Recall@K = 0.8785046728971962
    # DeepFool
    # PGD After attack correct count = 264, Total = 321, Recall@K = 0.822429906542056




