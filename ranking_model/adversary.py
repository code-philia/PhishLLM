from ranking_model.train import *
from ranking_model.torchattacks.attacks.fgsm import FGSM
from ranking_model.torchattacks.attacks.bim import BIM
from ranking_model.torchattacks.attacks.deepfool import DeepFool
from ranking_model.torchattacks.attacks.pgd import PGD
import math
import torch
import torch.nn.functional as F
from torch import nn
from clip.model import LayerNorm, OrderedDict, ResidualAttentionBlock, QuickGELU
import torchvision.transforms as transforms

class QuantizeGelu(nn.Module):
    def __init__(self, step_size=0.05):
        super().__init__()
        self.step_size = step_size
        self.qrelu = QuantizeRelu(step_size)

    def forward(self, x):
        out = self.qrelu(x.half())
        out = out * torch.sigmoid(1.702 * out)
        return out

class QuantizeRelu(nn.Module):
    def __init__(self, step_size=0.05):
        super().__init__()
        self.step_size = step_size

    def forward(self, x):
        out = torch.mul(torch.floor(x / self.step_size), self.step_size)  # Quantize by step_size
        return out

class ResidualAttentionBlockQuantize(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head).to(dtype=torch.half)  # Convert to 'half' data type
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4).to(dtype=torch.half)),  # Convert to 'half' data type
            ("gelu", QuantizeGelu()),
            ("c_proj", nn.Linear(d_model * 4, d_model).to(dtype=torch.half))  # Convert to 'half' data type
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.qrelu = QuantizeRelu()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x.half(), x.half(), x.half(), need_weights=False, attn_mask=self.attn_mask)[0]  # Convert to 'half' data type

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = self.qrelu(x.half()) + self.mlp(self.ln_2(x))
        return x

def protect_act(model):
    for name, module in model.named_modules():
        if module != model:
            if name == 'gelu':
                setattr(model, name, QuantizeGelu())
            elif name == 'relu':
                setattr(model, name, QuantizeRelu())
            elif isinstance(module, nn.Module) or isinstance(module, nn.Sequential):
                protect_act(module)

def protect_resnetblock(model):
    modules = []
    block_names = []
    for name, module in model.transformer.resblocks.named_modules():
        if isinstance(module, ResidualAttentionBlock):  # Replace with the actual ResNet block class name
            modules.append(module)
            block_names.append(name)

    if modules:
        module = modules[-1]
        name = block_names[-1]
        new_block = ResidualAttentionBlockQuantize(module.attn.embed_dim,
                                                   module.attn.num_heads, None)
        new_block.load_state_dict(module.state_dict())
        new_block.training = module.training
        setattr(model.transformer.resblocks, name, new_block)


def freeze_params(model):
    model.eval()
    model.token_embedding.eval()
    model.transformer.eval()
    model.ln_final.eval()
    model.positional_embedding.requires_grad = False
    model.text_projection.requires_grad = False
    model.logit_scale.requires_grad = False

    model.visual.ln_post.eval()
    model.visual.ln_pre.eval()
    model.visual.conv1.eval()
    model.visual.positional_embedding.requires_grad = False
    model.visual.proj.requires_grad = False


def print_model(model):
    for name, module in model.named_modules():
        print(module)


def tester_rank(model, test_dataset, device, protect_enabled, adv_attack=True):
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

        images = torch.stack(images).to(device)
        if adv_attack:
            # adversary attack
            target_labels = torch.zeros_like(labels)
            target_labels = target_labels.long().to(device)

            if protect_enabled:
                protect_act(model.visual)
                protect_resnetblock(model.visual)
                model = model.to(device)
            attack_cls = PGD(model, device=device)
            adv_images = attack_cls(images, labels, target_labels)
            images.detach()
            del attack_cls

        # perturbed prediction
        del model
        model, _ = clip.load("ViT-B/32", device=device)
        model.load_state_dict(torch.load("./checkpoints/epoch4_model.pt"))
        model = model.to(device)
        freeze_params(model)

        texts = clip.tokenize(["not a login button", "a login button"]).to(device)
        with torch.no_grad():
            if adv_attack:
                logits_per_image, logits_per_text = model(adv_images, texts)
                del adv_images
            else:
                logits_per_image, logits_per_text = model(images, texts)

        probs = logits_per_image.softmax(dim=-1)  # (N, C)
        conf = probs[torch.arange(probs.shape[0]), 1]  # take the confidence (N, 1)
        _, ind = torch.topk(conf, min(1, len(conf)))  # top1 index
        del images

        if (labels[ind] == 1).sum().item():  # has login button and it is reported
            perturb_correct += 1

        print(f"After attack correct count = {perturb_correct}, Total = {total}, Recall@K = {perturb_correct/total}")

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
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    state_dict = torch.load("./checkpoints/epoch{}_model.pt".format(4))
    model.load_state_dict(state_dict)

    protect_act(model.visual) #
    protect_resnetblock(model.visual)
    model = model.to(device)
    freeze_params(model)

    tester_rank(model, test_dataset, device, protect_enabled=True)
    # tester_rank(model, test_dataset, device, protect_enabled=True, adv_attack=False)

    # FGSM: After attack correct count = 210, Total = 321, Recall@K = 0.6542056074766355
    # BIM (iterative FGSM, but gradually increasing the perturbation magnitude) After attack correct count = 224, Total = 321, Recall@K = 0.6978193146417445
    # DeepFool After attack correct count = 27, Total = 321, Recall@K = 0.08411214953271028
    # PGD: Iterative FGSM, but the first step is random direction After attack correct count = 183, Total = 321, Recall@K = 0.5700934579439252

    ### With step-relu
    # FGSM After attack correct count = 291, Total = 321, Recall@K = 0.9065420560747663
    # BIM After attack correct count = 282, Total = 321, Recall@K = 0.8816
    # DeepFool After attack correct count = 109, Total = 321, Recall@K = 0.3395638629283489
    # PGD After attack correct count = 264, Total = 321, Recall@K = 0.822429906542056




