from torch import nn, optim
from PIL import Image
from torch.utils.data import DataLoader
import torch
import clip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from tqdm import tqdm
from scripts.data.data_utils import ButtonDataset, BalancedBatchSampler

def trainer(EPOCH, model, train_dataloader, device, LR):

    loss_img = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.visual.parameters(), lr=LR) # only change the image encoder part
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCH)
    model.train()
    model.token_embedding.eval()
    model.transformer.eval()
    model.ln_final.eval()

    model.positional_embedding.requires_grad = False
    model.text_projection.requires_grad = False
    model.logit_scale.requires_grad = False

    for epoch in range(EPOCH):
        print(f"running epoch {epoch}")
        step = 0
        tr_loss = 0

        pbar = tqdm(train_dataloader, leave=False)
        for batch in pbar:
            step += 1

            images, ground_truth, *_ = batch
            images = images.to(device)
            texts = clip.tokenize(["not a login button", "a login button"]).to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = ground_truth.to(device)
            total_loss = loss_img(logits_per_image, ground_truth) # only cross entropy for images

            optimizer.zero_grad()
            total_loss.backward()
            tr_loss += total_loss.item()
            if device == "cpu":
                optimizer.step()
                scheduler.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                scheduler.step()
                clip.model.convert_weights(model)
            pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
        tr_loss /= step

        torch.save(model.state_dict(), "./checkpoints/epoch{}_model.pt".format(epoch))
        print(f"epoch {epoch}, tr_loss {tr_loss}")

def convert_models_to_fp32(model):
    for p in model.visual.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCH = 5
    BATCH_SIZE = 128
    LR = 1e-5
    model, preprocess = clip.load("ViT-B/32", device=device)
    # https://github.com/openai/CLIP/issues/57
    if device == "cpu":
        model.float()

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    train_dataset = ButtonDataset(annot_path='./datasets/alexa_login_train.txt',
                                  root='./datasets/alexa_login',
                                  preprocess=preprocess)

    train_sampler = BalancedBatchSampler(train_dataset.labels, BATCH_SIZE)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    print(len(train_dataloader))

    trainer(EPOCH, model, train_dataloader, device, LR)