import os
from tqdm import tqdm
import clip
import torch
from PIL import Image
import torch
from torch import nn, optim

def trainer(EPOCH, model, train_dataloader, device, BATCH_SIZE):
    for epoch in range(EPOCH):
        print(f"running epoch {epoch}")
        step = 0
        tr_loss = 0
        model.train()
        model.token_embedding.zero_grad()
        model.transformer.zero_grad()
        model.ln_final.zero_grad()

        model.positional_embedding.requires_grad = False
        model.text_projection.requires_grad = False
        model.logit_scale.requires_grad = False
        pbar = tqdm(train_dataloader, leave=False)
        for batch in pbar:
            step += 1
            optimizer.zero_grad()

            images, texts, _ = batch
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(BATCH_SIZE).to(device)

            # total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss = loss_img(logits_per_image, ground_truth)
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

        torch.save(model.state_dict(), "epoch{}_model.pt".format(epoch))
        print(f"epoch {epoch}, tr_loss {tr_loss}")

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCH = 10
    BATCH_SIZE = 128
    model, preprocess = clip.load("ViT-B/32", device=device)
    # https://github.com/openai/CLIP/issues/57
    if device == "cpu":
        model.float()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.visual.parameters(), lr=1e-5) # only change the image encoder part
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCH)
    trainer(EPOCH, model, train_dataloader, device, BATCH_SIZE)