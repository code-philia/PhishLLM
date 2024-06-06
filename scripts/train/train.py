import os

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
import argparse

def trainer(epoch, model, train_dataloader, device, lr):
    loss_img = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.visual.parameters(), lr=lr)  # only change the image encoder part
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * epoch)
    model.train()
    model.token_embedding.eval()
    model.transformer.eval()
    model.ln_final.eval()

    model.positional_embedding.requires_grad = False
    model.text_projection.requires_grad = False
    model.logit_scale.requires_grad = False

    for ep in range(epoch):
        print(f"Running epoch {ep}")
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
            total_loss = loss_img(logits_per_image, ground_truth)  # only cross entropy for images

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
            pbar.set_description(f"Train batch CE: {total_loss.item()}", refresh=True)
        tr_loss /= step

        os.makedirs("./checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"./checkpoints/epoch{ep}_model.pt")
        print(f"Epoch {ep}, tr_loss {tr_loss}")
def convert_models_to_fp32(model):
    for p in model.visual.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Adjust model for CPU if necessary
    if device == "cpu":
        model.float()

    train_dataset = ButtonDataset(
        annot_path=args.annot_path,
        root=args.dataset_root,
        preprocess=preprocess
    )

    train_sampler = BalancedBatchSampler(train_dataset.labels, args.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
    print(f"Number of batches: {len(train_dataloader)}")

    trainer(args.epoch, model, train_dataloader, device, args.lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CLIP model on button images")
    parser.add_argument('--epoch', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--annot_path', type=str, required=True, help='Path to the annotation file')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of the dataset')

    args = parser.parse_args()
    main(args)