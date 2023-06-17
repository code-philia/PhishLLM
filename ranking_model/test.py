import torch


@torch.no_grad()
def tester(model, test_dataloader, device):
    model.eval()
    pbar = tqdm(test_dataloader, leave=False)

    for batch in pbar:
        images, texts, _ = batch
        images = images.to(device)
        texts = clip.tokenize(texts).to(device)
        # align
        logits_per_image, logits_per_text = model(images, texts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        pred = logits_per_image.argmax(dim=-1).cpu().numpy()[0]
        conf = probs[0, pred]