import torch
import clip

@torch.no_grad()
def tester(model, test_dataloader, device):
    button_cls = ["login button", "signup button", "other button"]

    model.eval()
    pbar = tqdm(test_dataloader, leave=False)
    correct = 0
    total = 0
    for batch in pbar:
        images, texts, _ = batch
        images = images.to(device)
        texts = clip.tokenize(texts).to(device)
        # align
        logits_per_image, logits_per_text = model(images, texts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        pred = logits_per_image.argmax(dim=-1).cpu().numpy()[0]
        conf = probs[0, pred]

        gt = torch.tensor([button_cls.index(txt) for txt in texts])
        correct += torch.eq(gt, pred).sum()
        total += images.shape[0]

    return correct, total

