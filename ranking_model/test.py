import torch
import clip
from ranking_model.train import *
import shutil
import time

@torch.no_grad()
def tester(model, test_dataloader, device):

    model.eval()
    correct = 0
    total = 0
    pred_pos = 0
    true_pos = 0
    pred_pos_and_true_pos = 0
    try:
        shutil.rmtree('./datasets/alexa_login_fp')
    except:
        pass
    try:
        shutil.rmtree('./datasets/alexa_login_tp')
    except:
        pass
    try:
        shutil.rmtree('./datasets/alexa_login_fn')
    except:
        pass

    os.makedirs('./datasets/alexa_login_fp', exist_ok=True)
    os.makedirs('./datasets/alexa_login_tp', exist_ok=True)
    os.makedirs('./datasets/alexa_login_fn', exist_ok=True)

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

        if pred.item() == 1 and ground_truth.item() == 0: # fp
            plt.imshow(Image.open(img_path[0]))
            plt.savefig(f"./datasets/alexa_login_fp/{url[0].split('https://')[1]+'_'+os.path.basename(img_path[0])}")

        if pred.item() == 1 and ground_truth.item() == 1: # tp
            plt.imshow(Image.open(img_path[0]))
            plt.savefig(
                f"./datasets/alexa_login_tp/{url[0].split('https://')[1] + '_' + os.path.basename(img_path[0])}")

        if pred.item() == 0 and ground_truth.item() == 1: # fn
            plt.imshow(Image.open(img_path[0]))
            plt.savefig(f"./datasets/alexa_login_fn/{url[0].split('https://')[1]+'_'+os.path.basename(img_path[0])}")

        pbar.set_description(f"test classification acc: {correct/total}, "
                             f"test precision: {pred_pos_and_true_pos/(pred_pos+1e-8)} "
                             f"test recall: {pred_pos_and_true_pos/(true_pos+1e-8)} ", refresh=True)

    print(f"test classification acc: {correct/total}, "
         f"test precision: {pred_pos_and_true_pos/(pred_pos+1e-8)} "
         f"test recall: {pred_pos_and_true_pos/(true_pos+1e-8)} ")
    return correct, total


@torch.no_grad()
def tester_rank(model, test_dataset, preprocess, device):
    try:
        shutil.rmtree('./datasets/debug')
    except:
        pass
    model.eval()
    correct = 0
    total = 0
    runtime = []

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

        if (labels == 1).sum().item():  # has login button
            total += 1
        else:
            continue

        images = torch.stack(images).to(device)
        texts = clip.tokenize(["not a login button", "a login button"]).to(device)
        start_time = time.time()
        logits_per_image, logits_per_text = model(images, texts)
        total_time = time.time() - start_time
        probs = logits_per_image.softmax(dim=-1) # (N, C)
        conf = probs[torch.arange(probs.shape[0]), 1] # take the confidence (N, 1)
        _, ind = torch.topk(conf, min(1, len(conf))) # top1 index

        runtime.append(total_time)
        if (labels[ind] == 1).sum().item(): # has login button and it is reported
            correct += 1
        # visualize
        # os.makedirs('./datasets/debug', exist_ok=True)
        # f, axarr = plt.subplots(4, 1)
        # for it in range(min(3, len(conf))):
        #     img_path_sorted = np.asarray(img_paths)[ind.cpu()]
        #     axarr[it].imshow(Image.open(img_path_sorted[it]))
        #     axarr[it].set_title(str(conf[ind][it].item()))
        #
        # gt_ind = torch.where(labels == 1)[0]
        # if len(gt_ind) > 1:
        #     gt_ind = gt_ind[0]
        # axarr[3].imshow(Image.open(np.asarray(img_paths)[gt_ind.cpu()]))
        # axarr[3].set_title('ground_truth'+str(conf[gt_ind].item()))
        #
        # plt.savefig(
        #     f"./datasets/debug/{url.split('https://')[1]}.png")
        # plt.close()

        print(correct, total)

    print(correct, total)
    print(correct/total)
    print(np.median(runtime))



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # https://github.com/openai/CLIP/issues/57
    if device == "cpu":
        model.float()

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




