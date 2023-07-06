import os.path

import torch
import clip
from ranking_model.train import *
import shutil
from xdriver.xutils.Regexes import Regexes
from xdriver.xutils.Logger import Logger
import base64
import io
import time
import yaml
from phishintention.src.crp_locator import login_config
import phishintention
from xdriver.XDriver import XDriver

def heuristic_find_dom(driver, topk):
    ct = 0
    page_text = driver.get_page_text()
    dom_element_list = []

    # no HTML text
    if not page_text or len(page_text) == 0:
        return dom_element_list

    page_text = page_text.split('\n')
    for line in page_text:  # iterate over html text
        if len(line.replace(' ', '')) > 300:
            continue
        # looking for keyword
        keyword_finder = re.findall(Regexes.CREDENTIAL_TAKING_KEYWORDS, " ".join(line.split()), re.IGNORECASE)

        if len(keyword_finder) > 0:
            ct += 1
            # clicking the text
            Logger.spit("Try clicking :{}".format(line), debug=True)
            elements = driver.get_clickable_elements_contains(line)
            prev_windows = driver.window_handles
            if len(elements):
                dom = driver.get_dompath(elements[0])
                if dom:
                    dom_element_list.append(dom)

        if len(dom_element_list) >= topk or ct >= topk:
            break

    return dom_element_list



def cv_find_dom(driver, crp_locator_model, topk):
    # CV-based login finder predict elements
    old_screenshot_img = Image.open(io.BytesIO(base64.b64decode(driver.get_screenshot_encoding())))
    old_screenshot_img = old_screenshot_img.convert("RGB")
    old_screenshot_img_arr = np.asarray(old_screenshot_img)
    old_screenshot_img_arr = np.flip(old_screenshot_img_arr, -1)  # RGB2BGR

    pred = crp_locator_model(old_screenshot_img_arr)
    pred_i = pred["instances"].to('cpu')
    pred_boxes = pred_i.pred_boxes.tensor  # Boxes coords
    login_buttons = pred_boxes.detach().cpu()

    dom_element_list = []
    # if no prediction at all
    if login_buttons is None or len(login_buttons) == 0:
        return dom_element_list

    login_buttons = login_buttons.detach().cpu().numpy()
    for bbox in login_buttons[: min(topk, len(login_buttons))]:  # only for top3 boxes
        x1, y1, x2, y2 = bbox
        element = driver.find_element_by_location((x1 + x2) // 2,
                                                  (y1 + y2) // 2)  # click center point of predicted bbox for safe
        Logger.spit("Try clicking point: ({}, {})".format((x1 + x2) // 2, (y1 + y2) // 2), debug=True)
        if element:
            dom = driver.get_dompath(element)
            if dom:
                dom_element_list.append(dom)

    return dom_element_list

def crp_locator(driver, url, topk, heuristic=False):

    try:
        driver.get(url)
        time.sleep(7)
        Logger.spit(f'URL = {url}', debug=True)
    except Exception as e:
        Logger.spit(e, debug=True)
        return

    if heuristic:
        dom_element_list = heuristic_find_dom(driver, topk)
    else:
        from distutils.sysconfig import get_python_lib
        with open(os.path.join(get_python_lib(), 'phishintention', 'configs.yaml')) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
        _, CRP_LOCATOR_MODEL = login_config(
            rcnn_weights_path=configs['CRP_LOCATOR']['WEIGHTS_PATH'],
            rcnn_cfg_path=configs['CRP_LOCATOR']['CFG_PATH'],
            device=device)
        dom_element_list = cv_find_dom(driver, CRP_LOCATOR_MODEL, topk)

    return dom_element_list



@torch.no_grad()
def tester_rank(model, test_dataset, preprocess, device):
    try:
        shutil.rmtree('./datasets/debug')
    except:
        pass
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

            total += 1

    print(correct, total)



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load("ViT-B/32", device=device)

    test_dataset = ButtonDataset(annot_path='./datasets/alexa_login_test.txt',
                                 root='./datasets/alexa_login',
                                 preprocess=preprocess)

    df = pd.DataFrame({'url': test_dataset.urls,
                       'dom': test_dataset.dom_paths,
                       'path': test_dataset.img_paths,
                       'label': test_dataset.labels})
    grp = df.groupby('url')
    grp = dict(list(grp), keys=lambda x: x[0])  # {url: List[dom_path, save_path]}

    # initiate driver
    XDriver.set_headless()
    Logger.set_debug_on()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(30)
    driver.set_page_load_timeout(30)
    time.sleep(3)  # fixme: you have to sleep sometime, otherwise the browser will keep crashing
    result_file = './datasets/crp_locator_baseline.txt'

    print(len(list(grp.keys())))
    for ct, (url, data) in enumerate(grp.items()):
        if url == 'keys':
            continue
        if os.path.exists(result_file) and url in open(result_file).read():
            continue

        dom, label = list(data.dom), list(data.label)
        ind = np.where(np.asarray(label) == 1)[0]
        if len(ind) == 0:
            continue

        gt_dom = np.asarray(dom)[ind]
        dom_element_list_heu = crp_locator(driver, url, topk=10, heuristic=True)
        if not dom_element_list_heu:
            dom_element_list_heu = ''

        dom_element_list_cv = crp_locator(driver, url, topk=10, heuristic=False)
        if not dom_element_list_cv:
            dom_element_list_cv = ''

        with open(result_file, 'a+') as f:
            f.write(url+'\t'+gt_dom[0]+'\t'+str(dom_element_list_heu)+'\t'+str(dom_element_list_cv)+'\n')

        # select one: another model
        if (ct + 1) % 100 == 0:
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(30)
            driver.set_page_load_timeout(30)
            time.sleep(3)




