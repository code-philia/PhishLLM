import os
import yaml
import torch
import numpy as np
from PIL import Image
import torch
from selection_model.dataloader import ShotDataset
from distutils.sysconfig import get_python_lib
import yaml
import os
from phishintention.src.AWL_detector import element_config, element_recognition, find_element_type
from phishintention.src.OCR_aided_siamese import pred_siamese_OCR, phishpedia_config_OCR_easy
from phishintention.src.OCR_siamese_utils.utils import brand_converter, resolution_alignment
from phishpedia.src.siamese_pedia.inference import pred_siamese
import time
from tqdm import tqdm
from tldextract import tldextract
import pickle
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['https_proxy'] = "http://127.0.0.1:7890" # proxy


@torch.no_grad()
def siamese_inference_OCR_reimplement(domain_map, reference_logo,
                                      SIAMESE_MODEL, OCR_MODEL,
                                      SIAMESE_THRE,
                                      LOGO_FEATS, LOGO_FILES):

    img_feat = pred_siamese_OCR(img=reference_logo,
                                model=SIAMESE_MODEL,
                                ocr_model=OCR_MODEL)
    print('logo feature returned')
    sim_list = np.matmul(LOGO_FEATS, img_feat.T)
    pred_brand_list = LOGO_FILES

    assert len(sim_list) == len(pred_brand_list)

    ## get top 3 brands
    idx = np.argsort(sim_list)[::-1][:3]
    pred_brand_list = np.array(pred_brand_list)[idx]
    sim_list = np.array(sim_list)[idx]

    # top1,2,3 candidate logos
    top3_logolist = [Image.open(x) for x in pred_brand_list]
    top3_brandlist = [brand_converter(os.path.basename(os.path.dirname(x))) for x in pred_brand_list]
    top3_domainlist = [domain_map[x] for x in top3_brandlist]
    top3_simlist = sim_list
    print('top3 similar logo returned')

    for j in range(3):

        ## If we are trying those lower rank logo, the predicted brand of them should be the same as top1 logo, otherwise might be false positive
        if top3_brandlist[j] != top3_brandlist[0]:
            continue

        ## If the largest similarity exceeds threshold
        if top3_simlist[j] >= SIAMESE_THRE:
            predicted_brand = top3_brandlist[j]
            predicted_domain = top3_domainlist[j]
            final_sim = top3_simlist[j]

        ## Else if not exceed, try resolution alignment, see if can improve
        else:
            cropped, candidate_logo = resolution_alignment(reference_logo, top3_logolist[j])
            img_feat = pred_siamese_OCR(img=cropped,
                                        model=SIAMESE_MODEL,
                                        ocr_model=OCR_MODEL)
            logo_feat = pred_siamese_OCR(img=candidate_logo,
                                        model=SIAMESE_MODEL,
                                        ocr_model=OCR_MODEL)
            final_sim = logo_feat.dot(img_feat)
            if final_sim >= SIAMESE_THRE:
                predicted_brand = top3_brandlist[j]
                predicted_domain = top3_domainlist[j]
            else:
                break  # no hope, do not try other lower rank logos

        ## If there is a prediction, do aspect ratio check
        if predicted_brand is not None:
            ratio_crop = reference_logo.size[0] / reference_logo.size[1]
            ratio_logo = top3_logolist[j].size[0] / top3_logolist[j].size[1]
            # aspect ratios of matched pair must not deviate by more than factor of 2.5
            if max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) > 2.5:
                continue  # did not pass aspect ratio check, try other
            # If pass aspect ratio check, report a match
            else:
                return predicted_brand, predicted_domain, final_sim

    return None, None, top3_simlist[0]

@torch.no_grad()
def siamese_inference_reimplement(domain_map, reference_logo,
                                  SIAMESE_MODEL,
                                  SIAMESE_THRE,
                                  LOGO_FEATS, LOGO_FILES):

    img_feat = pred_siamese(img=reference_logo,
                            model=SIAMESE_MODEL)
    print('logo feature returned')
    sim_list = np.matmul(LOGO_FEATS, img_feat.T)
    pred_brand_list = LOGO_FILES

    assert len(sim_list) == len(pred_brand_list)

    ## get top 3 brands
    idx = np.argsort(sim_list)[::-1][:3]
    pred_brand_list = np.array(pred_brand_list)[idx]
    sim_list = np.array(sim_list)[idx]

    # top1,2,3 candidate logos
    top3_logolist = [Image.open(x) for x in pred_brand_list]
    top3_brandlist = [brand_converter(os.path.basename(os.path.dirname(x))) for x in pred_brand_list]
    top3_domainlist = [domain_map[x] for x in top3_brandlist]
    top3_simlist = sim_list
    print('top3 similar logo returned')

    for j in range(3):

        ## If we are trying those lower rank logo, the predicted brand of them should be the same as top1 logo, otherwise might be false positive
        if top3_brandlist[j] != top3_brandlist[0]:
            continue

        ## If the largest similarity exceeds threshold
        if top3_simlist[j] >= SIAMESE_THRE:
            predicted_brand = top3_brandlist[j]
            predicted_domain = top3_domainlist[j]
            final_sim = top3_simlist[j]

        ## Else if not exceed, try resolution alignment, see if can improve
        else:
            cropped, candidate_logo = resolution_alignment(reference_logo, top3_logolist[j])
            img_feat = pred_siamese(img=cropped,
                                    model=SIAMESE_MODEL)
            logo_feat = pred_siamese(img=candidate_logo,
                                     model=SIAMESE_MODEL)
            final_sim = logo_feat.dot(img_feat)
            if final_sim >= SIAMESE_THRE:
                predicted_brand = top3_brandlist[j]
                predicted_domain = top3_domainlist[j]
            else:
                break  # no hope, do not try other lower rank logos

        ## If there is a prediction, do aspect ratio check
        if predicted_brand is not None:
            ratio_crop = reference_logo.size[0] / reference_logo.size[1]
            ratio_logo = top3_logolist[j].size[0] / top3_logolist[j].size[1]
            # aspect ratios of matched pair must not deviate by more than factor of 2.5
            if max(ratio_crop, ratio_logo) / min(ratio_crop, ratio_logo) > 2.5:
                continue  # did not pass aspect ratio check, try other
            # If pass aspect ratio check, report a match
            else:
                return predicted_brand, predicted_domain, final_sim

    return None, None, top3_simlist[0]

@torch.no_grad()
def brand_recognition_phishintention(screenshot_path,
                                     AWL_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE,
                                     LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH):

    screenshot_img = Image.open(screenshot_path).convert("RGB")
    screenshot_img_arr = np.asarray(screenshot_img)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1)  # RGB2BGR

    pred = AWL_MODEL(screenshot_img_arr)
    pred_i = pred["instances"].to('cpu')
    pred_classes = pred_i.pred_classes.detach().cpu()  # Boxes types
    pred_boxes = pred_i.pred_boxes.tensor.detach().cpu()  # Boxes coords

    if pred_boxes is None or len(pred_boxes) == 0:
        return ' '
    else:
        logo_pred_boxes, logo_pred_classes = find_element_type(pred_boxes=pred_boxes,
                                                               pred_classes=pred_classes,
                                                               bbox_type='logo')
        if len(logo_pred_boxes) == 0:
            return ' '
        else:
            logo_pred_boxes = logo_pred_boxes.detach().cpu().numpy()
            x1, y1, x2, y2 = logo_pred_boxes[0]
            reference_logo = screenshot_img.crop((x1, y1, x2, y2))
            reference_logo.save('./tmp.png')

    with open(DOMAIN_MAP_PATH, 'rb') as handle:
        domain_map = pickle.load(handle)

    target_this, domain_this, this_conf = siamese_inference_OCR_reimplement(domain_map, reference_logo,
                                                                            SIAMESE_MODEL, OCR_MODEL,
                                                                            SIAMESE_THRE,
                                                                            LOGO_FEATS, LOGO_FILES)

    if target_this is None:
        return ' '
    else:
        return domain_this[0]


@torch.no_grad()
def brand_recognition_phishpedia(screenshot_path,
                                 ELE_MODEL, SIAMESE_MODEL, SIAMESE_THRE,
                                 LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH):

    screenshot_img = Image.open(screenshot_path).convert("RGB")
    screenshot_img_arr = np.asarray(screenshot_img)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1)  # RGB2BGR

    pred = ELE_MODEL(screenshot_img_arr)
    pred_i = pred["instances"].to('cpu')
    pred_classes = pred_i.pred_classes  # tensor
    pred_boxes = pred_i.pred_boxes  # Boxes object
    logo_pred_boxes = pred_boxes[pred_classes == 1].tensor

    if len(logo_pred_boxes) == 0:
        return ' '
    else:
        logo_pred_boxes = logo_pred_boxes.detach().cpu().numpy()
        x1, y1, x2, y2 = logo_pred_boxes[0]
        reference_logo = screenshot_img.crop((x1, y1, x2, y2))
        reference_logo.save('./tmp.png')

    with open(DOMAIN_MAP_PATH, 'rb') as handle:
        domain_map = pickle.load(handle)

    target_this, domain_this, this_conf = siamese_inference_reimplement(domain_map, reference_logo,
                                                                            SIAMESE_MODEL,
                                                                            SIAMESE_THRE,
                                                                            LOGO_FEATS, LOGO_FILES)

    if target_this is None:
        return ' '
    else:
        return domain_this[0]


def test(result_file):

    ct = 0
    total = 0
    reported = 0
    runtime = []

    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        url, gt, pred, time = data
        total += 1
        runtime.append(float(time))

        if len(pred) > 1:
            reported += 1
            if gt in pred or tldextract.extract(gt).domain in pred:
                ct += 1
            elif tldextract.extract(pred).domain in gt:
                ct += 1
            else:
                print(data)

        # pbar.set_description(f"Recall (% brand recognized) = {ct / total} "
        #                      f"Precision (brand reported correct) = {ct / reported} ", refresh=True)

    print(f"Recall, i.e. % brand recognized = {ct / total} "
          f"Precision, i.e. % brand reported correct = {ct / reported} "
          f"Median runtime {np.median(runtime)}, Mean runtime {np.mean(runtime)}")


if __name__ == '__main__':
    dataset = ShotDataset(annot_path='./datasets/alexa_screenshots_orig.txt')

    # from phishintention.phishintention_config import load_config
    # AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config()
    # result_file = './datasets/alexa_brand_phishintention.txt'

    # from phishpedia.phishpedia_config import load_config
    # ELE_MODEL, SIAMESE_THRE, SIAMESE_MODEL, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config(None)
    # DOMAIN_MAP_PATH = '/home/ruofan/domain_map.pkl'
    # result_file = './datasets/alexa_brand_phishpedia.txt'

    # for url, shot_path, label in tqdm(zip(dataset.urls, dataset.shot_paths, dataset.labels)):
    #     if os.path.exists(result_file) and url in open(result_file).read():
    #         continue

        # start_time = time.time()
        # html_path = shot_path.replace('shot.png', 'index.html')
        # domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
        # # answer = brand_recognition_phishintention(shot_path,
        # #                                           AWL_MODEL, SIAMESE_MODEL, OCR_MODEL,
        # #                                           SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)
        # answer = brand_recognition_phishpedia(shot_path,
        #                                       ELE_MODEL, SIAMESE_MODEL,
        #                                       SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)
        # total_time = time.time() - start_time
        #
        # with open(result_file, 'a+') as f:
        #     f.write(url+'\t'+domain+'\t'+answer+'\t'+str(total_time)+'\n')

    # test(result_file='./datasets/alexa_brand_phishpedia.txt') # phishpedia Recall, i.e. % brand recognized = 0.03982300884955752 Precision, i.e. % brand reported correct = 0.782608695652174 Median runtime 0.28642261028289795, Mean runtime 0.282355929172672
    test(result_file='./datasets/alexa_brand_phishintention.txt') # PhishIntention Recall, i.e. % brand recognized = 0.04092920353982301 Precision, i.e. % brand reported correct = 0.8862275449101796 Median runtime 0.416703462600708, Mean runtime 0.49549826471942715
