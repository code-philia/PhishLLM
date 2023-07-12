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
from phishintention.phishintention_config import load_config
import time
from tqdm import tqdm
from tldextract import tldextract
import pickle
from brand_recognition.kb_baseline import BrandKnowledgeConstruction
from xdriver.XDriver import XDriver
from xdriver.xutils.Logger import Logger
import idna

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './brand_recognition/google_cloud.json'
# os.environ['https_proxy'] = "http://127.0.0.1:7890" # proxy

def expand_targetlist(domain_map_path, targetlist_path,
                      SIAMESE_MODEL, OCR_MODEL,
                      new_brand, new_domains, new_logos):
    # expand domain map
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)

    existing_brands = domain_map.keys()
    if new_brand in existing_brands:
        domain_in_target = True
    else:
        domain_in_target = False

    if not domain_in_target:  # if this domain is not in targetlist ==> add it
        domain_map[new_brand] = list(set(new_domains))
        with open(domain_map_path, 'wb') as handle:
            pickle.dump(domain_map, handle)

    # expand cached logo features list
    prev_logo_feats = np.load(
        os.path.join(os.path.dirname(targetlist_path), 'LOGO_FEATS.npy'))
    prev_file_name_list = np.load(
        os.path.join(os.path.dirname(targetlist_path), 'LOGO_FILES.npy'))


    # expand logo list
    valid_logo = [a for a in new_logos if a is not None]
    if len(valid_logo) == 0:  # no valid logo
        return prev_logo_feats, prev_file_name_list

    new_logo_save_folder = os.path.join(targetlist_path.split('.zip')[0], new_brand)
    os.makedirs(new_logo_save_folder, exist_ok=True)

    exist_num_files = len(os.listdir(new_logo_save_folder))
    new_logo_save_paths = []
    for ct, logo in enumerate(valid_logo):
        this_logo_save_path = os.path.join(new_logo_save_folder, '{}.png'.format(exist_num_files + ct))
        if os.path.exists(this_logo_save_path):
            this_logo_save_path = os.path.join(new_logo_save_folder, '{}_expand.png'.format(exist_num_files + ct))
        logo.save(this_logo_save_path)
        new_logo_save_paths.append(this_logo_save_path)


    new_logo_feats = []
    new_file_name_list = []

    for logo_path in new_logo_save_paths:
        new_logo_feats.append(pred_siamese_OCR(img=logo_path,
                                               model=SIAMESE_MODEL,
                                               ocr_model=OCR_MODEL,
                                               grayscale=False))
        new_file_name_list.append(str(logo_path))

    prev_logo_feats = prev_logo_feats.tolist()
    prev_file_name_list = prev_file_name_list.tolist()
    agg_logo_feats = prev_logo_feats + new_logo_feats
    agg_file_name_list = prev_file_name_list + new_file_name_list
    np.save(os.path.join(os.path.dirname(targetlist_path), 'LOGO_FEATS'),
            np.asarray(agg_logo_feats))
    np.save(os.path.join(os.path.dirname(targetlist_path), 'LOGO_FILES'),
            np.asarray(agg_file_name_list))

    # update reference list
    LOGO_FEATS = np.asarray(agg_logo_feats)
    LOGO_FILES = np.asarray(agg_file_name_list)

    return LOGO_FEATS, LOGO_FILES

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
def brand_siamese_classifier(reference_logo,
                             SIAMESE_MODEL, OCR_MODEL,
                             SIAMESE_THRE,
                             LOGO_FEATS, LOGO_FILES,
                             domain_map_path):
    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)

    target_this, domain_this, this_conf = siamese_inference_OCR_reimplement(domain_map, reference_logo,
                                                                            SIAMESE_MODEL, OCR_MODEL,
                                                                            SIAMESE_THRE,
                                                                            LOGO_FEATS, LOGO_FILES)
    return target_this, domain_this, this_conf

@torch.no_grad()
def brand_recognition_logo2brand(screenshot_path, url,
                                 kb_cls, kb_driver,
                                 AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH):

    with open(os.path.join(get_python_lib(), 'phishintention', 'configs.yaml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    SIAMESE_THRE_RELAX = 0.83
    TARGETLIST_PATH = configs['SIAMESE_MODEL']['TARGETLIST_PATH']

    screenshot_img = Image.open(screenshot_path).convert("RGB")
    screenshot_img_arr = np.asarray(screenshot_img)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1)  # RGB2BGR

    pred = AWL_MODEL(screenshot_img_arr)
    pred_i = pred["instances"].to('cpu')
    pred_classes = pred_i.pred_classes.detach().cpu()  # Boxes types
    pred_boxes = pred_i.pred_boxes.tensor.detach().cpu()  # Boxes coords

    if pred_boxes is None or len(pred_boxes) == 0:
        has_logo = False
    else:
        logo_pred_boxes, logo_pred_classes = find_element_type(pred_boxes=pred_boxes,
                                                               pred_classes=pred_classes,
                                                               bbox_type='logo')
        if len(logo_pred_boxes) == 0:
            has_logo = False
        else:
            has_logo = True
            logo_pred_boxes = logo_pred_boxes.detach().cpu().numpy()
            x1, y1, x2, y2 = logo_pred_boxes[0]
            reference_logo = screenshot_img.crop((x1, y1, x2, y2))
            reference_logo.save('./tmp.png')
            img_feat = pred_siamese_OCR(img=reference_logo,
                                         model=SIAMESE_MODEL,
                                         ocr_model=OCR_MODEL)
            sim_list = LOGO_FEATS @ img_feat.T  # take dot product for every pair of embeddings (Cosine Similarity)

    if not has_logo:
        return ' '

    if np.sum(sim_list >= SIAMESE_THRE_RELAX) == 0:  # not exceed siamese relaxed threshold, not in targetlist
        # KB
        query_domain, query_tld = tldextract.extract(url).domain, tldextract.extract(url).suffix
        _, new_domains, new_brand, new_logos, comment = kb_cls.runit_simplified(reference_logo, './tmp.png', kb_driver, query_domain, query_tld)
        # expand
        if len(new_domains) > 0 and np.sum([x is not None for x in new_logos]) > 0:
            # Ref* <- Ref* + <domain(w_target), rep(w_target)>
            LOGO_FEATS, LOGO_FILES = expand_targetlist(DOMAIN_MAP_PATH, TARGETLIST_PATH,
                                                      SIAMESE_MODEL, OCR_MODEL,
                                                      new_brand, new_domains, new_logos)
        else:
            return ' '

    target_this, domain_this, this_conf = brand_siamese_classifier(reference_logo,
                                                                   SIAMESE_MODEL, OCR_MODEL,
                                                                   SIAMESE_THRE,
                                                                   LOGO_FEATS, LOGO_FILES,
                                                                   DOMAIN_MAP_PATH
                                                                   )

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
            try:
                translated_domain = idna.encode(pred).decode('utf-8')
            except idna.core.InvalidCodepoint:
                translated_domain = pred
            if gt in pred or tldextract.extract(gt).domain in pred:
                ct += 1
            elif gt in translated_domain:
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

    result_file = './datasets/alexa_brand_baseline.txt'
    #
    # print(len(dataset))
    # SEARCH_ENGINE_ID, API_KEY = [x.strip() for x in open('./brand_recognition/google_search_key.txt').readlines()]
    # AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config()
    # SIAMESE_THRE_RELAX = 0.83
    # kb_cls = BrandKnowledgeConstruction(API_KEY, SEARCH_ENGINE_ID,
    #                                     AWL_MODEL, SIAMESE_MODEL, OCR_MODEL,
    #                                                       SIAMESE_THRE_RELAX)
    #
    # sleep_time = 3; timeout_time = 60
    # XDriver.set_headless()
    # Logger.set_debug_on()
    # kb_driver = XDriver.boot(chrome=True)
    # kb_driver.set_script_timeout(timeout_time/2)
    # kb_driver.set_page_load_timeout(timeout_time)
    # time.sleep(sleep_time)  # fixme: you have to sleep sometime, otherwise the browser will keep crashing
    #
    #
    # for url, shot_path, label in tqdm(zip(dataset.urls, dataset.shot_paths, dataset.labels)):
    #     if os.path.exists(result_file) and url in open(result_file).read():
    #         continue
    #
    #     while True:
    #         try:
    #             start_time = time.time()
    #             html_path = shot_path.replace('shot.png', 'index.html')
    #             domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
    #             answer = brand_recognition_logo2brand(shot_path, url, kb_cls, kb_driver,
    #                                                   AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)
    #             total_time = time.time() - start_time
    #             break
    #         except Exception as e:
    #             print(e)
    #             time.sleep(10)
    #
    #     with open(result_file, 'a+') as f:
    #         f.write(url+'\t'+domain+'\t'+answer+'\t'+str(total_time)+'\n')
    test(result_file) ## Recall, i.e. % brand recognized = 0.28926991150442477 Precision, i.e. % brand reported correct = 0.934763181411975
