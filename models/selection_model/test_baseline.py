
from phishintention.src.crp_classifier import html_heuristic, credential_classifier_mixed_al
import numpy as np
from PIL import Image
import torch
from selection_model.dataloader import ShotDataset
from distutils.sysconfig import get_python_lib
import yaml
import os
from phishintention.src.AWL_detector import element_config
from phishintention.src.crp_classifier import credential_config
import time
from tqdm import tqdm

@torch.no_grad()
def crp_classifier(screenshot_path, html_path):

    with open(os.path.join(get_python_lib(), 'phishintention', 'configs.yaml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    AWL_CONFIG, AWL_MODEL = element_config(rcnn_weights_path=configs['AWL_MODEL']['WEIGHTS_PATH'],
                                           rcnn_cfg_path=configs['AWL_MODEL']['CFG_PATH'])
    CRP_CLASSIFIER = credential_config(
        checkpoint=configs['CRP_CLASSIFIER']['WEIGHTS_PATH'],
        model_type=configs['CRP_CLASSIFIER']['MODEL_TYPE'])

    screenshot_img = Image.open(screenshot_path).convert("RGB")
    screenshot_img_arr = np.asarray(screenshot_img)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1)  # RGB2BGR

    if os.path.exists(html_path):
        cre_pred_heu = html_heuristic(html_path)
        cre_pred_heu = 'A' if cre_pred_heu == 0 else 'B'
    else:
        cre_pred_heu = 'B'

    pred = AWL_MODEL(screenshot_img_arr)
    pred_i = pred["instances"].to('cpu')
    pred_classes = pred_i.pred_classes.detach().cpu()  # Boxes types
    pred_boxes = pred_i.pred_boxes.tensor.detach().cpu()  # Boxes coords

    if pred_boxes is None or len(pred_boxes) == 0:
        return cre_pred_heu, 'B' # non-CRP

    cre_pred_cv, cred_conf, _ = credential_classifier_mixed_al(img=screenshot_path, coords=pred_boxes,
                                                                types=pred_classes, model=CRP_CLASSIFIER)
    cre_pred_cv = 'A' if cre_pred_cv == 0 else 'B'

    return cre_pred_heu, cre_pred_cv


def test(result_file):
    correct_heu = 0
    correct_cv = 0
    correct_heu_or_cv = 0

    total = 0
    pred_pos_heu = 0
    pred_pos_cv = 0
    pred_pos_heu_or_cv = 0
    true_pos = 0

    pred_pos_and_true_pos_heu = 0
    pred_pos_and_true_pos_cv = 0
    pred_pos_and_true_pos_heu_or_cv = 0

    result_lines = open(result_file).readlines()
    runtime = []
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        url, gt, pred_heu, pred_cv, time = data
        if pred_heu == 'A':
            pred = pred_heu
        else:
            pred = pred_cv

        pred_pos_heu += float(pred_heu=='A')
        pred_pos_cv += float(pred_cv=='A')
        pred_pos_heu_or_cv += float(pred=='A')

        true_pos += float(gt=='A')

        pred_pos_and_true_pos_heu += float(pred_heu=='A')*float(gt=='A')
        pred_pos_and_true_pos_cv += float(pred_cv=='A')*float(gt=='A')
        pred_pos_and_true_pos_heu_or_cv += float(pred=='A')*float(gt=='A')

        correct_heu += float(gt==pred_heu)
        correct_cv += float(gt==pred_cv)
        correct_heu_or_cv += float(gt==pred)

        total += 1
        runtime.append(float(time))

        pbar.set_description(f"Heuristic only test classification acc: {correct_heu / total}, "
                             f"Heuristic only test precision: {pred_pos_and_true_pos_heu / (pred_pos_heu + 1e-8)} "
                             f"Heuristic only test recall: {pred_pos_and_true_pos_heu / (true_pos + 1e-8)} "
                             f"CV only test classification acc: {correct_cv / total}, "
                             f"CV only test precision: {pred_pos_and_true_pos_cv / (pred_pos_cv + 1e-8)} "
                             f"CV only test recall: {pred_pos_and_true_pos_cv / (true_pos + 1e-8)} "
                             f"Heuristic+CV test classification acc: {correct_heu_or_cv / total}, "
                             f"Heuristic+CV test precision: {pred_pos_and_true_pos_heu_or_cv / (pred_pos_heu_or_cv + 1e-8)} "
                             f"Heuristic+CV test recall: {pred_pos_and_true_pos_heu_or_cv / (true_pos + 1e-8)} "
                             , refresh=True)


    print(f"Heuristic only test classification acc: {correct_heu / total}, "
         f"Heuristic only test precision: {pred_pos_and_true_pos_heu / (pred_pos_heu + 1e-8)} "
         f"Heuristic only test recall: {pred_pos_and_true_pos_heu / (true_pos + 1e-8)} \n"
         f"CV only test classification acc: {correct_cv / total}, "
         f"CV only test precision: {pred_pos_and_true_pos_cv / (pred_pos_cv + 1e-8)} "
         f"CV only test recall: {pred_pos_and_true_pos_cv / (true_pos + 1e-8)} \n"
         f"Heuristic+CV test classification acc: {correct_heu_or_cv / total}, "
         f"Heuristic+CV test precision: {pred_pos_and_true_pos_heu_or_cv / (pred_pos_heu_or_cv + 1e-8)} "
         f"Heuristic+CV test recall: {pred_pos_and_true_pos_heu_or_cv / (true_pos + 1e-8)} "
         f"Median runtime {np.median(runtime)}, Mean runtime {np.mean(runtime)}")

if __name__ == '__main__':
    '''run evaluation'''
    # dataset = ShotDataset(annot_path='./datasets/alexa_screenshots.txt')
    result_file = './datasets/crp_classifier_baseline.txt'
    #
    # for url, shot_path, label in tqdm(zip(dataset.urls, dataset.shot_paths, dataset.labels)):
    #     if os.path.exists(result_file) and url in open(result_file).read():
    #         continue
    #     start_time = time.time()
    #     html_path = shot_path.replace('shot.png', 'index.html')
    #     cre_pred_heu, cre_pred_cv = crp_classifier(shot_path, html_path)
    #
    #     total_time = time.time() - start_time
    #
    #     with open(result_file, 'a+') as f:
    #         f.write(url+'\t'+label+'\t'+cre_pred_heu+'\t'+cre_pred_cv+'\t'+str(total_time)+'\n')

    '''result'''
    test(result_file)
    # Heuristic only test classification acc: 0.8947368421052632, Heuristic only test precision: 0.8085501858660916 Heuristic only test recall: 0.7699115044179654
    # CV only test classification acc: 0.9376552970408855, CV only test precision: 0.8517298187738738 CV only test recall: 0.915044247779513
    # Heuristic+CV test classification acc: 0.9067088321662525, Heuristic+CV test precision: 0.7457162439976305 Heuristic+CV test recall: 0.9628318583985591


