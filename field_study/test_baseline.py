from model_chain.test_baseline import *
import argparse
from datetime import datetime
import cv2
from xdriver.xutils.Logger import Logger


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./datasets/field_study/2023-08-23/")
    parser.add_argument("--date", default="2023-08-23", help="%Y-%m-%d")
    parser.add_argument("--method", default='phishpedia', choices=['phishpedia', 'phishintention'])
    args = parser.parse_args()

    # PhishLLM
    phishintention_cls = PhishIntentionWrapper()
    base_cls = TestBaseline(phishintention_cls)

    # Xdriver
    sleep_time = 3; timeout_time = 60
    XDriver.set_headless()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)
    Logger.set_debug_on()

    os.makedirs('./field_study/results/', exist_ok=True)
    result_txt = './field_study/results/{}_{}.txt'.format(args.date, args.method)

    if not os.path.exists(result_txt):
        with open(result_txt, "w+") as f:
            f.write("folder" + "\t")
            f.write("phish_prediction" + "\t")
            f.write("target_prediction" + "\t")  # write top1 prediction only
            f.write("runtime" + "\n")

    for ct, folder in tqdm(enumerate(os.listdir(args.folder))):
        if folder in [x.split('\t')[0] for x in open(result_txt, encoding='ISO-8859-1').readlines()]:
            continue

        info_path = os.path.join(args.folder, folder, 'info.txt')
        html_path = os.path.join(args.folder, folder, 'html.txt')
        shot_path = os.path.join(args.folder, folder, 'shot.png')
        predict_path = os.path.join(args.folder, folder, 'predict_{}.png'.format(args.method))
        if not os.path.exists(shot_path):
            continue

        try:
            if len(open(info_path, encoding='ISO-8859-1').read()) > 0:
                url = open(info_path, encoding='ISO-8859-1').read()
            else:
                url = 'https://' + folder
        except:
            url = 'https://' + folder

        try:
            if args.method == 'phishpedia':
                pred, brand, runtime, plotvis = base_cls.test_phishpedia(url, shot_path)
            else:
                pred, brand, runtime, plotvis = base_cls.test_phishintention(url, shot_path, driver)
        except KeyError:
            continue

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(folder + "\t")
                f.write(str(pred) + "\t")
                f.write(str(brand) + "\t")  # write top1 prediction only
                f.write(str(runtime) + "\n")
            if pred == 1:
                cv2.imwrite(predict_path, plotvis)

        except UnicodeEncodeError:
            continue

        if (ct + 501) % 500 == 0:
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(timeout_time / 2)
            driver.set_page_load_timeout(timeout_time)
            time.sleep(sleep_time)

    driver.quit()