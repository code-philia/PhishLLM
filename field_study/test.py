from model_chain.test_llm import *
import argparse
from datetime import datetime
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./datasets/field_study/2023-08-05/")
    parser.add_argument("--date", default="2023-08-05", help="%Y-%m-%d")
    args = parser.parse_args()

    # PhishLLM
    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # set openai proxy

    # Xdriver
    sleep_time = 3; timeout_time = 60
    XDriver.set_headless()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)
    Logger.set_debug_on()

    os.makedirs('./field_study/results/', exist_ok=True)

    result_txt = './field_study/results/{}_phishllm.txt'.format(args.date)

    if not os.path.exists(result_txt):
        with open(result_txt, "w+") as f:
            f.write("folder" + "\t")
            f.write("phish_prediction" + "\t")
            f.write("target_prediction" + "\t")  # write top1 prediction only
            f.write("brand_recog_time" + "\t")
            f.write("crp_prediction_time" + "\t")
            f.write("crp_transition_time" + "\n")

    for ct, folder in tqdm(enumerate(os.listdir(args.folder))):
        if folder in [x.split('\t')[0] for x in open(result_txt, encoding='ISO-8859-1').readlines()]:
            continue
        # if folder not in ['g2.suupportfb-q.click']:
        #     continue

        info_path = os.path.join(args.folder, folder, 'info.txt')
        html_path = os.path.join(args.folder, folder, 'html.txt')
        shot_path = os.path.join(args.folder, folder, 'shot.png')
        predict_path = os.path.join(args.folder, folder, 'predict.png')
        if not os.path.exists(shot_path):
            continue

        try:
            if len(open(info_path, encoding='ISO-8859-1').read()) > 0:
                url = open(info_path, encoding='ISO-8859-1').read()
            else:
                url = 'https://' + folder
        except:
            url = 'https://' + folder

        _, reference_logo = phishintention_cls.predict_n_save_logo(shot_path)
        pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis = llm_cls.test(url, reference_logo,
                                                                                                        shot_path, html_path, driver,
                                                                                                        limit=1,
                                                                                                        # brand_recognition_do_validation=True
                                                                                                        )

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(folder + "\t")
                f.write(str(pred) + "\t")
                f.write(str(brand) + "\t")  # write top1 prediction only
                f.write(str(brand_recog_time) + "\t")
                f.write(str(crp_prediction_time) + "\t")
                f.write(str(crp_transition_time) + "\n")
            if plotvis and pred == 'phish':
                plotvis.save(predict_path)

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