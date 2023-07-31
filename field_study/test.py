from model_chain.test_llm import *
import argparse
from datetime import datetime
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()

    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    XDriver.set_headless()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you

    os.makedirs('./field_study/results/', exist_ok=True)
    if args.verbose:
        Logger.set_debug_on()

    today = datetime.today()
    today_date = today.strftime("%Y-%m-%d")
    result_txt = './field_study/results/{}_phishllm.txt'.format(today_date)

    if not os.path.exists(result_txt):
        with open(result_txt, "w+") as f:
            f.write("folder" + "\t")
            f.write("url" + "\t")
            f.write("phish_prediction" + "\t")
            f.write("target_prediction" + "\t")  # write top1 prediction only
            f.write("brand_recog_time" + "\t")
            f.write("crp_prediction_time" + "\t")
            f.write("crp_transition_time" + "\n")

    for ct, folder in tqdm(enumerate(os.listdir(args.folder))):
        if folder in [x.split('\t')[0] for x in open(result_txt, encoding='ISO-8859-1').readlines()]:
            continue

        info_path = os.path.join(args.folder, folder, 'info.txt')
        shot_path = os.path.join(args.folder, folder, 'shot.png')
        visualized_ocr_path = os.path.join(args.folder, folder, 'ocr.png')
        if not os.path.exists(shot_path):
            continue

        try:
            if len(open(info_path, encoding='ISO-8859-1').read()) > 0:
                url = open(info_path, encoding='ISO-8859-1').read()
            else:
                url = 'https://' + folder
        except:
            url = 'https://' + folder

        pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis = llm_cls.test(target, shot_path, html_path, driver)
        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(folder + "\t")
                f.write(str(pred) + "\t")
                f.write(str(brand) + "\t")  # write top1 prediction only
                f.write(str(brand_recog_time) + "\t")
                f.write(str(crp_prediction_time) + "\t")
                f.write(str(crp_transition_time) + "\n")
            cv2.imwrite(os.path.join(args.folder, folder, "predict_llm.png"), plotvis)

        except UnicodeEncodeError:
            continue

    driver.quit()