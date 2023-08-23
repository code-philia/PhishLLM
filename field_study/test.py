from model_chain.test_llm import *
import argparse
from datetime import datetime
import cv2
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="./datasets/field_study/2023-08-19/")
    parser.add_argument("--date", default="2023-08-19", help="%Y-%m-%d")
    args = parser.parse_args()

    # PhishLLM
    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls,
                      proxies={ "http": "http://127.0.0.1:7890",
                                 "https": "http://127.0.0.1:7890",
                     })
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
        # if ct <= 2984:
        #     continue
        if folder in [x.split('\t')[0] for x in open(result_txt, encoding='ISO-8859-1').readlines()]:
            continue
        # if folder not in [
        #                    'systemsblog.blog.test.wp.kolaybet.org',
        #                 ]:
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

        if url.startswith('cpanel'): # skip cpanel hosting
            continue

        logo_box, reference_logo = llm_cls.detect_logo(shot_path)
        pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, plotvis = llm_cls.test(url, reference_logo, logo_box,
                                                                                                        shot_path, html_path, driver,
                                                                                                        limit=1,
                                                                                                        brand_recognition_do_validation=True
                                                                                                        )

        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                f.write(folder + "\t")
                f.write(str(pred) + "\t")
                f.write(str(brand) + "\t")  # write top1 prediction only
                f.write(str(brand_recog_time) + "\t")
                f.write(str(crp_prediction_time) + "\t")
                f.write(str(crp_transition_time) + "\n")
            if pred == 'phish':
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
    # 482.91236.top, requests unionpay.com doesnt work, need to request cn.unionpay.com
    # device-54ddd6f1-5dc6-4480-8809-c75e99811e0d.remotewd.com: no text on the webpage that indicate the brand is bouyguestelecom.fr, also the phishing is using old version of logo
    # cicd-13365-azure-devops-pipeline-portal-core.australiaeast.azurecontainer.io: phishintention didnt report the logo from the original webpage, therefore cannot do comparison

    '''GPT problem'''
    # demo.nuxproservices.com: no prediction from gpt
    # cleyrop.cloud-iam.com: GPT prediction is observationtourisme.fr, but the gt is france-tourisme-observation.fr

    '''Google Image search problem'''
    # kbr.appwork.info, didnt pass logo validation, because of the inaccuracies of Google Image Search API, it returns the logos for enodiatherapies.com, however we are interested in the enodia.com

    '''OCR problem'''
    # paperless.fungamers-online.de: OCR prediction has typo: 'perless -ng Please sign in. Username Password Sign in'. While using Google OCR can solve the problem 'Paperless Username Password Please sign in. Sign in'
    # dashboard.nm.165-227-40-76.nip.io: OCR miss information 'N MAKER Home ! Create an Admin Password* Password Confirmation* CREATE ADMIN'. With google ocr '▬▬·目N- Username* Password* ETMAKER Create an Admin Password Confirmation* Home CREATE ADMIN'
