from model_chain.test_llm import *
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key2.txt').read()

if __name__ == '__main__':
    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    phishintention_cls = PhishIntentionWrapper()
    lm_cls = TestLLM(phishintention_cls,
                     param_dict=param_dict,
                     proxies={"http": "http://127.0.0.1:7890",
                              "https": "http://127.0.0.1:7890",
                              })
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    XDriver.set_headless()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you
    Logger.set_debug_on()

    root_folder = './datasets/alexa_login_test'
    result = './datasets/alexa7k_wo_validation.txt'

    for ct, target in enumerate(os.listdir(root_folder)):
        # if ct <= 5470:
        #     continue
        URL = 'https://{}'.format(target)

        target_folder = os.path.join(root_folder, target)
        if os.path.exists(result) and target in open(result).read():
            continue

        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')

        if os.path.exists(shot_path):
            logo_box, reference_logo = llm_cls.detect_logo(shot_path)
            pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, _ = llm_cls.test(URL,
                                                                                                    reference_logo,
                                                                                                    logo_box,
                                                                                                    shot_path,
                                                                                                    html_path,
                                                                                                    driver,
                                                                                                    brand_recognition_do_validation=True
                                                                                                    )
            with open(result, 'a+') as f:
                f.write(target+'\t'+str(pred)+'\t'+str(brand)+'\t'+str(brand_recog_time)+'\t'+str(crp_prediction_time)+'\t'+str(crp_transition_time)+'\n')

    driver.quit()






