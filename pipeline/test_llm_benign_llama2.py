from model_chain.test_llm_llama2 import *

if __name__ == '__main__':
    # load hyperparameters
    with open('./param_dict_llama2.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    PhishLLMLogger.set_debug_on()
    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLlama2(phishintention_cls,
                         param_dict=param_dict,
                         proxies={"http": "http://127.0.0.1:7890",
                                  "https": "http://127.0.0.1:7890",
                                  }
                         )
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    driver = CustomWebDriver.boot(proxy_server="http://127.0.0.1:7890")  # Using the proxy_url variable
    driver.set_script_timeout(timeout_time / 2)
    driver.set_page_load_timeout(timeout_time)

    link_list = [x.strip().split('\t')[0] for x in open('./datasets/alexa7k_wo_validation.txt').readlines()]
    root_folder = './datasets/alexa_login_test'
    result = './datasets/alexa7k_llama2.txt'
    os.makedirs(root_folder, exist_ok=True)

    for ct, target in enumerate(link_list):
        # if ct <= 5470:
        #     continue
        URL = 'https://{}'.format(target)
        target_folder = os.path.join(root_folder, target)
        os.makedirs(target_folder, exist_ok=True)

        if not os.path.exists(os.path.join(target_folder, 'shot.png')):
            print('Target URL = {}'.format(URL))
            try:
                driver.get(URL)
                time.sleep(sleep_time)
            except Exception as e:
                print('Error {} when getting the URL, exit..'.format(e))
                driver.quit()
                driver = CustomWebDriver.boot(proxy_server="http://127.0.0.1:7890")  # Using the proxy_url variable
                driver.set_script_timeout(timeout_time / 2)
                driver.set_page_load_timeout(timeout_time)
                continue

            try:
                with open(os.path.join(target_folder, 'index.html'), "w", encoding='utf-8') as f:
                    f.write(driver.page_source())
                driver.save_screenshot(os.path.join(target_folder, 'shot.png'))
            except Exception as e:
                print('Error {} when saving page source, exit..'.format(e))

        if os.path.exists(result) and target in open(result).read():
            continue

        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')

        if os.path.exists(shot_path):
            logo_box, reference_logo = llm_cls.detect_logo(shot_path)
            pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, _ = llm_cls.test(target,
                                                                                                      reference_logo,
                                                                                                      logo_box,
                                                                                                      shot_path,
                                                                                                      html_path,
                                                                                                      driver,
                                                                                                      )
            with open(result, 'a+') as f:
                f.write(target+'\t'+str(pred)+'\t'+str(brand)+'\t'+str(brand_recog_time)+'\t'+str(crp_prediction_time)+'\t'+str(crp_transition_time)+'\n')

        if (ct + 1) % 100 == 0:
            driver.quit()
            driver = CustomWebDriver.boot(proxy_server="http://127.0.0.1:7890")  # Using the proxy_url variable
            driver.set_script_timeout(timeout_time / 2)
            driver.set_page_load_timeout(timeout_time)

    driver.quit()






