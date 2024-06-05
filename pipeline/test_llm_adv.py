import os.path
import shutil
from pipeline.test_llm import *
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

if __name__ == '__main__':

    # load hyperparameters
    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)

    phishintention_cls = PhishIntentionWrapper()
    llm_cls = TestLLM(phishintention_cls, param_dict=param_dict,
                      proxies={"http": "http://127.0.0.1:7890",
                               "https": "http://127.0.0.1:7890",
                               }
                      )
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    driver = CustomWebDriver.boot(proxy_server="http://127.0.0.1:7890")  # Using the proxy_url variable
    driver.set_script_timeout(timeout_time / 2)
    driver.set_page_load_timeout(timeout_time)

    all_links = [x.strip().split(',')[-2] for x in open('./datasets/Brand_Labelled_130323.csv').readlines()[1:]]

    root_folder = './datasets/dynapd_adv'
    result = './datasets/dynapd_llm_adv.txt'
    os.makedirs(root_folder, exist_ok=True)

    for ct, target in enumerate(all_links):
        # if ct <= 5470:
        #     continue
        hash = target.split('/')[3]
        target_folder = os.path.join(root_folder, hash)
        os.makedirs(target_folder, exist_ok=True)
        if os.path.exists(result) and hash in open(result).read():
            continue
        if not os.path.exists(os.path.join('./datasets/dynapd', hash)):
            continue

        try:
            driver.get(target, click_popup=True, allow_redirections=False)
            time.sleep(5)
            print(f'Target URL = {target}')
        except Exception as e:
            print('Exception {}'.format(e))
            shutil.rmtree(target_folder)
            continue

        try:
            page_text = driver.get_page_text()
        except Exception as e:
            print('Exception {}'.format(e))
            shutil.rmtree(target_folder)
            continue

        try:
            error_free = web_func.page_error_checking(driver)
            if not error_free:
                print('Error page or White page')
                shutil.rmtree(target_folder)
                continue
        except Exception as e:
            print('Exception {}'.format(e))
            shutil.rmtree(target_folder)
            continue

        if "Index of" in page_text:
            try:
                # skip error URLs
                error_free = web_func.page_interaction_checking(driver)
                white_page = web_func.page_white_screen(driver, 1)
                if (error_free == False) or white_page:
                    print('Error page or White page')
                    shutil.rmtree(target_folder)
                    continue
                target = driver.current_url()
            except Exception as e:
                print('Exception {}'.format(e))
                shutil.rmtree(target_folder)
                continue

        if target.endswith('https/') or target.endswith('genWeb/'):
            shutil.rmtree(target_folder)
            continue

        try:
            driver.obfuscate_page()
            shot_path = os.path.join(target_folder, 'shot.png')
            html_path = os.path.join(target_folder, 'index.html')
            # take screenshots
            screenshot_encoding = driver.get_screenshot_encoding()
            screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
            screenshot_img.save(shot_path)
        except Exception as e:
            print('Exception {}'.format(e))
            shutil.rmtree(target_folder)
            continue

        try:
            # record HTML
            with open(html_path, 'w+', encoding='utf-8') as f:
                f.write(driver.page_source())
        except Exception as e:
            print('Exception {}'.format(e))
            pass

        if os.path.exists(shot_path):
            pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, _ = llm_cls.test(target, None, shot_path, html_path, driver)
            with open(result, 'a+') as f:
                f.write(hash+'\t'+pred+'\t'+brand+'\t'+str(brand_recog_time)+'\t'+str(crp_prediction_time)+'\t'+str(crp_transition_time)+'\n')

    driver.quit()






