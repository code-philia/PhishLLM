import os.path
import time

import selenium.common.exceptions
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
from xdriver.xutils.Logger import Logger
from xdriver.XDriver import XDriver
from tqdm import tqdm
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By

if __name__ == "__main__":
    sleep_time = 5; timeout_time = 60
    PhishIntentionWrapper._RETRIES = 1
    phishintention_cls = PhishIntentionWrapper()

    XDriver.set_headless()
    Logger.set_debug_on()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you have to sleep sometime, otherwise the browser will keep crashing

    alexa_urls = [x.strip().split(',')[1] for x in open('./datasets/top-1m.csv').readlines()]
    ct = 0
    annotations = './datasets/alexa_login.txt'
    os.makedirs('./datasets/alexa_login', exist_ok=True)
    for target in tqdm(alexa_urls[:5000]):
        ct += 1
        if ct <= 4860:
            continue
        target = 'https://{}'.format(target)
        # target = 'https://kissanime.com.ru'
        if os.path.exists(annotations) and target in open(annotations).read():
            continue

        Logger.spit('Target URL = {}'.format(target),
                    debug=True,
                    caller_prefix=PhishIntentionWrapper._caller_prefix)
        try:
            driver.get(target, accept_cookie=True, click_popup=True)
            time.sleep(3)
        except Exception as e:
            Logger.spit('Error {} when getting the URL, exit..'.format(e),
                        warning=True,
                        caller_prefix=PhishIntentionWrapper._caller_prefix)
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(timeout_time/2)
            driver.set_page_load_timeout(timeout_time)
            time.sleep(sleep_time)
            continue

        try:
            (btns, btns_dom),  \
                (links, links_dom), \
                (images, images_dom), \
                (others, others_dom) = driver.get_all_clickable_elements()
        except Exception as e:
            print(e)
            continue
        all_clickable = btns + links+ images + others
        all_clickable_dom = btns_dom + links_dom + images_dom + others_dom
        # save the element screenshot
        for it in range(min(300, len(all_clickable))):
            save_path = './datasets/alexa_login/{}/{}.png'.format(target.split('https://')[1], it)
            try:
                driver.scroll_to_top()
                x1, y1, x2, y2 = driver.get_location(all_clickable[it])
            except selenium.common.exceptions.TimeoutException as e:
                actions = ActionChains(driver)
                actions.send_keys(Keys.CONTROL + "l")
                actions.send_keys(target)
                actions.send_keys(Keys.ENTER)
                actions.perform()
            except Exception as e:
                continue

            if x2 - x1 <= 0 or y2 - y1 <= 0 or \
                    y2 >= driver.get_window_size()['height']//2: # invisible or at the bottom
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                all_clickable[it].screenshot(save_path)
            except Exception as e:
                actions = ActionChains(driver)
                actions.send_keys(Keys.CONTROL + "l")
                actions.send_keys(target)
                actions.send_keys(Keys.ENTER)
                actions.perform()
                Logger.spit(e, debug=True)
                continue
            with open(annotations, 'a+') as f:
                f.write(target+'\t'+all_clickable_dom[it]+'\t'+save_path+'\n')

        # the dom, the attributes? dom tree? the image => representation

        # align with the task (login but not signup)

        # rank

        # select one: another model
        if (ct+1) % 100 == 0:
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(timeout_time/2)
            driver.set_page_load_timeout(timeout_time)
            time.sleep(sleep_time)


    driver.quit()