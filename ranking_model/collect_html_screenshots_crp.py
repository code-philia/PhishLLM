import os.path
import time

import selenium.common.exceptions
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
from xdriver.xutils.Logger import Logger
from xdriver.XDriver import XDriver
from xdriver.xutils.state.StateClass import StateClass
from xdriver.xutils.action.StateAction import StateAction
from tqdm import tqdm

if __name__ == "__main__":
    sleep_time = 3; timeout_time = 60
    phishintention_cls = PhishIntentionWrapper()
    XDriver.set_headless()
    Logger.set_debug_on()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you have to sleep sometime, otherwise the browser will keep crashing

    os.makedirs('./datasets/alexa_login_crp', exist_ok=True)

    ct = 0
    for folder in tqdm(os.listdir('./datasets/alexa_login')):
        ct += 1
        target = 'https://{}'.format(folder)
        if os.path.exists(os.path.join('./datasets/alexa_login_crp', folder, 'shot.png')):
            continue
        os.makedirs(os.path.join('./datasets/alexa_login_crp', folder), exist_ok=True)
        Logger.spit('Target URL = {}'.format(target),
                    debug=True,
                    caller_prefix=PhishIntentionWrapper._caller_prefix)
        try:
            driver.get(target, accept_cookie=True, click_popup=True)
            time.sleep(sleep_time)
        except Exception as e:
            Logger.spit('Error {} when getting the URL, exit..'.format(e),
                        warning=True,
                        caller_prefix=PhishIntentionWrapper._caller_prefix)
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(timeout_time / 2)
            driver.set_page_load_timeout(timeout_time)
            time.sleep(sleep_time)
            continue

        try:
            # redirect to CRP page if staying on a non-CRP
            sa = StateAction(driver, phishintention_cls)  # execute certain action
            _, _, current_url = sa.CRP_transition()
            with open(os.path.join('./datasets/alexa_login_crp', folder, 'info.txt'), "w") as f:
                f.write(driver.current_url())
            with open(os.path.join('./datasets/alexa_login_crp', folder, 'index.html'), "w", encoding='utf-8') as f:
                f.write(driver.page_source())
            driver.save_screenshot(os.path.join('./datasets/alexa_login_crp', folder, 'shot.png'))
            print(folder)
        except Exception as e:
            Logger.spit('Error {} when saving page source, exit..'.format(e),
                        warning=True,
                        caller_prefix=PhishIntentionWrapper._caller_prefix)

        # select one: another model
        if (ct + 1) % 100 == 0:
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(timeout_time / 2)
            driver.set_page_load_timeout(timeout_time)
            time.sleep(sleep_time)

    driver.quit()

