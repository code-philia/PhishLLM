import os.path
import time
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
from xdriver.xutils.Logger import Logger
from xdriver.XDriver import XDriver
from tqdm import tqdm

if __name__ == "__main__":
    sleep_time = 5; timeout_time = 30
    PhishIntentionWrapper._RETRIES = 1
    phishintention_cls = PhishIntentionWrapper()

    XDriver.set_headless()
    Logger.set_debug_on()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you have to sleep sometime, otherwise the browser will keep crashing

    alexa_urls = [x.strip().split(',')[1] for x in open('./datasets/top-1m.csv').readlines()]
    ct = 0
    for target in tqdm(alexa_urls):
        ct += 1
        if ct <= 1967:
            continue
        if os.path.exists('./datasets/top-1m-phishintention.txt') and \
                'https://{}'.format(target) in [x.strip().split('\t')[0] \
                           for x in open('./datasets/top-1m-phishintention.txt').readlines()]:
            continue
        Logger.set_logfile('./datasets/phishintention_logs/{}.log'.format(target))
        target = 'https://{}'.format(target)

        Logger.spit('Target URL = {}'.format(target),
                    debug=True,
                    caller_prefix=PhishIntentionWrapper._caller_prefix)
        try:
            driver.get(target, accept_cookie=True, click_popup=True)
        except Exception as e:
            Logger.spit('Error {} when getting the URL, exit..'.format(e),
                        warning=True,
                        caller_prefix=PhishIntentionWrapper._caller_prefix)
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(timeout_time)
            driver.set_page_load_timeout(timeout_time)
            time.sleep(sleep_time)
            continue

        '''Run login finder'''
        try:
            # HTML heuristic based login finder
            reach_crp, orig_url, current_url = phishintention_cls.crp_locator_keyword_heuristic_reimplement(driver=driver)
            Logger.spit(
                'After HTML keyword finder, reach a CRP page ? {}, \n Original URL = {}, \n Current URL = {}'.format(
                    reach_crp, orig_url, current_url),
                debug=True,
                caller_prefix=PhishIntentionWrapper._caller_prefix)

            # If HTML login finder did not find CRP, call CV-based login finder
            if not reach_crp:
                reach_crp, orig_url, current_url = phishintention_cls.crp_locator_cv_reimplement(driver=driver)
                Logger.spit(
                    'After CV login finder, reach a CRP page ? {}, \n Original URL = {}, \n Current URL = {}'.format(
                        reach_crp, orig_url, current_url),
                    debug=True,
                    caller_prefix=PhishIntentionWrapper._caller_prefix)
        except Exception as e:
            Logger.spit('Error {} when getting the URL, exit..'.format(e),
                        warning=True,
                        caller_prefix=PhishIntentionWrapper._caller_prefix)
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(timeout_time)
            driver.set_page_load_timeout(timeout_time)
            time.sleep(sleep_time)
            continue

        with open('./datasets/top-1m-phishintention.txt', 'a+') as f:
            f.write(target + '\t' + current_url + '\n')

        if ct % 100 == 0:
            driver.quit()
            XDriver.set_headless()
            driver = XDriver.boot(chrome=True)
            driver.set_script_timeout(timeout_time)
            driver.set_page_load_timeout(timeout_time)
            time.sleep(sleep_time)

    driver.quit()
