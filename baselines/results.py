from llm.test_phishintention import *

if __name__ == '__main__':
    # results = open('./datasets/top-1m-phishintention.txt').readlines()
    # success = 0
    # failure = 0
    # total = 0
    # for line in results:
    #     orig_url, curr_url = line.strip().split('\t')
    #     curr_url = curr_url.lower()
    #     if 'login' in curr_url or 'signin' in curr_url or \
    #             'log-in' in curr_url or 'sign_in' in curr_url or \
    #             'log_in' in curr_url or 'sign-in' in curr_url:
    #         success += 1
    #     if curr_url == orig_url or abs(len(curr_url)-len(orig_url))<=2:
    #         failure += 1
    #     total += 1
    #
    # print(success, total, success/total)
    # print(failure, total, failure/total)

    '''debug'''

    sleep_time = 5;
    timeout_time = 30
    PhishIntentionWrapper._RETRIES = 1
    phishintention_cls = PhishIntentionWrapper()

    Logger.set_debug_on()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you have to sleep sometime, otherwise the browser will keep crashing

    target = 'https://iqiyi.com'

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


    '''Run login finder'''
    # HTML heuristic based login finder
    reach_crp, orig_url, current_url = phishintention_cls.crp_locator_keyword_heuristic_reimplement(
        driver=driver)
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

    driver.quit()
