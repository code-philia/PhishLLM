import time
from xdriver.XDriver import XDriver
import os
from model_chain.web_utils import WebUtil
from xdriver.xutils.Logger import Logger


if __name__ == '__main__':

    # web_func = WebUtil()
    #
    # sleep_time = 3; timeout_time = 60
    # XDriver.set_headless()
    # driver = XDriver.boot(chrome=True)
    # driver.set_script_timeout(timeout_time/2)
    # driver.set_page_load_timeout(timeout_time)
    # time.sleep(sleep_time)  # fixme: you
    # Logger.set_debug_on()
    #
    # all_links = [x.strip().split(',')[-2] for x in open('./datasets/Brand_Labelled_130323.csv').readlines()[1:]]
    #
    # for ct, target in enumerate(all_links):
    #     hash = target.split('/')[3]
    #
    #     try:
    #         driver.get(target, click_popup=True, allow_redirections=False)
    #         time.sleep(5)
    #         Logger.spit(f'Target URL = {target}', caller_prefix=XDriver._caller_prefix, debug=True)
    #     except Exception as e:
    #         Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
    #         continue
    #
    #     try:
    #         error_free = web_func.page_error_checking(driver)
    #         if not error_free:
    #             Logger.spit('Error page or White page', caller_prefix=XDriver._caller_prefix, debug=True)
    #             continue
    #     except Exception as e:
    #         Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
    #         continue
    #
    #     try:
    #         page_text = driver.get_page_text()
    #     except Exception as e:
    #         Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
    #         continue
    #
    #     if "Index of" in page_text:
    #         try:
    #             # skip error URLs
    #             error_free = web_func.page_interaction_checking(driver)
    #             white_page = web_func.page_white_screen(driver, 1)
    #             if (error_free == False) or white_page:
    #                 Logger.spit('Error page or White page', caller_prefix=XDriver._caller_prefix, debug=True)
    #                 continue
    #             target = driver.current_url()
    #         except Exception as e:
    #             Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
    #             continue
    #
    #     if target.endswith('https/') or target.endswith('genWeb/'): ## problematic kits
    #         continue
    #
    #     # try:
    #     #     # save screenshot
    #     #     driver.save_screenshot(shot_path)
    #     # except Exception as e:
    #     #     Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, warning=True)
    #     #     continue
    #
    #     # try:
    #     #     # save HTML
    #     #     with open(html_path, 'w+', encoding='utf-8') as f:
    #     #         f.write(driver.page_source())
    #     # except Exception as e:
    #     #     Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
    #     #     pass
    #
    #     ########## write your functions here ##########
    #
    #     ########## ########## ########## ##########
    #
    # driver.quit() # do not forget to quit driver, otherwise the chrome process will not be killed
    pass




