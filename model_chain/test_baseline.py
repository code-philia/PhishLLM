
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
from xdriver.XDriver import XDriver
import time
import os
os.environ['https_proxy'] = "http://127.0.0.1:7890" # proxy
os.environ['http_proxy'] = "http://127.0.0.1:7890" # proxy

class TestBase():
    def __int__(self, phishintention_cls):
        self.phishintention_cls = phishintention_cls

    def test_phishpedia(self, URL, screenshot_path):
        start_time = time.time()
        phish_category, phish_target, plotvis, siamese_conf, time_breakdown, pred_boxes, pred_classes = \
            self.phishintention_cls.test_orig_phishpedia(URL, screenshot_path)
        phishpedia_runtime = time.time() - start_time

        return phish_category, phish_target, str(phishpedia_runtime)

    def test_phishintention(self, URL, screenshot_path, dynamic_enabled=True):

        start_time = time.time()
        XDriver.set_headless()
        ph_driver = XDriver.boot(chrome=True)
        time.sleep(3)
        ph_driver.set_page_load_timeout(30)
        ph_driver.set_script_timeout(60)
        phish_category, phish_target, plotvis, siamese_conf, dynamic, time_breakdown, pred_boxes, pred_classes = \
            self.phishintention_cls.test_orig_phishintention(URL, screenshot_path, ph_driver)
        ph_driver.quit()
        phishintention_runtime = time.time() - start_time

        return phish_category, phish_target, str(phishintention_runtime)


if __name__ == '__main__':
    phishintention_cls = PhishIntentionWrapper()
    base_cls = TestBase(phishintention_cls)

    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_wo_validation.txt').readlines()]
    print(len(all_folders))


