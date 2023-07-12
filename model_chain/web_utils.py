import re
from xdriver.xutils.Logger import Logger
import time
from PIL import Image
import numpy as np
import io
import base64
import cv2

class WebUtil():
    home_page_heuristics = [
        ".*index.*\.htm",
        ".*login.*\.htm",
        ".*signin.*\.htm",
        ".*verif.*\.htm",
        ".*valid.*\.htm",
        ".*confir.*\.htm",
        "me.html"
    ]
    weaker_home_page_heuristics = [
        ".*index.*\.php",
        ".*login.*\.php",
        ".*signin.*\.php",
        ".*confir.*\.php",
        ".*verif.*\.php",
        "me.php"
    ]

    '''sort links by likelihood of being the index page'''

    def sort_files_lambda(self, x):
        LOGIN = "(log|sign)([^0-9a-zA-Z]|\s)*(in|on)|authenticat(e|ion)|/(my([^0-9a-zA-Z]|\s)*)?(user|account|profile|dashboard)"
        SIGNUP = "sign([^0-9a-zA-Z]|\s)*up|regist(er|ration)?|(create|new)([^0-9a-zA-Z]|\s)*(new([^0-9a-zA-Z]|\s)*)?(acc(ount)?|us(e)?r|prof(ile)?)|(forg(et|ot)|reset)([^0-9a-zA-Z]|\s)*((my|the)([^0-9a-zA-Z]|\s)*)?(acc(ount)?|us(e)?r|prof(ile)?|password)"
        SSO = "[^0-9a-zA-Z]+sso[^0-9a-zA-Z]+|oauth|openid"
        SUBMIT = "submit"
        VERIFY_VERBS = "verify|activate"
        AUTH = "%s|%s|%s|%s|%s|auth|(new|existing)([^0-9a-zA-Z]|\s)*(us(e)?r|acc(ount)?)|account|connect|profile|dashboard|next" % (
            LOGIN, SIGNUP, SSO, SUBMIT, VERIFY_VERBS)

        x = x.lower()
        if x.startswith('rename') or x.startswith('new') or x.startswith('hack'):
            return 6
        if any([True if re.search(rule, x, re.IGNORECASE) else False for rule in self.home_page_heuristics]):
            return 1
        elif x.endswith(".html") or x.endswith(".htm"):
            return 2
        elif any([True if re.search(rule, x, re.IGNORECASE) else False for rule in self.weaker_home_page_heuristics]):
            return 3
        elif re.search(AUTH, x, re.IGNORECASE):
            return 4
        elif x.startswith('_') or '~' in x or '@' in x or x.startswith(
                '.') or 'htaccess' in x or 'block' in x or 'anti' in x or x.lower == 'log/' or x.lower == 'off/':
            return 6
        else:
            return 5


    def page_error_checking(self, driver):
        source = driver.page_source()
        if source == "<html><head></head><body></body></html>":
            return False
        elif ("404 Not Found" in source
              or "The requested URL was not found on this server" in source
              or "Server at localhost Port" in source
              or "Not Found" in source
              or "Forbidden" in source
              or "no access to view" in source
              or "Bad request" in source
              or "Bad Gateway" in source
              or "Access denied" in source
        ):
            return False
        if source == "<html><head></head><body></body></html>":
            return False
        error = "(not found)|(no such file or directory)|(failed opening)|(refused to connect)|(invalid argument)|(undefined index)|(undefined property)|(undefined variable)|(syntax error)|(site error)|(parse error)"
        if re.search(error, source, re.IGNORECASE):
            return False
        return True



    def page_interaction_checking(self, driver):
        ct = 0
        hang = False
        while "Index of" in driver.get_page_text():
            links = driver.get_all_links_orig()

            links = [x for x in links if (not x[1].startswith('?')) and (not x[1].startswith('/')) and
                     (not x[1].endswith('png')) and (not x[1].endswith('jpg')) and (not x[1].endswith('txt')) and
                     (not x[1].endswith('.so'))]
            if len(links) == 0:
                hang = True
                break
            likelihood_sort = list(map(lambda x: self.sort_files_lambda(x[1]), links))
            if len(likelihood_sort) == 0:
                hang = True
                break
            sorted_index = np.argmin(likelihood_sort)
            sorted_link_by_likelihood = links[sorted_index]
            driver.click(sorted_link_by_likelihood[0])
            time.sleep(0.5)
            Logger.spit('After clicking URL={}'.format(driver.current_url()), debug=True)
            ct += 1
            if ct >= 10:
                hang = True
                break

        if hang:
            return False
        return self.page_error_checking(driver)

    '''check white screen'''
    def white_screen(self, shot_path):
        old_screenshot_img = Image.open(shot_path)
        old_screenshot_img = old_screenshot_img.convert("RGB")
        old_screenshot_img_arr = np.asarray(old_screenshot_img)
        old_screenshot_img_arr = np.flip(old_screenshot_img_arr, -1)  # RGB2BGR
        img = cv2.cvtColor(old_screenshot_img_arr, cv2.COLOR_BGR2GRAY)

        img_area = np.prod(img.shape)
        white_area = np.sum(img == 255)
        if white_area / img_area >= 0.99:  # skip white screenshots
            return True  # dirty
        return False


    def page_white_screen(self, driver, ts=1.0):
        old_screenshot_img = Image.open(io.BytesIO(base64.b64decode(driver.get_screenshot_encoding())))
        old_screenshot_img = old_screenshot_img.convert("RGB")
        old_screenshot_img_arr = np.asarray(old_screenshot_img)
        old_screenshot_img_arr = np.flip(old_screenshot_img_arr, -1)  # RGB2BGR
        img = cv2.cvtColor(old_screenshot_img_arr, cv2.COLOR_BGR2GRAY)

        img_area = np.prod(img.shape)
        white_area = np.sum(img == 255)
        if white_area / img_area >= ts:  # skip white screenshots
            return True  # dirty
        return False

