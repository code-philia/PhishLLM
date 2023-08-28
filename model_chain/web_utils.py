import re
from xdriver.xutils.Logger import Logger
import numpy as np
import cv2
import requests
import time
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
import base64
from webdriver_manager.chrome import ChromeDriverManager
from xdriver.XDriver import XDriver
from typing import *
from model_chain.logger_utils import PhishLLMLogger
from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper

class WebUtil():
    home_page_heuristics = [
        ".*ndex.*\.htm",
        ".*login.*\.htm",
        ".*signin.*\.htm",
        ".*verif.*\.htm",
        ".*valid.*\.htm",
        ".*confir.*\.htm",
        "me.html"
    ]
    weaker_home_page_heuristics = [
        ".*ndex.*\.php",
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
            time.sleep(2)
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


'''Validate the domain'''
def is_valid_domain(domain: str) -> bool:
    '''
        Check if the provided string is a valid domain
        :param domain:
        :return:
    '''
    # Regular expression to check if the string is a valid domain without spaces
    pattern = re.compile(
        r'^(?!-)'  # Cannot start with a hyphen
        r'(?!.*--)'  # Cannot have two consecutive hyphens
        r'(?!.*\.\.)'  # Cannot have two consecutive periods
        r'(?!.*\s)'  # Cannot contain any spaces
        r'[a-zA-Z0-9-]{1,63}'  # Valid characters are alphanumeric and hyphen
        r'(?:\.[a-zA-Z]{2,})+$'  # Ends with a valid top-level domain
    )
    it_is_a_domain = bool(pattern.fullmatch(domain))
    return it_is_a_domain

def is_alive_domain(domain: str, proxies: Optional[Dict]=None) -> bool:
    ct_limit = 0
    while ct_limit < 3:
        try:
            response = requests.get('https://' + domain, timeout=60, proxies=proxies)
            if response.status_code == 200:  # it is alive
                PhishLLMLogger.spit(f'Domain {domain} is valid and alive', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
                return True
            break
        except Exception as err:
            print(f'Error {err} when checking the aliveness of domain {domain}')
            ct_limit += 1
    PhishLLMLogger.spit(f'Domain {domain} is invalid or dead', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
    return False

'''Retrieve logo from a webpage'''
def url2logo(url, phishintention_cls):
    # Set up the driver (assuming ChromeDriver is in the current directory or PATH)
    options = webdriver.ChromeOptions()
    options.headless = True  # Run in headless mode
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    reference_logo = None
    try:
        driver.get(url)  # Visit the webpage
        time.sleep(2)
        screenshot_encoding = driver.get_screenshot_as_base64()
        screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
        logo_boxes = phishintention_cls.return_all_bboxes4type(screenshot_encoding, 'logo')
        if (logo_boxes is not None) and len(logo_boxes):
            logo_box = logo_boxes[0]  # get coordinate for logo
            x1, y1, x2, y2 = logo_box
            reference_logo = screenshot_img.crop((x1, y1, x2, y2))  # crop logo out
    except WebDriverException as e:
        print(f"Error accessing the webpage: {e}")
    except Exception as e:
        print(f"Failed to take screenshot: {e}")
    finally:
        driver.quit()  # Close the browser

    return reference_logo


'''Search for logo in Google Image'''
def query2image(query: str, SEARCH_ENGINE_API: str, SEARCH_ENGINE_ID: str, num: int=10, proxies: Optional[Dict]=None) -> List[str]:
    '''
        Retrieve the images from Google image search
        :param query:
        :param SEARCH_ENGINE_API:
        :param SEARCH_ENGINE_ID:
        :param num:
        :return:
    '''
    if len(query) == 0:
        return []

    URL = f"https://www.googleapis.com/customsearch/v1?key={SEARCH_ENGINE_API}&cx={SEARCH_ENGINE_ID}&q={query}&searchType=image&num={num}"
    while True:
        try:
            data = requests.get(URL, proxies=proxies).json()
            break
        except requests.exceptions.SSLError as e:
            print(e)
            time.sleep(1)

    if data.get('error', {}).get('code') == 429:
        raise RuntimeError("Google search exceeds quota limit")

    returned_urls = [item.get("image")["thumbnailLink"] for item in data.get("items", [])]

    return returned_urls


def download_image(url: str, proxies: Optional[Dict]=None) -> Optional[Image.Image]:
    '''
        Download images from given url (Google image context links)
        :param url:
        :return:
    '''
    try:
        response = requests.get(url, proxies=proxies, timeout=5)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            return img
    except requests.exceptions.Timeout:
        print("Request timed out after", 5, "seconds.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading image: {e}")

    return None

def get_images(image_urls: List[str], proxies: Optional[Dict]=None) -> List[Image.Image]:
    '''
        Run download_image in multiple threads
        :param image_urls:
        :return:
    '''
    images = []
    if len(image_urls) > 0:
        with ThreadPoolExecutor(max_workers=len(image_urls)) as executor:
            futures = [executor.submit(download_image, url, proxies) for url in image_urls]
            for future in futures:
                img = future.result()
                if img:
                    images.append(img)

    return images

'''Webdriver element clicking'''
def page_transition(driver: XDriver, dom: str, save_html_path: str, save_shot_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    '''
        Click an element and save the updated screenshot and HTML
        :param driver:
        :param dom:
        :param save_html_path:
        :param save_shot_path:
        :return:
    '''
    try:
        element = driver.find_elements_by_xpath(dom)
        if element:
            try:
                driver.execute_script("arguments[0].style.border='3px solid red'", element[0]) # hightlight the element to click
            except:
                pass
            driver.move_to_element(element[0])
            driver.click(element[0])
            time.sleep(7)  # fixme: must allow some loading time here, dynapd is slow
        current_url = driver.current_url()
    except Exception as e:
        PhishLLMLogger.spit('Exception {} when clicking the login button'.format(e), caller_prefix=PhishLLMLogger._caller_prefix, warning=True)
        return None, None, None

    try:
        driver.save_screenshot(save_shot_path)
        PhishLLMLogger.spit('CRP transition is successful! New screenshot has been saved', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        with open(save_html_path, "w", encoding='utf-8') as f:
            f.write(driver.page_source())
        return current_url, save_html_path, save_shot_path
    except Exception as e:
        PhishLLMLogger.spit('Exception {} when saving the new screenshot'.format(e), caller_prefix=PhishLLMLogger._caller_prefix, warning=True)
        return None, None, None


def get_screenshot_elements(phishintention_cls: PhishIntentionWrapper, driver: XDriver) -> List[int]:
    pred_boxes, pred_classes = phishintention_cls.return_all_bboxes(driver.get_screenshot_encoding())
    if pred_boxes is None:
        screenshot_elements = []
    else:
        screenshot_elements = pred_classes.numpy().tolist()
    return screenshot_elements

def has_page_content_changed(phishintention_cls: PhishIntentionWrapper, driver: XDriver, prev_screenshot_elements: List[int]) -> bool:
    screenshot_elements = get_screenshot_elements(phishintention_cls, driver)
    bincount_prev_elements = np.bincount(prev_screenshot_elements)
    bincount_curr_elements = np.bincount(screenshot_elements)
    set_of_elements = min(len(bincount_prev_elements), len(bincount_curr_elements))
    screenshot_ele_change_ts = np.sum(bincount_prev_elements) // 2 # half the different UI elements distribution has changed

    if np.sum(np.abs(bincount_curr_elements[:set_of_elements] - bincount_prev_elements[:set_of_elements])) > screenshot_ele_change_ts:
        PhishLLMLogger.spit(f"Webpage content has changed", caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        return True
    else:
        PhishLLMLogger.spit(f"Webpage content didn't change", caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        return False

