from tldextract import tldextract
import numpy as np
import re
from typing import *
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
from xdriver.xutils.Logger import Logger, TxtColors
import logging
import os
import re

'''LLM prompt'''
def question_template_prediction(html_text):
    return \
        {
            "role": "user",
            "content": f"Given the HTML webpage text: <start>{html_text}<end>, \n Question: A. This is a credential-requiring page. B. This is not a credential-requiring page. \n Answer: "
        }


def question_template_brand(html_text):
    return \
        {
            "role": "user",
            "content": f"Given the HTML webpage text: <start>{html_text}<end>, Question: What is the brand's domain? Answer: "
        }


def question_template_caption(logo_caption, logo_ocr):
    return \
        {
            "role": "user",
            "content": f"Given the following description on the brand's logo: '{logo_caption}', and the logo's OCR text: '{logo_ocr}', Question: What is the brand's domain? Answer: "
        }

'''Bbox utilities'''
def compute_overlap_areas_between_lists(bboxes1, bboxes2):
    # Convert bboxes lists to 3D arrays
    bboxes1 = np.array(bboxes1)[:, np.newaxis, :]
    bboxes2 = np.array(bboxes2)

    # Compute overlap for x and y axes separately
    overlap_x = np.maximum(0, np.minimum(bboxes1[:, :, 2], bboxes2[:, 2]) - np.maximum(bboxes1[:, :, 0], bboxes2[:, 0]))
    overlap_y = np.maximum(0, np.minimum(bboxes1[:, :, 3], bboxes2[:, 3]) - np.maximum(bboxes1[:, :, 1], bboxes2[:, 1]))

    # Compute overlapping areas for each pair
    overlap_areas = overlap_x * overlap_y
    return overlap_areas

def expand_bbox(bbox, image_width, image_height, expand_ratio):
    # Extract the coordinates
    x1, y1, x2, y2 = bbox

    # Calculate the center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate new width and height
    new_width = (x2 - x1) * expand_ratio
    new_height = (y2 - y1) * expand_ratio

    # Determine new coordinates
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # Ensure coordinates are legitimate
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]

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
            response = requests.get('https://' + domain, timeout=10, proxies=proxies)
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

'''Logging Utils'''
class PhishLLMLogger():

    _caller_prefix = "PhishLLMLogger"
    _verbose = True
    _logfile = None
    _debug = False # Off by default
    _warning = True

    @classmethod
    def set_verbose(cls, verbose):
        cls._verbose = verbose

    @classmethod
    def set_logfile(cls, logfile):
        # if os.path.isfile(logfile):
        #     os.remove(logfile)  # Remove the existing log file
        PhishLLMLogger._logfile = logfile

    @classmethod
    def unset_logfile(cls):
        PhishLLMLogger.set_logfile(None)

    @classmethod
    def set_debug_on(cls):
        PhishLLMLogger._debug = True

    @classmethod
    def set_debug_off(cls): # Call if need to turn debug messages off
        PhishLLMLogger._debug = False

    @classmethod
    def set_warning_on(cls):
        PhishLLMLogger._warning = True

    @classmethod
    def set_warning_off(cls): # Call if need to turn warnings off
        PhishLLMLogger._warning = False

    @classmethod
    def spit(cls, msg, warning=False, debug=False, error=False, exception=False, caller_prefix=""):
        logging.basicConfig(level=logging.DEBUG if PhishLLMLogger._debug else logging.WARNING)
        caller_prefix = f"[{caller_prefix}]" if caller_prefix else ""
        prefix = "[FATAL]" if error else "[DEBUG]" if debug else "[WARNING]" if warning else "[EXCEPTION]" if exception else ""
        logger = logging.getLogger("custom_logger")  # Choose an appropriate logger name
        if PhishLLMLogger._logfile:
            log_msg = re.sub(r"\033\[\d+m", "", msg)
            log_handler = logging.FileHandler(PhishLLMLogger._logfile, mode='a')
            log_formatter = logging.Formatter('%(message)s')
            log_handler.setFormatter(log_formatter)
            logger.addHandler(log_handler)
            logger.propagate = False
            logger.setLevel(logging.DEBUG if PhishLLMLogger._debug else logging.WARNING)
            logger.debug("%s%s %s" % (caller_prefix, prefix, log_msg))
            logger.removeHandler(log_handler)
        else:
            if PhishLLMLogger._verbose:
                txtcolor = TxtColors.FATAL if error else TxtColors.DEBUG if debug else TxtColors.WARNING if warning else "[EXCEPTION]" if exception else TxtColors.OK
                # if not debug or Logger._debug:
                if (not debug and not warning) or (debug and PhishLLMLogger._debug) or (warning and PhishLLMLogger._warning):
                    print("%s%s%s %s" % (txtcolor, caller_prefix, prefix, msg))
