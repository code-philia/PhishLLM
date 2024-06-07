import re
import numpy as np
import cv2
import requests
import time
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor
import base64
from webdriver_manager.chrome import ChromeDriverManager
from typing import *
from scripts.utils.logger_utils import PhishLLMLogger
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import *
from seleniumwire.webdriver import ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from scripts.utils.PhishIntentionWrapper import PhishIntentionWrapper
import json
from unidecode import unidecode
import urllib.parse

def lower(text):
	alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝ'
	return "translate(%s, '%s', '%s')" % (text, alphabet, alphabet.lower())

def replace_nbsp(text, by=' '):
	return "translate(%s, '\u00a0', %r)" % (text, by)

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
            print('After clicking URL={}'.format(driver.current_url()))
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

class CustomWebDriver(webdriver.Chrome):
    _MAX_RETRIES = 3
    _last_url = 'https://google.com'
    _forbidden_suffixes = r"\.(mp3|wav|wma|ogg|mkv|zip|tar|xz|rar|z|deb|bin|iso|csv|tsv|dat|txt|css|log|sql|xml|sql|mdb|apk|bat|bin|exe|jar|wsf|fnt|fon|otf|ttf|ai|bmp|gif|ico|jp(e)?g|png|ps|psd|svg|tif|tiff|cer|rss|key|odp|pps|ppt|pptx|c|class|cpp|cs|h|java|sh|swift|vb|odf|xlr|xls|xlsx|bak|cab|cfg|cpl|cur|dll|dmp|drv|icns|ini|lnk|msi|sys|tmp|3g2|3gp|avi|flv|h264|m4v|mov|mp4|mp(e)?g|rm|swf|vob|wmv|doc(x)?|odt|rtf|tex|txt|wks|wps|wpd)$"

    def __init__(self, proxy_server=None, *args, **kwargs):
        chrome_options = ChromeOptions()
        chrome_options.binary_location = "/usr/bin/google-chrome"
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

        prefs = {
            "download.default_directory": './trash',
            "download.prompt_for_download": False,  # To disable the download prompt and download automatically
            "download_restrictions": 3  # Attempt to restrict all downloads
        }
        chrome_options.add_experimental_option("prefs", prefs)

        self._proxy_server = proxy_server
        if proxy_server:
            chrome_options.add_argument(f"--proxy-server={proxy_server}")

        chrome_caps = DesiredCapabilities.CHROME
        chrome_caps['acceptSslCerts'] = True
        chrome_caps['acceptInsecureCerts'] = True

        super().__init__("./datasets/chromedriver",
                         chrome_options=chrome_options,
                         desired_capabilities=chrome_caps)

        self._RETRIES = kwargs.get("retries", {})
        self._REFS = kwargs.get("refs", {})
        self._recoverable_crashes = ["chrome not reachable", "page crash",
                                     "cannot determine loading status",
                                     "Message: unknown error"]
        with open('./scripts/utils/web_utils/web_utils_scripts.js', 'r') as f:
            js_content = f.read()
        self.add_script(js_content)

    def send(self, cmd, params={}):
        resource = "/session/%s/chromium/send_command_and_get_result" % self.session_id
        url = self.command_executor._url + resource
        body = json.dumps({'cmd': cmd, 'params': params})
        response = self.command_executor._request('POST', url, body)

    def add_script(self, script):
        self.send(cmd="Page.addScriptToEvaluateOnNewDocument", params={"source": script})

    @classmethod
    def boot(cls, **kwargs):
        return cls(**kwargs)

    def reboot(self):
        self.quit()
        new_instance = CustomWebDriver.boot(proxy_server=self._proxy_server)
        self.__dict__.update(new_instance.__dict__)

    '''Exception handling'''
    @staticmethod
    def stringify_exception(e, strip=False):
        exception_str = ""
        try:
            exception_str = str(e)
            if strip:  # Only return first line. Useful for webdriver exceptions
                exception_str = exception_str.split("\n")[0]
        except Exception as e:
            exception_str = "Could not stringify exception"
        return exception_str

    def enter_retry(self, method, max_retries=10):
        retries, max_retries = self._RETRIES.get(method, [0, max_retries])  # Necessary so nested `_invokes` of the same method won't reset the retry counter
        self._RETRIES[method] = [retries, max_retries]
        if retries <= max_retries:
            return True
        return False

    def exit_retry(self, method):
        self._RETRIES.pop(method, None)  # Only need to pop the method name

    def _invoke_exception_handler(self, handler, *args, **kwargs):
        try:
            return handler(*args, **kwargs)
        except UnexpectedAlertPresentException:
            self._invoke_exception_handler(self._UnexpectedAlertPresentException_handler)
        except NoAlertPresentException:  # This was raised during the _UnexpectedAlertPresentException_handler call.
            self.execute_script("window.alert = null;")
        except TimeoutException:
            return False

    def _UnexpectedAlertPresentException_handler(self):
        alert = self.switch_to.alert
        alert.dismiss()
        self.execute_script("window.alert = null;")  # Try to prevent the page from popping any more alerts
        self.switch_to.default_content()

    def _StaleElementReference_handler(self, webelement):
        element_ref = id(webelement)
        if element_ref not in self._REFS:
            return False
        method, args, kwargs = self._REFS[element_ref]
        kwargs["timeout"] = 3  # add some max timeout
        new_element = method(*args, **kwargs)  # refetch element
        if not new_element:  # The element is not in the DOM, it's not just stale
            return False

        webelement.__dict__.update(new_element.__dict__)  # Transparently update old webelement's reference
        return True

    def _TimeoutException_handler(self):
        print("Handling browser crash. Will try to restore webdriver.")
        self.reboot()  # We don't want to clear the _REFS, delete the profile or the proxy config
        return True

    def _invoke(self, method, *args, **kwargs):
        original_kwargs = dict(kwargs)  # In case we re-_invoke it, we need the original kwargs
        ex = None
        ret_val = None # used to record whether the run is successful or not

        try:
            if kwargs.pop("retry", True):  # By default, retry all methods if possible, otherwise explicitly requested
                self.enter_retry(method, max_retries=kwargs.pop("max_retries", self._MAX_RETRIES))
            web_element = kwargs.pop("webelement", None)
            ret_val = method(*args, **kwargs)
            if method == super(CustomWebDriver, self).get:
                ret_val = True  # Need to explicitly set the ret value for WebDriver's get
            self.exit_retry(method)  # The operation completed, no need to keep the retry counter
            return ret_val
        except UnexpectedAlertPresentException as ex:
            self._invoke_exception_handler(self._UnexpectedAlertPresentException_handler)
            if method == super(CustomWebDriver, self).get:  # Nothing more to do for a `get`
                ret_val = True
        except (InvalidSwitchToTargetException, NoSuchFrameException, NoSuchWindowException) as ex:
            if len(self.window_handles) == 0:  # If no windows remain for some reason, raise it
                raise
            self.switch_to.default_content()  # Return to the default handle
            ret_val = False
        except (InvalidSelectorException,
                InvalidElementStateException,
                ElementNotSelectableException,
                ElementNotVisibleException,
                MoveTargetOutOfBoundsException,
                JavascriptException) as ex:
            ret_val = False
        except (StaleElementReferenceException, NoSuchElementException) as ex:
            # Check _REFS for given WebElement.
            if not self._invoke_exception_handler(self._StaleElementReference_handler, web_element):
                raise
            ret_val = False
        except (TimeoutException, WebDriverException, ErrorInResponseException, RemoteDriverServerException,
                InvalidCookieDomainException, UnableToSetCookieException, ImeNotAvailableException,
                ImeActivationFailedException) as ex:
            str_ex = self.stringify_exception(ex)
            if ex is TimeoutException or any([crash in str_ex for crash in self._recoverable_crashes]):
                # Reboot browser, maintain state and retry the operation
                if not self._invoke_exception_handler(self._TimeoutException_handler):
                    raise
                if method != super(CustomWebDriver, self).get:
                    self.get(self._last_url)  # If it was `get`, it will be retried later on. For anything else, we need to manually go back to the last known URL
                ret_val = False
            else:
                raise

        retries, max_retries = self._RETRIES.get(method, [None, None])
        # If we are not in retry mode OR if the retries have exceeded the threshold, either return a default value (if set) or raise the exception to the caller
        if method not in self._RETRIES or retries >= max_retries:
            self.exit_retry(method)
            if ret_val != None:  # If a return value has been set, return it instead of raising the exception
                return ret_val
            raise  # These are considered fatal

        # About to re-invoke method. Increment retry counter
        self._RETRIES[method][0] += 1
        print("Retrying for the {}th time...".format(self._RETRIES[method][0]))
        return self._invoke(method, *args, **original_kwargs)

    def get_screenshot_encoding(self):
        return self._invoke(super(CustomWebDriver, self).get_screenshot_as_base64)

    def get_page_text(self):
        try:
            body = self.find_element_by_tag_name('html').text
            return body
        except:
            return ''

    def page_source(self):
        return self._invoke(self._page_source)

    def _page_source(self):
        return super(CustomWebDriver, self).page_source

    def obfuscate_page(self):
        self._invoke(self.execute_script, """
                  var script = document.createElement('script');
                  script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/0.5.0-beta4/html2canvas.min.js';
                  document.head.appendChild(script);
                """)
        time.sleep(1)
        return self._invoke(self.execute_script, """obfuscate_button();""")

    def current_url(self):
        return self._invoke(self._current_url)

    def _current_url(self):
        return super(CustomWebDriver, self).current_url

    def switch_to_window(self, window_handle):
        return self._invoke(self._switch_to_window, window_handle)

    def _switch_to_window(self, window_handle):
        # Default `switch_to.window` hangs in case there is an open alert, so we need a dummy op to trigger the alert handling
        self.execute_script("return 2;")
        self.switch_to.window(window_handle)
        return True

    '''Find elements'''

    def _webelement_find_element_by(self, element, by=By.ID, value=None):
        return element.find_element(by=by, value=value)

    def _webelement_find_elements_by(self, element, by=By.ID, value=None, *args, **kwargs):
        return element.find_elements(by=by, value=value, *args, **kwargs)

    def find_element(self, by=By.ID, value=None, timeout=0, visible=False, webelement=None):
        if timeout == 0 and visible is False:
            ret = self._invoke(super(CustomWebDriver, self).find_element, by=by, value=value) if webelement is None \
                else self._invoke(self._webelement_find_element_by, webelement, by=by, value=value)
        else:
            try:
                condition = EC.presence_of_element_located if not visible else EC.visibility_of_element_located
                ret = WebDriverWait(self, timeout).until(condition((by, value)))
            except TimeoutException:
                return None
        ref = id(ret)
        if ret:
            self._REFS[ref] = (self.find_element, (), {"by": by, "value": value, "timeout": timeout, "visible": visible, "webelement": webelement})
        return ret

    def find_elements(self, by=By.ID, value=None, timeout=0, visible=False, webelement=None, *args, **kwargs):
        if timeout > 0:
            self.find_element(by=by, value=value, timeout=timeout, visible=visible, webelement=webelement, *args, **kwargs)
        ret_elements = self._invoke(super(CustomWebDriver, self).find_elements, by=by, value=value, *args,
                                    **kwargs) if webelement is None \
            else self._invoke(self._webelement_find_elements_by, webelement, by=by, value=value, webelement=webelement,
                              *args, **kwargs)
        if ret_elements:
            to_remove = set()
            for el in ret_elements:
                try:
                    el_dompath = self.get_dompath(el)
                except StaleElementReferenceException:
                    to_remove.add(el)
                    continue
                # Make sure the returned elements are robust against StaleElementReferenceExceptions by simulating a `find_element_by_xpath`
                ref = id(el)
                if ref not in self._REFS:  # If not already previously fetched
                    self._REFS[ref] = (
                    self.find_element, (), {"by": By.XPATH, "value": el_dompath, "timeout": 3, "visible": False})

            for el in to_remove:
                ret_elements.remove(el)

        return ret_elements

    def find_elements_by_xpath(self, xpath_, timeout=0, visible=False, webelement=None, *args, **kwargs):
        return self.find_elements(By.XPATH, value=xpath_, timeout=timeout, visible=visible, webelement=webelement,
                                  *args, **kwargs)

    def find_element_by_location(self, point_x, point_y):
        return self._invoke(self.execute_script, 'return document.elementFromPoint(%r, %r);' % (point_x, point_y))

    def get_dompath(self, element):
        try:
            dompath = self._invoke(self.execute_script, "return get_dompath(arguments[0]).toLowerCase();", element,
                                   webelement=element)
        except Exception as e:
            raise  # Debug debug Debug
        return "//html%s" % "/".join([part if ":" not in part else "*" for part in dompath.split("/")]) if dompath else dompath

    def get_all_buttons(self):
        ret = self._invoke(self.execute_script, "return get_all_buttons();")
        interested_buttons = []
        interested_buttons_dom = []

        for button_ele in ret:
            button, button_dompath = button_ele
            interested_buttons.append(button)
            interested_buttons_dom.append(button_dompath)
            self._REFS[id(button)] = (self.find_element, (), {"by": By.XPATH, "value": button_dompath, "timeout": 3, "visible": False})

        return interested_buttons, interested_buttons_dom

    def get_all_links_orig(self):
        ret = self._invoke(self.execute_script, "return get_all_links();")
        interested_links = []
        for link_ele in ret:
            link, link_dompath, link_source = link_ele
            if re.search(CustomWebDriver._forbidden_suffixes, link_source, re.IGNORECASE):
                continue
            if link not in interested_links:
                interested_links.append([link, link_source])
                self._REFS[id(link)] = (self.find_element, (), {"by": By.XPATH, "value": link_dompath, "timeout": 3, "visible": False})

        return interested_links

    def get_all_links(self):
        ret = self._invoke(self.execute_script, "return get_all_links();")
        interested_links = []
        link_doms = []
        link_sources = []
        for link_ele in ret:
            link, link_dompath, link_source = link_ele
            interested_links.append(link)
            link_doms.append(link_dompath)
            link_sources.append(link_source)
            self._REFS[id(link)] = (self.find_element, (), {"by": By.XPATH, "value": link_dompath, "timeout": 3, "visible": False})

        return interested_links, link_doms, link_sources

    def get_all_clickable_images(self):
        ret = self._invoke(self.execute_script, "return get_all_clickable_imgs();")
        images = []
        images_dom = []

        for ele in ret:
            img, dompath = ele
            images.append(img)
            images_dom.append(dompath)
            self._REFS[id(img)] = (self.find_element, (), {"by": By.XPATH, "value": dompath, "timeout": 3, "visible": False})

        return images, images_dom

    def get_all_clickable_leaf_nodes(self):
        all_leaf_node_xpath = ["//span[not(*)]", "//div[not(*)]",
                               "//p[not(*)]", "//i[not(*)]"]
        leaf_elements_pre = []
        leaf_elements = []
        leaf_elements_dom = []

        for path in all_leaf_node_xpath:
            elements = self.find_elements_by_xpath(path)
            if elements:
                leaf_elements_pre.extend(elements)

        for ele in leaf_elements_pre:
            try:
                dompath = self.get_dompath(ele)
                leaf_elements_dom.append(dompath)
                leaf_elements.append(ele)
                self._REFS[id(ele)] = (self.find_element, (), {"by": By.XPATH, "value": dompath, "timeout": 3, "visible": False})
            except:
                continue

        return leaf_elements, leaf_elements_dom

    def get_all_clickable_elements(self):
        btns, btns_dom = self.get_all_buttons()
        links, links_dom, _ = self.get_all_links()
        images, images_dom = self.get_all_clickable_images()
        leaf_elements, leaf_elements_dom = self.get_all_clickable_leaf_nodes()

        return (btns, btns_dom), (links, links_dom), \
               (images, images_dom), (leaf_elements, leaf_elements_dom)

    def _get_elements_xpath_contains(self, patterns, tag=None, role=None, is_input=False, is_free_text=False):
        property_list = ["text()", "@class", "@title", "@value", "@label", "@aria-label"]

        # For regular elements
        tag_matching_regex = "//%s" % (tag if tag else "*")
        pattern_matching_regex = " or ".join(
            ["starts-with(normalize-space(%s), '%s')" % (lower(replace_nbsp(property)), patterns.lower()) for property
             in property_list])
        rule1 = tag_matching_regex + "[" + pattern_matching_regex + "]"

        # For elements with roles
        role_matching_regex = "//*[@role='%s']" % (role if role else "*")
        pattern_matching_regex = "[%s]" % ("starts-with(normalize-space(.), '%s')" % patterns)
        rule2 = role_matching_regex + pattern_matching_regex

        # For input elements with specific types
        if is_input:
            rule1 = "//input[@type='submit' or @type='button'][" + pattern_matching_regex + "]"

        # For free text elements
        if is_free_text:
            xpath_base = "//*[starts-with(normalize-space(%s), '%s')]" % (
            lower(replace_nbsp("text()")), patterns.lower())
            rule1 = '%s[not(self::script)][not(.%s)]' % (xpath_base, xpath_base)

        return [rule1, rule2]

    def get_clickable_elements_contains(self, patterns):

        button_xpaths = self._get_elements_xpath_contains(patterns, tag='button', role='button')
        link_xpaths = self._get_elements_xpath_contains(patterns, tag='a', role='link')
        input_xpath = self._get_elements_xpath_contains(patterns, is_input=True)
        free_text_xpath = self._get_elements_xpath_contains(patterns, is_free_text=True)

        element_list = []
        for path in button_xpaths + link_xpaths + input_xpath + free_text_xpath:
            elements = self.find_elements_by_xpath(path)
            if elements:
                element_list.extend(elements)

        return element_list

    def get_all_visible_username_password_inputs(self):
        ret_password, ret_username = self._invoke(self.execute_script,
                                                  "return get_all_visible_password_username_inputs();")
        return ret_password, ret_username

    # Get element location
    def get_location(self, element):
        try:
            loc = self._invoke(self.execute_script, 'return get_loc(arguments[0]);', element, webelement=element)
            return loc
        except StaleElementReferenceException as e:
            return [0, 0, 0, 0]

    # Get element text
    def get_text(self, element):
        try:
            text = self._invoke(self.execute_script, 'return arguments[0].innerText;', element, webelement=element)
            return text
        except StaleElementReferenceException as e:
            return ''

    # Get element's attribute
    def get_attribute(self, element, attribute):
        return self._invoke(self._get_attribute, element, attribute, webelement=element)

    def _get_attribute(self, element, attribute):
        try:
            attribute = element.get_attribute(attribute)
            return attribute
        except StaleElementReferenceException:
            return ''

    # Perform action on the element
    def move_to_element(self, element):
        return self._invoke(self._move_to_element, element, webelement=element)

    def _move_to_element(self, element):
        ActionChains(self).move_to_element(element).perform()
        return True

    def click(self, element):
        return self._invoke(self._click, element, webelement=element)

    def _click(self, element):
        ActionChains(self).move_to_element(element).click().perform()
        return True

    # Scroll to top of the page
    def scroll_to_top(self):
        try:
            self.execute_script("window.scrollTo(0, 0);")
            return True
        except Exception as e:
            return False



'''Validate the domain'''
def is_valid_domain(domain: Union[str, None]) -> bool:
    '''
        Check if the provided string is a valid domain
        :param domain:
        :return:
    '''
    # Regular expression to check if the string is a valid domain without spaces
    if domain is None:
        return False
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
   
    try:
        response = requests.head('https://www.' + domain, timeout=10, proxies=proxies)  # Reduced timeout and used HEAD
        PhishLLMLogger.spit(f'Domain {domain}, status code {response.status_code}', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        if response.status_code < 400 or response.status_code in [405, 429] or response.status_code >= 500:
            PhishLLMLogger.spit(f'Domain {domain} is valid and alive', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            return True
        elif response.history and any([r.status_code < 400 for r in response.history]):
            PhishLLMLogger.spit(f'Domain {domain} is valid and alive', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
            return True

    except Exception as err:
        PhishLLMLogger.spit(f'Error {err} when checking the aliveness of domain {domain}', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        return False

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
        logo_boxes = phishintention_cls.predict_all_uis4type(screenshot_encoding, 'logo')
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

'''Search for domain in Google Search'''
def query2url(query: str, SEARCH_ENGINE_API: str, SEARCH_ENGINE_ID: str, num: int=10, proxies: Optional[Dict]=None) -> List[str]:
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

    num = int(num)
    URL = f"https://www.googleapis.com/customsearch/v1?key={SEARCH_ENGINE_API}&cx={SEARCH_ENGINE_ID}&q={query}&num={num}&filter=1"
    while True:
        try:
            data = requests.get(URL, proxies=proxies).json()
            break
        except requests.exceptions.SSLError as e:
            print(e)
            time.sleep(1)

    if data.get('error', {}).get('code') == 429:
        raise RuntimeError("Google search exceeds quota limit")

    search_items = data.get("items")
    if search_items is None:
        return []

    returned_urls = [item.get("link") for item in search_items]

    return returned_urls


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

    num = int(num)
    URL = f"https://www.googleapis.com/customsearch/v1?key={SEARCH_ENGINE_API}&cx={SEARCH_ENGINE_ID}&q={query}&searchType=image&num={num}&filter=1"
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
def page_transition(driver: CustomWebDriver, dom: str, save_html_path: str, save_shot_path: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
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
        etext = None
        if element:
            try:
                driver.execute_script("arguments[0].style.border='3px solid red'", element[0]) # hightlight the element to click
                etext = driver.get_text(element[0])
                if (etext is None) or len(etext) == 0:
                    etext = driver.get_attribute(element[0], "value")
            except:
                pass
            driver.move_to_element(element[0])
            driver.click(element[0])
            time.sleep(12)  # fixme: must allow some loading time here
        current_url = driver.current_url()
    except Exception as e:
        print('Exception {} when clicking the login button'.format(e))
        return None, None, None, None

    try:
        css_script = """
            var style = document.createElement('style');
            document.head.appendChild(style);
            style.appendChild(document.createTextNode(`
              ::-webkit-input-placeholder { /* Chrome/Opera/Safari */
                color: black !important;
              }
              ::-moz-placeholder { /* Firefox 19+ */
                opacity: 1; /* Firefox has a lower opacity on placeholder by default */
                color: black !important;
              }
              :-ms-input-placeholder { /* IE 10+ */
                color: black !important;
              }
              ::placeholder { /* Universal Placeholder style */
                color: black !important;
              }
            `));
        """
        driver.execute_script(css_script)
    except Exception as e:
        print(e)

    try:
        driver.save_screenshot(save_shot_path)
        PhishLLMLogger.spit('CRP transition is successful! New screenshot has been saved', caller_prefix=PhishLLMLogger._caller_prefix, debug=True)
        with open(save_html_path, "w", encoding='utf-8') as f:
            f.write(driver.page_source())
        return etext, current_url, save_html_path, save_shot_path
    except Exception as e:
        PhishLLMLogger.spit('Exception {} when saving the new screenshot'.format(e), caller_prefix=PhishLLMLogger._caller_prefix, warning=True)
        return None, None, None, None


def get_screenshot_elements(phishintention_cls: PhishIntentionWrapper, driver: CustomWebDriver) -> List[int]:
    pred_boxes, pred_classes = phishintention_cls.predict_all_uis(driver.get_screenshot_encoding())
    if pred_boxes is None:
        screenshot_elements = []
    else:
        screenshot_elements = pred_classes.numpy().tolist()
    return screenshot_elements

def has_page_content_changed(phishintention_cls: PhishIntentionWrapper, driver: CustomWebDriver, prev_screenshot_elements: List[int]) -> bool:
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


def screenshot_element(clickable_element, clickable_dom, driver, clip_preprocess):
    candidate_ui = None
    candidate_ui_img = None
    candidate_ui_text = None

    try:
        driver.scroll_to_top()
        x1, y1, x2, y2 = driver.get_location(clickable_element)

        if x2 - x1 <= 0 or y2 - y1 <= 0:  # invisible
            return candidate_ui, candidate_ui_img, candidate_ui_text

        try:
            ele_screenshot_img = Image.open(io.BytesIO(clickable_element.screenshot_as_png))
            candidate_ui_img = clip_preprocess(ele_screenshot_img)
            candidate_ui = clickable_dom
            etext = driver.get_text(clickable_element)  # append the text
            if not etext:
                etext = driver.get_attribute(clickable_element, "value")
            candidate_ui_text = etext
        except Exception as e:
            try:
                full_screenshot = driver.get_screenshot_as_png()
                image = Image.open(io.BytesIO(full_screenshot))
                location = clickable_element.location
                size = clickable_element.size
                left, right = location['x'], location['x'] + size['width']
                top, bottom = location['y'], location['y'] + size['height']
                ele_screenshot_img = image.crop((left, top, right, bottom))
                candidate_ui_img = clip_preprocess(ele_screenshot_img)
                candidate_ui = clickable_dom
                etext = driver.get_text(clickable_element)  # append the text
                if not etext:
                    etext = driver.get_attribute(clickable_element, "value")
                candidate_ui_text = etext
            except Exception as e:
                print(f"Error processing element {clickable_dom}: {e}")

    except Exception as e:
        print(f"Error accessing element {clickable_dom}: {e}")

    return candidate_ui, candidate_ui_img, candidate_ui_text


