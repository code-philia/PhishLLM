import numpy as np
import cv2
from seleniumwire import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import re
import sys
from unidecode import unidecode
import shutil
import requests
import json


class SafeBrowsingInvalidApiKey(Exception):
    def __init__(self):
        Exception.__init__(self, "Invalid API key for Google Safe Browsing")

class SafeBrowsingPermissionDenied(Exception):
    def __init__(self, detail):
        Exception.__init__(self, detail)

class SafeBrowsingWeirdError(Exception):
    def __init__(self, code, status, message):
        self.message = "%s(%i): %s" % (
            status,
            code,
            message
        )
        Exception.__init__(self, message)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class SafeBrowsing(object):
    def __init__(self, key, api_url='https://safebrowsing.googleapis.com/v4/threatMatches:find'):
        self.api_key = key
        self.api_url = api_url

    def lookup_urls(self, urls, platforms=["ANY_PLATFORM"]):
        results = {}
        for urll in chunks(urls, 50):
            data = {
                "client": {
                    "clientId":      "pysafebrowsing",
                    "clientVersion": "1.5.2"
                },
                "threatInfo": {
                    "threatTypes":
                        [
                            "MALWARE",
                            "SOCIAL_ENGINEERING",
                            "THREAT_TYPE_UNSPECIFIED",
                            "UNWANTED_SOFTWARE",
                            "POTENTIALLY_HARMFUL_APPLICATION"
                        ],
                    "platformTypes": platforms,
                    "threatEntryTypes": ["URL"],
                    "threatEntries": [{'url': u} for u in urll],
                }
            } # include all threattypes
            headers = {'Content-type': 'application/json'}

            try:
                r = requests.post(
                        self.api_url,
                        data=json.dumps(data),
                        params={'key': self.api_key},
                        headers=headers
                )
            except requests.exceptions.ConnectionError:
                results.update(dict([(u, {"malicious": False}) for u in urls]))
                continue

            if r.status_code == 200:
                # Return clean results
                if r.json() == {}:
                    results.update(dict([(u, {"malicious": False}) for u in urls]))
                else:
                    for url in urll:
                        # Get matches
                        matches = [match for match in r.json()['matches'] if match['threat']['url'] == url]
                        if len(matches) > 0:
                            results[url] = {
                                'malicious': True,
                                'platforms': list(set([b['platformType'] for b in matches])),
                                'threats': list(set([b['threatType'] for b in matches])),
                                'cache': min([b["cacheDuration"] for b in matches])
                            }
                        else:
                            results[url] = {"malicious": False}
            else:
                results.update(dict([(u, {"malicious": False}) for u in urls]))
                continue

        return results

    def lookup_url(self, url, platforms=["ANY_PLATFORM"]):
        """
        Online lookup of a single url
        """
        r = self.lookup_urls([url], platforms=platforms)
        return r[url]

class OnlineForbiddenWord():
    IGNORE_DOMAINS = ['wikipedia', 'wiki',
                      'bloomberg', 'glassdoor',
                      'linkedin', 'jobstreet',
                      'facebook', 'twitter',
                      'instagram', 'youtube', 'org', 'accounting']

    # ignore those webhosting/domainhosting sites
    WEBHOSTING_TEXT = '(webmail.*)|(.*godaddy.*)|(.*roundcube.*)|(.*clouddns.*)|(.*namecheap.*)|(.*plesk.*)|(.*rackspace.*)|(.*cpanel.*)|(.*virtualmin.*)|(.*control.*webpanel.*)|(.*hostgator.*)|(.*mirohost.*)|(.*hostinger.*)|(.*bisecthosting.*)|(.*misshosting.*)|(.*serveriai.*)|(.*register\.to.*)|(.*appspot.*)|' \
                      '(.*weebly.*)|(.*serv5.*)|(.*weebly.*)|(.*umbler.*)|(.*joomla.*)' \
                      '(.*webnode.*)|(.*duckdns.*)|(.*moonfruit.*)|(.*netlify.*)|' \
                      '(.*glitch.*)|(.*herokuapp.*)|(.*yolasite.*)|(.*dynv6.*)|(.*cdnvn.*)|' \
                      '(.*surge.*)|(.*myshn.*)|(.*azurewebsites.*)|(.*dreamhost.*)|host|cloak|domain|block|isp|azure|wordpress|weebly|dns|network|shortener|server|helpdesk|laravel|jellyfin|portainer|reddit|storybook'

    WEBHOSTING_DOMAINS = ['godaddy', 'roundcube',
                          'clouddns', 'namecheap',
                          'plesk', 'rackspace', 'cpanel',
                          'virtualmin', 'control-webpanel',
                          'hostgator', 'mirohost', 'hostinger',
                          'bisecthosting', 'misshosting', 'serveriai',
                          'register', 'appspot', 'weebly', 'serv5',
                          'weebly', 'umbler', 'joomla', 'webnode', 'duckdns',
                          'moonfruit', 'netlify', 'glitch', 'herokuapp',
                          'yolasite', 'dynv6', 'cdnvn', 'surge', 'myshn',
                          'azurewebsites', 'dreamhost', 'proisp',
                          'accounting']



def initialize_chrome_settings(lang_txt:str):
    '''
    initialize chrome settings
    :return: chrome options
    '''
    # enable translation
    white_lists = {}

    with open(lang_txt) as langf:
        for i in langf.readlines():
            i = i.strip()
            text = i.split(' ')
            white_lists[text[1]] = 'en'
    prefs = {
        "translate": {"enabled": "true"},
        "translate_whitelists": white_lists,
        "download_restrictions": 3,
        "download.prompt_for_download": False,
        "download.default_directory": "/home/ruofan/git_space/phishing-research/trash",
    }
    options = webdriver.ChromeOptions()

    options.add_argument('--ignore-certificate-errors') # ignore errors
    options.add_argument('--ignore-ssl-errors')
    options.add_argument("--headless")
    options.add_argument('--no-proxy-server')
    options.add_argument("--proxy-server='direct://'")

    options.add_argument("--start-maximized")
    options.add_argument('--window-size=1920,1080') # fix screenshot size
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument('user-agent=Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0')

    # Add those options for Linux users
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')

    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option('useAutomationExtension', False)
    options.set_capability('unhandledPromptBehavior', 'dismiss') # dismiss

    capabilities = DesiredCapabilities.CHROME
    capabilities["goog:loggingPrefs"] = {"performance": "ALL"}  # chromedriver 75+
    capabilities["unexpectedAlertBehaviour"] = "dismiss"  # handle alert
    capabilities["pageLoadStrategy"] = "eager"  # eager mode #FIXME: set eager mode, may load partial webpage

    return options, capabilities

def clean_infobox(text):
    text = re.sub(r"\[\d\]", '', text)
    text = re.sub(r"\n", ' ', text)
    if sys.version_info[0] < 3:
        text = text.replace(u'\xa0', u' ')
    else:
        text = text.replace('\xa0', ' ')
    return text.strip()

def white_screen(img):
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    img_arr = np.flip(img_arr, -1)  # RGB2BGR
    img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_area = np.prod(img.shape)
    white_area = np.sum(img == 255)
    if white_area / img_area >= 0.9:  # skip white screenshots
        return True  # dirty
    return False

def sort_disambiguition_options(options):
    '''
        Sort by keywords
    '''
    likelihood = []
    for opt in options:
        if re.search("Corp|Inc\.|organization|company|\.com|service|brand|website|(sites web)", opt, re.IGNORECASE):
            likelihood.append(1)
        else:
            likelihood.append(2)
    return likelihood

def sort_by_lcs(reference_query, options):
    '''
        Soft by longest common substring
    '''
    def common(st1, st2):
        def _iter():
            for a, b in zip(st1, st2):
                if a == b:
                    yield a
                else:
                    return
        return ''.join(_iter())

    likelihood = []
    for opt in options:
        common_substring = common(reference_query.lower().replace(' ', ''),
                                  opt.lower().replace(' ', ''))
        likelihood.append(-len(common_substring))
    return likelihood


def special_character_replacement(query: str):
    query = unidecode(query)
    query.replace('\n', '')
    return query

def query_cleaning(query: str):

    if len(query) == 0:
        return ''
    if query.lower().startswith('index of') or \
        'forbidden' in query.lower() or \
        'access denied' in query.lower() or\
        'bad gateway' in query.lower() or \
        'not found' in query.lower():
        return ''
    if query.lower() in ['text', 'logo', 'graphics']:
        return ''
    if query.lower() == 'tm':
        return ''
    # remove noisy tokens
    for it, token in enumerate(query.split('\n')):
        if len(token) <= 1:
            continue
        # noisy token
        elif any(char.isdigit() for char in token) and any(char.isalpha() for char in token) and any(((not char.isalnum()) and (not char.isspace())) for char in token):
            continue
        else:
            query = ' '.join(query.split('\n')[it:])
            break

    for it, token in enumerate(query.split(' ')):
        if len(token) <= 2 or token.isnumeric():
            continue
        else:
            query = ' '.join(query.split(' ')[it:])
            break

    query = query.translate(str.maketrans('', '', r"""!"#$%'()*+,-/:;<=>?@[\]^_`{|}~"""))
    return query

# def undetected_chrome_options():
#     _chromeOpts = uc.ChromeOptions()
#     _chromeOpts.add_argument("--no-sandbox")
#     _chromeOpts.add_argument("--disable-dev-shm-usage")
#     _chromeOpts.add_argument('--disable-gpu')
#     _chromeOpts.add_argument('--headless')
#     _chromeOpts.add_experimental_option('useAutomationExtension', False)
#     _chromeOpts.add_experimental_option("excludeSwitches", ["enable-automation"])
#     _chromeOpts.add_argument("--disable-blink-features=AutomationControlled")
#     return _chromeOpts