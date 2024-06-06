import os
import requests
import tldextract
from PIL import Image
import base64
from google.cloud import vision
import io
from scripts.utils.dynaphish.utils import *
import datetime
from dateutil.parser import parse
import dateutil
import time
from scripts.utils.dynaphish.google_safebrowsing import SafeBrowsing
from urllib.parse import parse_qs, urljoin, urlparse, urlsplit, urlunsplit
import pandas as pd
from typing import Union
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class BrandKnowledgeConstruction():

    def __init__(self, SEARCH_ENGINE_API,
                 SEARCH_ENGINE_ID,
                 phishintention_wrapper,
                 proxies=None,
                 kb_driver_time_sleep=3):
        self.SEARCH_ENGINE_API = SEARCH_ENGINE_API  # API key for Gsearch
        self.SEARCH_ENGINE_ID = SEARCH_ENGINE_ID  # ID for Gsearch engine
        self.phishintention_wrapper = phishintention_wrapper
        self.client = vision.ImageAnnotatorClient()  # Google Cloud Vision
        self.kb_driver_time_sleep = kb_driver_time_sleep
        self.proxies = proxies

    @staticmethod
    def reduce_to_main_domain(urls_list):
        reduced_list = []
        for url in urls_list:
            parsed_url = urlparse(url)
            reduced_url = parsed_url.scheme + '://' + \
                          tldextract.extract(parsed_url.hostname).domain + '.' + \
                          tldextract.extract(parsed_url.hostname).suffix
            reduced_list.append(reduced_url)
        return reduced_list

    @staticmethod
    def get_web_pub_dates(search_item, url):
        '''
            Get the URL indexed/registered date
        '''
        try:  # check gooogle search returned publication time
            pub_date = parse(search_item.get('pagemap').get('metatags')[0].get('article:published_time')).replace(
                tzinfo=None)
        except (TypeError, AttributeError, dateutil.parser.ParserError):
            pub_date = ''
        return pub_date

    def predict_n_save_logo(self, shot_path):
        '''
            use AWL detector to crop and save logo
            Args:
                shot_path: path to screenshot
        '''
        if shot_path is None:
            return None, None
        with open(shot_path, "rb") as image_file:
            screenshot_encoding = base64.b64encode(image_file.read())

        reference_logo = self.phishintention_wrapper.predict_logos(screenshot_encoding=screenshot_encoding)
        if reference_logo is None or len(reference_logo) == 0:
            return None, None

        return None, reference_logo[0]

    def detect_brands(self, img: Union[str, Image.Image], max_retries: int = 3, retry_delay: int = 1):
        """
        Google Logo Detection API
        Args:
            img: path to logo or PIL image
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        client = self.client
        attempt = 0

        while attempt < max_retries:
            try:
                if isinstance(img, str):
                    with io.open(img, 'rb') as image_file:
                        content = image_file.read()
                else:
                    b = io.BytesIO()
                    img.save(b, 'jpeg')
                    content = b.getvalue()

                image = vision.Image(content=content)

                response = client.logo_detection(image=image)
                if response.error.message:
                    raise Exception(
                        '{}\nFor more info on error messages, check: '
                        'https://cloud.google.com/apis/design/errors'.format(
                            response.error.message))

                logos = response.logo_annotations
                print('Logos:')

                descriptions = []
                for logo in logos:
                    print(logo.description)
                    descriptions.append(logo.description)

                return descriptions

            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return []


    def detect_text(self, img: Union[str, Image.Image], max_retries: int = 3, retry_delay: int = 1):
        """
        Google OCR
        Args:
            img: path to image or PIL image
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        client = self.client
        attempt = 0

        while attempt < max_retries:
            try:
                if isinstance(img, str):
                    with io.open(img, 'rb') as image_file:
                        content = image_file.read()
                else:
                    b = io.BytesIO()
                    img.save(b, 'jpeg')
                    content = b.getvalue()

                image = vision.Image(content=content)

                response = client.text_detection(image=image, image_context={
                    "language_hints": ["en-t-i0-handwrit", "zh-t-i0-handwrit",
                                       "ja-t-i0-handwrit"]})  # support for Chinese and Japanese

                if response.error.message:
                    raise Exception(
                        '{}\nFor more info on error messages, check: '
                        'https://cloud.google.com/apis/design/errors'.format(
                            response.error.message))

                texts = response.text_annotations
                returned_strings = []
                for text in texts:
                    returned_strings.append(text.description)

                return returned_strings

            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return []


    def query2image(self, query, num=3, max_retries=3, retry_delay=1):
        """
        Retrieve the images from Google image search
        Args:
            query: query string
            num: number of results returned
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        Returns:
            returned_urls
            context_links: the source URLs for the images
        """
        returned_urls = []
        context_links = []
        if len(query) == 0:
            return returned_urls, context_links

        attempt = 0
        while attempt < max_retries:
            try:
                URL = f"https://www.googleapis.com/customsearch/v1?key={self.SEARCH_ENGINE_API}&cx={self.SEARCH_ENGINE_ID}&q={query}&searchType=image&num={num}&filter=1"
                data = requests.get(URL, proxies=self.proxies).json()

                if 'error' in data:
                    if data['error']['code'] == 429:
                        raise RuntimeError("Google search exceeds quota limit")
                    else:
                        raise Exception(data['error']['message'])

                search_items = data.get("items")
                if search_items is None:
                    return returned_urls, context_links

                # iterate over results found
                for search_item in search_items:
                    link = search_item.get("image")["thumbnailLink"]
                    context_link = search_item.get("image")['contextLink']
                    returned_urls.append(link)
                    context_links.append(context_link)

                return returned_urls, context_links

            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return [], []


    def query2url(self, query, allowed_domains=[], forbidden_domains=[], forbidden_subdomains=[], title_truncate=False,
                  num=5, max_retries=3, retry_delay=1):
        """
        Retrieve the URLs from Google search
        Args:
            query: query string
            allowed_domains: list of domains to include
            forbidden_domains: list of domains to exclude
            forbidden_subdomains: list of subdomains to exclude
            title_truncate: whether to truncate the title at the first dash
            num: number of results to return
            max_retries: maximum number of retry attempts
            retry_delay: delay between retries in seconds
        Returns:
            returned_urls: list of URLs
            returned_titles: list of titles
            returned_pub_dates: list of publication dates
        """
        returned_urls = []
        returned_titles = []
        returned_pub_dates = []
        if not query:
            return returned_urls, returned_titles, returned_pub_dates

        attempt = 0
        while attempt < max_retries:
            try:
                URL = f"https://www.googleapis.com/customsearch/v1?key={self.SEARCH_ENGINE_API}&cx={self.SEARCH_ENGINE_ID}&q={query}&num={num}"
                data = requests.get(URL, proxies=self.proxies).json()

                if 'error' in data:
                    if data['error']['code'] == 429:
                        raise RuntimeError("Google search exceeds quota limit")
                    else:
                        raise Exception(data['error']['message'])

                search_items = data.get("items")
                if not search_items:
                    return returned_urls, returned_titles, returned_pub_dates

                for search_item in search_items:
                    link = search_item.get("link")
                    title = search_item.get("title")
                    if title_truncate and ' - ' in title:
                        title = title.split(' - ')[0]

                    domain = tldextract.extract(link).domain
                    subdomain = tldextract.extract(link).subdomain

                    if domain in forbidden_domains or subdomain in forbidden_subdomains:
                        continue

                    if allowed_domains and domain not in allowed_domains:
                        continue

                    pub_date = self.get_web_pub_dates(search_item, link)  # Assuming this function is defined elsewhere

                    if tldextract.extract(query).domain in domain:
                        returned_titles.insert(0, title)
                        returned_urls.insert(0, link)
                        returned_pub_dates.insert(0, pub_date)
                    else:
                        returned_titles.append(title)
                        returned_urls.append(link)
                        returned_pub_dates.append(pub_date)

                return returned_urls[:num], returned_titles[:num], returned_pub_dates[:num]

            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return [], [], []


    def url2logo(self, driver, URL, url4logo=True):
        '''
            Get page's logo from an URL
            Args:
                URL:
                url4logo: the URL is a logo image already or not
        '''
        try:
            driver.get(URL)
            time.sleep(3)  # fixme: must allow some loading time here
        except Exception as e:
            return None, str(e)

        # the URL is for a webpage not the logo image
        if not url4logo:
            try:
                screenshot_encoding = driver.get_screenshot_encoding()
                all_logos_coords = self.phishintention_wrapper.predict_all_uis4type(screenshot_encoding, 'logo')
                if all_logos_coords is None:
                    return None, 'no_logo_prediction'
                else:
                    logo_coord = all_logos_coords[0]
                    # try:
                    screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
                    screenshot_img = screenshot_img.convert("RGB")
                    logo = screenshot_img.crop((int(logo_coord[0]), int(logo_coord[1]), int(logo_coord[2]), int(logo_coord[3])))
            except Exception as e:
                return None, str(e)

        else:  # the URL is directly pointing to a logo image
            try:
                logo_encoding = driver.find_element_by_tag_name('img').screenshot_as_base64
                logo = Image.open(io.BytesIO(base64.b64decode(logo_encoding)))
                logo = logo.convert("RGB")
            except Exception:
                try:
                    logo_encoding = driver.get_screenshot_encoding()
                    logo = Image.open(io.BytesIO(base64.b64decode(logo_encoding)))
                    logo = logo.convert("RGB")
                except Exception as e:
                    return None, str(e)

        return logo, 'success'

    def domain2brand(self, driver, q_domain, q_tld, reference_logo):
        '''
            Domain2brand branch (mainly designed for benign websites)
            Args:
                q_tld: tld domain for the testing site
                reference_logo: logo on the testing site as reference
                q_domain: domain for the testing site
            Returns:
                company_domains: knowledge domains
                company_logos: knowledge logos
                comment: comment on failure reasons
        '''

        company_urls = []
        company_logos = []
        company_titles = []
        comment = ''
        branch_time = 0

        # search for domain.tld directly in Google
        start_time = time.time()
        urls_from_gsearch, titles_from_gsearch, urls_pub_dates = self.query2url(query=q_domain + '.' + q_tld,
                                                                                forbidden_domains=OnlineForbiddenWord.WEBHOSTING_DOMAINS,
                                                                                forbidden_subdomains=OnlineForbiddenWord.WEBHOSTING_DOMAINS,
                                                                                num=2)
        branch_time += time.time() - start_time

        # only the main domain, not subdomain, not subpath
        urls_from_gsearch = self.reduce_to_main_domain(urls_from_gsearch)
        unique_indicies = pd.Series(urls_from_gsearch).drop_duplicates().index.tolist()
        urls_from_gsearch = [urls_from_gsearch[ind] for ind in unique_indicies]
        titles_from_gsearch = [titles_from_gsearch[ind] for ind in unique_indicies]
        urls_pub_dates = [urls_pub_dates[ind] for ind in unique_indicies]

        # get the logos from knowledge sites
        start_time = time.time()

        def fetch_logo_and_status(URL):
            return self.url2logo(driver=driver, URL=URL, url4logo=False)

        logos_from_gsearch = []
        logos_status = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit tasks to the executor
            future_to_url = {executor.submit(fetch_logo_and_status, URL): URL for URL in urls_from_gsearch}

            for future in as_completed(future_to_url):
                URL = future_to_url[future]
                try:
                    logo, status = future.result()  # Get the result from the future
                    logos_from_gsearch.append(logo)
                    logos_status.append(status)
                except Exception as exc:
                    print(f'URL {URL} generated an exception: {exc}')
                    logos_from_gsearch.append(None)
                    logos_status.append('Failed')

        print('Domain2brand crop the logo time:', time.time() - start_time)

        if len(urls_from_gsearch) == 0:
            comment = 'no_result_from_gsearch'
        elif np.sum([x is not None for x in logos_from_gsearch]) == 0:
            comment = 'cannot_crop_logo'

        # can return some results from Google search
        if len(urls_from_gsearch):

            # domain matching or logo matching
            domains_from_gsearch = [tldextract.extract(x).domain for x in urls_from_gsearch]
            tlds_from_gsearch = [tldextract.extract(x).suffix for x in urls_from_gsearch]

            start_time = time.time()
            domain_matched_indices, logo_matched_indices = self.knowledge_cleansing(
                reference_logo=reference_logo,
                logo_list=logos_from_gsearch,
                reference_domain=q_domain,
                domain_list=domains_from_gsearch,
                reference_tld=q_tld,
                tld_list=tlds_from_gsearch)
            branch_time += time.time() - start_time


            domain_or_logo_matched_indices = list(set(domain_matched_indices + logo_matched_indices))
            urls_from_gsearch = [urls_from_gsearch[a] for a in domain_or_logo_matched_indices]
            urls_pub_dates = [urls_pub_dates[a] for a in domain_or_logo_matched_indices]
            titles_from_gsearch = [titles_from_gsearch[a] for a in domain_or_logo_matched_indices]
            logos_from_gsearch = [logos_from_gsearch[a] for a in domain_or_logo_matched_indices]

            print('Domain2brand after domain match and logo matching take:', time.time() - start_time)
            if len(urls_from_gsearch) == 0:
                comment = 'doesnt_pass_validation'

            company_urls = urls_from_gsearch
            company_logos = logos_from_gsearch

        if len(company_urls):
            company_domains = [tldextract.extract(x).domain+'.'+tldextract.extract(x).suffix for x in company_urls]
        else:
            company_domains = []
        return company_domains, company_logos, comment, branch_time

    def logo2brand_ocr(self, driver, filePath, reference_logo, q_domain, q_tld):
        '''
            Logo2brand branch via OCR (mainly designed for suspicious websites)
            Args:
                q_tld: tld domain for the testing site
                reference_logo: logo on the testing site as reference
                q_domain: domain for the testing site
                filePath: path to logo
            Returns:
                company_domains: knowledge domains
                company_logos: knowledge logos
                comment: comment on failure reasons
        '''
        company_urls = []
        company_domains = []
        company_logos = []
        comment = ''
        branch_time = 0

        # Google OCR
        start_time = time.time()
        brand_name = self.detect_text(filePath)
        branch_time += time.time() - start_time
        if len(brand_name):
            brand_name = query_cleaning(brand_name[0])
            if len(brand_name) <= 1:
                comment = 'text_too_short_from_OCR'
                return company_domains, company_logos, brand_name, comment, branch_time
        else:
            comment = 'no_result_from_OCR'
            return company_domains, company_logos, brand_name, comment, branch_time
        print('Google OCR time:', time.time() - start_time)
        print('Brand name:', brand_name)

        # Search OCR text in Google
        start_time = time.time()
        urls_from_gsearch, _, urls_pub_dates = self.query2url(query=brand_name.lower(),
                                                              forbidden_domains=OnlineForbiddenWord.IGNORE_DOMAINS + OnlineForbiddenWord.WEBHOSTING_DOMAINS,
                                                              forbidden_subdomains=OnlineForbiddenWord.WEBHOSTING_DOMAINS,
                                                              num=1)
        branch_time += time.time() - start_time

        # only the main domain, not subdomain, not subpath
        urls_from_gsearch = self.reduce_to_main_domain(urls_from_gsearch)
        unique_indicies = pd.Series(urls_from_gsearch).drop_duplicates().index.tolist()
        urls_from_gsearch = [urls_from_gsearch[ind] for ind in unique_indicies]
        urls_pub_dates = [urls_pub_dates[ind] for ind in unique_indicies]

        # get the logos from knowledge sites

        def fetch_logo_and_status(URL):
            return self.url2logo(driver=driver, URL=URL, url4logo=False)

        # Assuming urls_from_gsearch is a list of URLs from which you want to fetch logos
        logos_from_gsearch = []
        logos_status = []

        # Use ThreadPoolExecutor to execute calls to url2logo in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Create a dictionary of future submissions
            future_to_url = {executor.submit(fetch_logo_and_status, URL): URL for URL in urls_from_gsearch}

            for future in as_completed(future_to_url):
                URL = future_to_url[future]
                try:
                    logo, status = future.result()  # Get the result from future
                    logos_from_gsearch.append(logo)
                    logos_status.append(status)
                except Exception as exc:
                    print(f'URL {URL} generated an exception: {exc}')
                    # Handle failed URL if necessary, e.g., by appending None or a custom error indicator
                    logos_from_gsearch.append(None)
                    logos_status.append('Failed')

        print('crop the logo time:', time.time() - start_time)

        if len(urls_from_gsearch) == 0:  # no result from google search
            comment = 'no_result_from_gsearch'
        elif np.sum([x is not None for x in logos_from_gsearch]) == 0:  # has result, but cannot crop the logo
            comment = 'cannot_crop_logo'

        if len(urls_from_gsearch):
            # Domain matching OR Logo matching
            logo_domains = [tldextract.extract(x).domain for x in urls_from_gsearch]
            logo_tlds = [tldextract.extract(x).suffix for x in urls_from_gsearch]

            start_time = time.time()
            domain_matched_indices, logo_matched_indices = self.knowledge_cleansing(reference_logo=reference_logo,
                                                                                    logo_list=logos_from_gsearch,
                                                                                    reference_domain=q_domain,
                                                                                    domain_list=logo_domains,
                                                                                    reference_tld=q_tld,
                                                                                    tld_list=logo_tlds)
            branch_time += time.time() - start_time

            domain_or_logo_matched_indices = list(set(domain_matched_indices + logo_matched_indices))
            if len(domain_or_logo_matched_indices) == 0:
                # use EXACT string matching as backup
                for it, logo in enumerate(logos_from_gsearch):
                    if logo is not None:
                        brand_name_knowledge_site = self.detect_text(logo)
                        if len(brand_name_knowledge_site) > 0 and \
                                brand_name_knowledge_site[0].lower().replace(' ', '') == brand_name.lower().replace(' ', ''):
                            domain_or_logo_matched_indices.append(it)

            logos_from_gsearch = [logos_from_gsearch[a] for a in domain_or_logo_matched_indices]
            urls_from_gsearch = [urls_from_gsearch[a] for a in domain_or_logo_matched_indices]
            urls_pub_dates = [urls_pub_dates[a] for a in domain_or_logo_matched_indices]
            if len(urls_from_gsearch) == 0:
                comment = 'doesnt_pass_validation'

            company_urls = urls_from_gsearch
            company_logos = logos_from_gsearch

            if len(company_urls) > 0 and np.sum([x is not None for x in
                                                 logos_from_gsearch]) == 0:  # pass all validation but that logo cannot be cropped
                comment = 'cannot_crop_logo'

        if len(company_urls):
            company_domains = [tldextract.extract(x).domain+'.'+tldextract.extract(x).suffix for x in company_urls]
        else:
            company_domains = []
        return company_domains, company_logos, brand_name, comment, branch_time

    def logo2brand_logodet(self, driver, filePath, reference_logo, q_domain, q_tld):
        '''
            Logo2brand branch Logodet (mainly designed for suspicious page)
            Args:
                q_tld: tld domain for the testing site
                reference_logo: logo on the testing site as reference
                q_domain: domain for the testing site
                filePath: path to logo
            Returns:
                company_domains: knowledge domains
                company_logos: knowledge logos
                comment: comment on failure reasons
        '''

        company_urls = []
        company_domains = []
        company_logos = []
        comment = ''

        # Google OCR
        start_time = time.time()
        brand_name = self.detect_brands(filePath)
        if len(brand_name) == 0:
            comment = 'no_result_from_logo_detection'
            return company_domains, company_logos, brand_name, comment
        brand_name = brand_name[0]
        print('Logo2brand Google logo detection time:', time.time() - start_time)
        print('Logo2brand brand name:', brand_name)

        # Search brand in Google
        start_time = time.time()
        urls_from_gsearch, _, urls_pub_dates = self.query2url(query=brand_name.lower(),
                                                              forbidden_domains=OnlineForbiddenWord.WEBHOSTING_DOMAINS,
                                                              forbidden_subdomains=OnlineForbiddenWord.WEBHOSTING_DOMAINS,
                                                              num=3)
        print('Logo2brand Google search time:', time.time() - start_time)

        # only the main domain, not subdomain, not subpath
        urls_from_gsearch = self.reduce_to_main_domain(urls_from_gsearch)
        unique_indicies = pd.Series(urls_from_gsearch).drop_duplicates().index.tolist()
        urls_from_gsearch = [urls_from_gsearch[ind] for ind in unique_indicies]
        urls_pub_dates = [urls_pub_dates[ind] for ind in unique_indicies]

        # get the logos from knowledge sites
        start_time = time.time()
        logos_from_gsearch = []
        logos_status = []
        for URL in urls_from_gsearch:
            logo, status = self.url2logo(driver=driver, URL=URL, url4logo=False)
            logos_from_gsearch.append(logo)
            logos_status.append(status)
        print('Logo2brand crop the logo time:', time.time() - start_time)

        if len(urls_from_gsearch) == 0:  # no result from google search
            comment = 'no_result_from_gsearch'
        elif np.sum([x is not None for x in logos_from_gsearch]) == 0:  # has result, but cannot crop the logo
            comment = 'cannot_crop_logo'

        # has result
        if len(urls_from_gsearch):
            # Domain matching OR Logo matching
            start_time = time.time()
            logo_domains = [tldextract.extract(x).domain for x in urls_from_gsearch]
            logo_tlds = [tldextract.extract(x).suffix for x in urls_from_gsearch]

            domain_matched_indices, logo_matched_indices = self.knowledge_cleansing(reference_logo=reference_logo,
                                                                                    logo_list=logos_from_gsearch,
                                                                                    reference_domain=q_domain,
                                                                                    domain_list=logo_domains,
                                                                                    reference_tld=q_tld,
                                                                                    tld_list=logo_tlds)

            domain_or_logo_matched_indices = list(set(domain_matched_indices + logo_matched_indices))
            if len(domain_or_logo_matched_indices) == 0:
                # use EXACT string matching as backup
                for it, logo in enumerate(logos_from_gsearch):
                    if logo is not None:
                        brand_name_knowledge_site = self.detect_brands(logo)
                        if len(brand_name_knowledge_site) > 0 and brand_name_knowledge_site[0] == brand_name:
                            domain_or_logo_matched_indices.append(it)

            logos_from_gsearch = [logos_from_gsearch[a] for a in domain_or_logo_matched_indices]
            urls_from_gsearch = [urls_from_gsearch[a] for a in domain_or_logo_matched_indices]
            urls_pub_dates = [urls_pub_dates[a] for a in domain_or_logo_matched_indices]
            print('Logo2brand after domain match and logo matching take:', time.time() - start_time)
            if len(urls_from_gsearch) == 0:
                comment = 'doesnt_pass_validation'

            # (1) Filter by Google safe browsing + (2) Filter by website published date
            start_time = time.time()
            results_from_gsb = self.gsb_filter(urls_from_gsearch)
            for j in range(len(urls_from_gsearch)):
                if results_from_gsb[urls_from_gsearch[j]]["malicious"] is False:
                    # alive for more than 3 months
                    if urls_pub_dates[j] is not None and urls_pub_dates[j] != '' and (
                            datetime.datetime.today() - urls_pub_dates[j]).days >= 90:
                        company_urls.append(urls_from_gsearch[j])
                        company_logos.append(logos_from_gsearch[j])

                    elif urls_pub_dates[j] is None or urls_pub_dates[j] == '':  # cannot get publication date
                        company_urls.append(urls_from_gsearch[j])
                        company_logos.append(logos_from_gsearch[j])
            print('Logo2brand filter by gsb and publication date:', time.time() - start_time)
            if len(urls_from_gsearch) > 0 and len(company_urls) == 0:  # pass validation but not gsb or publicate date
                comment = 'doesnt_pass_gsb_or_date'
            if len(company_urls) > 0 and np.sum([x is not None for x in
                                                 logos_from_gsearch]) == 0:  # pass all validation but that logo cannot be cropped
                comment = 'cannot_crop_logo'

        if len(company_urls):
            company_domains = [tldextract.extract(x).domain+'.'+tldextract.extract(x).suffix for x in company_urls]
        else:
            company_domains = []
        return company_domains, company_logos, brand_name, comment

    def knowledge_cleansing(self, reference_logo, logo_list,
                            reference_domain, domain_list,
                            reference_tld, tld_list):
        '''
            Knowledge cleansing with Logo matcher and Domain matcher
            Args:
                reference_logo: logo on the testing website as reference
                logo_list: list of logos to check, logos_status
                reference_domain: domain for the testing website
                reference_tld: top-level domain for the testing website
                domain_list: list of domains to check
                ts: logo matching threshold
                strict: strict domain matching or not
            Returns:
                domain_matched_indices
                logo_matched_indices
        '''

        def match_domain(ct):
            if re.search('^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$', domain_list[ct]):
                return None  # skip if an IP address
            if domain_list[ct] == reference_domain and tld_list[ct] == reference_tld:
                return ct
            return None

        def match_logo(ct):
            if re.search('^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$', domain_list[ct]):
                return None  # skip if an IP address
            if logo_list[ct] is not None:
                logo_feat = self.phishintention_wrapper.return_logo_feat(logo_list[ct])
                if reference_logo_feat @ logo_feat >= self.phishintention_wrapper.SIAMESE_THRE_RELAX:
                    return ct
            return None

        if reference_logo is not None:
            reference_logo_feat = self.phishintention_wrapper.return_logo_feat(reference_logo)

        with ThreadPoolExecutor() as executor:
            domain_results = list(executor.map(match_domain, range(len(domain_list))))
            logo_results = list(executor.map(match_logo, range(len(logo_list))))

        domain_matched_indices = [result for result in domain_results if result is not None]
        logo_matched_indices = [result for result in logo_results if result is not None]

        return domain_matched_indices, logo_matched_indices

    def gsb_filter(self, urls):
        '''
            Google safebrowsing lookup
            Args:
                urls: list of URLs to lookup
        '''
        gsb = SafeBrowsing(self.SEARCH_ENGINE_API)
        results = gsb.lookup_urls(urls)
        return results

    def runit_simplified(self, driver, shot_path,
                         query_domain, query_tld, type='logo2brand'):
        # Get Logo
        _, reference_logo = self.predict_n_save_logo(shot_path)
        filePath = './temp.png'
        reference_logo.save(filePath)

        if type == 'logo2brand':
            if reference_logo is None:  # no way to run logo2brand
                return None, None, None, None, None, 'failure_nologo'
            company_domains, company_logos, brand_name, comment, branch_time = self.logo2brand_ocr(driver, filePath, reference_logo,
                                                                                      query_domain, query_tld)
        elif type == 'domain2brand':
            company_domains, company_logos, comment, branch_time = \
                self.domain2brand(driver, query_domain, query_tld, reference_logo)

        else:
            raise NotImplementedError

        # can discover the brand knowledge
        if len(company_domains) and np.sum([x is not None for x in company_logos]) > 0:
            if 'host' not in company_domains[0].lower():  # TODO: avoid domain hosting website
                if type == 'domain2brand':
                    brand_name = company_domains[0]
                    return reference_logo, company_domains, brand_name, company_logos, branch_time, 'success_domain2brand'
                elif type == 'logo2brand':
                    brand_name = company_domains[0]
                    return reference_logo, company_domains, brand_name, company_logos, branch_time, 'success_logo2brand'

        elif len(company_domains):  # if can discover the brand, but cannot report the logo
            brand_name = company_domains[0]
            return reference_logo, company_domains, brand_name, [], branch_time, 'failure_' + comment

        return reference_logo, [], None, [], branch_time, 'failure_' + comment


if __name__ == '__main__':
    from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
    from tqdm import tqdm

    config_path = './ablation_study/configs.yaml'
    phishintention_wrapper = PhishIntentionWrapper()
    phishintention_wrapper.reset_model(config_path)

    API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./knowledge_expansion/api_key.txt').readlines()]
    bkb = BrandKnowledgeConstruction(API_KEY, SEARCH_ENGINE_ID, phishintention_wrapper)

    targetlist_logo_features = np.load('./ablation_study/LOGO_FEATS.npy')

    test_folder = './datasets/Alexa_bottom10k'
    print(len(os.listdir(test_folder)))
    #
    from xdriver.XDriver import XDriver

    driver = XDriver.boot(chrome=True)

    for num, folder in tqdm(enumerate(os.listdir(test_folder))):
        # folder = "https://backoffice.hungrydata.com.au/"
        # folder = "https://adam.zeloof.xyz/"
        query_domain = tldextract.extract(folder).domain
        query_tld = tldextract.extract(folder).suffix
        company_domains_B2, company_logos_B2 = bkb.domain2brand(driver=driver, q_domain=query_domain, q_tld=query_tld,
                                                                reference_logo=None)
        print(company_domains_B2)
        break

    driver.quit()

    # os.makedirs('./field_study_domain2brand/expand_targetlist/{}'.format(company_domains_B2[0]), exist_ok=True)
    # company_logos_B2[0].save('./field_study_domain2brand/expand_targetlist/{}/0.png'.format(company_domains_B2[0]))