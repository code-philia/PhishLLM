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
                # if r.status_code == 400:
                #     print(r.json())
                #     if r.json()['error']['message'] == 'API key not valid. Please pass a valid API key.':
                #         raise SafeBrowsingInvalidApiKey()
                #     else:
                #         raise SafeBrowsingWeirdError(
                #             r.json()['error']['code'],
                #             r.json()['error']['status'],
                #             r.json()['error']['message'],
                #         )
                # elif r.status_code == 403:
                #     raise SafeBrowsingPermissionDenied(r.json()['error']['message'])
                # else:
                #     raise SafeBrowsingWeirdError(r.status_code, "", "", "")
                results.update(dict([(u, {"malicious": False}) for u in urls]))
                continue

        return results

    def lookup_url(self, url, platforms=["ANY_PLATFORM"]):
        """
        Online lookup of a single url
        """
        r = self.lookup_urls([url], platforms=platforms)
        return r[url]


if __name__ == '__main__':
    from tqdm import tqdm
    import os
    import tldextract
    API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./knowledge_expansion/api_key.txt').readlines()]
    gsb = SafeBrowsing(API_KEY)

    # test_folder = './datasets/phish_sample_30k'
    # lookup_urls = []
    # lookup_folders = []
    # for num, folder in tqdm(enumerate(os.listdir(test_folder))):
    #
    #     shot_path = os.path.join(test_folder, folder, 'shot.png')
    #     info_path = os.path.join(test_folder, folder, 'info.txt')
    #     url = eval(open(info_path, encoding='ISO 8859-1').read())['url']
    #     query_domain = tldextract.extract(url).domain
    #     query_tld = tldextract.extract(url).suffix
    #
    #     lookup_url = 'https://'+query_domain+'.'+query_tld
    #     lookup_urls.append(lookup_url)
    #     lookup_folders.append(folder)

    # for ct, urll in enumerate(chunks(lookup_urls, 50)):
        # results = gsb.lookup_urls(urll)
        # with open("./datasets/gsb_lookup_phish30k_{}.json".format(ct), "w") as fp:
        #     json.dump(results, fp)

    # results = {}
    # for chunk_ct, urll in enumerate(chunks(lookup_urls, 50)):
    #     with open("./datasets/gsb_lookup_phish30k_{}.json".format(chunk_ct), "r") as fp:
    #         results_this = json.load(fp)
    #     results.update(results_this)
    #
    # ct = 0
    # for num, folder in tqdm(enumerate(os.listdir(test_folder))):
    #
    #     shot_path = os.path.join(test_folder, folder, 'shot.png')
    #     info_path = os.path.join(test_folder, folder, 'info.txt')
    #     url = eval(open(info_path, encoding='ISO 8859-1').read())['url']
    #     query_domain = tldextract.extract(url).domain
    #     query_tld = tldextract.extract(url).suffix
    #
    #     lookup_url = 'https://'+query_domain+'.'+query_tld
    #     if results[lookup_url]['malicious'] == False:
    #         ct += 1
    #         # shutil.rmtree(os.path.join())
    # print(ct)

    print(gsb.lookup_url('https://whisperingpinescottages.ca/'))