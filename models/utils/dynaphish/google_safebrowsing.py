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

