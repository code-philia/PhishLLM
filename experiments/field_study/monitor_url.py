import gspread # gspread==5.10.0 gspread-dataframe==3.0.8
from oauth2client.service_account import ServiceAccountCredentials
import pprint
import pandas as pd
from gspread import Cell
import os
import requests
import socket
from datetime import datetime
from tldextract import tldextract
import whois # pip install python-whois
import time
import http.client
import math
import ast
import json
# os.environ['ipinfo_token'] = open('./datasets/ipinfo_token.txt').read()
# os.environ['abstract_api'] = open('./datasets/abstract_api.txt').read()

class gwrapper_monitor():
    def __init__(self):
        scope = [
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/drive.file'
        ]
        file_name = './datasets/google_cloud.json'
        creds = ServiceAccountCredentials.from_json_keyfile_name(file_name, scope)
        client = gspread.authorize(creds)

        # Fetch the sheet
        self.sheet = client.open('phishllm monitor').sheet1

    def get_records(self):
        print('get records')
        return self.sheet.get_all_records()

    def update_list(self, to_update):
        self.sheet.append_rows(to_update)

    def update_cell(self, question_id, country, geo_loc, org, domain_age, sector):
        # update google sheet yes, no, unsure columns
        cells = []
        id = int(question_id)
        cells.append(Cell(row=id, col=4, value=country))
        cells.append(Cell(row=id, col=5, value=geo_loc))
        cells.append(Cell(row=id, col=6, value=org))
        cells.append(Cell(row=id, col=7, value=domain_age))
        cells.append(Cell(row=id, col=9, value=sector))

        self.sheet.update_cells(cells)

def get_status_code(url):
    try:
        response = requests.get(url, timeout=30)
        return response.status_code
    except requests.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
        return None

def resolve_url_to_ips(url):
    try:
        ip_address = socket.gethostbyname(url)
        return ip_address
    except socket.gaierror:
        return None

def get_ip_geolocation(ip):
    access_token = os.getenv("ipinfo_token")
    url = f"https://ipinfo.io/{ip}/json?token={access_token}"
    inference_done = False
    while not inference_done:
        try:
            response = requests.get(url)
            data = response.json()
            inference_done = True
            try:
                return data['country'], data['loc'], data['org']
            except KeyError:
                return 0,0,0
        except:
            time.sleep(5)
            continue

def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        if w.creation_date:
            if type(w.creation_date) is list:
                creation_date = w.creation_date[0]
            else:
                creation_date = w.creation_date
            age = (datetime.now() - creation_date).days // 365 # in year
            return age
        else:
            return None
    except Exception as e:
        return None

def classify_url(brand_url):

    api_key = os.getenv('abstract_api')
    inference_done = False
    while not inference_done:
        try:
            response = requests.get(
                f"https://companyenrichment.abstractapi.com/v1/?api_key={api_key}&domain={brand_url}")
            inference_done = True
            data = json.loads(response.content.decode("utf-8"))
            return data["industry"]
        except:
            time.sleep(5)
            continue

if __name__ == '__main__':

    try:
        while True:
            base = "./datasets/phishing_TP_examples"
            gs_sheet = gwrapper_monitor()

            # update sheets
            rows = gs_sheet.get_records()
            folder_names = list(map(lambda x: x['foldername'], rows))
            to_update = []
            for i in os.listdir(base):
                folder = os.path.join(base, i)
                if datetime.strptime(i, "%Y-%m-%d").date() <= datetime.strptime('2023-08-06', "%Y-%m-%d").date():
                    continue
                df = pd.read_csv('./field_study/results/{}_phishllm.txt'.format(i), sep='\t', encoding='ISO-8859-1')
                for j in os.listdir(folder):
                    data_folder = os.path.join(folder, j)
                    if j in folder_names:
                        continue
                    else:
                        brand = df[df['folder'] == j]['target_prediction'].values
                        brand = brand[0]
                        if isinstance(brand, float) and (math.isinf(brand) or math.isnan(brand)):
                            continue
                        if brand in open('./datasets/hosting_blacklists.txt').read():
                            continue
                        info_file = os.path.join(data_folder, 'info.txt')
                        if os.path.exists(info_file):
                            with open(info_file, 'r') as f:
                                url = f.read()
                        else:
                            url = "Cannot find info file"
                        to_update.append([i, url, j, 0, 0, 0, 0, brand, 0])

            print(to_update)
            gs_sheet.update_list(to_update)

            ## update cell
            rows = gs_sheet.get_records()
            for it, row in enumerate(rows):
                id = it + 2
                url = row['url']
                domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
                country = row['country']
                geo_loc = row['geo_loc']
                org = row['org']
                domain_age = row['domain_age']
                brand = row['brand']
                sector = row['sector']
                date = row['date']

                if row['country'] == 0:
                    print(id, row)
                    ip = resolve_url_to_ips(domain)
                    print('IP: ', ip)

                    # country
                    if ip and (row['country'] == 0 or row['geo_loc'] == 0 or row['org'] == 0):
                        country, geo_loc, org = get_ip_geolocation(ip)
                        print('From country: ', country)
                        print('From loc: ', geo_loc)
                        print('From org: ', org)

                    # domain_age
                    if row['domain_age'] == 0:
                        age = get_domain_age(domain)
                        if age:
                            domain_age = age
                            print('Domain age: ', domain_age)

                    if row['sector'] == 0:
                        sector = classify_url(brand)
                        time.sleep(1)
                        print('Sector: ', sector)
                        sector = str(sector)
                    gs_sheet.update_cell(id, country, geo_loc, org, domain_age, sector)
                    time.sleep(0.5)


    except KeyboardInterrupt:
        print("\nLoop interrupted by user. Exiting.")
