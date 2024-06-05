
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import re
from experiments.field_study.tele.gsheets import gwrapper
import collections
import pickle

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_pos_site(result_txt):

    df = [x.strip().split('\t') for x in open(result_txt, encoding='ISO-8859-1').readlines()]
    df_pos = [x for x in df if (x[1] == 'phish' or x[1] == '1')]
    webhosting_domains = open('./datasets/hosting_blacklists.txt').readlines()
    df_pos = [x for x in df_pos if
              (not any([x[0].startswith(yy) for yy in ['libreddit', 'ebay', 'autodiscover', 'outlook',
                                                     'vineasx', 'office', 'onlyoffice',
                                                     'portainer', 'rss']])) and
              (not any([yy.strip().lower() in x[2].lower() for yy in webhosting_domains]))]
    return df_pos

def get_pos_site_dynaphish(result_txt, domain_map_path):

    df = [x.strip().split('\t') for x in open(result_txt, encoding='ISO-8859-1').readlines()]
    df_pos = [x for x in df if (x[1] == 'phish' or x[1] == '1')]
    webhosting_domains = open('./datasets/hosting_blacklists.txt').readlines()

    df_pos = [x for x in df_pos if
              (not any([x[0].startswith(yy) for yy in ['libreddit', 'ebay', 'autodiscover', 'outlook',
                                                     'vineasx', 'office', 'onlyoffice',
                                                     'portainer', 'rss']])) ]

    with open(domain_map_path, 'rb') as handle:
        domain_map = pickle.load(handle)

    filtered_df_pos  = []
    for x in df_pos:
        if x[2] in domain_map.keys():
            if not any([yy.strip().lower() in x[2].lower() for yy in webhosting_domains]):
                if not any([yy in webhosting_domains for yy in domain_map[x[2]]]):
                    filtered_df_pos.append(x)
        else:
            filtered_df_pos.append(x)
    return filtered_df_pos


def compute_precision_recall(reported_folders, gt_folders):
    intersection = set(reported_folders).intersection(set(gt_folders))
    print(f'Number of reported and verified phishing = {len(intersection)}')
    print(list(set(reported_folders) - set(intersection)))
    # print(list(set(gt_folders) - set(intersection)))
    # [print(x) for x in list(set(gt_folders) - set(intersection))]
    recall = len(intersection) / len(gt_folders)
    if len(reported_folders) > 0:
        precision = len(intersection) / len(reported_folders)
    else:
        precision = 1.
    return precision, recall

def compute_overall_precision_recall(reported_folders, gt_folders):
    intersection = set(reported_folders).intersection(set(gt_folders))
    recall = len(intersection) / len(set(gt_folders))
    if len(reported_folders) > 0:
        precision = len(intersection) / len(set(reported_folders))
    else:
        precision = 1.
    return precision, recall

def runtime_llm(result_txt):
    df = [x.strip().split('\t') for x in open(result_txt, encoding='ISO-8859-1').readlines()[1:]]
    runtime = [eval(x[-1]) + eval(x[-2]) + eval(x[-3]) for x in df if eval(x[-1]) + eval(x[-2]) + eval(x[-3]) != 0]
    print(f'Median runtime LLM = {np.median(runtime)}')

def runtime_base(result_txt):
    df = [x.strip().split('\t') for x in open(result_txt, encoding='ISO-8859-1').readlines()[1:]]
    runtime = [eval(x[-1]) for x in df if eval(x[-1]) != 0]
    print(f'Median runtime = {np.median(runtime)}')

if __name__ == '__main__':
    # modified_domain_map_path = '/home/ruofan/git_space/ScamDet/model_chain/dynaphish/domain_map.pkl'
    # # modified_domain_map_path = '/home/ruofan/anaconda3/envs/myenv/lib/python3.8/site-packages/phishintention/src/phishpedia_siamese/domain_map.pkl'
    #
    # with open(modified_domain_map_path, "rb") as handle:
    #     domain_map = pickle.load(handle)
    # #
    # # domain_map['atlasformen TAILLÉ POUR LAVENTURE'].extend(['atlasformen.ch', 'atlasformen.pl', 'atlasformen.ca'])
    # domain_map['inronline.net'].extend(['inronline.ca'])
    # #
    # with open(modified_domain_map_path, "wb") as handle:
    #     pickle.dump(domain_map, handle)
    # exit()
    #

    '''Move reported phishing into a seperate folder, for labeling'''
    # os.makedirs('./datasets/phishing_TP_examples', exist_ok=True)
    # start_date = date(2024, 1, 29)
    # # today = datetime.today().date()
    # # end_date = today + timedelta(days=1)
    # end_date = date(2024, 1, 30)
    # pedia_total = 0
    # intention_total = 0
    # dyna_total = 0
    # llm_total = 0
    #
    # pedia_pos_list = []
    # intention_pos_list = []
    # dyna_pos_list = []
    # llm_pos_list = []
    #
    # for single_date in daterange(start_date, end_date):
    #     today_date = single_date.strftime("%Y-%m-%d")
    #     df_pos = get_pos_site(f'./field_study/results/{today_date}_phishllm.txt')
    #     df_pos_folders = [x[0] for x in df_pos]
    #     print(f'Date {today_date} Phishllm # phishing = {len(df_pos)}')
    #     llm_total += len(df_pos)
    #     llm_pos_list.extend(df_pos_folders)
    #
    #     if os.path.exists('./field_study/results/{}_phishpedia.txt'.format(today_date)):
    #         pedia_pos = get_pos_site('./field_study/results/{}_phishpedia.txt'.format(today_date))
    #         print(f'Phishpedia # phishing = {len(pedia_pos)}')
    #         pedia_total += len(pedia_pos)
    #         pedia_pos_folders = [x[0] for x in pedia_pos]
    #         pedia_pos_list.extend(pedia_pos_folders)
    #
    #     if os.path.exists('./field_study/results/{}_phishintention.txt'.format(today_date)):
    #         intention_pos = get_pos_site('./field_study/results/{}_phishintention.txt'.format(today_date))
    #         print(f'PhishIntention # phishing = {len(intention_pos)}')
    #         intention_total += len(intention_pos)
    #         intention_pos_folders = [x[0] for x in intention_pos]
    #         intention_pos_list.extend(intention_pos_folders)
    #
    #     if os.path.exists('./field_study/results/{}_dynaphish.txt'.format(today_date)):
    #         dynaphish_pos = get_pos_site_dynaphish('./field_study/results/{}_dynaphish.txt'.format(today_date),
    #                                                domain_map_path='./model_chain/dynaphish/domain_map.pkl')
    #         print(f'DynaPhish # phishing = {len(dynaphish_pos)}')
    #         dyna_total += len(dynaphish_pos)
    #         dyna_pos_folders = [x[0] for x in dynaphish_pos]
    #         dyna_pos_list.extend(dyna_pos_folders)
    #
    #     os.makedirs(os.path.join('./datasets/phishing_TP_examples', today_date), exist_ok=True)
    #     for folder in set(df_pos_folders).union(set(dyna_pos_folders)).union(set(intention_pos_folders)).union(set(pedia_pos_folders)):
    #         try:
    #             shutil.copytree(os.path.join('./datasets/field_study', today_date, folder),
    #                             os.path.join('./datasets/phishing_TP_examples', today_date, folder))
    #         except (FileExistsError, FileNotFoundError):
    #             continue
    # print(
    #     f'Total Phishllm # phishing = {llm_total}, Phishpedia # phishing = {pedia_total}, PhishIntention # phishing = {intention_total}, DynaPhish # phishing = {dyna_total} \n')
    # exit()

    '''Save reported phishing'''
    g = gwrapper()
    all_records = g.get_records()
    pedia_pos_list = []
    pedia_target = []

    intention_pos_list = []
    intention_target = []

    dyna_pos_list = []
    dyna_target = []

    llm_pos_list = []
    llm_target = []
    gt_phish_list = []
    start_date = date(2024, 1, 25)
    end_date = date(2024, 1, 30)

    for single_date in daterange(start_date, end_date):
        date_ = single_date.strftime("%Y-%m-%d")
        llm_txt = get_pos_site('./field_study/results/{}_phishllm.txt'.format(date_))
        llm_pos_folders = [x[0] for x in llm_txt]

        pedia_txt = get_pos_site('./field_study/results/{}_phishpedia.txt'.format(date_))
        pedia_pos_folders = [x[0] for x in pedia_txt]

        intention_txt = get_pos_site('./field_study/results/{}_phishintention.txt'.format(date_))
        intention_pos_folders = [x[0] for x in intention_txt]

        dyna_txt = get_pos_site_dynaphish('./field_study/results/{}_dynaphish.txt'.format(date_), domain_map_path='./model_chain/dynaphish/domain_map.pkl')
        dyna_pos_folders = [x[0] for x in dyna_txt]

        # # get the labels for today
        todays_labels = [x for x in all_records if x['date'] == date_]
        todays_pos = [x for x in todays_labels if x['yes']>0]
        todays_ignore = [x for x in todays_labels if x['unsure']>0]
        today_pos_folders = [x['foldername'] for x in todays_pos]
        today_ignore_folders = [x['foldername'] for x in todays_ignore]
        # print(f'Total gt positive = {len(set(today_pos_folders))}')

        # '''compute precision & recall'''
        llm_pos_folders = list(set(llm_pos_folders) - set(today_ignore_folders))
        pedia_pos_folders = list(set(pedia_pos_folders) - set(today_ignore_folders))
        intention_pos_folders = list(set(intention_pos_folders) - set(today_ignore_folders))
        dyna_pos_folders = list(set(dyna_pos_folders) - set(today_ignore_folders))

        llm_pos_list.extend(llm_pos_folders)
        pedia_pos_list.extend(pedia_pos_folders)
        intention_pos_list.extend(intention_pos_folders)
        dyna_pos_list.extend(dyna_pos_folders)
        gt_phish_list.extend(today_pos_folders)

        intention_target.extend([x[2] for x in intention_txt if x[0] in today_pos_folders])
        pedia_target.extend([x[2] for x in pedia_txt if x[0] in today_pos_folders])
        dyna_target.extend([x[2] for x in dyna_txt if x[0] in today_pos_folders])
        llm_target.extend([x[2] for x in llm_txt if x[0] in today_pos_folders])


    llm_precision, llm_recall = compute_precision_recall(llm_pos_list, gt_phish_list)
    # {'cigaroslot95e.xyz',
    # 'livecrmcopy.harmanmotors.xyz', 'stimagtest.zeno-online.nl',
    # 'structuredproductsonline.com', 'wel24.weeklynewschannel.club',
    # 'mspaeth.hth.staging.sellsite.scireum.com',
    # 'www.nt-schadenportal.de',
    # 'www.tradebeefx.me.kamalvasini.com', 'feature-359-miquillsouth.aspens.services',
    # 'sumatosoft-alliance.cc', 'instylebt.xyz',
    # 'ws909.appwsh.top', 'clients.siscog.com', 'exchange.xencon.net',
    # 'www.cigaroslot93e.xyz', 'mail.paperhelp.vip',
    # 'mail.antispampbx.xyz', 'mona4dsenin.xyz',
    # 'wwwwww.media.moodle-optivet-u3030.vm.elestio.app',
    # 'megacity.vps1918171.fastwebserver.de',
    # 'avenc.casalsonline.net', 'blog.blog.old.shop.adaptivehelp.com',
    # 'fujiclean-onlinestore.com', 'cigaroslot797a.xyz', 'www.cigaroslot95e.xyz', 'portal-staging.plan2book.com',
    # 'tipografiaformertest.it', 'damascussteel.store.insilicomediacompany.com', 'support.groometrans.com',
    # 'netflixclone.jamesryandcrz.com', 'pp-hot-4064-4093.pozyczkaplus-pl.avgr.it',
    # 'pve01.proxcloud.cc', 'sber.avito.yandex.yandex.8eod5qap3riswow.system.adaptivehelp.com',
    # 'cpanel.flsafetytowingandrepairs.com', 'www.ticket.maxxi.archeoares.it',
    #  'www.jiaoyupan.vip', 'imprendiaservizisas.ranocchiworldservices.com', 'rajaku4djaya.shop',
    #  'device-9126ceb2-dc66-42d9-ba4c-56c782b60851.remotewd.com', 'blissfullcouple.com.blissfullmind.in',
    #  'www.www.media.moodle-optivet-u3030.vm.elestio.app', 'device-77da372b-104f-437c-916f-f487805fc8df.remotewd.com',
    #  'vpn.westmakina.com.tr-mvgrbmkghdc.dynamic-m.com',
    #  'sabnzbd.z.doofus.club', 'sberuniver.merchservice.ru'}

    print(f'PhishLLM precision = {llm_precision}')
    runtime_llm('./field_study/results/{}_phishllm.txt'.format(date_))
    print(f'No. distinct target brands = {len(set(llm_target))}')

    pedia_precision, pedia_recall = compute_precision_recall(pedia_pos_list, gt_phish_list)
    print(f'Phishpedia precision = {pedia_precision}')
    runtime_base('./field_study/results/{}_phishpedia.txt'.format(date_))
    print(f'No. distinct target brands = {len(set(pedia_target))}')

    intention_precision, intention_recall = compute_precision_recall(intention_pos_list, gt_phish_list)
    print(f'PhishIntention precision = {intention_precision}')
    runtime_base('./field_study/results/{}_phishintention.txt'.format(date_))
    print(f'No. distinct target brands = {len(set(intention_target))}')

    dyna_precision, dyna_recall = compute_precision_recall(dyna_pos_list, gt_phish_list)
    print(f'DynaPhish precision = {dyna_precision}')
    runtime_base('./field_study/results/{}_dynaphish.txt'.format(date_))
    print(f'No. distinct target brands = {len(set(dyna_target))}')
    # exit()

    # print((llm_total-pedia_total)/pedia_total)
    # print((llm_total-pedia_total)/30)
    # pedia_not_llm = set(pedia_pos_list) - set(llm_pos_list)
    # intention_not_llm = set(intention_pos_list) - set(llm_pos_list)
    # print(len(pedia_not_llm))
    # print(len(intention_not_llm))
    #
    # # Phishpedia reported positives
    # for folder in pedia_pos_list:
    #     exist = False
    #     for dirpath, dirnames, filenames in os.walk('./datasets/field_study'):
    #         # Check if the folder name exists in the current directory's subdirectories
    #         if folder in dirnames:
    #             exist = True
    #             break
    #     if exist:
    #         shot_path = os.path.join(dirpath, folder, 'shot.png')
    #         print(shot_path)
    #         os.makedirs('./datasets/field_study_pedia_reported', exist_ok=True)
    #         shutil.copyfile(shot_path, os.path.join('./datasets/field_study_pedia_reported', folder+'.png'))
    #
    # # PhishIntention reported positives
    # for folder in intention_pos_list:
    #     exist = False
    #     for dirpath, dirnames, filenames in os.walk('./datasets/field_study'):
    #         # Check if the folder name exists in the current directory's subdirectories
    #         if folder in dirnames:
    #             exist = True
    #             break
    #     if exist:
    #         shot_path = os.path.join(dirpath, folder, 'shot.png')
    #         print(shot_path)
    #         os.makedirs('./datasets/field_study_intention_reported', exist_ok=True)
    #         shutil.copyfile(shot_path, os.path.join('./datasets/field_study_intention_reported', folder+'.png'))

    # Phishpedia but not LLM:
    # PhishIntention but not LLM:
    # for folder in pedia_not_llm:
    #     exist = False
    #     for dirpath, dirnames, filenames in os.walk('./datasets/field_study'):
    #         # Check if the folder name exists in the current directory's subdirectories
    #         if folder in dirnames:
    #             exist = True
    #             break
    #     if exist:
    #         shot_path = os.path.join(dirpath, folder, 'shot.png')
    #         print(shot_path)
    #         os.makedirs('./datasets/field_study_pedia_not_llm', exist_ok=True)
    #         shutil.copyfile(shot_path, os.path.join('./datasets/field_study_pedia_not_llm', folder+'.png'))
    #
    # for folder in intention_not_llm:
    #     exist = False
    #     for dirpath, dirnames, filenames in os.walk('./datasets/field_study'):
    #         # Check if the folder name exists in the current directory's subdirectories
    #         if folder in dirnames:
    #             exist = True
    #             break
    #     if exist:
    #         shot_path = os.path.join(dirpath, folder, 'shot.png')
    #         os.makedirs('./datasets/field_study_intention_not_llm', exist_ok=True)
    #         shutil.copyfile(shot_path, os.path.join('./datasets/field_study_intention_not_llm', folder+'.png'))

    # {'redminepmtool.ainqaplatform.in', 'www.formti.arteti.com', 'sell.amazon.fr',
    #  'argo-hytos-dev1-pingfed-runtime.cloud.thingworx.com', 'orangehomeloans.smartonline.com.au', 'vo.aximplatform.com'}

    '''How many are not in the 277 brands'''
    # load the domain_map
    # import pickle
    # from tldextract import tldextract
    # with open('/home/ruofan/anaconda3/envs/myenv/lib/python3.8/site-packages/phishintention/src/phishpedia_siamese/domain_map.pkl', 'rb') as handle:
    #     domain_map = pickle.load(handle)

    # total_counts = 0
    # unique_unseen_brands = []
    # unique_counts = 0
    # for k in llm_target:
    #     if k not in str(list(domain_map.values())):
    #         total_counts += 1
    #         if k not in unique_unseen_brands:
    #             unique_counts += 1
    #             unique_unseen_brands.append(k)
    # print(total_counts)
    # print(unique_counts)

    # total_counts = 0
    # unique_unseen_brands = []
    # unique_counts = 0
    # for k in dyna_target:
    #     if k not in str(list(domain_map.values())):
    #         total_counts += 1
    #         if k not in unique_unseen_brands:
    #             unique_counts += 1
    #             unique_unseen_brands.append(k)
    # print(total_counts)
    # print(unique_counts)

    '''How many are non-english sites'''
    # from bs4 import BeautifulSoup
    # from langdetect import detect
    # from collections import Counter
    #
    # # Function to determine if the webpage text is in English
    # def detect_language(html_content):
    #     # Parse the HTML content
    #     soup = BeautifulSoup(html_content, 'html.parser')
    #
    #     # Extract text from the parsed HTML
    #     text = soup.get_text(strip=True)
    #
    #     try:
    #         # Detect the language of the text
    #         language = detect(text)
    #         return language
    #     except Exception as e:
    #         print(f"Language detection error: {e}")
    #         return None
    #
    #
    # language_counter = Counter()
    # count = 0
    # # Traverse the directory structure
    # for dirpath, dirnames, filenames in os.walk('./datasets/field_study_TP_password_is_llm'):
    #     for name in filenames:
    #         if name.endswith('html.txt'):
    #             file_path = os.path.join(dirpath, name)
    #             with open(file_path, 'r', encoding='utf-8') as file:
    #                 content = file.read()
    #                 # Detect language and update the counter
    #                 language = detect_language(content)
    #                 if language:
    #                     language_counter[language] += 1
    #             count += 1
    #
    # # Get the top-5 languages and their frequencies
    # top_5_languages = language_counter.most_common(5)
    #
    # # Print the results
    # for lang, freq in top_5_languages:
    #     print(f"Language: {lang}, Frequency: {freq/count}")
    # Language: en, Frequency: 342/502
    # Language: de, Frequency: 29/502
    # Language: es, Frequency: 22/502
    # Language: zh-cn, Frequency: 19/502
    # Language: pt, Frequency: 17/502