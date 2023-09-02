
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import re
from field_study.tele.gsheets import gwrapper
import collections

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


def compute_precision_recall(reported_folders, gt_folders):
    intersection = set(reported_folders).intersection(set(gt_folders))
    # print(list(set(reported_folders) - set(intersection)))
    print(list(set(gt_folders) - set(intersection)))
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
    runtime = [eval(x[-1]) + eval(x[-2]) + eval(x[-3]) for x in df]
    print(f'Median runtime LLM = {np.median(runtime)}')

def runtime_base(result_txt):
    df = [x.strip().split('\t') for x in open(result_txt, encoding='ISO-8859-1').readlines()[1:]]
    runtime = [eval(x[-1]) for x in df]
    print(f'Median runtime = {np.median(runtime)}')

if __name__ == '__main__':
    '''Save reported phishing'''
    g = gwrapper()
    all_records = g.get_records()

    date_ = "2023-08-11"
    llm_txt = get_pos_site('./field_study/results/{}_phishllm.txt'.format(date_))
    llm_pos_folders = [x[0] for x in llm_txt]

    pedia_txt = get_pos_site('./field_study/results/{}_phishpedia.txt'.format(date_))
    pedia_pos_folders = [x[0] for x in pedia_txt]

    intention_txt = get_pos_site('./field_study/results/{}_phishintention.txt'.format(date_))
    intention_pos_folders = [x[0] for x in intention_txt]

    # # get the labels for today
    todays_labels = [x for x in all_records if x['date'] == date_]
    todays_pos = [x for x in todays_labels if x['yes']>0]
    todays_ignore = [x for x in todays_labels if x['unsure']>0]
    today_pos_folders = [x['foldername'] for x in todays_pos]
    today_ignore_folders = [x['foldername'] for x in todays_ignore]

    # '''compute precision & recall'''
    llm_pos_folders = list(set(llm_pos_folders) - set(today_ignore_folders))
    pedia_pos_folders = list(set(pedia_pos_folders) - set(today_ignore_folders))
    intention_pos_folders = list(set(intention_pos_folders) - set(today_ignore_folders))

    llm_precision, llm_recall = compute_precision_recall(llm_pos_folders, today_pos_folders)
    print(f'PhishLLM precision = {llm_precision}, PhishLLM recall = {llm_recall}')
    runtime_llm('./field_study/results/{}_phishllm.txt'.format(date_))
    pedia_precision, pedia_recall = compute_precision_recall(pedia_pos_folders, today_pos_folders)
    print(f'Phishpedia precision = {pedia_precision}, Phishpedia recall = {pedia_recall}')
    runtime_base('./field_study/results/{}_phishpedia.txt'.format(date_))
    intention_precision, intention_recall = compute_precision_recall(intention_pos_folders, today_pos_folders)
    print(f'PhishIntention precision = {intention_precision}, PhishIntention recall = {intention_recall}')
    runtime_base('./field_study/results/{}_phishintention.txt'.format(date_))

    '''Move reported phishing into a seperate folder, for labeling'''
    os.makedirs('./datasets/phishing_TP_examples', exist_ok=True)
    start_date = date(2023, 8, 7)
    today = datetime.today().date()
    end_date = today + timedelta(days=1)
    for single_date in daterange(start_date, end_date):
        today_date = single_date.strftime("%Y-%m-%d")
        df_pos = get_pos_site(f'./field_study/results/{today_date}_phishllm.txt')
        df_pos_folders = [x[0] for x in df_pos]

        if os.path.exists('./field_study/results/{}_phishpedia.txt'.format(today_date)) and \
                os.path.exists('./field_study/results/{}_phishintention.txt'.format(today_date)):
            pedia_pos = get_pos_site('./field_study/results/{}_phishpedia.txt'.format(today_date))
            intention_pos = get_pos_site('./field_study/results/{}_phishintention.txt'.format(today_date))
            print(f'Date {today_date} Phishllm # phishing = {len(df_pos)}, Phishpedia # phishing = {len(pedia_pos)}, PhishIntention # phishing = {len(intention_pos)} \n')

        os.makedirs(os.path.join('./datasets/phishing_TP_examples', today_date), exist_ok=True)
        for folder in df_pos_folders:
            try:
                shutil.copytree(os.path.join('./datasets/field_study', today_date, folder),
                            os.path.join('./datasets/phishing_TP_examples', today_date, folder))
            except FileExistsError:
                continue
            except FileNotFoundError:
                continue
    print()
