
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
    df_pos = [x for x in df_pos if not any([yy.strip().lower() in x[2].lower() for yy in webhosting_domains])]

    return df_pos



def compute_precision_recall(reported_folders, gt_folders):
    intersection = set(reported_folders).intersection(set(gt_folders))
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

def runtime(result_txt):
    df = [x.strip().split('\t') for x in open(result_txt, encoding='ISO-8859-1').readlines()[1:]]
    df = [x[-1].split('|') for x in df if (len(x) >= 3) if not ((x[-1].split('|')[0] == '0') and (x[-1].split('|')[1] == '0') and
                                                                (x[-1].split('|')[2] == '0') and (x[-1].split('|')[3] == '0'))]
    phishintention_time = [eval(x[0]) for x in df]
    knowledge_time = [eval(x[1]) for x in df]
    interaction_time = [eval(x[2]) for x in df]
    return phishintention_time, knowledge_time, interaction_time



if __name__ == '__main__':
    '''Save reported phishing'''
    start_date = date(2023, 8, 1)
    end_date = date(2023, 8, 2)
    g = gwrapper()
    all_records = g.get_records()
    for single_date in daterange(start_date, end_date):
        date_ = single_date.strftime("%Y-%m-%d")

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
        pedia_precision, pedia_recall = compute_precision_recall(pedia_pos_folders, today_pos_folders)
        print(f'Phishpedia precision = {pedia_precision}, Phishpedia recall = {pedia_recall}')
        intention_precision, intention_recall = compute_precision_recall(intention_pos_folders, today_pos_folders)
        print(f'PhishIntention precision = {intention_precision}, PhishIntention recall = {intention_recall}')

    '''Move reported phishing into a seperate folder, for labeling'''
    os.makedirs('./datasets/phishing_TP_examples', exist_ok=True)
    start_date = date(2023, 8, 1)
    end_date = date(2023, 8, 7)
    for single_date in daterange(start_date, end_date):
        today_date = single_date.strftime("%Y-%m-%d")
        df_pos = get_pos_site(f'./field_study/results/{today_date}_phishllm.txt')
        df_pos_folders = [x[0] for x in df_pos]

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
