import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    df_llm = pd.read_table('./datasets/dynapd_wo_validation.txt', sep='\t', header=None)
    df_llm.columns = ["hash", 'pred', 'brand', 'brand_recog_time',  'crp_pred_time', 'crp_transit_time']

    df_llm_adv = pd.read_table('./datasets/dynapd_llm_adv.txt', sep='\t', header=None)
    df_llm_adv.columns = ["hash", 'pred', 'brand', 'brand_recog_time',  'crp_pred_time', 'crp_transit_time']

    df_pedia = pd.read_table('./datasets/dynapd_phishpedia.txt', sep='\t', header=None)
    df_pedia.columns = ["hash", 'pred', 'brand', 'time']

    df_tention = pd.read_table('./datasets/dynapd_phishintention.txt', sep='\t', header=None)
    df_tention.columns = ["hash", 'pred', 'brand', 'time']

    common_hashes = set(df_llm['hash']).intersection(set(df_pedia['hash'])).intersection(set(df_tention['hash'])).intersection(set(df_llm_adv['hash']))
    common_hashes = [x for x in common_hashes if os.path.exists(os.path.join('./datasets/dynapd', x))]

    df_llm = df_llm[df_llm['hash'].isin(common_hashes)]
    df_pedia = df_pedia[df_pedia['hash'].isin(common_hashes)]
    df_tention = df_tention[df_tention['hash'].isin(common_hashes)]
    df_llm_adv = df_llm_adv[df_llm_adv['hash'].isin(common_hashes)]

    total = len(df_llm)
    count_llm = sum(df_llm['pred'] == 'phish')
    count_llm_adv = sum(df_llm_adv['pred'] == 'phish')
    count_pedia = sum(df_pedia['pred'] == 1)
    count_tention = sum(df_tention['pred'] == 1)

    print(np.median(df_llm['brand_recog_time'] + df_llm['crp_pred_time'] + df_llm['crp_transit_time']))
    print(np.median(df_pedia['time']))
    print(np.median(df_tention['time']))


    df_llm_b = pd.read_table('./datasets/alexa7k_wo_validation.txt', sep='\t', header=None)
    df_llm_b.columns = ["hash", 'pred', 'brand', 'brand_recog_time',  'crp_pred_time', 'crp_transit_time']

    df_pedia_b = pd.read_table('./datasets/alexa7k_phishpedia.txt', sep='\t', header=None)
    df_pedia_b.columns = ["hash", 'pred', 'brand', 'time']

    df_tention_b = pd.read_table('./datasets/alexa7k_phishintention.txt', sep='\t', header=None)
    df_tention_b.columns = ["hash", 'pred', 'brand', 'time']

    count_llm_b = sum(df_llm_b['pred'] == 'phish')
    count_pedia_b = sum(df_pedia_b['pred'] == 1)
    count_tention_b = sum(df_tention_b['pred'] == 1)
    print(f"Total = {total}, "
          f"LLM recall = {count_llm/total}, "
          f"LLM HTML obfuscation recall = {count_llm_adv/total} \n"
          f"Phishpedia recall = {count_pedia/total}, " 
          f"PhishIntention recall = {count_tention/total}\n"
          f"LLM precision = {count_llm/(count_llm_b+count_llm)}, "
          f"Phishpedia precision = {count_pedia/(count_pedia_b+count_pedia)}, "
          f"PhishIntention precision = {count_tention/(count_tention_b+count_tention)}, ")
    # Total = 6334 LLM recall = 0.7586043574360594, Phishpedia recall = 0.2826018313861699, PhishIntention recall = 0.25970950426270917

