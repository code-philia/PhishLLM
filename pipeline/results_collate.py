import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    df_llm = pd.read_table('./datasets/dynapd_llm.txt', sep='\t', header=None)
    df_llm.columns = ["hash", 'pred', 'brand', 'brand_recog_time',  'crp_pred_time', 'crp_transit_time']

    df_llama2 = pd.read_table('./datasets/dynapd_llama2.txt', sep='\t', header=None)
    df_llama2.columns = ["hash", 'pred', 'brand', 'brand_recog_time',  'crp_pred_time', 'crp_transit_time']

    df_llm_adv = pd.read_table('./datasets/dynapd_llm_adv.txt', sep='\t', header=None)
    df_llm_adv.columns = ["hash", 'pred', 'brand', 'brand_recog_time',  'crp_pred_time', 'crp_transit_time']

    df_pedia = pd.read_table('./datasets/dynapd_phishpedia.txt', sep='\t', header=None)
    df_pedia.columns = ["hash", 'pred', 'brand', 'time']

    df_tention = pd.read_table('./datasets/dynapd_phishintention.txt', sep='\t', header=None)
    df_tention.columns = ["hash", 'pred', 'brand', 'time']

    common_hashes = set(df_llm['hash']).intersection(set(df_pedia['hash'])).intersection(set(df_tention['hash'])).intersection(set(df_llm_adv['hash'])).intersection(set(df_llama2['hash']))
    common_hashes = [x for x in common_hashes if os.path.exists(os.path.join('./datasets/dynapd', x))]

    df_llm = df_llm[df_llm['hash'].isin(common_hashes)]
    df_llama2 = df_llama2[df_llama2['hash'].isin(common_hashes)]
    df_pedia = df_pedia[df_pedia['hash'].isin(common_hashes)]
    df_tention = df_tention[df_tention['hash'].isin(common_hashes)]
    df_llm_adv = df_llm_adv[df_llm_adv['hash'].isin(common_hashes)]

    total = len(df_llm)
    count_llm = sum(df_llm['pred'] == 'phish')
    count_llama2 = sum(df_llama2['pred'] == 'phish')
    count_llm_adv = sum(df_llm_adv['pred'] == 'phish')
    count_pedia = sum(df_pedia['pred'] == 1)
    count_tention = sum(df_tention['pred'] == 1)

    print(np.median(df_llm['brand_recog_time'] + df_llm['crp_pred_time'] + df_llm['crp_transit_time']))
    print(np.median(df_llama2['brand_recog_time'] + df_llama2['crp_pred_time'] + df_llama2['crp_transit_time']))

    df_llm_b = pd.read_table('./datasets/alexa7k_wo_validation.txt', sep='\t', header=None)
    df_llm_b.columns = ["hash", 'pred', 'brand', 'brand_recog_time',  'crp_pred_time', 'crp_transit_time']

    df_llm_b_llama2 = pd.read_table('./datasets/alexa7k_llama2.txt', sep='\t', header=None)
    df_llm_b_llama2.columns = ["hash", 'pred', 'brand', 'brand_recog_time',  'crp_pred_time', 'crp_transit_time']

    df_pedia_b = pd.read_table('./datasets/alexa7k_phishpedia.txt', sep='\t', header=None)
    df_pedia_b.columns = ["hash", 'pred', 'brand', 'time']

    df_tention_b = pd.read_table('./datasets/alexa7k_phishintention.txt', sep='\t', header=None)
    df_tention_b.columns = ["hash", 'pred', 'brand', 'time']

    # downsample
    common_hashes_b = set(df_llm_b['hash']).intersection(set(df_pedia_b['hash'])).intersection(set(df_tention_b['hash'])).intersection(set(df_llm_b_llama2['hash']))
    common_hashes_b = set(list(common_hashes_b)[:total])
    total_b = len(common_hashes_b)

    df_llm_b = df_llm_b[df_llm_b['hash'].isin(common_hashes_b)]
    df_llm_b_llama2 = df_llm_b_llama2[df_llm_b_llama2['hash'].isin(common_hashes_b)]
    df_pedia_b = df_pedia_b[df_pedia_b['hash'].isin(common_hashes_b)]
    df_tention_b = df_tention_b[df_tention_b['hash'].isin(common_hashes_b)]

    count_llm_b = sum(df_llm_b['pred'] == 'phish')
    count_llm_b_llama2 = sum(df_llm_b_llama2['pred'] == 'phish')
    count_pedia_b = sum(df_pedia_b['pred'] == 1)
    count_tention_b = sum(df_tention_b['pred'] == 1)

    print(f"Total phishing = {total}, "
          f"LLM recall = {count_llm/total}, "
          f"LLM Llama2 recall = {count_llama2/total}, "
          f"Phishpedia recall = {count_pedia/total}, " 
          f"PhishIntention recall = {count_tention/total} \n"
          f"Total benign = {total_b}, "
          f"LLM precision = {count_llm/(count_llm_b+count_llm)}, "
          f"LLM Llama2 precision = {count_llama2/(count_llama2+count_llm_b_llama2)}, "
          f"Phishpedia precision = {count_pedia/(count_pedia_b+count_pedia)}, "
          f"PhishIntention precision = {count_tention/(count_tention_b+count_tention)}, ")


