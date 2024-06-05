import os
from models.utils.dynaphish.brand_knowledge_utils import BrandKnowledgeConstruction
from pipeline.test_dynaphish import SubmissionButtonLocator, DynaPhish
import torch
from models.utils.PhishIntentionWrapper import PhishIntentionWrapper
from models.utils.logger_utils import PhishLLMLogger
from models.utils.web_utils import WebUtil, CustomWebDriver
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"


if __name__ == '__main__':
    import pickle
    with open('./model_chain/dynaphish/domain_map.pkl', 'rb') as handle:
        domain_map = pickle.load(handle)
    domain_map['cetelem'] = ['cetelem.fr']
    with open('./model_chain/dynaphish/domain_map.pkl', 'wb') as handle:
        pickle.dump(domain_map, handle)

    method = 'phishintention'
    result_file = f'./datasets/cost_benchmarking_dynaphish+{method}.txt'
    # if not os.path.exists(result_file):
    #     with open(result_file, 'a+') as f:
    #         f.write('folder' +
    #                 '\tdomain2brand_cost' +
    #                 '\tlogo2brand_cost' +
    #                 '\tdomain2brand_time' +
    #                 '\tlogo2brand_time' +
    #                 '\tele_detector_time' +
    #                 '\tsiamese_time' +
    #                 '\tcrp_time' +
    #                 '\tdynamic_time' +
    #                 '\n')
    #
    # PhishLLMLogger.set_debug_on()
    # phishintention_config_path = '/home/ruofan/git_space/ScamDet/model_chain/dynaphish/configs.yaml'
    # PhishIntention = PhishIntentionWrapper()
    # PhishIntention.reset_model(phishintention_config_path, False)
    #
    # API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]
    # KnowledgeExpansionModule = BrandKnowledgeConstruction(API_KEY, SEARCH_ENGINE_ID, PhishIntention)
    #
    # dynaphish_cls = DynaPhish(PhishIntention,
    #                           phishintention_config_path,
    #                           None,
    #                           KnowledgeExpansionModule)
    #
    # web_func = WebUtil()
    #
    # sleep_time = 3; timeout_time = 60
    # driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")  # Using the proxy_url variable
    # driver.set_script_timeout(timeout_time / 2)
    # driver.set_page_load_timeout(timeout_time)
    #
    # for brand in tqdm(os.listdir('./datasets/alexa_middle_5k/')[:50]):
    #     if brand.startswith('.'):
    #         continue
    #
    #     folder = os.path.join('./datasets/alexa_middle_5k/', brand)
    #     if os.path.exists(result_file) and folder in [x.strip().split('\t')[0] for x in open(result_file).readlines()]:
    #         continue
    #
    #     shot_path = os.path.join(folder, 'shot.png')
    #     html_path = os.path.join(folder, 'index.html')
    #     info_path = os.path.join(folder, 'info.txt')
    #     if os.path.exists(info_path):
    #         URL = open(info_path, encoding='utf-8').read()
    #     else:
    #         URL = f'http://{brand}'
    #
    #     if not os.path.exists(shot_path):
    #         continue
    #
    #     _, reference_logo = KnowledgeExpansionModule.predict_n_save_logo(shot_path)
    #     if reference_logo is None:
    #         continue
    #
    #     (domain2brand_cost, logo2brand_cost), \
    #         (domain2brand_time, logo2brand_time) , \
    #         (ele_detector_time, siamese_time, crp_time, dynamic_time) = dynaphish_cls.estimate_cost(URL, driver, shot_path, base_model=method)
    #
    #     with open(result_file, 'a+') as f:
    #         f.write(folder + '\t' +
    #                 str(domain2brand_cost) + '\t' +
    #                 str(logo2brand_cost)  + '\t' +
    #                 str(domain2brand_time) + '\t' +
    #                 str(logo2brand_time) + '\t' +
    #                 str(ele_detector_time) + '\t' +
    #                 str(siamese_time) + '\t' +
    #                 str(crp_time) + '\t' +
    #                 str(dynamic_time) + '\n')
    #
    # driver.quit()

    result_file = f'./datasets/cost_benchmarking_dynaphish+{method}.txt'
    df = pd.read_csv(result_file, delimiter='\t')
    time_columns = [col for col in df.columns if (col.endswith('_time'))]
    df_time = df[time_columns]
    df_time = df_time.astype(float)
    df_time['total_processing_time'] = df_time[time_columns].sum(axis=1)
    df_time[f'{method}_time'] = df_time[['ele_detector_time', 'siamese_time', 'crp_time', 'dynamic_time']].sum(axis=1)

    column_mapping = {
        'domain2brand_time': 'Brand Knowledge Expansion (H1)',
        'logo2brand_time': 'Brand Knowledge Expansion (H2)',
        f'{method}_time': method.capitalize(),
        'total_processing_time': 'Total'
    }

    # Rename columns in the DataFrame
    df_time = df_time.rename(columns=column_mapping)
    # Reorder columns to place 'Total Processing' last
    df_time = df_time[[col for col in column_mapping.values() if col not in ['Total', 'ele_detector_time', 'siamese_time', 'crp_time', 'dynamic_time']] + ['Total']]

    df_melted = df_time.melt(var_name='Processing Time Type', value_name='Time')
    plt.figure(figsize=(14, 7))  # Adjusted figure size for better aspect ratio
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")
    ax = sns.boxplot(x='Processing Time Type', y='Time', data=df_melted, palette=palette)

    # Enhance the plot aesthetics
    plt.xlabel('')
    plt.ylabel('Time (seconds)', fontsize=20)
    labels = [label.get_text() for label in ax.get_xticklabels()]
    wrapped_labels = ['\n'.join(wrap(label, 15)) for label in labels]  # Wrap labels at 15 characters
    ax.set_xticklabels(wrapped_labels, ha='center', fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Retrieve the correct order of categories
    categories = df_melted['Processing Time Type'].unique()
    # Annotate the medians
    medians = df_melted.groupby('Processing Time Type')['Time'].median().loc[categories].values
    for i, median in enumerate(medians):
        ax.annotate(f'Median = {median:.2f}', xy=(i, median), xycoords='data',
                    xytext=(0, 10), textcoords='offset points',  # Offset the annotation by 10 points above the median
                    ha='center', va='center', fontsize=15, color='black',
                    bbox=dict(facecolor='white', alpha=0.3, edgecolor='black', boxstyle='round,pad=0.5'))

    # Adjust layout for better fit
    plt.tight_layout()
    plt.savefig(f'./field_study/plots/dynaphish+{method}_cost.png')
    plt.close()

    df = pd.read_csv(result_file, delimiter='\t')
    cost_columns = [col for col in df.columns if col.endswith('_cost')]
    df_cost = df[cost_columns]
    df_cost = df_cost.astype(float)
    df_cost = df_cost.multiply(1000)
    df_cost['total_cost'] = df_cost[cost_columns].sum(axis=1)

    # Mapping old column names to more descriptive ones
    column_mapping = {
        'domain2brand_cost': 'Google API Cost for Brand Knowledge Expansion (H1)',
        'logo2brand_cost': 'Google API Cost for Brand Knowledge Expansion (H2)',
        'total_cost': 'Total API Cost'
    }

    # Rename columns in the DataFrame according to the mapping
    df_cost = df_cost.rename(columns=column_mapping)

    # Melting the DataFrame for easier plotting
    df_melted = df_cost.melt(var_name='Processing Cost Type', value_name='Cost')

    # Creating a boxplot
    plt.figure(figsize=(14, 7))  # Adjusted figure size for better aspect ratio
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2")
    ax = sns.boxplot(x='Processing Cost Type', y='Cost', data=df_melted, palette=palette)

    # Enhance the plot aesthetics
    plt.xlabel('')
    plt.ylabel(r'Cost (in USD) per $\mathbf{1000}$ Websites', fontsize=20)
    labels = [label.get_text() for label in ax.get_xticklabels()]
    wrapped_labels = ['\n'.join(wrap(label, 15)) for label in labels]  # Wrap labels at 15 characters
    ax.set_xticklabels(wrapped_labels, ha='center', fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Annotate medians on the plot
    categories = df_melted['Processing Cost Type'].unique()
    medians = df_melted.groupby('Processing Cost Type')['Cost'].median().loc[categories].values
    for i, median in enumerate(medians):
        ax.annotate(f'Median = {median:.2f}', xy=(i, median), xycoords='data',
                    xytext=(0, 10), textcoords='offset points',  # Offset the annotation by 10 points above the median
                    ha='center', va='center', fontsize=15, color='black',
                    bbox=dict(facecolor='white', alpha=0.3, edgecolor='black', boxstyle='round,pad=0.5'))

    # Adjust layout for better fit and save the plot
    plt.tight_layout()
    plt.savefig(f'./field_study/plots/dynaphish+{method}_money.png')
    plt.close()
