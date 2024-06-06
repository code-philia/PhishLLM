
import os
import yaml
import openai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"

if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy

    with open('./param_dict.yaml') as file:
        param_dict = yaml.load(file, Loader=yaml.FullLoader)
    #
    model = "gpt-3.5-turbo-16k"
    result_file = './datasets/cost_benchmarking.txt'

    # phishintention_cls = PhishIntentionWrapper()
    # llm_cls = TestLLM(phishintention_cls,
    #                   param_dict=param_dict,
    #                   proxies={"http": "http://127.0.0.1:7890",
    #                            "https": "http://127.0.0.1:7890",
    #                            }
    #                   )
    # PhishLLMLogger.set_debug_on()
    # PhishLLMLogger.set_verbose(True)


    # sleep_time = 3; timeout_time = 60
    # driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")  # Using the proxy_url variable
    # driver.set_script_timeout(timeout_time)
    # driver.set_page_load_timeout(timeout_time)
    #
    # if not os.path.exists(result_file):
    #     with open(result_file, 'a+') as f:
    #         f.write('folder' +
    #                 '\tbrand_llm_cost' +
    #                 '\tcrp_llm_cost' +
    #                 '\tbrand_validation_cost' +
    #                 '\tpopularity_validation_cost'
    #                 '\tocr_processing_time' +
    #                 '\timage_caption_processing_time' +
    #                 '\tbrand_recog_time' + '\t' +
    #                 'crp_prediction_time' +
    #                 '\tclip_prediction_time'+
    #                 '\tbrand_validation_searching_time'+
    #                 '\tbrand_validation_matching_time'+
    #                 '\tpopularity_validation_time\n')
    #
    #
    # for brand in tqdm(os.listdir('./datasets/public_phishing_feeds/2024-05-15/')[:50]):
    #     if brand.startswith('.'):
    #         continue
    #
    #     folder = os.path.join('./datasets/public_phishing_feeds/2024-05-15/', brand)
    #     if os.path.exists(result_file) and folder in [x.strip().split('\t')[0] for x in open(result_file).readlines()]:
    #         continue
    #
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
    #     logo_box, reference_logo = llm_cls.detect_logo(shot_path)
    #     if reference_logo is None:
    #         continue
    #
    #     (brand_llm_cost, crp_llm_cost, brand_validation_cost, popularity_validation_cost), \
    #     (ocr_processing_time, image_caption_processing_time,
    #      brand_recog_time, crp_prediction_time, clip_prediction_time,
    #      brand_validation_searching_time, brand_validation_matching_time, popularity_validation_time) = llm_cls.estimate_cost_phishllm(url=URL,
    #                                                                                                                                    reference_logo=reference_logo,
    #                                                                                                                                    logo_box=logo_box,
    #                                                                                                                                    shot_path=shot_path,
    #                                                                                                                                    html_path=html_path,
    #                                                                                                                                    driver=driver)
    #
    #     with open(result_file, 'a+') as f:
    #         f.write(folder + '\t' +
    #                 str(brand_llm_cost) + '\t' +
    #                 str(crp_llm_cost)  + '\t' +
    #                 str(brand_validation_cost) + '\t' +
    #                 str(popularity_validation_cost) + '\t' +
    #                 str(ocr_processing_time) + '\t' +
    #                 str(image_caption_processing_time) + '\t' +
    #                 str(brand_recog_time) + '\t' +
    #                 str(crp_prediction_time) + '\t' +
    #                 str(clip_prediction_time) + '\t' +
    #                 str(brand_validation_searching_time) + '\t' +
    #                 str(brand_validation_matching_time) + '\t' +
    #                 str(popularity_validation_time) + '\n')
    #
    # driver.quit()
    # exit()

    df = pd.read_csv(result_file, delimiter='\t')
    # Process time data
    time_columns = [col for col in df.columns if (col.endswith('_time') and col != 'brand_validation_searching_time' and col != 'brand_validation_matching_time')]
    df_time = df[time_columns].astype(float)
    df_time['total_processing_time'] = df_time.sum(axis=1)

    # Mapping for time data
    time_column_mapping = {
        'ocr_processing_time': 'OCR Processing',
        'image_caption_processing_time': 'Image Captioning Processing',
        'brand_recog_time': 'Brand Recognition',
        'crp_prediction_time': 'CRP Prediction',
        'clip_prediction_time': 'CRP Transition Prediction',
        'popularity_validation_time': 'Popularity Validation',
        'total_processing_time': 'Total'
    }

    df_time.rename(columns=time_column_mapping, inplace=True)

    df_melted = df_time.melt(var_name='Processing Time Type', value_name='Time')
    plt.figure(figsize=(10, 4))
    sns.set(style="whitegrid")
    palette = sns.color_palette("Paired")
    ax = sns.boxplot(x='Processing Time Type', y='Time', data=df_melted, palette=palette, width=0.5)

    plt.xlabel('')
    plt.ylabel('Time (seconds)', fontsize=13, fontweight='bold', color='black')
    labels = [label.get_text() for label in ax.get_xticklabels()]
    wrapped_labels = ['\n'.join(wrap(label, 10)) for label in labels]
    ax.set_xticklabels(wrapped_labels, ha='center', fontsize=13, fontweight='bold', color='black')
    plt.yticks(fontsize=13, fontweight='bold', color='black')
    plt.ylim(bottom=-0.2, top=6.0)
    plt.grid(True, linestyle='--', alpha=0.7)

    categories = df_melted['Processing Time Type'].unique()
    medians = df_melted.groupby('Processing Time Type')['Time'].median().loc[categories].values
    for i, median in enumerate(medians):
        ax.annotate(f'Median = {median:.2f}', xy=(i, median), xycoords='data',
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='center', fontsize=10, color='black',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.savefig('./experiments/field_study/plots/phishllm_cost.png', dpi=300)
    plt.close()

    # df = pd.read_csv(result_file, delimiter='\t')
    # cost_columns = [col for col in df.columns if col.endswith('_cost')]
    # df_cost = df[cost_columns]
    # df_cost = df_cost.astype(float)
    # df_cost = df_cost.multiply(1000)
    # df_cost = df_cost[['brand_llm_cost', 'brand_validation_cost', 'crp_llm_cost']]
    # df_cost['total_cost'] = df_cost[['brand_llm_cost', 'brand_validation_cost', 'crp_llm_cost']].sum(axis=1)
    #
    # print(df_cost.describe())
