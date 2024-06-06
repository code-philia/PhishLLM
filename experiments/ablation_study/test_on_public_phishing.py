import shutil

import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from pathlib import Path

os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read().strip()
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if __name__ == '__main__':

    # # load hyperparameters
    # with open('./param_dict.yaml') as file:
    #     param_dict = yaml.load(file, Loader=yaml.FullLoader)
    #
    root_folder = './datasets/public_phishing_feeds'
    #
    # today = date.today()
    # day = "2024-05-13"
    # result = f'./datasets/public_phishing/{day}.txt'
    # os.makedirs("./datasets/public_phishing", exist_ok=True)
    #
    # phishintention_cls = PhishIntentionWrapper()
    # llm_cls = TestLLM(phishintention_cls,
    #                   param_dict=param_dict,
    #                   proxies={"http": "http://127.0.0.1:7890",
    #                            "https": "http://127.0.0.1:7890",
    #                            }
    #                   )
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.proxy = "http://127.0.0.1:7890" # proxy
    #
    # sleep_time = 3; timeout_time = 60
    # driver = CustomWebDriver.boot(proxy_server="127.0.0.1:7890")  # Using the proxy_url variable
    # driver.set_script_timeout(timeout_time)
    # driver.set_page_load_timeout(timeout_time)
    #
    # PhishLLMLogger.set_debug_on()
    # PhishLLMLogger.set_verbose(True)
    # # PhishLLMLogger.set_logfile(f'./datasets/public_phishing/{datetime}_phishllm.log')
    #
    # for it, folder in tqdm(enumerate(os.listdir(os.path.join(root_folder, day))[::-1])):
    #     # if it <= 241:
    #     #     continue
    #     target_folder = os.path.join(root_folder, day, folder)
    #
    #     # if os.path.exists(result) and folder in [x.strip().split('\t')[0] for x in open(result).readlines()]:
    #     #      continue
    #
    #     if folder not in [
    #         'payment-advice.pages.dev',
    #                      ]:
    #        continue
    #
    #     shot_path = os.path.join(target_folder, 'shot.png')
    #     html_path = os.path.join(target_folder, 'index.html')
    #     info_path = os.path.join(target_folder, 'info.txt')
    #     if os.path.exists(info_path):
    #         URL = open(info_path, encoding='utf-8').read()
    #     else:
    #         URL = f'http://{folder}'
    #
    #     if os.path.exists(shot_path):
    #         logo_box, reference_logo = llm_cls.detect_logo(shot_path)
    #         PhishLLMLogger.spit(URL)
    #         pred, brand, brand_recog_time, crp_prediction_time, crp_transition_time, _ = llm_cls.test(URL,
    #                                                                                                     reference_logo,
    #                                                                                                     logo_box,
    #                                                                                                     shot_path,
    #                                                                                                     html_path,
    #                                                                                                     driver,
    #                                                                                                   )
    #         # with open(result, 'a+') as f:
    #         #     f.write(folder+'\t'+str(pred)+'\t'+str(brand)+'\t'+str(brand_recog_time)+'\t'+str(crp_prediction_time)+'\t'+str(crp_transition_time)+'\n')
    #     else:
    #         shutil.rmtree(target_folder)
    #
    #     driver.delete_all_cookies()
    #
    # driver.quit()

    # '''Clean the FN cases'''
    root_folder = './datasets/public_phishing_feeds'
    # result = open(result).readlines()
    # os.makedirs(f"./datasets/{day}-FN-debug", exist_ok=True)
    # for line in result:
    #     line = line.strip()
    #     if "benign" in line:
    #         domain = line.split('\t')[0]
    #         if os.path.exists(os.path.join(os.path.join(root_folder, day, domain), 'shot.png')):
    #             shutil.copyfile(os.path.join(os.path.join(root_folder, day, domain), 'shot.png'),
    #                         os.path.join(f"./datasets/{day}-FN-debug", f"{domain}.png"))
    #
    # phish_ct = 0
    # total_ct = 0
    # for line in result:
    #     if "phish" in line:
    #         phish_ct += 1
    #         total_ct += 1
    #     elif line.split('\t')[0] in [x.split(".png")[0] for x in os.listdir(f'./datasets/{day}-FN-debug')]:
    #         total_ct += 1
    #     else:
    #         pass
    #
    # print(day, phish_ct, total_ct, phish_ct / total_ct)

    start_date = datetime(2024, 5, 10)
    end_date = datetime(2024, 5, 21)
    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)

    phishllm_ct = 0
    phishpedia_ct = 0
    phishintention_ct = 0
    total_ct = 0
    web_hosting_ct = 0

    phishllm_reported = []
    phishpedia_reported = []
    phishintention_reported = []

    phishllm_fn_dir = Path(f"./datasets/phishllm_fns")
    phishllm_fn_dir.mkdir(parents=True, exist_ok=True)
    phishpedia_fn_dir = Path(f"./datasets/phishpedia_fns")
    phishpedia_fn_dir.mkdir(parents=True, exist_ok=True)
    phishintention_fn_dir = Path(f"./datasets/phishintention_fns")
    phishintention_fn_dir.mkdir(parents=True, exist_ok=True)

    for date in date_range:
        phishllm_fns = []
        phishpedia_fns = []
        phishintention_fns = []

        day = date.strftime("%Y-%m-%d")
        result_llm = open(f'./datasets/public_phishing/{day}.txt')
        result_pedia = open(f'./datasets/public_phishing/{day}_phishpedia.txt')
        result_intention = open(f'./datasets/public_phishing/{day}_phishintention.txt')

        valid_folders = []
        for line in result_llm:
            if "phish" in line:
                phishllm_ct += 1; total_ct += 1
                valid_folders.append(line.split('\t')[0])
                phishllm_reported.append(day + line.split('\t')[0])
                if any([x in line for x in ["github.io", "pages.dev", "duckdns.org", "r2.dev", "hubspotpagebuilder.com",
                                            "blogspot.am", "vercel.app", "blogspot", "workers.dev", "weebly.com",
                                            "wixsite.com", "webflow.io", "weeblysite.com", "mybluehost.me"]]):
                    web_hosting_ct += 1
            elif line.split('\t')[0] in [x.split(".png")[0] for x in os.listdir(f'./datasets/{day}-FN-debug')]:
                total_ct += 1 # FN incurred by PhishLLM
                phishllm_fns.append(line.split('\t')[0])
                valid_folders.append(line.split('\t')[0])
                if any([x in line for x in ["github.io", "pages.dev", "duckdns.org", "r2.dev", "hubspotpagebuilder.com",
                                            "blogspot.am", "vercel.app", "blogspot", "workers.dev", "weebly.com",
                                            "wixsite.com", "webflow.io", "weeblysite.com", "mybluehost.me"]]):
                    web_hosting_ct += 1

        for line in result_pedia:
            if ("\t1\t" in line) and (line.split('\t')[0] in valid_folders):
                phishpedia_ct += 1
                phishpedia_reported.append(day + line.split('\t')[0])
            elif (line.split('\t')[0] in valid_folders):
                phishpedia_fns.append(line.split('\t')[0])

        for line in result_intention:
            if ("\t1\t" in line) and (line.split('\t')[0] in valid_folders):
                phishintention_ct += 1
                phishintention_reported.append(day + line.split('\t')[0])
            elif (line.split('\t')[0] in valid_folders):
                phishintention_fns.append(line.split('\t')[0])

        for fn in phishllm_fns:
            if os.path.exists(os.path.join(os.path.join(root_folder, day, fn), 'shot.png')):
                shutil.copyfile(os.path.join(os.path.join(root_folder, day, fn), 'shot.png'),
                                        phishllm_fn_dir / f"{day}_{fn}.png")
        # for fn in phishpedia_fns:
        #     if os.path.exists(os.path.join(os.path.join(root_folder, day, fn), 'shot.png')):
        #         shutil.copyfile(os.path.join(os.path.join(root_folder, day, fn), 'shot.png'),
        #                         phishpedia_fn_dir / f"{day}_{fn}.png")
        # for fn in phishintention_fns:
        #     if os.path.exists(os.path.join(os.path.join(root_folder, day, fn), 'shot.png')):
        #         shutil.copyfile(os.path.join(os.path.join(root_folder, day, fn), 'shot.png'),
        #                         phishintention_fn_dir / f"{day}_{fn}.png")

        print()

    print(f"web hosting phishing ct = {web_hosting_ct}")
    print(f"Total = {total_ct}, PhishLLM Recall = {phishllm_ct/total_ct}, Phishpedia Recall = {phishpedia_ct/total_ct}, PhishIntention Recall = {phishintention_ct/total_ct}")
    print(phishllm_ct, phishpedia_ct, phishintention_ct)
    print(f"FN count = {total_ct - phishllm_ct}")
    #
    # # Convert lists to sets for Venn diagram
    phishllm_only = len(set(phishllm_reported) - set(phishintention_reported) - set(phishpedia_reported))
    phishpedia_only = len(set(phishpedia_reported) - set(phishintention_reported) - set(phishllm_reported))
    phishllm_and_phishpedia = len(set(phishllm_reported).intersection(set(phishpedia_reported)) - set(phishintention_reported))
    phishintention_only = len(set(phishintention_reported) - set(phishllm_reported) - set(phishpedia_reported))
    phishllm_and_phishintention = len(set(phishllm_reported).intersection(set(phishintention_reported)) - set(phishpedia_reported))
    phishintention_and_phishpedia = len(set(phishintention_reported).intersection(set(phishpedia_reported)) - set(phishllm_reported))
    all_three = len(set(phishllm_reported).intersection(set(phishpedia_reported)).intersection(set(phishintention_reported)))

    # 创建 Venn 图
    plt.figure(figsize=(10, 10))
    venn_diagram = venn3(
        subsets=(phishllm_only, phishpedia_only, phishllm_and_phishpedia,
                 phishintention_only, phishllm_and_phishintention, phishintention_and_phishpedia, all_three),
        set_labels=('PhishLLM', 'Phishpedia', 'PhishIntention'),
        set_colors=('#ff9999', '#66b3ff', '#99ff99'),  # Distinctive color scheme
        alpha=0.5  # Adjusted transparency for better overlap visibility
    )

    # 高亮显示三个圆圈的边界
    venn_circles = venn3_circles(
        subsets=(phishllm_only, phishpedia_only, phishllm_and_phishpedia,
                 0, 0, phishintention_and_phishpedia, all_three),
        linestyle='solid'
    )
    for circle in venn_circles:
        circle.set_edgecolor('black')
        circle.set_linewidth(1.5)

    # 手动调整标签位置
    labels = venn_diagram.subset_labels
    if labels[2]:  # phishllm_and_phishpedia
        labels[2].set_position((labels[2].get_position()[0] - 0.05, labels[2].get_position()[1] + 0.05))
    if labels[1]:  # phishpedia_only
        labels[1].set_position((labels[1].get_position()[0], labels[1].get_position()[1] + 0.15))

    # 设置其他标签的字体粗细
    for label in labels:
        if label:
            label.set_fontweight('bold')
            label.set_fontsize(25)

    # 设置集合标签的字体大小和粗细
    set_labels = venn_diagram.set_labels
    for set_label in set_labels:
        if set_label:
            set_label.set_fontsize(25)
            set_label.set_fontweight('bold')

    # 添加标题
    plt.tight_layout()
    plt.savefig('./debug.png', dpi=300)
    plt.close()
    #
    # phishintention_not_phishllm_dir = Path(f"./datasets/phishintention_not_phishllm")
    # phishintention_not_phishllm_dir.mkdir(parents=True, exist_ok=True)
    # phishintention_not_phishllm_set = set_phishintention - set_phishllm
    # for target_path in phishintention_not_phishllm_set:
    #     for dirpath, dirnames, filenames in os.walk(root_folder):
    #         if target_path in dirnames:
    #             target_dir = os.path.join(dirpath, target_path)
    #             shot_path = os.path.join(target_dir, 'shot.png')
    #             if os.path.exists(shot_path):
    #                 shutil.copyfile(shot_path,
    #                                 phishintention_not_phishllm_dir / f"{target_path}.png")
    #                 break

    # phishpedia_not_phishllm_dir = Path(f"./datasets/phishpedia_not_phishllm")
    # phishpedia_not_phishllm_dir.mkdir(parents=True, exist_ok=True)
    # phishpedia_not_phishllm_set = set_phishpedia - set_phishllm
    # for target_path in phishpedia_not_phishllm_set:
    #     for dirpath, dirnames, filenames in os.walk(root_folder):
    #         if target_path in dirnames:
    #             target_dir = os.path.join(dirpath, target_path)
    #             shot_path = os.path.join(target_dir, 'shot.png')
    #             if os.path.exists(shot_path):
    #                 shutil.copyfile(shot_path,
    #                                 phishpedia_not_phishllm_dir / f"{target_path}.png")
    #                 break

