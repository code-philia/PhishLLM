
from model_chain.test_baseline import *

if __name__ == '__main__':

    phishintention_cls = PhishIntentionWrapper()
    base_cls = TestBaseline(phishintention_cls)

    root_folder = './datasets/alexa_login_test'
    # result = './datasets/alexa7k_phishpedia.txt'
    result = './datasets/alexa7k_phishintention.txt'
    XDriver.set_headless()


    for ct, target in enumerate(os.listdir(root_folder)):
        # if ct <= 5470:
        #     continue
        URL = 'https://{}'.format(target)
        target_folder = os.path.join(root_folder, target)
        if os.path.exists(result) and target in open(result).read():
            continue

        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        if os.path.exists(shot_path):
            # pred, brand, runtime = base_cls.test_phishpedia(URL, shot_path)
            pred, brand, runtime = base_cls.test_phishintention(URL, shot_path)
            with open(result, 'a+') as f:
                f.write(target + '\t' + str(pred) + '\t' + str(brand) + '\t' + str(runtime) + '\n')




