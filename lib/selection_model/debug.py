
import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    # annotation_not_login = []
    # annotation_login = []
    # all_folders = set()
    # annotation_train = './datasets/alexa_screenshots.txt'
    #
    # path_set = set()
    # for line in open('./datasets/alexa_login.txt').readlines()[::-1]:
    #     url, dom, save_path = line.strip().split('\t')
    #     if url in path_set:
    #         continue
    #     else:
    #         path_set.add(url)
    #         with open(annotation_train, 'a+') as f:
    #             f.write(url+'\t'+save_path.replace(os.path.basename(save_path), 'shot.png')+'\t'+str(0)+'\n')

    # os.makedirs('./datasets/alexa_shot_aug', exist_ok=True)
    # for folder in os.listdir('./datasets/alexa_login_crp'):
    #     shot_path = os.path.join('./datasets/alexa_login_crp', folder, 'shot.png')
    #     shutil.copyfile(shot_path,
    #                     os.path.join('./datasets/alexa_shot_aug', folder+'.png'))

    # os.makedirs('./datasets/alexa_shot_crp_aug', exist_ok=True)
    # for img in os.listdir('./datasets/alexa_shot_aug'):
    #     folder = img.split('.png')[0]
    #     if (os.path.exists(os.path.join('./datasets/alexa_login_crp', folder))):
    #         shutil.copytree(os.path.join('./datasets/alexa_login_crp', folder),
    #                           os.path.join('./datasets/alexa_shot_crp_aug', folder))

    # annotation_train = './datasets/alexa_screenshots.txt'
    # for folder in os.listdir('./datasets/alexa_shot_crp'):
    #     if os.path.exists(os.path.join('./datasets/alexa_login', folder.split('.png')[0], 'shot.png')):
    #         url = 'https://' + folder.split('.png')[0]
    #         with open(annotation_train, 'a+') as f:
    #             f.write(url + '\t' + os.path.join('./datasets/alexa_login', folder.split('.png')[0], 'shot.png') + '\t' + 'A' + '\n')

    # for folder in os.listdir('./datasets/alexa_shot_noncrp'):
    #     if os.path.exists(os.path.join('./datasets/alexa_login', folder.split('.png')[0], 'shot.png')):
    #         url = 'https://' + folder.split('.png')[0]
    #         with open(annotation_train, 'a+') as f:
    #             f.write(url + '\t' + os.path.join('./datasets/alexa_login', folder.split('.png')[0], 'shot.png') + '\t' + 'B' + '\n')

    # for folder in os.listdir('./datasets/alexa_shot_crp_aug'):
    #     url = open(os.path.join('./datasets/alexa_shot_crp_aug', folder, 'info.txt')).read()
    #     with open(annotation_train, 'a+') as f:
    #         f.write(url + '\t' + os.path.join('./datasets/alexa_shot_crp_aug', folder, 'shot.png') + '\t' + 'A' + '\n')

    def check_all_pixels_equal(image):
        # Convert the image to a NumPy array
        image_array = np.array(image)
        # Check if all pixels are equal
        return np.all(image_array == image_array[0])

    os.makedirs('./datasets/dynapd_shot', exist_ok=True)
    root_folder = './datasets/dynapd'
    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_wo_validation.txt').readlines()]
    ct = 0
    for hash in tqdm(all_folders):
        target_folder = os.path.join(root_folder, hash)
        shot_path = os.path.join(target_folder, 'shot.png')
        result = check_all_pixels_equal(Image.open(shot_path))
        if result:
            ct += 1
        # shutil.copy(shot_path, os.path.join('./datasets/dynapd_shot', hash+'.png'))
    print(ct)



