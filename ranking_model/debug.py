

import os
import shutil

if __name__ == '__main__':

    annotation_not_login = []
    annotation_login = []
    all_folders = set()
    annotation_train = './datasets/alexa_login_test.txt'

    path_set = set()
    for line in open('./datasets/alexa_login.txt').readlines()[::-1]:
        url, dom, save_path = line.strip().split('\t')
        if (url, dom) in path_set:
            continue
        else:
            path_set.add((url, dom))
            if os.path.exists(save_path):
                if os.path.exists(os.path.dirname(save_path)) and \
                        os.path.exists(os.path.dirname(save_path).replace('alexa_login', 'alexa_login_test_annot')):
                    all_folders.add(os.path.dirname(save_path))
                    if os.path.exists(save_path.replace('alexa_login', 'alexa_login_test_annot')):
                        annotation_not_login.append([url, dom, save_path])
    #                     with open(annotation_train, 'a+') as f:
    #                         f.write(url + '\t' + dom + '\t' + save_path + '\t' + '0' +'\n')
                    else:
                        annotation_login.append([url, dom, save_path])
    #                     with open(annotation_train, 'a+') as f:
    #                         f.write(url + '\t' + dom + '\t' + save_path + '\t' + '1' + '\n')
                # else:
                    # os.makedirs('./datasets/alexa_login_test', exist_ok=True)
                    # try:
                    #     shutil.copytree(os.path.dirname(save_path), os.path.dirname(save_path).replace('alexa_login', 'alexa_login_test'))
                    # except FileExistsError:
                    #     pass

    '''Training'''
    print(len(all_folders)) # 3047
    print(len(annotation_not_login), len(annotation_login)) # 112623, 2242, roughly 50:1

    '''Testing'''
    # print(len(all_folders)) # 593
    # print(len(annotation_not_login), len(annotation_login)) # 22869, 405, roughly 50:1


    '''Clean training again'''
    # for line in open('./datasets/alexa_login_train.txt').readlines()[::-1]:
    #     url, dom, save_path, label = line.strip().split('\t')
    #     if label == '1': # not login
    #         os.makedirs('./datasets/train_login/', exist_ok=True)
    #         shutil.copyfile(save_path, './datasets/train_login/{}'.format(url.split('https://')[1] + '_' + os.path.basename(save_path)))




