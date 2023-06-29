
import os
import shutil

if __name__ == '__main__':
    # ct = 0
    # total = 0
    # for folder in os.listdir('./datasets/alexa_login'):
    #     if len(os.listdir(os.path.join('./datasets/alexa_login', folder))) <= 3: # blank page or block page
    #         ct += 1
    #         shutil.rmtree(os.path.join('./datasets/alexa_login', folder))
    #         print(os.path.join('./datasets/alexa_login', folder))
    #     total += 1
    # print(ct, total)

    ct = 0
    for folder in os.listdir('./datasets/alexa_login'):
        if folder not in os.listdir('./datasets/alexa_login_p1'):
            ct += 1
    print(ct)


# for line in open('./datasets/alexa_login.txt').readlines():
    #     path = line.strip().split('\t')[-1]
    #     if not (os.path.exists(path)):
    #         print(path)
