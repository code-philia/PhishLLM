import torch
from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
from paddleocr import PaddleOCR
# import os
# # git clone https://github.com/lindsey98/LAVIS.git
# # cd LAVIS
# # pip install -e .
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = "./brand_recognition/test_case/img_2.png"
raw_image = Image.open(img_path).convert("RGB")

model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
result = model.generate({"image": image})
print(result)

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)  # need to run only once to download and load model into memory
result = ocr.ocr(img_path, cls=True)
print(result)

# ocr_text, ocr_coord = get_ocr_text_coord("./brand_recognition/test_case/shot.png")
# ocr_text = ' '.join(ocr_text)
# print(ocr_text)

# image_i = transparent_text_injection(raw_image, 'abc.com')
# image_i.convert("RGB").save('./brand_recognition/test_case/img_10i.png')
# image_i = vis_processors["eval"](image_i).unsqueeze(0).to(device)
# result = model.generate({"image": image_i})
# print(result)

# ['the usenix association logo']
# ["a starbucks logo with a woman's face in the center"]
# ['a white letter t on a pink background']
# ['a black and white logo with a knot in the middle']
# ["a red square with a yellow mcdonald's logo"]
# ['an apple logo on a black background']
# ['a red, white and blue pepsi logo']
# ['a black and white image of a monkey with headphones']
# ['a black and white photo of a clock tower']
# ['tiffany & co logo on a blue background']
# ['the new york yankees logo is shown']
# ['a blue and white wordpress logo']

# from xdriver.XDriver import XDriver
# from xdriver.xutils.Logger import Logger
# from xdriver.xutils.forms.Form import Form
# from mmocr.apis import MMOCRInferencer
# from xdriver.xutils.forms.SubmissionButtonLocator import SubmissionButtonLocator
# from xdriver.xutils.PhishIntentionWrapper import PhishIntentionWrapper
# import time
# import numpy as np

# if __name__ == '__main__':
    # orig_url = "https://reg.ebay.com/reg/PartialReg"
    # sleep_time = 3; timeout_time = 60
    # XDriver.set_headless()
    # driver = XDriver.boot(chrome=True)
    # driver.set_script_timeout(timeout_time / 2)
    # driver.set_page_load_timeout(timeout_time)
    # time.sleep(sleep_time)
    # Logger.set_debug_on()
    #
    # # load phishintention, mmocr, button_locator_model
    # phishintention_cls = PhishIntentionWrapper()
    # mmocr_model = MMOCRInferencer(det=None,
    #                     rec='ABINet',
    #                     device='cuda')
    # button_locator_model = SubmissionButtonLocator(
    #     button_locator_config='/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/config.yaml',
    #     button_locator_weights_path='/home/ruofan/git_space/MyXdriver_pub/xutils/forms/button_locator_models/model_final.pth')
    #
    # # initialization
    # Logger.spit('URL={}'.format(orig_url), caller_prefix=XDriver._caller_prefix, debug=True)
    # try:
    #     driver.get(orig_url, allow_redirections=True)
    #     time.sleep(sleep_time)  # fixme: wait until page is fully loaded
    # except Exception as e:
    #     Logger.spit('Exception when getting the URL {}'.format(e), caller_prefix=XDriver._caller_prefix,
    #                 warning=True)
    #     raise
    #
    # form = Form(driver, phishintention_cls, mmocr_model,
    #             button_locator_model, obfuscate=False)  # initialize form
    #
    # # form filling and form submission
    # filled_values = form.fill_all_inputs()
    #
    # # scrolling only happens at the first time, otherwise the screenshot changes just because we scroll it, e.g.: deepl.com
    # # button maybe at the bottom, need to decide when to scroll
    # if (not form._button_visibilities[0]):
    #     Logger.spit("Scroll to the bottom since the buttons are invisible", debug=True,
    #                 caller_prefix=XDriver._caller_prefix)
    #     driver.scroll_to_bottom()
    #     # scrolling change the screenshot
    #     form.button_reinitialize()
    #
    # form.submit(1)  # form submission
    # driver.quit()

    # brand_recog_time = [eval(x.strip().split('\t')[-3]) for x in open('./field_study/results/2023-08-21_phishllm.txt', encoding='ISO-8859-1').readlines()[1:]]
    # print(np.median(brand_recog_time))

