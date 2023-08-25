
from brand_recognition.test_llm import *
import idna
import pandas as pd
from model_chain.web_utils import WebUtil, is_valid_domain
from xdriver.xutils.Logger import Logger
from xdriver.XDriver import XDriver
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()

def test(result_file):

    orig_ct = 0
    adv_ct = 0
    total = 0

    result_lines = open(result_file).readlines()
    pbar = tqdm(result_lines, leave=False)
    for line in pbar:
        data = line.strip().split('\t')
        url, orig_pred, adv_pred, time = data
        total += 1

        pattern = re.compile(
            r'^(?!-)'  # Cannot start with a hyphen
            r'(?!.*--)'  # Cannot have two consecutive hyphens
            r'(?!.*\.\.)'  # Cannot have two consecutive periods
            r'(?!.*\s)'  # Cannot contain any spaces
            r'[a-zA-Z0-9-]{1,63}'  # Valid characters are alphanumeric and hyphen
            r'(?:\.[a-zA-Z]{2,})+$'  # Ends with a valid top-level domain
        )
        it_is_a_domain_orig = bool(pattern.fullmatch(orig_pred))
        it_is_a_domain_adv = bool(pattern.fullmatch(adv_pred))

        if it_is_a_domain_orig:
            orig_ct += 1
        if it_is_a_domain_adv:
            adv_ct += 1

        pbar.set_description(f"Original Recall (% brand recognized) = {orig_ct/total} \n"
                             f"After adv Recall (% brand recognized) = {adv_ct/total}", refresh=True)

    print(f"Original Recall (% brand recognized) = {orig_ct/total} \n"
          f"After adv Recall (% brand recognized) = {adv_ct/total}")


def get_results(logo_box, reference_logo, screenshot_img, ocr_text, ocr_coord, html_text):
    if reference_logo is not None:
        # generation caption for logo
        logo_caption = get_caption(reference_logo)
        # expand the logo bbox a bit to see the surrounding region
        expand_logo_box = expand_bbox(logo_box, image_width=screenshot_img.size[0],
                                      image_height=screenshot_img.size[1], expand_ratio=(5, 5))
        extra_description = ''
        if len(ocr_coord):
            # get the OCR text description surrounding the logo
            overlap_areas = compute_overlap_areas_between_lists([expand_logo_box], ocr_coord)
            extra_description = np.array(ocr_text)[overlap_areas[0] > 0].tolist()
            extra_description = ' '.join(extra_description)
    else:
        logo_caption = ''
        extra_description = ' '.join(ocr_text)

    print('Logo Caption: ', logo_caption)
    print('Logo OCR: ', extra_description)

    if len(logo_caption)>0 or len(extra_description)>0:
        industry = ask_industry("gpt-3.5-turbo-16k", html_text)

        question = question_template_caption_industry(logo_caption, extra_description, industry)
        with open('./brand_recognition/prompt_caption.json', 'rb') as f:
            prompt = json.load(f)
        new_prompt = prompt
        new_prompt.append(question)

        # example token count from the OpenAI API
        inference_done = False
        while not inference_done:
            try:
                start_time = time.time()
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=new_prompt,
                    temperature=0,
                    max_tokens=50,  # we're only counting input tokens here, so let's not waste tokens on the output
                )
                inference_done = True
            except Exception as e:
                print(f"Error was: {e}")
                new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
                time.sleep(10)
        total_time = time.time() - start_time

        orig_answer = ''.join([choice["message"]["content"] for choice in response['choices']])
        return orig_answer, total_time
    else:
        return '', 0

if __name__ == '__main__':

    openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.proxy = "http://127.0.0.1:7890" # proxy
    model = "gpt-3.5-turbo-16k"

    root_folder = './datasets/dynapd'
    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_wo_validation.txt').readlines()]
    df = pd.read_csv('./datasets/Brand_Labelled_130323.csv')
    result_file = './datasets/dynapd_llm_caption_adv_brand.txt'

    phishintention_cls = PhishIntentionWrapper()
    web_func = WebUtil()

    sleep_time = 3; timeout_time = 60
    # XDriver.set_headless()
    driver = XDriver.boot(chrome=True)
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)
    Logger.set_debug_on()

    for hash in tqdm(all_folders):
        target_folder = os.path.join(root_folder, hash)
        if os.path.exists(result_file) and hash in open(result_file).read():
            continue

        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')
        if os.path.exists(shot_path):
            pk_info = df.loc[df['HASH'] == hash]
            try:
                URL = list(pk_info['URL'])[0]
            except IndexError:
                URL = f'http://127.0.0.5/{hash}'

            # report logo
            screenshot_img = Image.open(shot_path)
            screenshot_img = screenshot_img.convert("RGB")
            with open(shot_path, "rb") as image_file:
                screenshot_encoding = base64.b64encode(image_file.read())
            logo_boxes = phishintention_cls.return_all_bboxes4type(screenshot_encoding, 'logo')

            orig_answer = 'no prediction'
            answer = 'no prediction'
            total_time = 0

            # has logo
            if (logo_boxes is not None) and len(logo_boxes):
                # get extra description on the webpage
                ocr_text, ocr_coord = get_ocr_text_coord(shot_path)
                html_text = ' '.join(ocr_text)

                for logo_box in logo_boxes[:min(len(logo_boxes), 3)]: # Top-3 logo box
                    '''original prediction '''
                    x1, y1, x2, y2 = logo_box
                    reference_logo = screenshot_img.crop((x1, y1, x2, y2))  # crop logo out
                    orig_answer, total_time = get_results(logo_box, reference_logo, screenshot_img, ocr_text, ocr_coord, html_text)
                    print('Original answer: ', orig_answer)

                    if is_valid_domain(orig_answer):
                        '''perform adversarial attack '''
                        print('Adversarial attack')
                        injected_logo = transparent_text_injection(reference_logo.convert('RGB'), 'abc.com')
                        # get image caption for injected logo
                        screenshot_img.paste(injected_logo, (int(x1), int(y1)))
                        adv_shot_path = shot_path.replace('shot.png', 'shot_adv.png')
                        screenshot_img.save(adv_shot_path)

                        # get extra description on the webpage
                        adv_ocr_text, adv_ocr_coord = get_ocr_text_coord(adv_shot_path)
                        adv_html_text = ' '.join(adv_ocr_text)
                        answer, total_time = get_results(logo_box, injected_logo, screenshot_img, adv_ocr_text, adv_ocr_coord, adv_html_text)
                        print('After attack answer: ', answer)
                        break
            else:
                try:
                    driver.get(URL, click_popup=True, allow_redirections=False)
                    time.sleep(5)
                    Logger.spit(f'Target URL = {URL}', caller_prefix=XDriver._caller_prefix, debug=True)
                    page_text = driver.get_page_text()
                    error_free = web_func.page_error_checking(driver)
                    if not error_free:
                        Logger.spit('Error page or White page', caller_prefix=XDriver._caller_prefix, debug=True)
                        continue

                    if "Index of" in page_text:
                        # skip error URLs
                        error_free = web_func.page_interaction_checking(driver)
                        if not error_free:
                            Logger.spit('Error page or White page', caller_prefix=XDriver._caller_prefix,
                                        debug=True)
                            continue
                        target = driver.current_url()

                    # take screenshots
                    screenshot_encoding = driver.get_screenshot_encoding()
                    screenshot_img = Image.open(io.BytesIO(base64.b64decode(screenshot_encoding)))
                    screenshot_img.save(shot_path)
                except Exception as e:
                    Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                    continue

                try:
                    # record HTML
                    with open(html_path, 'w+', encoding='utf-8') as f:
                        f.write(driver.page_source())
                except Exception as e:
                    Logger.spit('Exception {}'.format(e), caller_prefix=XDriver._caller_prefix, debug=True)
                    pass

                # report logo
                screenshot_img = Image.open(shot_path)
                screenshot_img = screenshot_img.convert("RGB")
                with open(shot_path, "rb") as image_file:
                    screenshot_encoding = base64.b64encode(image_file.read())
                logo_boxes = phishintention_cls.return_all_bboxes4type(screenshot_encoding, 'logo')

                # get extra description on the webpage
                ocr_text, ocr_coord = get_ocr_text_coord(shot_path)
                html_text = ' '.join(ocr_text)

                if logo_boxes is not None and len(logo_boxes)>0:  # Top-3 logo box
                    '''original prediction '''
                    logo_box = logo_boxes[0]
                    x1, y1, x2, y2 = logo_box
                    reference_logo = screenshot_img.crop((x1, y1, x2, y2))  # crop logo out
                else:
                    logo_box = None
                    reference_logo = None
                orig_answer, total_time = get_results(logo_box, reference_logo, screenshot_img, ocr_text,
                                                      ocr_coord, html_text)
                print('Original answer: ', orig_answer)

                if is_valid_domain(orig_answer):
                    '''perform adversarial attack '''
                    print('Adversarial attack')
                    if reference_logo:
                        injected_logo = transparent_text_injection(reference_logo.convert('RGB'), 'abc.com')
                        # get image caption for injected logo
                        screenshot_img.paste(injected_logo, (int(x1), int(y1)))
                        adv_shot_path = shot_path.replace('shot.png', 'shot_adv.png')
                        screenshot_img.save(adv_shot_path)

                        # get extra description on the webpage
                        adv_ocr_text, adv_ocr_coord = get_ocr_text_coord(adv_shot_path)
                        adv_html_text = ' '.join(adv_ocr_text)
                        answer, total_time = get_results(logo_box, injected_logo, screenshot_img, adv_ocr_text,
                                                         adv_ocr_coord, adv_html_text)
                    else:
                        answer = orig_answer
                    print('After attack answer: ', answer)

            with open(result_file, 'a+') as f:
                f.write(hash+'\t'+orig_answer+'\t'+answer+'\t'+str(total_time)+'\n')

    driver.quit()
    test(result_file)