
from brand_recognition.test_caption_llm import *
import idna
import pandas as pd
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key3.txt').read()

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
    # generation caption for logo
    logo_caption = get_caption(reference_logo)

    # expand the logo bbox a bit to see the surrounding region
    expand_logo_box = expand_bbox(logo_box, image_width=screenshot_img.size[0],
                                  image_height=screenshot_img.size[1], expand_ratio=1.5)

    extra_description = ''
    if len(ocr_coord):
        # get the OCR text description surrounding the logo
        overlap_areas = compute_overlap_areas_between_lists([expand_logo_box], ocr_coord)
        extra_description = np.array(ocr_text)[overlap_areas[0] > 0].tolist()
        extra_description = ' '.join(extra_description)

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
    print('Original answer: ', orig_answer)
    return orig_answer

if __name__ == '__main__':

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy
    model = "gpt-3.5-turbo-16k"

    root_folder = './datasets/dynapd'
    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_wo_validation.txt').readlines()]
    df = pd.read_csv('./datasets/Brand_Labelled_130323.csv')
    result_file = './datasets/dynapd_llm_caption_adv_brand.txt'

    phishintention_cls = PhishIntentionWrapper()

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

            # has logo
            if (logo_boxes is not None) and len(logo_boxes):
                # get extra description on the webpage
                ocr_text, ocr_coord = get_ocr_text_coord(shot_path)
                html_text = ' '.join(ocr_text)

                for logo_box in logo_boxes[:min(len(logo_boxes), 3)]: # Top-3 logo box
                    '''original prediction '''
                    x1, y1, x2, y2 = logo_box
                    reference_logo = screenshot_img.crop((x1, y1, x2, y2))  # crop logo out
                    orig_answer = get_results(logo_box, reference_logo, screenshot_img, ocr_text, ocr_coord, html_text)

                    if is_valid_domain(orig_answer):
                        '''perform adversarial attack '''
                        injected_logo = transparent_text_injection(reference_logo.convert('RGB'), 'abc.com')
                        # get image caption for injected logo
                        screenshot_img.paste(injected_logo, (int(x1), int(y1)))
                        adv_shot_path = shot_path.replace('shot', 'shot_adv')
                        screenshot_img.save(adv_shot_path)

                        # get extra description on the webpage
                        adv_ocr_text, adv_ocr_coord = get_ocr_text_coord(adv_shot_path)
                        adv_html_text = ' '.join(adv_ocr_text)
                        answer = get_results(logo_box, injected_logo, screenshot_img, adv_ocr_text, adv_ocr_coord, adv_html_text)
                        break

            with open(result_file, 'a+') as f:
                f.write(hash+'\t'+orig_answer+'\t'+answer+'\t'+str(total_time)+'\n')