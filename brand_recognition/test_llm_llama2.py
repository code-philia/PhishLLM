from typing import List, Optional
from llama import Llama, Dialog
import os
from brand_recognition.test_llm import *
import contextlib
import json
from tqdm import tqdm
import tldextract
import time
from concurrent.futures import ThreadPoolExecutor

@contextlib.contextmanager
def set_env_vars(vars):
    original_vars = {var: os.environ.get(var) for var in vars}
    os.environ.update(vars)
    yield
    for var, value in original_vars.items():
        if value is None:
            del os.environ[var]
        else:
            os.environ[var] = value

def initialize_llama2():
    with set_env_vars({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12355', 'RANK': '0', 'WORLD_SIZE': '1'}):
        return Llama.build(
            ckpt_dir='../llama/llama-2-7b-chat',
            tokenizer_path='../llama/tokenizer.model',
            max_seq_len=512,
            max_batch_size=8,
        )

def prepare_webpage(img_path, phishintention_cls):
    # report logo
    screenshot_img = Image.open(img_path)
    screenshot_img = screenshot_img.convert("RGB")
    with open(img_path, "rb") as image_file:
        screenshot_encoding = base64.b64encode(image_file.read())
    logo_boxes = phishintention_cls.predict_all_uis4type(screenshot_encoding, 'logo')
    caption = ''
    extra_description = ''
    ocr_text = []
    reference_logo = None

    if (logo_boxes is not None) and len(logo_boxes):
        logo_box = logo_boxes[0]  # get coordinate for logo
        x1, y1, x2, y2 = logo_box
        reference_logo = screenshot_img.crop((x1, y1, x2, y2))  # crop logo out
        # generation caption for logo
        caption = get_caption(reference_logo)

        # get extra description on the webpage
        ocr_text, ocr_coord = get_ocr_text_coord(img_path)
        # expand the logo bbox a bit to see the surrounding region
        expand_logo_box = expand_bbox(logo_box, image_width=screenshot_img.size[0],
                                      image_height=screenshot_img.size[1], expand_ratio=1.5)

        if len(ocr_coord):
            # get the OCR text description surrounding the logo
            overlap_areas = compute_overlap_areas_between_lists([expand_logo_box], ocr_coord)
            extra_description = np.array(ocr_text)[overlap_areas[0] > 0].tolist()
            extra_description = ' '.join(extra_description)

    return caption, extra_description, ' '.join(ocr_text), reference_logo

if __name__ == "__main__":
    llama2_model = initialize_llama2()

    proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }
    API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]

    dataset = ShotDataset_Caption(annot_path='./datasets/alexa_screenshots_orig.txt')
    print(len(dataset))
    result_file = './datasets/alexa_brand_llama2.txt'
    phishintention_cls = PhishIntentionWrapper()

    for it in tqdm(range(len(dataset))):
        img_path = dataset.shot_paths[it]
        url = dataset.urls[it]
        label = dataset.labels[it]

        if os.path.exists(result_file) and url in open(result_file).read():
            continue

        logo_caption, logo_ocr, html_text, reference_logo = prepare_webpage(img_path, phishintention_cls)
        print('Logo caption: ', logo_caption)
        print('Logo OCR: ', logo_ocr)
        domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix

        if len(logo_caption) or len(logo_ocr):
            question = question_template_caption(logo_caption, logo_ocr)

            with open('./brand_recognition/prompt.json', 'rb') as f:
                prompt = json.load(f)
            new_prompt = prompt
            new_prompt.append(question)

            # example token count from the OpenAI API
            start_time = time.time()
            results = llama2_model.chat_completion(
                [new_prompt],  # type: ignore
                max_gen_len=None,
                temperature=0,
                top_p=0.9,
            )
            total_time = time.time() - start_time

            answer = results[0]['generation']['content'].strip().lower()
            print(answer)
            if len(answer) > 0 and is_valid_domain(answer):

                validation_success = False
                start_time = time.time()
                API_KEY, SEARCH_ENGINE_ID = [x.strip() for x in open('./datasets/google_api_key.txt').readlines()]
                returned_urls = query2image(query=answer + ' logo',
                                            SEARCH_ENGINE_ID=SEARCH_ENGINE_ID, SEARCH_ENGINE_API=API_KEY,
                                            num=5, proxies=proxies
                                            )
                logos = get_images(returned_urls, proxies=proxies)
                d_logo = url2logo(f'https://{answer}', phishintention_cls)
                if d_logo:
                    logos.append(d_logo)
                print('Crop the logo time:', time.time() - start_time)

                if reference_logo and len(logos) > 0:
                    reference_logo_feat = pred_siamese_OCR(img=reference_logo,
                                                           model=phishintention_cls.SIAMESE_MODEL,
                                                           ocr_model=phishintention_cls.OCR_MODEL)
                    start_time = time.time()
                    sim_list = []
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(pred_siamese_OCR, logo,
                                                   phishintention_cls.SIAMESE_MODEL,
                                                   phishintention_cls.OCR_MODEL) for logo in logos]
                        for future in futures:
                            logo_feat = future.result()
                            matched_sim = reference_logo_feat @ logo_feat
                            sim_list.append(matched_sim)
                    if any([x > 0.7 for x in sim_list]):
                        validation_success = True

                if not validation_success:
                    answer = 'failure in logo matching'
            else:
                answer = 'no prediction'
        else:
            answer = 'no prediction'

        with open(result_file, 'a+') as f:
            f.write(url + '\t' + domain + '\t' + answer + '\t' + str(total_time) + '\n')

    test(result_file)
    # LLAMA2 Completeness (% brand recognized) = 0.5107853982300885 Median runtime 0.19560885429382324, Mean runtime 0.29112398195846945