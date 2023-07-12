from brand_recognition.test_baseline_logo2brand import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './brand_recognition/google_cloud.json'
os.environ['https_proxy'] = "http://127.0.0.1:7890" # proxy
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

@torch.no_grad()
def brand_recognition_domain2brand(screenshot_path, url,
                      kb_cls, kb_driver,
                      AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH):

    with open(os.path.join(get_python_lib(), 'phishintention', 'configs.yaml')) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    SIAMESE_THRE_RELAX = 0.83
    TARGETLIST_PATH = configs['SIAMESE_MODEL']['TARGETLIST_PATH']

    screenshot_img = Image.open(screenshot_path).convert("RGB")
    screenshot_img_arr = np.asarray(screenshot_img)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1)  # RGB2BGR

    pred = AWL_MODEL(screenshot_img_arr)
    pred_i = pred["instances"].to('cpu')
    pred_classes = pred_i.pred_classes.detach().cpu()  # Boxes types
    pred_boxes = pred_i.pred_boxes.tensor.detach().cpu()  # Boxes coords

    if pred_boxes is None or len(pred_boxes) == 0:
        has_logo = False
    else:
        logo_pred_boxes, logo_pred_classes = find_element_type(pred_boxes=pred_boxes,
                                                               pred_classes=pred_classes,
                                                               bbox_type='logo')
        if len(logo_pred_boxes) == 0:
            has_logo = False
        else:
            has_logo = True
            logo_pred_boxes = logo_pred_boxes.detach().cpu().numpy()
            x1, y1, x2, y2 = logo_pred_boxes[0]
            reference_logo = screenshot_img.crop((x1, y1, x2, y2))
            reference_logo.save('./tmp.png')
            img_feat = pred_siamese_OCR(img=reference_logo,
                                         model=SIAMESE_MODEL,
                                         ocr_model=OCR_MODEL)
            sim_list = LOGO_FEATS @ img_feat.T  # take dot product for every pair of embeddings (Cosine Similarity)

    if not has_logo:
        return ' '

    if np.sum(sim_list >= SIAMESE_THRE_RELAX) == 0:  # not exceed siamese relaxed threshold, not in targetlist
        # KB
        query_domain, query_tld = tldextract.extract(url).domain, tldextract.extract(url).suffix
        _, new_domains, new_brand, new_logos, comment = kb_cls.runit_simplified(reference_logo, './tmp.png', kb_driver, query_domain, query_tld,
                                                                                type='domain2brand')
        # expand
        if len(new_domains) > 0 and np.sum([x is not None for x in new_logos]) > 0:
            # Ref* <- Ref* + <domain(w_target), rep(w_target)>
            LOGO_FEATS, LOGO_FILES = expand_targetlist(DOMAIN_MAP_PATH, TARGETLIST_PATH,
                                                      SIAMESE_MODEL, OCR_MODEL,
                                                      new_brand, new_domains, new_logos)
        else:
            return ' '

    target_this, domain_this, this_conf = brand_siamese_classifier(reference_logo,
                                                                   SIAMESE_MODEL, OCR_MODEL,
                                                                   SIAMESE_THRE,
                                                                   LOGO_FEATS, LOGO_FILES,
                                                                   DOMAIN_MAP_PATH
                                                                   )

    if target_this is None:
        return ' '
    else:
        return domain_this[0]


if __name__ == '__main__':

    dataset = ShotDataset(annot_path='./datasets/alexa_screenshots_orig.txt')

    result_file = './datasets/alexa_brand_baseline_domain2brand.txt'

    print(len(dataset))
    SEARCH_ENGINE_ID, API_KEY = [x.strip() for x in open('./brand_recognition/google_search_key.txt').readlines()]
    AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH = load_config()
    SIAMESE_THRE_RELAX = 0.83
    kb_cls = BrandKnowledgeConstruction(API_KEY, SEARCH_ENGINE_ID,
                                        AWL_MODEL, SIAMESE_MODEL, OCR_MODEL,
                                                          SIAMESE_THRE_RELAX)

    sleep_time = 3; timeout_time = 60
    XDriver.set_headless()
    Logger.set_debug_on()
    kb_driver = XDriver.boot(chrome=True)
    kb_driver.set_script_timeout(timeout_time/2)
    kb_driver.set_page_load_timeout(timeout_time)
    time.sleep(sleep_time)  # fixme: you have to sleep sometime, otherwise the browser will keep crashing


    for url, shot_path, label in tqdm(zip(dataset.urls, dataset.shot_paths, dataset.labels)):
        if os.path.exists(result_file) and url in open(result_file).read():
            continue

        while True:
            try:
                start_time = time.time()
                html_path = shot_path.replace('shot.png', 'index.html')
                domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
                answer = brand_recognition_domain2brand(shot_path, url, kb_cls, kb_driver,
                                                      AWL_MODEL, CRP_CLASSIFIER, CRP_LOCATOR_MODEL, SIAMESE_MODEL, OCR_MODEL, SIAMESE_THRE, LOGO_FEATS, LOGO_FILES, DOMAIN_MAP_PATH)
                total_time = time.time() - start_time
                break
            except Exception as e:
                print(e)
                time.sleep(10)

        with open(result_file, 'a+') as f:
            f.write(url+'\t'+domain+'\t'+answer+'\t'+str(total_time)+'\n')
