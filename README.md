# PhishLLM
Official repository for "Less Defined Knowledge and More True Alarms: Reference-based Phishing Detection without a Pre-defined Reference List".
Published in USENIX Security 2024. 

<p align="center">

  • <a href="">Read our Paper</a> •

  • <a href="https://sites.google.com/view/phishllm">Visit our Website</a> •

  • <a href="https://sites.google.com/view/phishllm/experimental-setup-datasets?authuser=0#h.r0fy4h1fw7mq">Download our Datasets</a>  •

  • <a href="#citation">Cite our Paper</a> •

</p>

## Introduction
Existing reference-based phishing detection:

- :x: Relies on a pre-defined reference list, which is lack of comprehensiveness and incurs high maintenance cost 
- :x: Does not fully make use of the textual semantics present on the webpage

In our PhishLLM, we build a reference-based phishing detection framework:

- ✅ **Without the pre-defined reference list**: Modern LLMs have encoded far more extensive brand-domain information than any predefined list
- ✅ **Chain-of-thought credential-taking prediction**: Reasoning the credential-taking status in a step-by-step way by looking at the text

## Framework
<img src="./figures/phishllm.png"/>

```Input```: a URL and its screenshot, ```Output```: Phish/Benign, Phishing target

- **Step 1: Brand recognition model**
  - Input: Logo caption, Logo OCR Results
  - Intermediate Output: LLM's predicted brand
  - Output: Validated predicted brand, confirmed through Google Images
  
- **Step 2: Credential-Requiring-Page classification model**
  - Input: Webpage OCR results
  - Output: LLM chooses from A. Credential-Taking Page or B. Non-Credential-Taking Page
  - Go to step 4 if LLM chooses 'A', otherwise go to step 3.
  
- **Step 3: Credential-Requiring-Page transition model (activates if LLM chooses 'B' from the last step)**
  - Input: All clickable UI elements
  - Intermediate Output: Top-1 most likely login UI
  - Output: Webpage after clicking that UI, **go back to Step 1** with the updated webpage and URL

- **Step 4: Output step** 
  - _Case 1_: If the domain is from a web hosting domain: it is flagged as **phishing** if
    (i) LLM predicts a targeted brand inconsistent with the webpage's domain
  and  (ii) LLM chooses 'A' from Step 2
  
  - _Case 2_: If the domain is not from a web hosting domain: it is flagged as **phishing** if
  (i) LLM predicts a targeted brand inconsistent with the webpage's domain
  (ii) LLM chooses 'A' from Step 2
  and (iii) the domain is not a popular domain indexed by Google
  
  - _Otherwise_: reported as **benign**

## Project structure

<pre>
scripts/ 
├── infer/
│   └──test.py             # inference script
├── pipeline/             
│   └──test_llm.py # TestLLM class
└── utils/ # other utitiles such as web interaction utility functions 

prompts/ 
├── brand_recog_prompt.json 
└── crp_pred_prompt.json
└── crp_trans_prompt.json

</pre>

## Setup

### Step 1: **Install Requirements**. 
- A new conda environment "phishllm" will be created after this step, type for "All" when prompted.
```bash
  cd PhishLLM/
  chmod +x ./setup.sh
  export ENV_NAME="phishllm"
  ./setup.sh
```

### Step 2: **Install ChromeDriver**. 
- Look for output logs in previous step, you should see something like 
```console
[+] google-chrome-stable is installed. (version: Google Chrome 133.0.6943.98 ). 
```
- Here, this "133.0.6943.98" is your installed Chrome version. Based on the version you have, find the corresponding chromedriver file in https://github.com/dreamshao/chromedriver.
Unzip it and place the .exe file under "./chromedriver-linux64/chromedriver". 

### Step 3: Register **Two API Keys**. 

- **OpenAI API key**, [See Tutorial here](https://platform.openai.com/docs/quickstart). Paste the API key to './datasets/openai_key.txt'.

- **Google Programmable Search API Key**, [See Tutorial here](https://meta.discourse.org/t/google-search-for-discourse-ai-programmable-search-engine-and-custom-search-api/307107). 
Paste your API Key (in the first line) and Search Engine ID (in the second line) to "./datasets/google_api_key.txt":
     ```text 
      [API_KEY]
      [SEARCH_ENGINE_ID]
     ```

## Prepare the Dataset
To test on your own dataset, you need to prepare the dataset in the following structure:
<pre>
testing_dir/
├── aaa.com/
│   ├── shot.png  # save the webpage screenshot
│   ├── info.txt  # save the webpage URL
│   └── html.txt  # save the webpage HTML source
├── bbb.com/
│   ├── shot.png  # save the webpage screenshot
│   ├── info.txt  # save the webpage URL
│   └── html.txt  # save the webpage HTML source
├── ccc.com/
│   ├── shot.png  # save the webpage screenshot
│   ├── info.txt  # save the webpage URL
│   └── html.txt  # save the webpage HTML source
</pre>


## Inference: Run PhishLLM 
```bash
  conda activate phishllm
  python -m scripts.infer.test --folder [folder to test, e.g., ./testing_dir]
```

## Understand the Output
- You will see the console is printing logs like the following <details><summary> Expand to see the sample log</summary>
  <pre><code>
    [PhishLLMLogger][DEBUG] Folder ./datasets/field_study/2023-09-01/device-862044b2-5124-4735-b6d5-f114eea4a232.remotewd.com
    [PhishLLMLogger][DEBUG] Time taken for LLM brand prediction: 0.9699530601501465 Detected brand: sonicwall.com
    [PhishLLMLogger][DEBUG] Domain sonicwall.com is valid and alive
    [PhishLLMLogger][DEBUG] Time taken for LLM CRP classification: 2.9195783138275146 	 CRP prediction: A. This is a credential-requiring page.
    [❗️] Phishing discovered, phishing target is sonicwall.com
  </code></pre></details>
  
- Meanwhile, a txt file named "[today's date]_phishllm.txt" is being created, it has the following columns: 
  - "folder": name of the folder
  - "phish_prediction": "phish" | "benign"
  - "target_prediction": phishing target brand's domain, e.g. paypal.com, meta.com
  - "brand_recog_time": time taken for brand recognition
  - "crp_prediction_time": time taken for CRP prediction
  - "crp_transition_time": time taken for CRP transition

## Citations
```bibtex
  @inproceedings {299838,
  author = {Ruofan Liu and Yun Lin and Xiwen Teoh and Gongshen Liu and Zhiyong Huang and Jin Song Dong},
  title = {Less Defined Knowledge and More True Alarms: Reference-based Phishing Detection without a Pre-defined Reference List},
  booktitle = {33rd USENIX Security Symposium (USENIX Security 24)},
  year = {2024},
  isbn = {978-1-939133-44-1},
  address = {Philadelphia, PA},
  pages = {523--540},
  url = {https://www.usenix.org/conference/usenixsecurity24/presentation/liu-ruofan},
  publisher = {USENIX Association},
  month = aug
  }
```
If you have any issues running our code, you can raise a Github issue or email us liu.ruofan16@u.nus.edu, lin_yun@sjtu.edu.cn, dcsdjs@nus.edu.sg.
