# PhishLLM
Official repository for "Less Defined Knowledge and More True Alarms: Reference-based Phishing Detection without a Pre-defined Reference List".
Published in USENIX Security 2024. 

<p align="center">

  • <a href="">Paper</a> •

  • <a href="https://sites.google.com/view/phishllm">Website</a> •

  • <a href="https://sites.google.com/view/phishllm/experimental-setup-datasets?authuser=0#h.r0fy4h1fw7mq">Datasets</a>  •

  • <a href="#citation">Citation</a> •

</p>

## Introduction
Existing reference-based phishing detection:

- :x: Relies on a pre-defined reference list, which is lack of comprehensiveness and incurs high maintenance cost 
- :x: Does not fully make use of the textual semantics present on the webpage

In our PhishLLM, we build a reference-based phishing detection framework:

- ✅ **Without the pre-defined reference list**: Modern LLMs have encoded far more extensive brand-domain information than any predefined list
- ✅ **Chain-of-thought credential-taking prediction**: Reasoning the credential-taking status in a step-by-step way by looking at the text

## Framework
<embed src="./figures/phishllm.pdf" width="100%" height="600px" />

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
  
  - _Other cases_: reported as **benign**

## Project structure

<pre>
models/ (defining the brand recognition model, CRP prediction model, and CRP transition model)
├── brand_recognition/      # brand recognition model
├── selection_model/        # CRP prediction model
└── ranking_model/          # CRP transition model

pipeline/ (chaining all the components together)
└── test_llm.py             # defining the TestLLM class

experiments/
├── ablation_study/ 
│   ├── adapt_to_cryptocurrency_phishing.py  # exploration of VLM
│   ├── cost_benchmarking/                   # benchmarking the runtime of PhishLLM
│   ├── domain_alias/                        # domain alias experiment in RQ2
│   ├── test_on_middle_ranked_benign.py      # lower-rank Alexa experiment in RQ2
│   └── test_on_public_phishing/             # public phishing study in RQ4
└── field_study/                             # Large/Small-scale field study in RQ4
    └── test.py                              # main testing script
</pre>

## Setup
- Step 1: Clone the Repository and **Install Requirements**. A new conda environment "phishllm" will be created
```bash
    cd PhishLLM/
    chmod +x ./setup.sh
    export ENV_NAME="phishllm" && ./setup.sh
```
- Step 2: Register **OpenAI API Key**. See [OpenAI Official Docs](https://platform.openai.com/). Paste the API key to './datasets/openai_key.txt'.

- Step 3: Register a **Google Programmable Search API Key**
  - Go to [Google Cloud Console]((https://console.cloud.google.com/)) and set up billing details.
  - Create a project and enable the "Custom Search API".
  - Obtain the API Key and Search Engine ID for "Custom Search API" following this [guide](https://developers.google.com/custom-search/v1/overview).
  - Create a blank text file in the directory "./datasets/google_api_key.txt" and paste your API Key (in the first line) and Search Engine ID (in the second line) as follows:
     ```text 
      [YOUR_API_KEY]
      [YOUR_SEARCH_ENGINE_ID]
     ```
    
- Step 4 (Optional): Edit Hyperparameters. All hyperparameter configurations are stored in param_dict.yaml. Edit this file to experiment with different parameter combinations.

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
    python -m experiments.field_study.test --folder [folder to test, e.g., ./testing_dir]
  ```

## Understand the Output
- You will see the console is printing logs like the following <details><summary> Click to see the sample log</summary>
    <pre><code>
      [PhishLLMLogger][DEBUG] Folder ./datasets/field_study/2023-09-01/device-862044b2-5124-4735-b6d5-f114eea4a232.remotewd.com
      [PhishLLMLogger][DEBUG] Logo caption: the logo for sonicwall network security appliance
      [PhishLLMLogger][DEBUG] Logo OCR: SONICWALL Network Security Appliance Username
      [PhishLLMLogger][DEBUG] Industry: Technology
      [PhishLLMLogger][DEBUG] LLM prediction time: 0.9699530601501465
      [PhishLLMLogger][DEBUG] Detected brand: sonicwall.com
      [PhishLLMLogger][DEBUG] Domain sonicwall.com is valid and alive
      [PhishLLMLogger][DEBUG] CRP prediction: There is no confusing token. Then we find the keywords that are related to login: LOG IN. Additionally, the presence of "Username" suggests that this page requires credentials. Therefore, the answer would be A.
      [💥] Phishing discovered, phishing target is sonicwall.com
      [PhishLLMLogger][DEBUG] Folder ./datasets/field_study/2023-09-01/lp.aldooliveira.com
      [PhishLLMLogger][DEBUG] Logo caption: a black and white photo of the word hello world
      [PhishLLMLogger][DEBUG] Logo OCR: Hello world! Welcome to WordPress. This is your first post. Edit or delete it, then start writing! dezembro 2, 2021 publicado
      [PhishLLMLogger][DEBUG] Industry: Uncategorized
      [PhishLLMLogger][DEBUG] LLM prediction time: 0.8813009262084961
      [PhishLLMLogger][DEBUG] Detected brand: wordpress.com
      [PhishLLMLogger][DEBUG] Domain wordpress.com is valid and alive
      [PhishLLMLogger][DEBUG] CRP prediction: There is no token or keyword related to login or sensitive information. Therefore the answer would be B.
      [PhishLLMLogger][DEBUG] No candidate login button to click
       [✅] Benign
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
```

