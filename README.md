# PhishLLM

<p align="center">

[//]: # (  • <a href="">Paper</a> •)

[//]: # (  • <a href="">Website</a> •)

[//]: # (  • <a href="https://drive.google.com/drive/folders/1x6N6QEt_34B-pMStbBANUrjim-2ixG6T?usp=sharing">Datasets</a>  •)

[//]: # (  • <a href="#citation">Citation</a> •)

</p>

## Introductions
Existing reference-based phishing detection:

- :x: Relies on a pre-defined reference list
- :x: Necessitates a massive amount of high-quality, diverse annotated data
- :x: Does not fully utilize the textual information present on the webpage

In our PhishLLM, we build a reference-based phishing detection framework:

- ✅ Without a pre-defined reference list
- ✅ Requires lightweight training
- ✅ Fully explainable, as it mirrors the human cognitive process during web interaction and provides natural language explanations at every step

## Framework
<img src="./figures/phishllm.png">

- Step 1: Brand recognition model
  - Input: logo caption, Logo OCR Results, industry sector (optional)
  - Intermediate Output: LLM's predicted brand
  - Output: Validated predicted brand, confirmed through Google Images
- Step 2: Credential-Requiring-Page classification model
  - Input: webpage OCR results
  - Output: LLM chooses from A. Credential-Taking Page or B. Non-Credential-Taking Page
- Step 3.1: CRP transition model (activate if LLM chooses 'B' from the last step)
  - Input: webpage clickable UI elements (the webpage must be live)
  - Intermediate Output: most likely UI element being a login button
  - Output: The page after clicking the UI 
- Step 3.2: Termination
  - A phishing alarm will be raised if:
  LLM predicts a targeted brand inconsistent with the webpage's domain
  **AND** LLM chooses 'A' from Step 2
  - A benign decision will be reached if:
  LLM cannot predict a targeted brand
  **OR** the targeted brand aligns with the webpage domain
  **OR** LLM consistently chooses 'B' even after running Step 3.1 multiple times.

## Project structure
```
|_ brand_recognition
|_ selection_model (i.e. credential-requiring-page classification model)
|_ ranking_model
|_ model_chain (chaining all the components)
  |_ test_llm.py: main class
|_ field_study 
   |_ test.py: main script
```

## Setup
- Step 1: Clone the Repository and Install Requirements
```bash
    cd PhishLLM/
    chmod +x ./setup.sh
    ./setup.sh
```
- Step 2: Register OpenAI API Key. See [OpenAI Official Docs](https://platform.openai.com/). Save the API key to './datasets/openai_key2.txt'.
- Step 3: Create a Google Cloud Service Account
  - Go to [Google Cloud Console]((https://console.cloud.google.com/)) and set up billing details.
  - Create a project and enable the "Custom Search API".
  - Obtain the API Key and Search Engine ID for "Custom Search API" following this [guide](https://developers.google.com/custom-search/v1/overview).
  - Create a blank text file in the directory "./datasets/google_api_key.txt" and paste your API Key and Search Engine ID as follows:
     ```text 
      [YOUR_API_KEY]
      [YOUR_SEARCH_ENGINE_ID]
     ```
- Step 4 (Optional): Edit Hyperparameters. All hyperparameter configurations are stored in param_dict.yaml. Edit this file to experiment with different parameter combinations.
- Step 5: Run the Code
```bash
    conda activate myenv
    python -m field_study.test --folder [folder to test, e.g., ./datasets/field_study/2023-08-21/] --date [e.g., 2023-08-21]
```

<details>
<summary> A .log file will be created during the run, which will log the explanations for each model prediction, click to see the sampled log</summary>
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
    </code></pre>
</details>

## Findings
<details>
  <summary>Number of phishing caught</summary>
  <img src="./field_study/plots/num_phish.png">
</details>
<details>
  <summary>Phishing domain age distribution</summary>
  <img src="./field_study/plots/domain_age.png">
</details>
<details>
  <summary>Phishing domain TLD distribution</summary>
  
  | Top-5 TLD | Frequency      |
  |----------------| --------------- |
  | .com | 447 occurrences |
  | .de | 60 occurrences |
  | .online | 58 occurrences |
  | .info | 52 occurrences |
  | .xyz | 52 occurrences |

</details>
<details>
  <summary>Top phishing targets, and top targeted sectors</summary>
  <img src="./field_study/plots/brand_freq.png">
  <img src="./field_study/plots/brand_sector.png">
</details>
<details>
  <summary>Geolocations of phishing IPs</summary>
  <img src="./field_study/plots/geo.png">
</details>
<details>
  <summary>Phishing campaign analysis</summary>
  <img src="./field_study/plots/campaign.png">
</details>


## Updates
- [👏2023-09-07] Packaged phishing websites reported from field study: link.
- [🛠️2023-08-28] Added functions to determine whether the webpage state has been updated (best method is to check the webpage screenshot, not the URL).
- [🛠️2023-08-28] Updated CRP transition logic: if the webpage state hasn't changed, the system will attempt to click lower-ranked buttons instead of repeatedly clicking the top-ranked button.
- [🤔2023-08-27] Found that providing the industry sector to the brand recognition model further improves brand recognition without affecting robustness.
- [🤔2023-08-24] Added an option for relaxed result validation by checking if the predicted domain is alive.
- [🛠️2023-08-20] Modified the brand recognition model.
- [🛠️2023-08-07] To prevent PhishLLM from reporting brands offering cloud services (e.g., domain hosting, web hosting, etc.), we maintain a list of such domains in ./datasets/hosting_blacklists.txt. This list will continue to grow.