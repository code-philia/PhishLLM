# PhishLLM

## Introductions

## Framework
- Step 1: Brand Recognition Model to predict the targeted brand
  - Input: Logo Caption, Logo OCR results, Industry sector (optional)
  - Intermediate output: LLM's predicted brand
  - Output: Validated predicted brand, validated through Google Images
- Step 2: Credential-requiring-page Classification Model 
  - Input: Webpage OCR results
  - Output: LLM chooses from A. credential-taking page B. non-credential-taking page
- Step 3.1 Ranking Model (Activate if LLM chooses B from last step): 
  - Input: Webpage clickable UI elements (the webpage must be alive)
  - Intermediate output: The most likely UI being a login button
  - Output: The page after clicking the UI
- Step 3.2: Termination
  - Phishing alarm will be raised if 
    - LLM predicts a targeted brand that is not consistent with the webpage's domain 
    - **AND** the LLM chooses A from step 2
  - Benign decision will be reached if 
    - LLM cannot predict a targeted brand
    - **OR** the targeted brand aligns with the webpage domain
    - **OR** the LLM consistently chooses B even after running step 3.1 for multiple times.

## Project structure
```
|_ brand_recognition
|_ selection_model (i.e. credential-requiring-page classification model)
|_ ranking_model
|_ model_chain (chaining all the components)
|_ field_study 
   |_ test.py: main script
```

## Setup
- Step 1: Install requirements
```
    git clone https://github.com/lindsey98/PhishLLM.git
    cd PhishLLM/
    chmod +x ./setup.sh
    ./setup.sh
```
- Step 2: Register openai API key: See https://platform.openai.com/. Save the API key to './datasets/openai_key2.txt'
- Step 3: Run!
```
    conda activate myenv
    python -m field_study.test --folder [folder to test, e.g. ./datasets/field_study/2023-08-21/] --date [e.g. 2023-08-21]
```