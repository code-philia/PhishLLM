import time
import openai
from brand_recognition.dataloader import *
import pandas as pd
os.environ['OPENAI_API_KEY'] = open('./datasets/openai_key.txt').read()
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.proxy = "http://127.0.0.1:7890" # proxy

    model = "gpt-3.5-turbo-16k"
    root_folder = './datasets/dynapd'
    all_folders = [x.strip().split('\t')[0] for x in open('./datasets/dynapd_wo_validation.txt').readlines()]
    df = pd.read_csv('./datasets/Brand_Labelled_130323.csv')

    result = './datasets/dynapd_llm_adv_brand_defense.txt'

    for hash in tqdm(all_folders):
        target_folder = os.path.join(root_folder, hash)
        if os.path.exists(result) and hash in open(result).read():
            continue
        # if hash != '1d3556c7dd3b90a8d4de96380916a1c4':
        #     continue

        shot_path = os.path.join(target_folder, 'shot.png')
        html_path = os.path.join(target_folder, 'index.html')

        html_text = get_ocr_text(shot_path, html_path)
        if len(html_text):
            question = question_template_adversary(html_text, 'abc.com')

            with open('./brand_recognition/prompt_defense2.json', 'rb') as f: # domain prediction with explanation
                prompt = json.load(f)
            new_prompt = prompt
            new_prompt.append(question)

            # original response
            start_time = time.time()
            inference_done = False
            while not inference_done:
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=new_prompt,
                        temperature=0,
                        max_tokens=100,  # we're only counting input tokens here, so let's not waste tokens on the output
                    )
                    inference_done = True
                except Exception as e:
                    print(f"Error was: {e}")
                    new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
                    time.sleep(10)
            total_time = time.time() - start_time

            answer = ''.join([choice["message"]["content"] for choice in response['choices']])
            print(answer)

            if 'abc.com' in answer:
                # delete the most influential tokens then predict again
                print('Enter correction branch')
                if 'Indicative tokens' in answer:
                    indicative_tokens = answer.split('Indicative tokens: ')[1].split(' ')
                else:
                    indicative_tokens = [answer]
                for x in indicative_tokens:
                    question['content'] = question['content'].replace(x, '')
                with open('./brand_recognition/prompt_defense2.json', 'rb') as f:
                    prompt = json.load(f)
                new_prompt = prompt
                new_prompt.append(question)

                start_time = time.time()
                inference_done = False
                while not inference_done:
                    try:
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=new_prompt,
                            temperature=0,
                            max_tokens=100,
                            # we're only counting input tokens here, so let's not waste tokens on the output
                        )
                        inference_done = True
                    except Exception as e:
                        print(f"Error was: {e}")
                        new_prompt[-1]['content'] = new_prompt[-1]['content'][:len(new_prompt[-1]['content']) // 2]
                        time.sleep(10)
                total_time = time.time() - start_time

                answer = ''.join([choice["message"]["content"] for choice in response['choices']])
                print(answer)
                answer = answer.split('\n')[0]

        else:
            answer = ''
        with open(result, 'a+') as f:
            f.write(hash+'\t'+answer+'\t'+str(total_time)+'\n')


