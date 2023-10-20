import uuid
from threading import Thread
from urllib.parse import unquote
from flask import Flask, Response, request, session, jsonify, render_template
from flask_cors import CORS
from flask_session import Session
from model_chain.test_llm import *
from apscheduler.schedulers.background import BackgroundScheduler
import shutil
import random
import concurrent.futures
# os.environ['proxy_url'] = "http://127.0.0.1:7890"
announcers = {}

class Config:
    CURRENT_DIR = os.path.dirname(__file__)
    LOGS_DIR = os.path.join(CURRENT_DIR, "logs")
    REQUESTS_DIR = os.path.join(CURRENT_DIR, "requests")
    PARAM_PATH = os.path.join(CURRENT_DIR, "../param_dict.yaml")
    SESSION_TYPE = "filesystem"
    SESSION_PERMANENT = False
    TIMEOUT_TIME = 60

# Initialize server and PhishLLM
app = Flask(__name__)
app.config['SESSION_TYPE'] = Config.SESSION_TYPE
app.config['SESSION_PERMANENT'] = Config.SESSION_PERMANENT
CORS(app)
Session(app)

os.makedirs(Config.LOGS_DIR, exist_ok=True)
os.makedirs(Config.REQUESTS_DIR, exist_ok=True)

def clear_directories():
    if os.path.exists(Config.LOGS_DIR):
        shutil.rmtree(Config.LOGS_DIR)
    if os.path.exists(Config.REQUESTS_DIR):
        shutil.rmtree(Config.REQUESTS_DIR)
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    os.makedirs(Config.REQUESTS_DIR, exist_ok=True)

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=clear_directories, trigger="interval", days=1)  # clear every day
scheduler.start()

# load hyperparameters
with open(Config.PARAM_PATH) as file:
    param_dict = yaml.load(file, Loader=yaml.FullLoader)

# PhishLLM
proxy_url = os.environ.get('proxy_url', None)
phishintention_cls = PhishIntentionWrapper()
llm_cls = TestLLM(phishintention_cls, param_dict=param_dict,
                  proxies={"http": proxy_url,
                           "https": proxy_url,
                           }
                  )
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxy = proxy_url

@app.route("/")
def interface():
    session['ready'] = True
    return render_template("index.html")

# STEP 1: Crawl URL and take a screenshot, return screenshot in base64
# frontend url -> backend crawling -> frontend display
@app.route("/crawl", methods=["POST"])
def crawl():
    url = str(request.json["url"])
    driver = get_xdriver()
    id = uuid.uuid4().hex
    html_path = os.path.join(Config.REQUESTS_DIR, f"{id}.txt")
    screenshot_path = os.path.join(Config.REQUESTS_DIR, f"{id}.png")
    success = False

    try:
        driver.delete_all_cookies()
        driver.get(url)
        time.sleep(1.5) # wait for the page to be fully loaded
        success = driver.save_screenshot(screenshot_path)
        with open(html_path, "w") as f:
            f.write(driver.page_source)
        driver.quit()
    except:
        driver.quit()
        return jsonify(success=False), 400

    if success:
        session['url'] = url
        session['html_path'] = html_path
        session['screenshot_path'] = screenshot_path
        with open(screenshot_path, "rb") as image_file:
            im_b64 = base64.b64encode(image_file.read()).decode()
        return jsonify(success=True, screenshot=im_b64, url=url, screenshot_path=screenshot_path, html_path=html_path), 200
    else:
        return jsonify(success=False), 400


# STEP 2: Perform PhishLLM inference
# frontend url,screenshot_path -> backend inference -> frontend announcer
@app.route('/listen', methods=['GET'])
def listen():
    id = request.args.get('id')  # get request id
    params = json.loads(request.args.get('params'))  # get params
    # update LLM
    uodate_LLM_params(params)
    url = unquote(request.args.get('url'))
    screenshot_path = unquote(request.args.get('screenshot_path'))
    html_path = unquote(request.args.get('html_path'))

    print("url,", url, "screenshot_path", screenshot_path)

    if not (id and os.path.exists(screenshot_path)):
        return jsonify(success=False), 400

    # 为每个id创建一个新的announcer对象
    if id not in announcers:
        announcers[id] = Announcer()
    announcer = announcers[id]

    def stream(url, screenshot_path, html_path):
        Thread(target=get_inference, args=(url, screenshot_path, html_path, announcer)).start()
        messages = announcer.message_queue
        while True:
            msg = messages.get()  # blocks until a new message arrives
            yield msg

    return Response(stream(url, screenshot_path, html_path), mimetype='text/event-stream')

def uodate_LLM_params(form_data):
    # Partially updating param_dict based on form data
    for section, params in form_data.items():
        if section in param_dict:
            for param, value in params.items():
                if param in param_dict[section]:
                    try:
                        if isinstance(value, float):
                            param_dict[section][param] = float(value)
                        elif isinstance(value, bool):
                            param_dict[section][param] = bool(value)
                        elif isinstance(value, int):
                            param_dict[section][param] = int(value)
                        print(section, param, value)
                    except ValueError:
                        return 

    # Update internal state of TestLLM
    llm_cls.update_params(param_dict)  # Assumes such a method exists


def get_xdriver():
    timeout_time = Config.TIMEOUT_TIME  # Moved to Config class
    driver = CustomWebDriver.boot(proxy_server=proxy_url)  # Using the proxy_url variable
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    return driver

def get_inference(url, screenshot_path, html_path, announcer):
    start_time = time.time()
    driver = get_xdriver()
    msg = f'Time taken for initializing webdriver: {time.time()-start_time}'
    announcer.spit(msg, AnnouncerEvent.RESPONSE)
    time.sleep(0.5)
    start_time = time.time()
    logo_box, reference_logo = llm_cls.detect_logo(screenshot_path)
    msg = f'Time taken for logo detection: {time.time()-start_time}'
    announcer.spit(msg, AnnouncerEvent.RESPONSE)
    time.sleep(0.5)

    llm_cls.test(
        url,
        reference_logo,
        logo_box,
        screenshot_path,
        html_path,
        driver,
        announcer=announcer
    )
    driver.quit()

# Function to fetch sampled URLs from the server
def fetch_sampled_urls(url="https://openphish.com/feed.txt"):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.text.split('\n')  # Split by newlines to get a list
        return data
    except requests.RequestException as e:
        raise Exception("Failed to fetch URLs from the feed.") from e

# Function to check the aliveness of URLs
def keep_alive_urls(urls):
    # Do a downsample first
    if len(urls) > 50:
        urls = random.sample(urls, 50)

    alive_urls = []

    def check_url_alive(url):
        try:
            response = requests.head(url, timeout=1)
            if response.status_code == 200:
                return url
        except requests.exceptions.RequestException:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(check_url_alive, urls))

    # Remove None values (failed requests) and limit to 20 alive URLs
    alive_urls = [url for url in results if url is not None]

    return alive_urls

@app.route('/sample_urls', methods=['POST'])
def sample_urls():
    # Sample URLs
    sampled_urls = fetch_sampled_urls()
    alive_urls = keep_alive_urls(sampled_urls)
    return jsonify({'sampled_urls': alive_urls})

if __name__ == "__main__":

    app.run("0.0.0.0", port=6789, debug=True)
