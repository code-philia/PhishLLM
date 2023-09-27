import os
import uuid
import time
import base64
from threading import Thread
from datetime import datetime

from flask import Flask, Response, request, session, jsonify, render_template
from flask_cors import CORS
from flask_session import Session
from selenium import webdriver
from seleniumwire.webdriver import ChromeOptions

from server.announcer import Announcer
from model_chain.test_llm import *

class Config:
    CURRENT_DIR = os.path.dirname(__file__)
    LOGS_DIR = os.path.join(CURRENT_DIR, "logs")
    REQUESTS_DIR = os.path.join(CURRENT_DIR, "requests")
    PARAM_PATH = os.path.join(CURRENT_DIR, "../param_dict.yaml")
    SESSION_TYPE = "filesystem"
    SESSION_PERMANENT = False

# Initialize server and PhishLLM
app = Flask(__name__)
app.config['SESSION_TYPE'] = Config.SESSION_TYPE 
app.config['SESSION_PERMANENT'] = Config.SESSION_PERMANENT
CORS(app)
Session(app)

os.makedirs(Config.LOGS_DIR, exist_ok=True)
os.makedirs(Config.REQUESTS_DIR, exist_ok=True)

PhishLLMLogger.set_debug_on()
PhishLLMLogger.set_logfile(os.path.join(Config.LOGS_DIR, f"{datetime.now().strftime('%Y-%d-%m_%H%M%S')}.log"))

# load hyperparameters
with open('./param_dict.yaml') as file:
    param_dict = yaml.load(file, Loader=yaml.FullLoader)

# PhishLLM
phishintention_cls = PhishIntentionWrapper()
llm_cls = TestLLM(phishintention_cls, param_dict=param_dict,
                  proxies={"http": "http://127.0.0.1:7890",
                           "https": "http://127.0.0.1:7890",
                           }
                  ) # todo
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxy = "http://127.0.0.1:7890" # # todo

@app.route("/")
def interface():
    session['ready'] = True
    return render_template("index.html")

# STEP 1: Crawl URL and take a screenshot, return screenshot in base64
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
        time.sleep(3) # wait for the page to be fully loaded
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
        return jsonify(success=True, screenshot=im_b64), 200
    else:
        return jsonify(success=False), 400

# STEP 2: Perform PhishLLM inference
@app.route('/listen', methods=['GET'])
def listen():
    if not (
        session.get('screenshot_path')
        and session.get('html_path')
        and session.get('url')
    ):
        return jsonify(success=False), 400

    url = session['url']
    screenshot_path = session['screenshot_path']
    html_path = session['html_path']

    def stream(url, screenshot_path, html_path):
        announcer = Announcer()
        Thread(target=get_inference, args=(url, screenshot_path, html_path, announcer)).start()
        messages = announcer.message_queue
        while True:
            msg = messages.get()  # blocks until a new message arrives
            yield msg

    return Response(stream(url, screenshot_path, html_path), mimetype='text/event-stream')

def get_xdriver():
    timeout_time = 60
    driver = CustomWebDriver.boot()
    driver.set_script_timeout(timeout_time/2)
    driver.set_page_load_timeout(timeout_time)
    return driver

def get_inference(url, screenshot_path, html_path, announcer):
    driver = get_xdriver()
    logo_box, reference_logo = llm_cls.detect_logo(screenshot_path)
    llm_cls.test(
        url,
        reference_logo,
        logo_box,
        screenshot_path,
        html_path,
        driver,
        brand_recognition_do_validation=False,
        announcer=announcer
    )
    driver.quit()

if __name__ == "__main__":
    app.run("0.0.0.0", port=6789, debug=True)