import queue

class AnnouncerPrompt:
    @staticmethod
    def question_template_prediction(html_text):
        return f"Given the HTML webpage text, Question: \
                 A. This is a credential-requiring page. B.\
                 This is not a credential-requiring page.<br> Answer: "

    @staticmethod
    def question_template_brand(logo_caption, logo_ocr):
        return f"Given the description on the brand's logo, and the logo's OCR text \
                 Question: What is the brand's domain? <br> Answer: "

    @staticmethod
    def question_template_brand_industry(logo_caption, logo_ocr, industry):
        return f"Given the description on the brand's logo, the logo's OCR text, and the industry sector.\
                Question: What is the brand's domain? <br> Answer: "

    @staticmethod
    def question_template_industry(html_text):
        return f"Your task is to predict the industry sector given webpage content. \
                Only give the industry sector, do not output any explanation.\
                Given the webpage text, Question: What is the webpage's industry sector? <br> Answer: "

class AnnouncerEvent:
    PROMPT = "prompt"
    RESPONSE = "response"
    FAIL = "fail"
    SUCCESS = "success"

class Announcer:
    def __init__(self):
        """
        Each Announcer instance keeps track of a single user session.
        Creates a queue to store server-side events (SSE) messages.
        Sends output logs from PhishLLM to the user via SSE. 
        """
        self.spit = self._spit
        self.message_queue = queue.Queue(maxsize=1)

    def format(self, msg: str, event: str) -> str:
        """
        Format a string as a SSE message.
        """
        msg = msg.replace(" \n ", "<br>")
        return f'event: {event}\ndata: {msg}\n\n'

    @staticmethod
    def spit(msg: str, event: str):
        """
        Ignore PhishLLM output log if Announcer is not an instance (no defined user session)
        """
        return

    def _spit(self, msg: str, event: str):
        """
        Convert PhishLLM output log into SSE message, send to user
        """
        try:
            msg = self.format(msg, event)
            self.message_queue.put_nowait(msg)
        except:
            pass

announcer = Announcer()