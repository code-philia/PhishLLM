const PROMPT_COLOR = "#D24317";
const RESPONSE_COLOR = "#00895A";

const urlForm = document.getElementById("url-form");
const urlSubmitButton = document.getElementById("url-search-button");
const urlLoading = document.getElementById("url-loading");
const urlResult = document.getElementById("url-result");
const urlScreenshot = document.getElementById("url-screenshot");
const urlSuccess = document.getElementById("url-success");
const urlFail = document.getElementById("url-fail");

const inferenceLoading = document.getElementById("inference-loading");
const inferenceResult = document.getElementById("inference-result");
const inferenceLogs = document.getElementById("inference-logs");
const inferenceSuccess = document.getElementById("inference-success");
const inferenceFail = document.getElementById("inference-fail");
let url = "";
let eventNumber = 0;
let eventQueue = [];

urlForm.addEventListener("submit", async function (e) {
  e.preventDefault();
  var formData = new FormData(e.target);
  url = formData.get("url");
  await getScreenshot();
});

const scrollIntoView = (element) => {
  element.scrollIntoView({ behavior: "smooth", block: "end" });
}

const resetAll = () => {
  urlForm.reset();
  urlResult.style.display = "none";
  inferenceResult.style.display = "none";
  scrollIntoView(urlForm);
}

const getScreenshot = async () => {
  urlResult.style.display = "block";
  urlSuccess.style.display = "none";
  urlFail.style.display = "none";
  urlLoading.style.display = "block";
  urlSubmitButton.disabled = true;
  urlScreenshot.replaceChildren();
  
  try {
    const response = await fetch(`/crawl`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    if (response?.ok) {
      const responseBody = await response.json();
      var screenshot = document.createElement("img");
      screenshot.src = "data:image/png;base64," + responseBody.screenshot;
      urlScreenshot.appendChild(screenshot);
      urlSuccess.style.display = "block";
      scrollIntoView(urlSuccess);
    } else {
      urlFail.style.display = "block";
      scrollIntoView(urlFail);
    }
  } catch(error) {
    console.log(error);
    urlFail.style.display = "block";
    scrollIntoView(urlFail);
  }

  urlLoading.style.display = "none";
  urlSubmitButton.disabled = false;
}

const getInference = () => {
  const eventSource = new EventSource(`/listen`);
  inferenceResult.style.display = "block";
  inferenceSuccess.style.display = "none";
  inferenceFail.style.display = "none";
  inferenceLoading.style.display = "block";
  inferenceLogs.replaceChildren();
  scrollIntoView(inferenceResult);

  const queueEvent = async (event) => {
    eventQueue.push(event);
    if (eventQueue.length > 1) return;
    while (eventQueue.length) {
      try {
        await runEvent();
      } catch(err) {
        console.log(err);
      }
    }
  }

  const runEvent = async () => {
    eventNumber += 1;
    try {
      let event = eventQueue[0];
      await eventHandlers[event.type](event);
    } finally {
      eventQueue.shift();
    }
  }

  const handlePromptEvent = async (event) => {
    await showMessage(true, event.data);
  }

  const handleResponseEvent = async (event) => {
    await showMessage(false, event.data);
  }

  const handleSuccessEvent = async (event) => {
    await showMessage(false, event.data);
    inferenceSuccess.style.display = "block";
    inferenceLoading.style.display = "none";
    scrollIntoView(inferenceSuccess);
    eventSource.close();
  }

  const handleFailEvent = async (event) => {
    inferenceFail.style.display = "block";
    inferenceLoading.style.display = "none";
    scrollIntoView(inferenceFail);
    eventSource.close();
  }

  const showMessage = async (isPrompt, msg) => {
    var inferenceNode = document.createElement('div');
    var color = isPrompt ? PROMPT_COLOR : RESPONSE_COLOR;
    var chat_caption = isPrompt ? 'Prompt' : 'Response'
    
    inferenceNode.innerHTML = `<hgroup><kbd style='background-color: ${color};'>${chat_caption}</kbd><div id=msg-${eventNumber}></div></hgroup><hr />`;
    inferenceLogs.appendChild(inferenceNode);
    scrollIntoView(inferenceResult);

    var lines = msg.split("<br>");
    var msgNode = document.getElementById(`msg-${eventNumber}`);
    for (const [i, line] of lines.entries()) {
      if (!isPrompt) await typeWriter(msgNode, line);
      else msgNode.innerHTML += line;
      
      if (i < lines.length - 1) msgNode.innerHTML += "<br><br>\n";
      scrollIntoView(inferenceResult);
    }

    await new Promise(r => setTimeout(r, 1000)); 
  }

  const typeWriter = async (element, str) => {
    return new Promise((resolve) => {
      var chars = str.split("");
      var prevHeight = document.body.scrollHeight;
  
      (function animate() {
        if (chars.length > 0) {
          element.innerHTML += chars.shift();
          var running = setTimeout(animate, 20);
        } else {
          clearTimeout(running);
          resolve();
        }

        if (prevHeight != document.body.scrollHeight) {
          scrollIntoView(inferenceResult);
          prevHeight = document.body.scrollHeight
        }
      })();
    });
  };

  const eventHandlers = {
    "prompt": handlePromptEvent,
    "response": handleResponseEvent,
    "success": handleSuccessEvent,
    "fail": handleFailEvent
  }

  eventSource.addEventListener("response", queueEvent);
  eventSource.addEventListener("prompt", queueEvent);
  eventSource.addEventListener("success", queueEvent);
  eventSource.addEventListener("fail", queueEvent);

  eventSource.onerror = (err) => {
    console.error("EventSource failed:", err);
    inferenceFail.style.display = "block";
    inferenceLoading.style.display = "none";
    scrollIntoView(inferenceFail);
    eventSource.close();
  };

}

