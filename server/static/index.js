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
const responseSound = new Audio('/static/facebookchatone.mp3');

// Function to show values when scrollin
document.addEventListener("DOMContentLoaded", function() {
  // Initialize range value display
  const sliders = document.querySelectorAll('.slider');
  sliders.forEach((slider) => {
    const valueElement = document.getElementById(`value-${slider.id}`);
    valueElement.innerText = slider.value;
    slider.addEventListener('input', function() {
      valueElement.innerText = this.value;
    });
  });
});


document.addEventListener('DOMContentLoaded', () => {
  const wrapper = document.querySelector('.wrapper');
  const handleRight = document.getElementById('resizeHandleRight');
  const handleBottom = document.getElementById('resizeHandleBottom');
  const handleCorner = document.getElementById('resizeHandleCorner');
  
  let isResizingRight = false;
  let isResizingBottom = false;
  let isResizingCorner = false;

  // Handle right resizer
  handleRight.addEventListener('mousedown', (e) => {
    isResizingRight = true;
    document.addEventListener('mousemove', handleMouseMoveRight);
    document.addEventListener('mouseup', () => {
      isResizingRight = false;
      document.removeEventListener('mousemove', handleMouseMoveRight);
    });
  });

  // Handle bottom resizer
  handleBottom.addEventListener('mousedown', (e) => {
    isResizingBottom = true;
    document.addEventListener('mousemove', handleMouseMoveBottom);
    document.addEventListener('mouseup', () => {
      isResizingBottom = false;
      document.removeEventListener('mousemove', handleMouseMoveBottom);
    });
  });

  // Handle corner resizer
  handleCorner.addEventListener('mousedown', (e) => {
    isResizingCorner = true;
    document.addEventListener('mousemove', handleMouseMoveCorner);
    document.addEventListener('mouseup', () => {
      isResizingCorner = false;
      document.removeEventListener('mousemove', handleMouseMoveCorner);
    });
  });

  function handleMouseMoveRight(e) {
    if (isResizingRight) {
      const newWidth = e.clientX - wrapper.getBoundingClientRect().left;
      wrapper.style.width = `${newWidth}px`;
    }
  }

  function handleMouseMoveBottom(e) {
    if (isResizingBottom) {
      const newHeight = e.clientY - wrapper.getBoundingClientRect().top;
      wrapper.style.height = `${newHeight}px`;
    }
  }

  function handleMouseMoveCorner(e) {
    if (isResizingCorner) {
      const newWidth = e.clientX - wrapper.getBoundingClientRect().left;
      const newHeight = e.clientY - wrapper.getBoundingClientRect().top;
      wrapper.style.width = `${newWidth}px`;
      wrapper.style.height = `${newHeight}px`;
    }
  }
});



document.getElementById('param-form').addEventListener('submit', function(event) {
  event.preventDefault();

  const formData = new FormData(event.target);

  const formObject = {};
  formData.forEach((value, key) => {
    const [group, subkey] = key.split('[').map(str => str.replace(']', ''));

    if (!(group in formObject)) {
      formObject[group] = {};
    }

    const parsedValue = isNaN(parseFloat(value)) ? value : parseFloat(value);

    formObject[group][subkey] = parsedValue;
  });

  // Explicitly set the value for brand_valid['activate']
  const toggleSwitch = document.getElementById('brand-valid-activate');
  if (toggleSwitch) {
    if ('brand_valid' in formObject) {
      formObject['brand_valid']['activate'] = toggleSwitch.checked;
    }
  }

  console.log(formObject);
  // Now, send `formObject` to your server to update the parameters
  fetch('/update_params', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(formObject)
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      // Update was successful
      alert('Parameters updated successfully');
    } else {
      // Update failed
      alert('Failed to update parameters');
    }
  })
  .catch(error => {
    console.error('Error:', error);
  });
});





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
     responseSound.play();
     await showMessage(true, event.data);
  }

  const handleResponseEvent = async (event) => {
    responseSound.play();
    await showMessage(false, event.data);
  }

  const handleSuccessEvent = async (event) => {
    responseSound.play();
    await showMessage(false, event.data);
    inferenceSuccess.style.display = "block";
    inferenceLoading.style.display = "none";
    scrollIntoView(inferenceSuccess);
    eventSource.close();
  }

  const handleFailEvent = async (event) => {
    responseSound.play();
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
          var running = setTimeout(animate, 10);
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

