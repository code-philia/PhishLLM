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
const inferenceLoading2 = document.getElementById("inference-loading2");
const inferenceResult = document.getElementById("inference-result");
const inferenceLogs = document.getElementById("inference-logs");
const inferenceSuccess = document.getElementById("inference-success");
const inferenceFail = document.getElementById("inference-fail");
const resetButton = document.getElementById("reset-button");
let eventNumber = 0;
let eventQueue = [];
const responseSound = new Audio('/static/facebookchatone.mp3');

let url = "";
let back_url = ""
let screenshot_path = ""
let html_path = ""

// resize the hyperparameter panel
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

// Function to show values when scrolling the hyperparameters slider
document.addEventListener("DOMContentLoaded", function () {
  // Initialize range value display
  const sliders = document.querySelectorAll('.slider');
  sliders.forEach((slider) => {
    const valueElement = document.getElementById(`value-${slider.id}`);
    valueElement.innerText = slider.value;
    slider.addEventListener('input', function () {
      valueElement.innerText = this.value;
    });
  });
});

// Function to disable or enable all buttons and input[type="submit"] or input[type="button"]
function toggleButtons(disabled) {
  const buttons = document.querySelectorAll("button, input[type='submit'], input[type='button']");
  buttons.forEach(button => button.disabled = disabled);
}


// Add click event listener to reset button
resetButton.addEventListener("click", async function (e) {
  e.preventDefault();

  // Disable all buttons and show loading
  toggleButtons(true);

  // Reset sliders and checkboxes to their default values
  document.getElementById("common-temperature").value = 0;
  document.getElementById("brand-valid-activate").checked = false;
  document.getElementById("brand-valid-k").value = 5;
  document.getElementById("siamese-thre").value = 0.7;
  document.getElementById("rank-depth-limit").value = 1;

  // Update label values to match the default values
  document.getElementById("value-common-temperature").innerText = "0";
  document.getElementById("value-brand-valid-k").innerText = "5";
  document.getElementById("value-siamese-thre").innerText = "0.7";
  document.getElementById("value-rank-depth-limit").innerText = "1";

  toggleButtons(false);

});

// Crawl the screenshot for a given URL
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
  urlScreenshot.replaceChildren();
  // Disable all buttons and show loading
  toggleButtons(true);

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
      back_url = responseBody.url
      screenshot_path = responseBody.screenshot_path
      html_path = responseBody.html_path
      urlSuccess.style.display = "block";
      scrollIntoView(urlSuccess);
    } else {
      urlFail.style.display = "block";
      scrollIntoView(urlFail);
    }
  } catch (error) {
    console.log(error);
    urlFail.style.display = "block";
    scrollIntoView(urlFail);
  }
  finally {
    toggleButtons(false);
    urlLoading.style.display = "none";
  }
}

function generateUniqueId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
}

const getInference = () => {
  // Disable all buttons and show loading
  toggleButtons(true);
  const id = generateUniqueId(); // 生成唯一ID的函数

  const defaultParams = {
    brand_recog: { temperature: document.getElementById("common-temperature").value },
    brand_valid: {
      activate: document.getElementById("brand-valid-activate").checked,
      k: document.getElementById("brand-valid-k").value,
      siamese_thre: document.getElementById("siamese-thre").value
    },
    crp_pred: { temperature: document.getElementById("common-temperature").value },
    rank: { depth_limit: document.getElementById("rank-depth-limit").value }
  };

  const eventSource = new EventSource(`/listen?id=${id}&params=${JSON.stringify(defaultParams)}&url=${encodeURIComponent(back_url)}&screenshot_path=${encodeURIComponent(screenshot_path)}&html_path=${encodeURIComponent(html_path)}`);

  inferenceResult.style.display = "block";
  inferenceSuccess.style.display = "none";
  inferenceFail.style.display = "none";
  inferenceLoading.style.display = "block";
  inferenceLoading2.style.display = "block";
  inferenceLogs.replaceChildren();
  scrollIntoView(inferenceResult);

  const queueEvent = async (event) => {
    eventQueue.push(event);
    if (eventQueue.length > 1) return;
    while (eventQueue.length) {
      try {
        await runEvent();
      } catch (err) {
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
    toggleButtons(true);
    responseSound.play();
    await showMessage(true, event.data);
  }

  const handleResponseEvent = async (event) => {
    toggleButtons(true);
    responseSound.play();
    await showMessage(false, event.data);
  }

  const handleSuccessEvent = async (event) => {
    toggleButtons(true);
    responseSound.play();
    await showMessage(false, event.data);
    inferenceSuccess.style.display = "block";
    inferenceLoading.style.display = "none";
    inferenceLoading2.style.display = "none";
    scrollIntoView(inferenceSuccess);
    eventSource.close();
    // Re-enable buttons after message is shown
    toggleButtons(false);
  }

  const handleFailEvent = async (event) => {
    toggleButtons(true);
    responseSound.play();
    inferenceFail.style.display = "block";
    inferenceLoading.style.display = "none";
    inferenceLoading2.style.display = "none";
    scrollIntoView(inferenceFail);
    eventSource.close();
    // Re-enable buttons after message is shown
    toggleButtons(false);
  }

  const showMessage = async (isPrompt, msg) => {
    inferenceLoading.style.display = "none"; // hide the processing event

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

    // Re-enable buttons after message is shown
    toggleButtons(false);
    await new Promise(r => setTimeout(r, 1000));
  }

  const typeWriter = async (element, str) => {
    return new Promise((resolve) => {
      var chars = str.split("");
      var prevHeight = document.body.scrollHeight;

      (function animate() {
        if (chars.length > 0) {
          element.innerHTML += chars.shift();
          var running = setTimeout(animate, 1);
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
    inferenceLoading2.style.display = "none";
    scrollIntoView(inferenceFail);
    eventSource.close();
    toggleButtons(false); // Re-enable buttons when an error occurs
  };

}

$(document).ready(function() {
    // Function to fetch sampled URLs from the server
    function fetchSampledUrls() {
        $.post('/sample_urls', function(data) {
            const urlSuggestions = $('#url-suggestions');
            urlSuggestions.empty();

            // Populate the datalist with the sampled URLs
            data['sampled_urls'].forEach(function(url) {
                urlSuggestions.append('<option value="' + url + '">');
            });
        });
    }

    // Call the fetchSampledUrls function to populate the datalist initially
    fetchSampledUrls();

    // Autocomplete functionality for the input field
    $('#url-search-bar').on('input', function() {
        const inputText = $(this).val();
        const urlSuggestions = $('#url-suggestions');

        // Filter and display suggestions based on user input
        urlSuggestions.empty();
        fetchSampledUrls(); // Refresh the list of suggestions

        data['sampled_urls'].forEach(function(url) {
            if (url.startsWith(inputText)) {
                urlSuggestions.append('<option value="' + url + '">');
            }
        });
    });

    // Event handler for input box click
    $('#url-search-bar').on('click', function() {
        // Refresh the list of suggestions when the input box is clicked
        fetchSampledUrls();
    });
});

