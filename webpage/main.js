const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const messageInput = document.getElementById("userInput");
const imageInput = document.getElementById("imageInput");
const chatContainer = document.getElementById("chat");
const historyLimitInput = document.getElementById("recordLimitInput");

let abortController = null;
let chatHistory = [];
const defaultHistoryLimit = 10;

// 控制最多显示的条数
function enforceHistoryLimit() {
  let limit = parseInt(historyLimitInput.value);
  if (isNaN(limit) || limit < 1 || limit > 20) {
    alert("保留条数必须在 1 到 20 之间，已自动调整为合法值");
    limit = Math.min(20, Math.max(1, limit));
    historyLimitInput.value = limit;
  }

  while (chatHistory.length > limit) {
    chatContainer.removeChild(chatHistory[0]);
    chatHistory.shift();
  }
}

// 添加消息到聊天容器
function appendMessage(role, content, isImage = false) {
  const messageElem = document.createElement("div");
  messageElem.className = `message ${role}`;

  if (isImage) {
    const img = document.createElement("img");
    img.src = content;
    img.style.maxWidth = "200px";
    img.style.maxHeight = "200px";
    img.onload = () => enforceHistoryLimit();
    messageElem.appendChild(img);
  } else {
    messageElem.textContent = content;
  }

  chatContainer.appendChild(messageElem);
  chatHistory.push(messageElem);
  enforceHistoryLimit();
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 将图片文件转换为Base64字符串
function readImageAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1]; // 去除前缀
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// 与后端通信
async function sendToServer(text, imageBase64) {
  const serverUrl = 'http://192.168.137.100:8080/rkllm_chat';

  const messages = [];

  if (text) {
    messages.push({
      role: "user",
      content: text
    });
  }

  if (imageBase64) {
    messages.push({
      role: "user",
      content: {
        type: "imagedata",
        imagedata: `data:image/jpeg;base64,${imageBase64}`
      }
    });
  }

  const payload = {
    model: "your_model_deploy_with_RKLLM_Server",
    messages: messages,
    stream: false
  };

  abortController = new AbortController();

  const response = await fetch(serverUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload),
    signal: abortController.signal
  });

  if (!response.ok) {
    throw new Error(`服务器错误: ${response.status}`);
  }

  const data = await response.json();
  return data;
}

// 显示逐字消息
function displayReplyGradually(text) {
  const messageElem = document.createElement("div");
  messageElem.className = "message bot";
  chatContainer.appendChild(messageElem);
  chatHistory.push(messageElem);
  enforceHistoryLimit();

  let index = 0;
  function step() {
    if (index < text.length) {
      messageElem.textContent += text[index++];
      chatContainer.scrollTop = chatContainer.scrollHeight;
      setTimeout(step, 30);
    }
  }
  step();
}

// 点击发送按钮
sendBtn.addEventListener("click", async () => {
  const text = messageInput.value.trim();
  const imageFile = imageInput.files[0];

  if (!text && !imageFile) return;

  sendBtn.disabled = true;
  messageInput.disabled = true;
  imageInput.disabled = true;
  stopBtn.disabled = false;

  if (text) appendMessage("user", text);
  let imageBase64 = null;
  if (imageFile) {
      imageBase64 = await readImageAsBase64(imageFile);
      const imageDataUrl = `data:${imageFile.type};base64,${imageBase64}`;
      appendMessage("user", imageDataUrl, true);
  }

  try {
    const response = await sendToServer(text, imageBase64);

    let reply = "";
    if (response && response.choices && response.choices.length > 0) {
      reply = response.choices[0].message.content || "[无回复内容]";
    } else {
      reply = "[格式错误或无响应]";
    }

    displayReplyGradually(reply);
  } catch (err) {
    console.error("请求失败:", err);
    appendMessage("bot", `[请求失败] ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    messageInput.disabled = false;
    imageInput.disabled = false;
    stopBtn.disabled = true;
    abortController = null;
    messageInput.value = "";
    imageInput.value = "";
  }
});

// 停止按钮中止请求
stopBtn.addEventListener("click", () => {
  if (abortController) {
    abortController.abort();
    appendMessage("bot", "[请求已手动中止]");
    sendBtn.disabled = false;
    messageInput.disabled = false;
    imageInput.disabled = false;
    stopBtn.disabled = true;
  }
});
