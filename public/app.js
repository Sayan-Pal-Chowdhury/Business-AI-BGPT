const chatForm = document.getElementById("chatForm");
const promptInput = document.getElementById("promptInput");
const chatLog = document.getElementById("chatLog");
const statusText = document.getElementById("statusText");
const newChat = document.getElementById("newChat");
const chatPanel = document.getElementById("chatPanel");

function addMessage(role, text) {
  document.getElementById("welcomePanel")?.remove();
  chatPanel?.classList.remove("is-empty");
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = role === "user" ? "You" : "BGPT";
  article.innerHTML = `<span>${label}</span><p></p>`;
  article.querySelector("p").textContent = text;
  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
  return article;
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    statusText.textContent = data.checkpoint ? "BGPT online" : "BGPT warming up";
  } catch {
    statusText.textContent = "Offline";
  }
}

chatForm.addEventListener("submit", async event => {
  event.preventDefault();
  const message = promptInput.value.trim();
  if (!message) return;

  addMessage("user", message);
  promptInput.value = "";
  const thinking = addMessage("bot", "Generating...");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        max_new_tokens: 140,
        temperature: 0.72,
        top_k: 18,
        mode: "hybrid",
        include_core_sample: false,
      }),
    });
    const data = await res.json();
    thinking.querySelector("p").textContent = data.reply || data.detail || "No output.";
  } catch {
    thinking.querySelector("p").textContent = "The BGPT API is not reachable.";
  }
});

promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
});

promptInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

newChat?.addEventListener("click", () => {
  chatPanel?.classList.add("is-empty");
  chatLog.innerHTML = `
    <div id="welcomePanel" class="welcome-panel">
      <h2>What business problem are you working on?</h2>
      <p>Ask about sales, inventory, credit, profit, customers, orders, billing, delivery, or operations.</p>
    </div>
  `;
});

checkHealth();
