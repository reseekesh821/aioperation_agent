const API_BASE = window.AIOPS_API_BASE || "http://127.0.0.1:8000";

const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const chatBox = document.getElementById("chat-box");
const button = form.querySelector("button");

function addMessage(sender, message) {
    const p = document.createElement("p");
    if (sender === "You") {
        p.classList.add("user-message");
    }
    p.textContent = `${sender}: ${message}`;
    chatBox.appendChild(p);
    chatBox.scrollTop = chatBox.scrollHeight;
    return p;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = input.value.trim();
    if (!message) return;

    addMessage("You", message);
    input.value = "";
    button.disabled = true;
    const pendingMessage = addMessage("Assistant", "Thinking...");

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message})
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        pendingMessage.textContent = `Assistant: ${data.answer}`;
    } catch (error) {
        pendingMessage.textContent = "System: Unable to reach the AI agent.";
    } finally {
        button.disabled = false;
    }
});
