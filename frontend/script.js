const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const chatBox = document.getElementById("chat-box");
const button = form.querySelector("button");
const loadEmailsButton = document.getElementById("load-emails-button");
const supportStatus = document.getElementById("support-inbox-status");
const supportEmailList = document.getElementById("support-email-list");

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

function setSupportStatus(message, isError = false) {
    supportStatus.textContent = message;
    supportStatus.classList.toggle("is-error", Boolean(isError));
}

function renderSupportEmails(emails) {
    supportEmailList.replaceChildren();

    if (!emails.length) {
        setSupportStatus("No labeled support emails found.");
        return;
    }

    setSupportStatus(`Loaded ${emails.length} support email(s).`);

    for (const email of emails) {
        const article = document.createElement("article");
        article.className = "support-email";

        const meta = document.createElement("p");
        meta.className = "support-email-meta";
        meta.textContent = `From: ${email.sender || "(Unknown sender)"}`;

        const subject = document.createElement("p");
        subject.className = "support-email-subject";
        subject.textContent = email.subject || "(No subject)";

        const body = document.createElement("pre");
        body.className = "support-email-body";
        body.textContent = email.body || "(No message body)";

        article.append(meta, subject, body);
        supportEmailList.appendChild(article);
    }
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
        const response = await fetch("http://127.0.0.1:8000/chat", {
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

loadEmailsButton.addEventListener("click", async () => {
    loadEmailsButton.disabled = true;
    supportEmailList.replaceChildren();
    setSupportStatus("Loading support emails...");

    try {
        const response = await fetch("http://127.0.0.1:8000/support/emails");
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            const detail = data.detail || `HTTP ${response.status}`;
            throw new Error(detail);
        }

        renderSupportEmails(Array.isArray(data.emails) ? data.emails : []);
    } catch (error) {
        supportEmailList.replaceChildren();
        setSupportStatus(
            error instanceof Error
                ? error.message
                : "Unable to load support emails.",
            true
        );
    } finally {
        loadEmailsButton.disabled = false;
    }
});