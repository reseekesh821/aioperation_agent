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

function appendFindingBlock(container, label, value) {
    if (!value) return;
    const block = document.createElement("div");
    block.className = "support-finding-block";

    const title = document.createElement("p");
    title.className = "support-finding-label";
    title.textContent = label;

    const content = document.createElement("pre");
    content.className = "support-finding-text";
    content.textContent = value;

    block.append(title, content);
    container.appendChild(block);
}

function renderDraftPanel(container, messageId) {
    const panel = document.createElement("div");
    panel.className = "support-draft-panel";

    const draftButton = document.createElement("button");
    draftButton.type = "button";
    draftButton.className = "generate-draft-button";
    draftButton.textContent = "Generate Draft";

    const draftStatus = document.createElement("p");
    draftStatus.className = "support-draft-status";
    draftStatus.hidden = true;

    const draftEditor = document.createElement("textarea");
    draftEditor.className = "support-draft-editor";
    draftEditor.rows = 6;
    draftEditor.hidden = true;
    draftEditor.setAttribute("aria-label", "Editable customer response draft");

    const draftActions = document.createElement("div");
    draftActions.className = "support-draft-actions";
    draftActions.hidden = true;

    const notSent = document.createElement("span");
    notSent.className = "support-draft-not-sent";
    notSent.textContent = "Not sent";

    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.className = "copy-draft-button";
    copyButton.textContent = "Copy";

    const saveDraftButton = document.createElement("button");
    saveDraftButton.type = "button";
    saveDraftButton.className = "save-gmail-draft-button";
    saveDraftButton.textContent = "Save to Gmail Drafts";

    let savedDraftFingerprint = null;

    function currentDraftFingerprint() {
        return draftEditor.value.trim();
    }

    function syncSaveButtonState() {
        const text = currentDraftFingerprint();
        saveDraftButton.disabled = !text || text === savedDraftFingerprint;
    }

    draftEditor.addEventListener("input", () => {
        if (savedDraftFingerprint !== null && currentDraftFingerprint() !== savedDraftFingerprint) {
            notSent.textContent = "Not sent";
            draftStatus.classList.remove("is-error");
            if (!draftStatus.textContent.includes("Generating")) {
                draftStatus.textContent = "Draft edited. Ready to save again.";
            }
        }
        syncSaveButtonState();
    });

    copyButton.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(draftEditor.value);
            copyButton.textContent = "Copied";
            setTimeout(() => {
                copyButton.textContent = "Copy";
            }, 1200);
        } catch (_error) {
            draftStatus.hidden = false;
            draftStatus.classList.add("is-error");
            draftStatus.textContent = "Unable to copy draft to clipboard.";
        }
    });

    saveDraftButton.addEventListener("click", async () => {
        const editedText = draftEditor.value.trim();
        if (!editedText) {
            draftStatus.hidden = false;
            draftStatus.classList.add("is-error");
            draftStatus.textContent = "Draft text cannot be empty.";
            return;
        }
        if (editedText === savedDraftFingerprint) {
            return;
        }

        saveDraftButton.disabled = true;
        draftButton.disabled = true;
        draftStatus.hidden = false;
        draftStatus.classList.remove("is-error");
        draftStatus.textContent = "Saving to Gmail Drafts...";

        try {
            const response = await fetch(
                `http://127.0.0.1:8000/support/emails/${encodeURIComponent(messageId)}/save-draft`,
                {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({draft_text: editedText})
                }
            );
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.detail || `HTTP ${response.status}`);
            }

            savedDraftFingerprint = editedText;
            notSent.textContent = "Saved to Gmail Drafts — Not sent";
            draftStatus.classList.remove("is-error");
            draftStatus.textContent = data.draft_id
                ? `Saved to Gmail Drafts — Not sent (draft ${data.draft_id}).`
                : "Saved to Gmail Drafts — Not sent.";
        } catch (error) {
            draftStatus.classList.add("is-error");
            draftStatus.textContent = error instanceof Error
                ? error.message
                : "Unable to save Gmail draft.";
        } finally {
            draftButton.disabled = false;
            syncSaveButtonState();
        }
    });

    draftActions.append(notSent, copyButton, saveDraftButton);

    draftButton.addEventListener("click", async () => {
        draftButton.disabled = true;
        saveDraftButton.disabled = true;
        draftStatus.hidden = false;
        draftStatus.classList.remove("is-error");
        draftStatus.textContent = "Generating draft...";
        draftEditor.hidden = true;
        draftActions.hidden = true;
        savedDraftFingerprint = null;

        try {
            const response = await fetch(
                `http://127.0.0.1:8000/support/emails/${encodeURIComponent(messageId)}/draft`,
                {method: "POST"}
            );
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.detail || `HTTP ${response.status}`);
            }

            draftEditor.value = data.draft || "";
            draftEditor.hidden = false;
            draftActions.hidden = false;
            notSent.textContent = "Not sent";
            draftStatus.classList.remove("is-error");
            draftStatus.textContent = data.used_fallback
                ? "Draft ready (evidence-backed fallback). Not sent."
                : "Draft ready for review. Not sent.";
            syncSaveButtonState();
        } catch (error) {
            draftEditor.hidden = true;
            draftActions.hidden = true;
            draftStatus.classList.add("is-error");
            draftStatus.textContent = error instanceof Error
                ? error.message
                : "Draft generation failed.";
        } finally {
            draftButton.disabled = false;
        }
    });

    panel.append(draftButton, draftStatus, draftEditor, draftActions);
    container.appendChild(panel);
}

function renderInvestigationFindings(container, findings, messageId) {
    container.replaceChildren();
    container.classList.remove("is-error");

    const heading = document.createElement("p");
    heading.className = "support-finding-heading";
    heading.textContent = "Investigation findings";
    container.appendChild(heading);

    const presentation = findings.presentation || {};
    const intent = findings.intent || presentation.intent || "unknown";
    const orderId = findings.order_id != null
        ? String(findings.order_id)
        : (findings.olist_order_id || "none");
    const dataScope = presentation.data_scope === "historical"
        ? "Historical records"
        : "Live operations";

    appendFindingBlock(container, "Intent", intent);
    appendFindingBlock(container, "Order ID", orderId);
    appendFindingBlock(container, "Data scope", dataScope);
    appendFindingBlock(container, "Status", presentation.status || "");
    appendFindingBlock(container, "Order details", presentation.order_details || "");
    appendFindingBlock(
        container,
        "Payment details",
        presentation.payment_details || ""
    );
    appendFindingBlock(
        container,
        "Policy information",
        presentation.policy_information || findings.policy_information || ""
    );

    const uncertainties = Array.isArray(presentation.uncertainties)
        ? presentation.uncertainties.filter(Boolean)
        : (Array.isArray(findings.uncertainties)
            ? findings.uncertainties.filter(Boolean)
            : []);
    if (uncertainties.length) {
        appendFindingBlock(
            container,
            "Uncertainties",
            uncertainties.map((item) => `- ${item}`).join("\n")
        );
    }

    appendFindingBlock(
        container,
        "Summary",
        presentation.summary || findings.summary || ""
    );
    renderDraftPanel(container, messageId);
}

async function investigateSupportEmail(messageId, button, findingsEl) {
    button.disabled = true;
    findingsEl.classList.remove("is-error");
    findingsEl.replaceChildren();
    findingsEl.hidden = false;

    const loading = document.createElement("p");
    loading.className = "support-finding-loading";
    loading.textContent = "Investigating...";
    findingsEl.appendChild(loading);

    try {
        const response = await fetch(
            `http://127.0.0.1:8000/support/emails/${encodeURIComponent(messageId)}/investigate`,
            {method: "POST"}
        );
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }
        renderInvestigationFindings(findingsEl, data, messageId);
    } catch (error) {
        findingsEl.replaceChildren();
        findingsEl.classList.add("is-error");
        const errorText = document.createElement("p");
        errorText.className = "support-finding-error";
        errorText.textContent = error instanceof Error
            ? error.message
            : "Investigation failed.";
        findingsEl.appendChild(errorText);
    } finally {
        button.disabled = false;
    }
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

        const actions = document.createElement("div");
        actions.className = "support-email-actions";

        const investigateButton = document.createElement("button");
        investigateButton.type = "button";
        investigateButton.className = "investigate-button";
        investigateButton.textContent = "Investigate";

        const findings = document.createElement("div");
        findings.className = "support-investigation";
        findings.hidden = true;

        investigateButton.addEventListener("click", () => {
            if (!email.id) {
                findings.hidden = false;
                findings.classList.add("is-error");
                findings.textContent = "Missing Gmail message ID.";
                return;
            }
            investigateSupportEmail(email.id, investigateButton, findings);
        });

        actions.appendChild(investigateButton);
        article.append(meta, subject, body, actions, findings);
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