const API_BASE = "http://127.0.0.1:8000";

const loadEmailsButton = document.getElementById("load-emails-button");
const supportStatus = document.getElementById("support-inbox-status");
const supportEmailList = document.getElementById("support-email-list");
const syncCasesButton = document.getElementById("sync-cases-button");
const refreshCasesButton = document.getElementById("refresh-cases-button");
const supportCasesStatus = document.getElementById("support-cases-status");
const supportCasesSummary = document.getElementById("support-cases-summary");
const supportCasesList = document.getElementById("support-cases-list");
const supportCaseDetail = document.getElementById("support-case-detail");
const inboxPanel = document.getElementById("support-inbox");
const casesPanel = document.getElementById("support-cases");
const navInboxButton = document.getElementById("nav-inbox");
const navCasesButton = document.getElementById("nav-cases");

function showOwnerView(view) {
    const selected = view === "cases" ? "cases" : "inbox";
    inboxPanel.hidden = selected !== "inbox";
    casesPanel.hidden = selected !== "cases";

    navInboxButton.classList.toggle("is-active", selected === "inbox");
    navCasesButton.classList.toggle("is-active", selected === "cases");
    navInboxButton.setAttribute("aria-pressed", selected === "inbox" ? "true" : "false");
    navCasesButton.setAttribute("aria-pressed", selected === "cases" ? "true" : "false");
}

navInboxButton.addEventListener("click", () => {
    showOwnerView("inbox");
});

navCasesButton.addEventListener("click", () => {
    showOwnerView("cases");
});

showOwnerView("inbox");

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
                `${API_BASE}/support/emails/${encodeURIComponent(messageId)}/save-draft`,
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
                `${API_BASE}/support/emails/${encodeURIComponent(messageId)}/draft`,
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
            `${API_BASE}/support/emails/${encodeURIComponent(messageId)}/investigate`,
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

loadEmailsButton.addEventListener("click", async () => {
    loadEmailsButton.disabled = true;
    supportEmailList.replaceChildren();
    setSupportStatus("Loading support emails...");

    try {
        const response = await fetch(`${API_BASE}/support/emails`);
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

function setCasesStatus(message, isError = false) {
    supportCasesStatus.textContent = message;
    supportCasesStatus.classList.toggle("is-error", Boolean(isError));
}

function orderLabelForCase(caseItem) {
    if (caseItem.order_id != null) {
        return String(caseItem.order_id);
    }
    if (caseItem.olist_order_id) {
        return caseItem.olist_order_id;
    }
    return "—";
}

function renderCasesSummary(counts, total) {
    supportCasesSummary.replaceChildren();
    const summary = document.createElement("p");
    summary.className = "support-cases-counts";
    const parts = [`Total: ${total}`];
    const order = [
        "new",
        "investigated",
        "draft_ready",
        "draft_saved",
        "needs_attention",
        "closed",
    ];
    for (const status of order) {
        parts.push(`${status}: ${(counts && counts[status]) || 0}`);
    }
    summary.textContent = parts.join(" · ");
    supportCasesSummary.appendChild(summary);
}

function renderCasesList(cases) {
    supportCasesList.replaceChildren();
    if (!cases.length) {
        const empty = document.createElement("p");
        empty.className = "support-cases-empty";
        empty.textContent = "No support cases yet. Sync from Gmail to create cases.";
        supportCasesList.appendChild(empty);
        return;
    }

    for (const caseItem of cases) {
        const article = document.createElement("article");
        article.className = "support-case";

        const subject = document.createElement("p");
        subject.className = "support-case-subject";
        subject.textContent = caseItem.subject || "(No subject)";

        const meta = document.createElement("p");
        meta.className = "support-case-meta";
        meta.textContent = [
            `Status: ${caseItem.status || "unknown"}`,
            `Order: ${orderLabelForCase(caseItem)}`,
            `Updated: ${caseItem.updated_at || "—"}`,
        ].join(" · ");

        const actions = document.createElement("div");
        actions.className = "support-case-actions";

        const openButton = document.createElement("button");
        openButton.type = "button";
        openButton.className = "open-case-button";
        openButton.textContent = "Open";
        openButton.addEventListener("click", () => {
            openSupportCase(caseItem.case_id, openButton);
        });

        actions.appendChild(openButton);
        article.append(subject, meta, actions);
        supportCasesList.appendChild(article);
    }
}

function renderCaseDetail(detail) {
    supportCaseDetail.hidden = false;
    supportCaseDetail.replaceChildren();

    const caseItem = detail.case || {};
    const heading = document.createElement("p");
    heading.className = "support-finding-heading";
    heading.textContent = `Case #${caseItem.case_id}`;

    const subject = document.createElement("p");
    subject.className = "support-case-subject";
    subject.textContent = caseItem.subject || "(No subject)";

    const meta = document.createElement("pre");
    meta.className = "support-finding-text";
    meta.textContent = [
        `Status: ${caseItem.status || "unknown"}`,
        `Intent: ${caseItem.intent || "—"}`,
        `Order: ${orderLabelForCase(caseItem)}`,
        `Gmail message: ${caseItem.gmail_message_id || "—"}`,
        `Gmail draft: ${caseItem.gmail_draft_id || "—"}`,
        `Updated: ${caseItem.updated_at || "—"}`,
    ].join("\n");

    const actions = document.createElement("div");
    actions.className = "support-case-actions";

    const needsAttentionButton = document.createElement("button");
    needsAttentionButton.type = "button";
    needsAttentionButton.textContent = "Needs Attention";
    needsAttentionButton.addEventListener("click", () => {
        updateCaseStatus(caseItem.case_id, "needs_attention", needsAttentionButton);
    });

    const closeButton = document.createElement("button");
    closeButton.type = "button";
    closeButton.textContent = "Close Case";
    closeButton.addEventListener("click", () => {
        updateCaseStatus(caseItem.case_id, "closed", closeButton);
    });

    actions.append(needsAttentionButton, closeButton);

    const historyHeading = document.createElement("p");
    historyHeading.className = "support-finding-label";
    historyHeading.textContent = "Activity history";

    const history = document.createElement("pre");
    history.className = "support-finding-text";
    const events = Array.isArray(detail.events) ? detail.events : [];
    history.textContent = events.length
        ? events.map((event) => {
            const when = event.created_at || "";
            const type = event.event_type || "";
            const detailText = event.detail || "";
            return `- [${when}] ${type}: ${detailText}`;
        }).join("\n")
        : "No activity recorded yet.";

    supportCaseDetail.append(
        heading,
        subject,
        meta,
        actions,
        historyHeading,
        history
    );
}

async function loadSupportCases({quiet = false} = {}) {
    if (!quiet) {
        setCasesStatus("Loading cases...");
    }
    const response = await fetch(`${API_BASE}/support/cases`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}`);
    }
    const cases = Array.isArray(data.cases) ? data.cases : [];
    const counts = data.counts || {};
    const total = data.total != null ? data.total : cases.length;
    renderCasesSummary(counts, total);
    renderCasesList(cases);
    setCasesStatus(`Loaded ${total} case(s).`);
}

async function openSupportCase(caseId, button) {
    button.disabled = true;
    setCasesStatus(`Opening case #${caseId}...`);
    try {
        const response = await fetch(
            `${API_BASE}/support/cases/${encodeURIComponent(caseId)}`
        );
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }
        renderCaseDetail(data);
        setCasesStatus(`Viewing case #${caseId}.`);
    } catch (error) {
        setCasesStatus(
            error instanceof Error ? error.message : "Unable to open case.",
            true
        );
    } finally {
        button.disabled = false;
    }
}

async function updateCaseStatus(caseId, status, button) {
    button.disabled = true;
    try {
        const response = await fetch(
            `${API_BASE}/support/cases/${encodeURIComponent(caseId)}/status`,
            {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({status}),
            }
        );
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }
        await loadSupportCases({quiet: true});
        const detailResponse = await fetch(
            `${API_BASE}/support/cases/${encodeURIComponent(caseId)}`
        );
        const detail = await detailResponse.json().catch(() => ({}));
        if (detailResponse.ok) {
            renderCaseDetail(detail);
        }
        setCasesStatus(`Case #${caseId} marked ${status}.`);
    } catch (error) {
        setCasesStatus(
            error instanceof Error ? error.message : "Unable to update case.",
            true
        );
    } finally {
        button.disabled = false;
    }
}

syncCasesButton.addEventListener("click", async () => {
    syncCasesButton.disabled = true;
    refreshCasesButton.disabled = true;
    setCasesStatus("Syncing labeled Gmail messages into cases...");
    try {
        const response = await fetch(`${API_BASE}/support/cases/sync`, {
            method: "POST",
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || `HTTP ${response.status}`);
        }
        renderCasesSummary(data.counts || {}, data.total || 0);
        renderCasesList(Array.isArray(data.cases) ? data.cases : []);
        setCasesStatus(
            `Synced ${data.synced_count || 0} labeled message(s) into cases.`
        );
    } catch (error) {
        setCasesStatus(
            error instanceof Error ? error.message : "Case sync failed.",
            true
        );
    } finally {
        syncCasesButton.disabled = false;
        refreshCasesButton.disabled = false;
    }
});

refreshCasesButton.addEventListener("click", async () => {
    refreshCasesButton.disabled = true;
    syncCasesButton.disabled = true;
    try {
        await loadSupportCases();
    } catch (error) {
        supportCasesList.replaceChildren();
        supportCasesSummary.replaceChildren();
        setCasesStatus(
            error instanceof Error ? error.message : "Unable to load cases.",
            true
        );
    } finally {
        refreshCasesButton.disabled = false;
        syncCasesButton.disabled = false;
    }
});
