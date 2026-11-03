"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("experiment-form");
  const transcript = $("transcript");
  const initialTranscript = transcript.cloneNode(true);
  const numericFields = ["max_turns", "message_words", "temperature", "top_p", "seed", "history_window", "num_ctx", "num_predict"];
  const activeStatuses = new Set(["running", "paused", "stopping"]);
  const defaultTopic = "Develop a reliable, concise way to cooperate using whatever text-based method is most efficient.";
  const dirtyFields = new Set();
  let topicPresets = [{ id: "cooperation", label: "Efficient cooperation", topic: defaultTopic }];
  let csrfToken = "";
  let currentSession = null;
  let archivedSession = null;
  let archivedId = "";
  let renderedSignature = "";
  let activitySignature = "";
  let statePending = false;
  let statusPending = false;
  let actionPending = false;
  let didApplyDefaults = false;
  let lastRunIdentity = "";
  let toastTimer;
  let pollFailures = 0;
  let historyRequest = 0;
  let actionError = "";

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function errorMessage(error) {
    return error instanceof Error ? error.message : String(error || "An unexpected error occurred.");
  }

  function showNotice(message) {
    $("notice").textContent = message || "";
    $("notice").hidden = !message;
  }

  function toast(message) {
    clearTimeout(toastTimer);
    $("toast").textContent = message;
    $("toast").hidden = false;
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
  }

  async function api(path, body, timeout = 15000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const options = { signal: controller.signal, cache: "no-store", credentials: "same-origin" };
      if (body !== undefined) {
        options.method = "POST";
        options.headers = { "Content-Type": "application/json", "X-Lab-Token": csrfToken };
        options.body = JSON.stringify(body);
      }
      const response = await fetch(path, options);
      let data;
      try { data = await response.json(); } catch { throw new Error(`The application returned an unreadable response (${response.status}).`); }
      if (!response.ok) {
        const detail = typeof data.error === "object" ? data.error?.message : data.error;
        throw new Error(detail || data.message || `Request failed (${response.status}).`);
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The request timed out. Check the application terminal and connection.");
      if (error instanceof TypeError) throw new Error("Cannot reach the local application. Check that app.py is still running.");
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  function selectedMode() {
    return form.querySelector('input[name="mode"]:checked').value;
  }

  function updateMode() {
    const network = selectedMode() === "network";
    $("network-fields").hidden = !network;
    $("peer_url").required = network;
    $("peer_token").required = network;
    $("mode-description").textContent = network
      ? "Agent A runs here. Agent B runs on the connected computer."
      : "Two independent perspectives, one local Ollama model.";
  }

  function updateExperiment() {
    const coordination = $("experiment").value === "coordination";
    $("topic-preset-field").hidden = coordination;
    $("max_turns").step = coordination ? "2" : "1";
    $("max_turns").title = coordination ? "Use an even number of turns so each game round has a sender and receiver." : "Maximum conversation turns";
    const explainer = $("experiment-explainer");
    explainer.querySelector(".mini-label").textContent = coordination ? "A REASON TO COMMUNICATE" : "ROOM TO EXPLORE";
    explainer.querySelector("p").textContent = coordination
      ? "One agent sees a private ordered pair. The other sees candidate arrangements. They exchange English and receive each round’s result."
      : "Agents take turns exploring a shared objective, negotiating useful conventions as the conversation develops.";
    $("topic-label").textContent = coordination ? "Shared objective" : "Conversation topic";
    $("topic").placeholder = coordination
      ? "Develop clear, economical English conventions for solving the task together."
      : defaultTopic;
  }

  function syncTopicPreset() {
    const topic = $("topic").value.trim();
    const preset = topicPresets.find((entry) => entry.topic.trim() === topic);
    $("topic-preset").value = preset ? preset.id : "custom";
  }

  function renderTopicPresets(presets) {
    if (Array.isArray(presets)) {
      const seen = new Set();
      const valid = presets.filter((entry) => {
        if (!entry || typeof entry.id !== "string" || entry.id === "custom" || seen.has(entry.id)
          || typeof entry.label !== "string" || typeof entry.topic !== "string") return false;
        seen.add(entry.id);
        return true;
      });
      if (valid.length) topicPresets = valid;
    }
    const options = topicPresets.map((preset) => {
      const option = element("option", "", preset.label);
      option.value = preset.id;
      return option;
    });
    const custom = element("option", "", "Custom");
    custom.value = "custom";
    $("topic-preset").replaceChildren(...options, custom);
    syncTopicPreset();
  }

  function updateTokenBounds() {
    $("num_predict").max = String(Math.min(8192, Math.floor(Number($("num_ctx").value || 8192) / 2)));
  }

  function collectConfig() {
    const data = new FormData(form);
    const config = {};
    for (const key of ["mode", "model", "experiment", "peer_url", "peer_token", "topic", "think", "initial_message"]) {
      config[key] = String(data.get(key) || "").trim();
    }
    for (const key of numericFields) config[key] = Number(data.get(key));
    config.compress = $("compress").checked;
    return config;
  }

  function applyDefaults(defaults) {
    if (didApplyDefaults || !defaults) return;
    didApplyDefaults = true;
    const config = defaults.config || defaults;
    for (const [key, value] of Object.entries(config)) {
      const input = $(key);
      if (!input || key === "peer_token" || dirtyFields.has(key)) continue;
      if (input.type === "checkbox") input.checked = Boolean(value);
      else if (value !== null && value !== undefined) input.value = String(value);
    }
    if (config.mode && !dirtyFields.has("mode")) {
      for (const radio of form.querySelectorAll('input[name="mode"]')) radio.checked = radio.value === config.mode;
    }
    updateMode();
    updateExperiment();
    updateTokenBounds();
    syncTopicPreset();
    renderMetrics(null);
  }

  function modelNames(models) {
    if (!Array.isArray(models)) return [];
    return models.map((model) => typeof model === "string" ? model : model.name || model.model).filter(Boolean);
  }

  async function loadStatus(manual = false) {
    if (statusPending) return;
    statusPending = true;
    $("check-ollama").disabled = true;
    if (manual) $("ollama-status").replaceChildren(element("span", "status-dot pending"), document.createTextNode("Checking local Ollama…"));
    try {
      const status = await api("/api/status");
      const ollama = status.ollama || {};
      const names = modelNames(ollama.models);
      const selected = $("model").value || "gpt-oss:20b";
      const defaultModel = (status.defaults?.config || status.defaults)?.model || "gpt-oss:20b";
      const choices = [...new Set(["gpt-oss:20b", ...names, selected, defaultModel])];
      $("model").replaceChildren(...choices.map((name) => {
        const option = element("option", "", name);
        option.value = name;
        return option;
      }));
      $("model").value = selected;
      applyDefaults(status.defaults);
      renderTopicPresets(status.topic_presets);
      let message;
      let dotClass;
      if (!ollama.ok) {
        message = ollama.error || "Ollama is unavailable. Start Ollama on this computer.";
        dotClass = "status-dot error";
      } else if (!names.includes($("model").value)) {
        message = "Ollama connected · selected model is not downloaded";
        dotClass = "status-dot pending";
      } else {
        message = "Ollama connected · model available locally";
        dotClass = "status-dot";
      }
      $("ollama-status").replaceChildren(element("span", dotClass), document.createTextNode(message));
      renderPeerSetup(status.peer || {});
    } catch (error) {
      $("ollama-status").replaceChildren(element("span", "status-dot error"), document.createTextNode(errorMessage(error)));
      if (manual) toast("Ollama check could not complete.");
    } finally {
      statusPending = false;
      $("check-ollama").disabled = false;
    }
  }

  function renderPeerSetup(peer) {
    $("peer-listener-status").textContent = peer.enabled
      ? `Peer listener active · port ${peer.port || 8766}`
      : "Peer listener is disabled.";
    $("peer-disabled-help").hidden = Boolean(peer.enabled);
    $("peer-enabled-info").hidden = !peer.enabled;
    $("local-peer-token").value = peer.token || "";
    if (!peer.enabled) return;
    const addresses = Array.isArray(peer.addresses) ? peer.addresses : [];
    $("peer-addresses").replaceChildren(...addresses.map((address) => element("li", "", typeof address === "string" ? address : address.url || address.address || "")));
    if (!addresses.length) $("peer-addresses").append(element("li", "", `http://<this computer’s LAN IP>:${peer.port || 8766}`));
  }

  async function checkPeer() {
    const address = $("peer_url").value.trim();
    const token = $("peer_token").value.trim();
    const result = $("peer-check-result");
    result.classList.remove("error");
    if (!address || !token) {
      result.textContent = "Enter the other computer’s peer address and token first.";
      result.classList.add("error");
      return;
    }
    $("check-peer").disabled = true;
    result.textContent = "Checking peer and its local model…";
    try {
      const data = await api("/api/check-peer", { peer_url: address, peer_token: token }, 30000);
      if (data.ok === false) throw new Error(data.error || "The peer is not ready.");
      result.textContent = `Peer connected${data.model ? ` · ${data.model}` : ""}. Ready for conversation.`;
    } catch (error) {
      result.classList.add("error");
      result.textContent = errorMessage(error);
    } finally {
      $("check-peer").disabled = false;
    }
  }

  function normalizedSession(data) {
    if (data && Object.prototype.hasOwnProperty.call(data, "session")) return data.session;
    return data && data.id ? data : null;
  }

  function sessionSignature(session) {
    if (!session) return "empty";
    return JSON.stringify([session.id, session.status, session.updated_at, session.turns?.length, session.feedback?.length,
      session.active?.content, session.active?.phase, session.active?.speaker, session.error, session.metrics]);
  }

  function shortDate(value) {
    if (!value) return "";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  }

  function timeLabel(value) {
    if (!value) return "";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? "" : date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }

  function displayNumber(value, digits = 1) {
    return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
  }

  function renderMetrics(session) {
    const metrics = session?.metrics || {};
    const count = metrics.turn_count ?? session?.turns?.length ?? 0;
    const limit = session?.config?.max_turns ?? Number($("max_turns").value || 40);
    $("metric-turns").replaceChildren(document.createTextNode(String(count)), element("span", "metric-unit", ` / ${limit}`));
    $("metric-accuracy").textContent = typeof metrics.accuracy === "number" ? `${Math.round(metrics.accuracy * 100)}%` : "—";
    $("metric-accuracy").title = metrics.rounds_completed ? `${metrics.correct_rounds ?? 0} correct of ${metrics.rounds_completed} completed rounds` : "Scored after each completed reference-game round";
    $("metric-words").textContent = count ? displayNumber(metrics.mean_words) : "—";
    $("metric-diversity").textContent = count ? displayNumber(metrics.lexical_diversity, 2) : "—";
    $("metric-diversity").title = "Unique words divided by total words. Descriptive; not an emergence score.";
  }

  function messageNode(turn, active = false) {
    const speaker = turn.speaker === "B" ? "B" : "A";
    const node = element("article", `message agent-${speaker.toLowerCase()}${active ? " active" : ""}`);
    node.setAttribute("aria-label", `Agent ${speaker}${active ? ", generating" : `, turn ${turn.number || ""}`}`);
    const heading = element("div", "message-heading");
    heading.append(element("span", "speaker-badge", speaker), element("span", "speaker-name", `Agent ${speaker}`));
    const role = turn.role ? turn.role.replaceAll("_", " ") : "";
    if (role && role !== "dialogue") heading.append(element("span", "", `· ${role}`));
    if (turn.number) heading.append(element("span", "message-time", `TURN ${String(turn.number).padStart(2, "0")}${turn.round ? ` / ROUND ${turn.round}` : ""}`));
    const body = element("div", "message-body", turn.content || "");
    node.append(heading, body);
    const foot = element("div", "message-foot");
    if (active) {
      const dots = element("span", "typing-dots");
      dots.setAttribute("aria-hidden", "true");
      dots.append(element("i"), element("i"), element("i"));
      foot.append(document.createTextNode(turn.content ? "RECEIVING FINAL ANSWER" : "MODEL WORKING · REASONING PRIVATE"), dots);
    } else {
      if (turn.words !== undefined) foot.append(element("span", "", `${turn.words} words${turn.budget_words ? ` / ${turn.budget_words} budget` : ""}`));
      if (turn.generation?.model) foot.append(element("span", "", turn.generation.model));
      if (turn.created_at || turn.at) foot.append(element("span", "", timeLabel(turn.created_at || turn.at)));
      if (turn.clipped) foot.append(element("span", "clipped-note", "CLIPPED TO WORD BUDGET"));
      if (turn.generation?.done_reason === "length") foot.append(element("span", "clipped-note", "TOKEN LIMIT REACHED"));
    }
    node.append(foot);
    if (turn.prompts) {
      const details = element("details", "context-details");
      details.append(element("summary", "", "Inspect delivered context"), element("pre", "", JSON.stringify(turn.prompts, null, 2)));
      node.append(details);
    }
    return node;
  }

  function feedbackNode(feedback) {
    const resultClass = feedback.correct === true ? "correct" : feedback.correct === false ? "incorrect" : "";
    return element("div", `feedback ${resultClass}`, feedback.content || `Round ${feedback.round} · ${feedback.correct ? "correct" : "incorrect"}`);
  }

  function renderTranscript(session) {
    const signature = sessionSignature(session);
    if (signature === renderedSignature) return;
    const priorId = transcript.dataset.sessionId || "";
    const newId = session?.id || "";
    const follow = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 100 || priorId !== newId;
    const oldScroll = transcript.scrollTop;
    const oldLength = Number(transcript.dataset.turnCount || 0);
    const fragment = document.createDocumentFragment();
    const turns = session?.turns || [];
    if (!turns.length && !session?.active) {
      if (session) {
        const pending = element("div", "empty-state");
        pending.append(element("p", "eyebrow", "THE CONDITIONS ARE SET"), element("h3", "", session.status === "error" ? "The conversation needs attention." : "Waiting for the first exchange."));
        pending.append(element("p", "", session.status === "running" ? "Loading the model and preparing an independent context…" : "Messages will appear here when the experiment begins."));
        fragment.append(pending);
      } else {
        for (const child of initialTranscript.childNodes) fragment.append(child.cloneNode(true));
      }
    } else {
      const byTurn = new Map();
      for (const feedback of session.feedback || []) {
        const list = byTurn.get(feedback.after_turn) || [];
        list.push(feedback);
        byTurn.set(feedback.after_turn, list);
      }
      for (const turn of turns) {
        fragment.append(messageNode(turn));
        for (const feedback of byTurn.get(turn.number) || []) fragment.append(feedbackNode(feedback));
      }
      if (session.active) fragment.append(messageNode(session.active, true));
    }
    transcript.replaceChildren(fragment);
    transcript.dataset.sessionId = newId;
    transcript.dataset.turnCount = String(turns.length);
    if (follow) {
      transcript.scrollTop = transcript.scrollHeight;
      $("new-activity").hidden = true;
    } else {
      transcript.scrollTop = oldScroll;
      if (turns.length > oldLength || session?.active) $("new-activity").hidden = false;
    }
    renderedSignature = signature;
  }

  function updateControls() {
    const status = currentSession?.status;
    const active = activeStatuses.has(status);
    $("start-run").disabled = active || actionPending;
    $("pause-run").disabled = !["running", "paused"].includes(status) || actionPending;
    $("pause-run").textContent = status === "paused" ? "Resume" : "Pause";
    $("stop-run").disabled = !["running", "paused"].includes(status) || actionPending;
    $("start-run").replaceChildren(document.createTextNode(active ? "Conversation in progress" : "Start conversation"), element("span", "", active ? "·" : "↗"));
    $("return-live").hidden = !archivedId;
  }

  function renderSession() {
    const session = archivedId ? archivedSession : currentSession;
    const status = session?.status || "ready";
    $("session-status").textContent = archivedId ? `Archive · ${status}` : status;
    $("session-status").className = `status-pill ${status}`;
    if (session) {
      const experiment = session.config?.experiment === "coordination" ? "Cooperative reference game" : "Open-ended conversation";
      const connection = session.config?.mode === "network" ? "two computers" : "self-conversation";
      $("run-meta").textContent = `${experiment} · ${connection} · ${shortDate(session.created_at)}`;
      $("run-meta").title = session.id;
    } else {
      $("run-meta").textContent = "A place to watch conventions take shape.";
      $("run-meta").removeAttribute("title");
    }
    renderMetrics(session);
    renderTranscript(session);
    if (session?.error) showNotice(typeof session.error === "string" ? session.error : session.error.message || JSON.stringify(session.error));
    else if (pollFailures === 0) showNotice(actionError);
    for (const format of ["json", "markdown"]) {
      const link = $(`export-${format}`);
      link.setAttribute("aria-disabled", session ? "false" : "true");
      if (session) link.href = `/api/runs/${encodeURIComponent(session.id)}/export?format=${format}`;
      else link.removeAttribute("href");
    }
    const text = {
      ready: "Waiting for an experiment", running: session?.active ? `Agent ${session.active.speaker || "A"} is generating` : "Preparing the next turn",
      paused: "Paused · context preserved", stopping: "Stopping the active generation…", stopped: "Conversation stopped",
      completed: "Conversation complete", error: "Experiment interrupted by an error", interrupted: "Interrupted when the application exited"
    };
    $("stream-status").textContent = archivedId ? "Saved conversation" : text[status] || status;
    updateControls();
  }

  function renderPeerActivity(activity, active) {
    const entries = Array.isArray(activity) ? activity : [];
    const signature = JSON.stringify([entries, active]);
    if (signature === activitySignature) return;
    activitySignature = signature;
    $("peer-activity-section").hidden = entries.length === 0 && !active;
    const nodes = entries.slice(-12).map((entry) => {
      const turn = { ...entry, generation: { model: entry.model }, created_at: entry.at };
      const node = messageNode(turn);
      if (entry.parent_session) node.querySelector(".message-foot").append(element("span", "", `RUN ${String(entry.parent_session).slice(0, 12)}`));
      return node;
    });
    if (active) nodes.push(messageNode({ ...active, number: active.turn }, true));
    $("peer-activity-list").replaceChildren(...nodes);
  }

  async function loadState() {
    if (statePending) return;
    statePending = true;
    try {
      const state = await api("/api/state", undefined, 8000);
      currentSession = normalizedSession(state);
      pollFailures = 0;
      renderSession();
      renderPeerActivity(state.peer_activity, state.peer_active);
      const identity = `${currentSession?.id || ""}:${currentSession?.status || ""}`;
      if (identity !== lastRunIdentity) {
        lastRunIdentity = identity;
        void loadHistory();
      }
    } catch (error) {
      pollFailures += 1;
      if (pollFailures >= 2) {
        showNotice(errorMessage(error));
        $("stream-status").textContent = "Connection lost · retrying locally";
      }
    } finally {
      statePending = false;
    }
  }

  async function loadHistory() {
    try {
      const data = await api("/api/runs");
      const runs = Array.isArray(data) ? data : data.runs || [];
      const select = $("run-history");
      const options = [element("option", "", "Current conversation")];
      options[0].value = "";
      for (const run of runs) {
        const experiment = (run.experiment || run.config?.experiment) === "coordination" ? "Reference game" : "Conversation";
        const turns = typeof run.turns === "number" ? run.turns : Array.isArray(run.turns) ? run.turns.length : run.turn_count ?? run.metrics?.turn_count;
        const label = `${shortDate(run.created_at)} · ${experiment}${turns !== undefined ? ` · ${turns} turns` : ""} · ${run.status || "saved"}`;
        const option = element("option", "", label);
        option.value = run.id;
        options.push(option);
      }
      select.replaceChildren(...options);
      select.value = archivedId;
    } catch {
      // A history refresh does not interrupt an active conversation.
    }
  }

  async function selectArchive(id) {
    const request = ++historyRequest;
    actionError = "";
    if (!id) {
      archivedId = "";
      archivedSession = null;
      $("run-history").value = "";
      renderedSignature = "";
      renderSession();
      return;
    }
    try {
      const data = await api(`/api/runs/${encodeURIComponent(id)}`);
      if (request !== historyRequest) return;
      archivedSession = normalizedSession(data);
      if (!archivedSession) throw new Error("This saved run has no readable session record.");
      archivedId = id;
      $("run-history").value = id;
      renderedSignature = "";
      renderSession();
    } catch (error) {
      if (request === historyRequest) {
        actionError = errorMessage(error);
        showNotice(actionError);
        $("run-history").value = archivedId;
      }
    }
  }

  async function runAction(action, body = {}) {
    if (actionPending) return;
    actionPending = true;
    actionError = "";
    updateControls();
    showNotice("");
    try {
      const state = await api(`/api/${action}`, body);
      currentSession = normalizedSession(state);
      if (action === "start") {
        archivedId = "";
        archivedSession = null;
        $("run-history").value = "";
        renderedSignature = "";
      }
      renderSession();
      void loadHistory();
      if (action === "pause") toast("Pause requested. The current turn will finish first.");
      if (action === "resume") toast("Conversation resumed.");
      if (action === "stop") toast("Stopping the conversation.");
    } catch (error) {
      actionError = errorMessage(error);
      showNotice(actionError);
    } finally {
      actionPending = false;
      updateControls();
    }
  }

  async function copyPeerToken() {
    const input = $("local-peer-token");
    if (!input.value) return;
    try {
      await navigator.clipboard.writeText(input.value);
      toast("Peer token copied.");
    } catch {
      const oldType = input.type;
      input.type = "text";
      input.select();
      let copied = false;
      try { copied = document.execCommand("copy"); } catch { /* Manual copy remains available. */ }
      input.type = oldType;
      toast(copied ? "Peer token copied." : "Select and copy the peer token manually.");
    }
  }

  const markEdited = (event) => {
    if (event.target.name) dirtyFields.add(event.target.name);
  };
  form.addEventListener("input", markEdited);
  form.addEventListener("change", markEdited);
  $("topic").addEventListener("input", syncTopicPreset);
  $("topic-preset").addEventListener("change", () => {
    const preset = topicPresets.find((entry) => entry.id === $("topic-preset").value);
    if (!preset) return;
    $("topic").value = preset.topic;
    dirtyFields.add("topic");
    syncTopicPreset();
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (form.reportValidity()) void runAction("start", collectConfig());
  });
  form.querySelectorAll('input[name="mode"]').forEach((radio) => radio.addEventListener("change", updateMode));
  $("experiment").addEventListener("change", updateExperiment);
  $("num_ctx").addEventListener("input", updateTokenBounds);
  $("check-ollama").addEventListener("click", () => { void loadStatus(true); });
  $("check-peer").addEventListener("click", () => { void checkPeer(); });
  $("model").addEventListener("change", () => { void loadStatus(); });
  $("max_turns").addEventListener("input", () => { if (!currentSession && !archivedId) renderMetrics(null); });
  $("pause-run").addEventListener("click", () => { void runAction(currentSession?.status === "paused" ? "resume" : "pause"); });
  $("stop-run").addEventListener("click", () => { void runAction("stop"); });
  $("run-history").addEventListener("change", (event) => { void selectArchive(event.target.value); });
  $("return-live").addEventListener("click", () => { void selectArchive(""); });
  $("reveal-token").addEventListener("click", () => {
    const show = $("local-peer-token").type === "password";
    $("local-peer-token").type = show ? "text" : "password";
    $("reveal-token").textContent = show ? "Hide" : "Show";
    $("reveal-token").setAttribute("aria-pressed", String(show));
  });
  $("copy-token").addEventListener("click", () => { void copyPeerToken(); });
  $("new-activity").addEventListener("click", () => {
    transcript.scrollTo({ top: transcript.scrollHeight, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
    $("new-activity").hidden = true;
  });
  transcript.addEventListener("scroll", () => {
    if (transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 80) $("new-activity").hidden = true;
  }, { passive: true });
  document.addEventListener("click", (event) => {
    for (const menu of document.querySelectorAll(".export-menu[open]")) if (!menu.contains(event.target)) menu.open = false;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") for (const menu of document.querySelectorAll(".export-menu[open]")) menu.open = false;
  });

  async function initialize() {
    updateMode();
    updateExperiment();
    try {
      const bootstrap = await api("/api/bootstrap");
      csrfToken = bootstrap.csrf_token || "";
      if (!csrfToken) throw new Error("The local session could not initialize. Reload this page.");
      await Promise.allSettled([loadStatus(), loadState(), loadHistory()]);
      setInterval(() => { void loadState(); }, 1000);
      setInterval(() => { void loadStatus(); }, 30000);
      setInterval(() => { void loadHistory(); }, 15000);
    } catch (error) {
      showNotice(errorMessage(error));
      $("start-run").disabled = true;
      $("stream-status").textContent = "Reload after starting the local application";
    }
  }

  void initialize();
})();
