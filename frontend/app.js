const $ = (selector) => document.querySelector(selector);
let analyzeAudioBlob = null;
let journalAudioBlob = null;
let recorder = null;
let chunks = [];

function esc(value) {
  const element = document.createElement("div");
  element.textContent = value ?? "";
  return element.innerHTML;
}

function go(page) {
  document.querySelectorAll(".page").forEach((item) => item.classList.toggle("active-page", item.id === page));
  document.querySelectorAll("[data-nav]").forEach((item) => item.classList.toggle("active", item.dataset.nav === page));
  history.replaceState(null, "", `#${page}`);
  if (page === "journal") loadEntries();
}

function setCaptureTab(prefix, tab) {
  document.querySelectorAll(`[data-${prefix}-tab], [data-capture]`).forEach((item) => {
    if (item.dataset[`${prefix}Tab`] || item.dataset.capture) {
      item.classList.toggle("active", (item.dataset[`${prefix}Tab`] || item.dataset.capture) === tab);
    }
  });
  const recordPanel = $(`#${prefix === "journal" ? "journal-" : ""}record-panel`);
  const uploadPanel = $(`#${prefix === "journal" ? "journal-" : ""}upload-panel`);
  recordPanel.classList.toggle("hidden", tab !== "record");
  uploadPanel.classList.toggle("hidden", tab !== "upload");
}

function setPreview(prefix, blob) {
  const isJournal = prefix === "journal";
  const preview = $(`#${isJournal ? "journal-" : ""}audio-preview`);
  const button = $(`#${isJournal ? "save-journal-button" : "analyze-button"}`);
  preview.classList.remove("hidden");
  preview.querySelector("audio").src = URL.createObjectURL(blob);
  button.disabled = false;
  if (isJournal) journalAudioBlob = blob;
  else analyzeAudioBlob = blob;
}

function clearPreview(prefix) {
  const isJournal = prefix === "journal";
  const preview = $(`#${isJournal ? "journal-" : ""}audio-preview`);
  preview.classList.add("hidden");
  $(`#${isJournal ? "save-journal-button" : "analyze-button"}`).disabled = true;
  if (isJournal) journalAudioBlob = null;
  else analyzeAudioBlob = null;
}

function showStatus(id, text, error = false) {
  const status = $(`#${id}`);
  status.textContent = text;
  status.className = `status${error ? " error" : ""}`;
  status.classList.remove("hidden");
}

async function recordInto(prefix) {
  const isJournal = prefix === "journal";
  const button = $(`#${isJournal ? "journal-" : ""}record-button`);
  const label = $(`#${isJournal ? "journal-" : ""}record-label`);
  if (recorder?.state === "recording") {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (event) => chunks.push(event.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      setPreview(prefix, new Blob(chunks, { type: "audio/webm" }));
      button.classList.remove("recording");
      label.textContent = "Record again";
    };
    recorder.start();
    button.classList.add("recording");
    label.textContent = "Stop recording";
  } catch {
    showStatus(isJournal ? "journal-status" : "analysis-status", "Microphone permission is unavailable. You can upload audio instead.", true);
  }
}

async function submitAudio(prefix, endpoint, source, successMessage) {
  const isJournal = prefix === "journal";
  const blob = isJournal ? journalAudioBlob : analyzeAudioBlob;
  if (!blob) return;
  const button = $(`#${isJournal ? "save-journal-button" : "analyze-button"}`);
  const form = new FormData();
  form.append("audio", blob, "recording.webm");
  showStatus(isJournal ? "journal-status" : "analysis-status", "Transcribing locally with Whisper…");
  button.disabled = true;
  try {
    const response = await fetch(`${endpoint}?source=${source}`, { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Analysis failed");
    if (isJournal) {
      $("#journal-modal").classList.add("hidden");
      clearPreview("journal");
      showStatus("analysis-status", "Thought saved to your journal.");
      go("journal");
    } else {
      renderResult(data);
      showStatus("analysis-status", successMessage);
    }
  } catch (error) {
    showStatus(isJournal ? "journal-status" : "analysis-status", error.message, true);
  } finally {
    button.disabled = false;
  }
}

function renderResult(entry) {
  const result = $("#result");
  result.classList.remove("hidden");
  result.innerHTML = `<span class="eyebrow">YOUR SUMMARY</span><h3>${esc(entry.title || "Audio analysis")}</h3><div class="summary">${esc(entry.summary)}</div><p>${(entry.topics || []).map((topic) => `<span class="tag">${esc(topic)}</span>`).join("")} <span class="tag mood">${esc(entry.mood)}</span></p><button class="reveal" id="reveal">Reveal full transcript ↓</button><p id="transcript" class="hidden">${esc(entry.transcript)}</p>`;
  $("#reveal").onclick = () => {
    $("#transcript").classList.remove("hidden");
    $("#reveal").remove();
  };
}

async function loadEntries() {
  const params = new URLSearchParams({ search: $("#search").value, semantic: $("#semantic-search").value, topic: $("#topic-filter").value, mood: $("#mood-filter").value, has_actions: $("#actions-filter").checked });
  const response = await fetch(`/api/entries?${params}`);
  const data = await response.json();
  $("#topic-filter").innerHTML = `<option value="">All topics</option>${(data.facets.topics || []).map((item) => `<option>${esc(item)}</option>`).join("")}`;
  $("#mood-filter").innerHTML = `<option value="">All moods</option>${(data.facets.moods || []).map((item) => `<option>${esc(item)}</option>`).join("")}`;
  renderEntries(data.entries);
}

async function loadVisuals() {
  const [calendarResponse, graphResponse] = await Promise.all([fetch("/api/calendar"), fetch("/api/graph")]);
  renderCalendar((await calendarResponse.json()).entries);
  renderGraph(await graphResponse.json());
}

function renderCalendar(entries) {
  const grouped = entries.reduce((result, entry) => {
    (result[entry.date] ||= []).push(entry);
    return result;
  }, {});
  $("#calendar-view").innerHTML = Object.keys(grouped).sort().reverse().map((date) =>
    `<div class="calendar-day"><strong>${new Date(`${date}T00:00:00`).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}</strong>${grouped[date].map((entry) => `<span>${esc(entry.title)} <em>${esc(entry.mood)}</em></span>`).join("")}</div>`
  ).join("") || "<div class=\"empty\">Your calendar will fill up as you save thoughts.</div>";
}

function renderGraph(data) {
  const svg = $("#topic-graph");
  if (!data.nodes.length) {
    svg.innerHTML = "";
    return;
  }
  const center = { x: 380, y: 210 };
  const radius = Math.min(150, 45 + data.nodes.length * 12);
  const positions = Object.fromEntries(data.nodes.map((node, index) => {
    const angle = (index / data.nodes.length) * Math.PI * 2;
    return [node.id, { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius }];
  }));
  svg.innerHTML = data.links.map((link) => `<line x1="${positions[link.source].x}" y1="${positions[link.source].y}" x2="${positions[link.target].x}" y2="${positions[link.target].y}" stroke="#d8def0" stroke-width="${1 + link.value}"/>`).join("") +
    data.nodes.map((node) => `<g><circle cx="${positions[node.id].x}" cy="${positions[node.id].y}" r="${8 + node.count * 2}" fill="#4169f5"/><text x="${positions[node.id].x}" y="${positions[node.id].y + 4}" text-anchor="middle" fill="#17202e">${esc(node.id)}</text></g>`).join("");
}

function setJournalView(view) {
  document.querySelectorAll(".view-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === view));
  $("#timeline").classList.toggle("hidden", view !== "timeline");
  $("#calendar-view").classList.toggle("hidden", view !== "calendar");
  $("#graph-view").classList.toggle("hidden", view !== "graph");
  if (view !== "timeline") loadVisuals();
}

function renderEntries(entries) {
  const timeline = $("#timeline");
  if (!entries.length) {
    timeline.innerHTML = "<div class=\"empty\"><strong>No thoughts here yet.</strong><br>Save a thought above and it will appear in this timeline.</div>";
    return;
  }
  timeline.innerHTML = entries.map((entry) => `<article class="entry"><div class="entry-head"><span>${new Date(entry.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })} · ${esc(entry.source)}</span><span>${esc(entry.mood)}</span></div><h3>${esc(entry.title)}</h3><p>${esc(entry.summary)}</p><div>${(entry.topics || []).map((topic) => `<span class="tag">${esc(topic)}</span>`).join("")}</div>${entry.action_items?.length ? `<div class="actions">☐ ${entry.action_items.map(esc).join("<br>☐ ")}</div>` : ""}<details><summary>Read transcript</summary><p>${esc(entry.transcript)}</p></details></article>`).join("");
}

function openJournalModal(tab = "record") {
  $("#journal-modal").classList.remove("hidden");
  setCaptureTab("journal", tab);
}

document.querySelectorAll("[data-nav]").forEach((item) => item.onclick = (event) => { event.preventDefault(); go(item.dataset.nav); });
document.querySelectorAll("[data-capture]").forEach((item) => item.onclick = () => setCaptureTab("capture", item.dataset.capture));
document.querySelectorAll("[data-journal-tab]").forEach((item) => item.onclick = () => setCaptureTab("journal", item.dataset.journalTab));
document.querySelectorAll("[data-journal-capture]").forEach((item) => item.onclick = () => openJournalModal(item.dataset.journalCapture));
document.querySelectorAll("[data-go]").forEach((item) => item.onclick = () => go(item.dataset.go));
$("#record-button").onclick = () => recordInto("capture");
$("#journal-record-button").onclick = () => recordInto("journal");
$("#audio-file").onchange = (event) => { if (event.target.files[0]) setPreview("capture", event.target.files[0]); };
$("#journal-audio-file").onchange = (event) => { if (event.target.files[0]) setPreview("journal", event.target.files[0]); };
$("#clear-audio").onclick = () => clearPreview("capture");
$("#clear-journal-audio").onclick = () => clearPreview("journal");
$("#analyze-button").onclick = () => submitAudio("capture", "/api/analyze", $("#upload-panel").classList.contains("hidden") ? "microphone" : "upload", "Analysis complete. This result was not saved to your journal.");
$("#save-journal-button").onclick = () => submitAudio("journal", "/api/journal/entries", $("#journal-upload-panel").classList.contains("hidden") ? "microphone" : "upload", "Thought saved to your journal.");
$("#close-journal-modal").onclick = () => $("#journal-modal").classList.add("hidden");
["search", "semantic-search", "topic-filter", "mood-filter", "actions-filter"].forEach((id) => $(`#${id}`).addEventListener("input", loadEntries));
document.querySelectorAll(".view-tab").forEach((tab) => tab.onclick = () => setJournalView(tab.dataset.view));
$("#ask-button").onclick = async () => {
  const question = $("#question").value.trim();
  if (!question) return;
  $("#answer").textContent = "Searching your notes…";
  $("#answer").classList.remove("hidden");
  const response = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
  const data = await response.json();
  $("#answer").textContent = data.answer || data.detail;
};
window.addEventListener("hashchange", () => go(location.hash.slice(1) || "analyze"));
fetch("/api/health").then((response) => response.json()).then((data) => $("#health").textContent = data.status === "ok" ? "Ready · models load when used" : "Offline").catch(() => $("#health").textContent = "API unavailable");
go(location.hash.slice(1) || "analyze");
