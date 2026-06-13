/**
 * DeepShield KYC  ·  app.js  (v2 — Command Center UI)
 * Handles: step flow, webcam, WebSocket, illumination,
 *          animated layer updates, result rendering.
 */
"use strict";

// ── Config ───────────────────────────────────────────────
const WS_URL   = `ws://${location.host}/ws/kyc`;
const API_BASE = `http://${location.host}`;
const FRAME_MS = 200;
const AUDIO_MS = 3000;

// ── State ────────────────────────────────────────────────
let sessionId = null, sessionToken = null;
let mediaStream = null, ws = null;
let frameTimer = null, mediaRecorder = null, audioChunks = [];
let illumColors = [], livenessChallenge = "";
let currentStep = 1;

// ── DOM ──────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Helpers ──────────────────────────────────────────────
const sleep = ms => new Promise(r => setTimeout(r, ms));

function now() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
}

function addLog(msg, type = "") {
  const body = $("logBody");
  const row  = document.createElement("div");
  row.className = `log-row ${type}`;
  row.innerHTML = `<span class="log-time font-mono">${now()}</span><span class="log-msg">${msg}</span>`;
  body.appendChild(row);
  body.scrollTop = body.scrollHeight;
  // Keep last 40 entries
  while (body.children.length > 40) body.removeChild(body.firstChild);
}

function toast(msg, type = "") {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast${type ? ' '+type : ''}`;
  el.classList.remove("hidden");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.add("hidden"), 3500);
}

// ── Step navigation ──────────────────────────────────────
function goTo(n) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  $(`step${n}`).classList.add("active");

  for (let i = 1; i <= 4; i++) {
    const nav = $(`nav${i}`);
    nav.classList.remove("active", "done");
    if (i < n)  nav.classList.add("done"), nav.querySelector(".snav-dot").innerHTML = "✓";
    if (i === n) nav.classList.add("active");
  }
  currentStep = n;
}

// ── Step 1 ───────────────────────────────────────────────
$("step1Next").addEventListener("click", async () => {
  const name = $("applicantName").value.trim();
  if (!name) return toast("Please enter your full name.", "error");
  if (!$("termsCheck").checked) return toast("Please agree to the terms.", "error");

  addLog("Creating session...", "info");
  try {
    const res  = await fetch(`${API_BASE}/api/kyc/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ applicant_name: name, id_type: $("idType").value }),
    });
    const data = await res.json();
    sessionId         = data.session_id;
    sessionToken      = data.session_token;
    illumColors       = data.illum_challenge_colors;
    livenessChallenge = data.liveness_challenge;

    addLog(`Session created — ID: ${sessionId.slice(0,8)}...`, "pass");
    goTo(2);
  } catch (e) {
    addLog("Server error starting session", "fail");
    toast("Server error. Is the backend running?", "error");
  }
});

// ── Step 2 ───────────────────────────────────────────────
const zone   = $("uploadZone");
const fileIn = $("docFile");

zone.addEventListener("click", () => fileIn.click());
zone.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag-over"); });
zone.addEventListener("dragleave", ()  => zone.classList.remove("drag-over"));
zone.addEventListener("drop",      e  => { e.preventDefault(); zone.classList.remove("drag-over"); handleDoc(e.dataTransfer.files[0]); });
fileIn.addEventListener("change",  ()  => handleDoc(fileIn.files[0]));

async function handleDoc(file) {
  if (!file) return;
  addLog(`Uploading document: ${file.name}`, "info");
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", sessionId);
  try {
    const res  = await fetch(`${API_BASE}/api/kyc/upload-doc`, { method: "POST", body: form });
    const data = await res.json();
    if (data.ok) {
      $("docFileName").textContent = file.name;
      $("docPreview").classList.remove("hidden");
      $("step2Next").disabled = false;
      addLog("Document uploaded & OCR queued", "pass");
      toast("Document uploaded successfully.", "success");
    } else {
      addLog("Document upload failed", "fail");
      toast(data.detail || "Upload failed.", "error");
    }
  } catch {
    addLog("Upload error", "fail");
    toast("Upload error.", "error");
  }
}

$("step2Next").addEventListener("click", () => {
  goTo(3);
  startVideo();
});

// ── Step 3: Video KYC ────────────────────────────────────
async function startVideo() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: false },
    });
    $("videoFeed").srcObject = mediaStream;

    $("recTag").textContent  = "RECORDING";
    $("recTag").classList.add("live");
    $("liveTag").style.display = "flex";

    addLog("Camera & microphone active", "pass");
    openWS();
    streamFrames();
    streamAudio();

    if (livenessChallenge) {
      $("challengeBox").style.display = "flex";
      $("challengeText").textContent  = livenessChallenge;
      addLog(`Liveness challenge: ${livenessChallenge}`, "info");
    }

    // Enable submit after 8s
    setTimeout(() => {
      $("submitKYC").disabled = false;
      addLog("Ready to submit — click 'Submit for AI Analysis'", "info");
    }, 8000);

  } catch (e) {
    addLog("Camera/microphone denied", "fail");
    toast("Camera access denied.", "error");
  }
}

// WebSocket
function openWS() {
  ws = new WebSocket(`${WS_URL}/${sessionId}?token=${sessionToken}`);
  ws.onopen    = () => addLog("WebSocket connected", "info");
  ws.onmessage = e  => handleWS(JSON.parse(e.data));
  ws.onerror   = ()  => addLog("WebSocket error", "fail");
  ws.onclose   = ()  => addLog("WebSocket closed", "");
}

function handleWS(msg) {
  switch (msg.type) {
    case "LAYER_UPDATE":
      animateLayer(msg.payload);
      break;
    case "CHALLENGE_READY":
      runIllum(msg.payload.colors);
      break;
    case "ANALYSIS_COMPLETE":
      stop();
      showResult(msg.payload);
      break;
    case "ERROR":
      toast(msg.payload.detail, "error");
      addLog(msg.payload.detail, "fail");
      break;
  }
}

// Frame streaming
function streamFrames() {
  const canvas = $("videoCanvas"), video = $("videoFeed");
  const ctx    = canvas.getContext("2d");
  frameTimer   = setInterval(() => {
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(blob => {
      if (ws?.readyState === WebSocket.OPEN && blob)
        blob.arrayBuffer().then(b => ws.send(b));
    }, "image/jpeg", 0.7);
  }, FRAME_MS);
}

// Audio streaming
function streamAudio() {
  const aStream  = new MediaStream(mediaStream.getAudioTracks());
  mediaRecorder  = new MediaRecorder(aStream, { mimeType: "audio/webm" });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    audioChunks = [];
    const form  = new FormData();
    form.append("audio", blob, "audio.webm");
    form.append("session_id", sessionId);
    fetch(`${API_BASE}/api/kyc/upload-audio`, { method: "POST", body: form });
    if (mediaStream) { mediaRecorder.start(); }
  };
  mediaRecorder.start();
  setInterval(() => { if (mediaRecorder?.state === "recording") mediaRecorder.stop(); }, AUDIO_MS);
}

// Illumination challenge
async function runIllum(colors) {
  animateLayer({ layer: "illum", status: "running", detail: "Showing color flashes..." });
  addLog("Illumination challenge started", "info");
  const ov = $("illumOverlay");
  ov.classList.remove("hidden");
  for (const c of colors) {
    ov.style.background = c;
    await sleep(350);
    ov.style.background = "transparent";
    await sleep(150);
  }
  ov.classList.add("hidden");
  addLog("Illumination challenge complete", "pass");
  ws?.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: "CHALLENGE_COMPLETE", session_id: sessionId }));
}

// Submit
$("submitKYC").addEventListener("click", () => {
  $("submitKYC").disabled = true;
  addLog("Submitting for full AI analysis...", "info");
  ws?.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: "SUBMIT", session_id: sessionId }));
  runIllum(illumColors);

  // Trigger all layers to "running"
  ["deepfake","rppg","acoustic","illum","facematch","hw"].forEach(k => {
    animateLayer({ layer: k, status: "running", detail: "Analyzing..." });
  });
});

// Stop capture
function stop() {
  clearInterval(frameTimer);
  mediaStream?.getTracks().forEach(t => t.stop());
  ws?.close();
}

// ── Layer animations ─────────────────────────────────────
const LAYER_LABELS = {
  deepfake:  "Deepfake classifier",
  rppg:      "Biological signal",
  acoustic:  "Acoustic profiling",
  illum:     "Illumination challenge",
  facematch: "ArcFace face match",
  hw:        "Hardware auth",
};

function animateLayer({ layer, status, detail }) {
  const card   = $(`card-${layer}`);
  const badge  = $(`badge-${layer}`);
  const detEl  = $(`detail-${layer}`);
  const bar    = $(`bar-${layer}`);
  if (!card) return;

  card.classList.remove("pass", "fail");
  badge.classList.remove("idle", "running", "pass", "fail");

  if (status === "running") {
    badge.classList.add("running");
    badge.textContent = "scanning";
    bar.style.width = "40%";
    bar.style.background = "var(--cyan)";
    addLog(`${LAYER_LABELS[layer]}: scanning...`, "info");
  } else if (status === "pass") {
    card.classList.add("pass");
    badge.classList.add("pass");
    badge.textContent = "pass";
    bar.style.width   = "100%";
    bar.style.background = "var(--green)";
    addLog(`${LAYER_LABELS[layer]}: PASSED`, "pass");
  } else if (status === "fail") {
    card.classList.add("fail");
    badge.classList.add("fail");
    badge.textContent = "fail";
    bar.style.width   = "100%";
    bar.style.background = "var(--red)";
    addLog(`${LAYER_LABELS[layer]}: FAILED — ${detail}`, "fail");
  }

  if (detEl && detail) detEl.textContent = detail.slice(0, 55);

  // Update risk meter
  updateMeter();
}

function updateMeter() {
  const failCount = document.querySelectorAll(".lcard.fail").length;
  const passCount = document.querySelectorAll(".lcard.pass").length;
  const total     = 6;
  const score     = Math.round((failCount / total) * 100);
  const fill      = $("rmFill");
  const val       = $("rmVal");
  if (!fill) return;
  fill.style.width = score + "%";
  fill.style.background = score >= 70 ? "var(--red)" : score >= 40 ? "var(--amber)" : "var(--green)";
  val.textContent = score;
}

// ── Step 4: Result ───────────────────────────────────────
const LAYER_META = {
  deepfake_result:     { name: "Deepfake classifier", ico: "c" },
  rppg_result:         { name: "Biological signal",   ico: "p" },
  acoustic_result:     { name: "Acoustic profiling",  ico: "a" },
  illumination_result: { name: "Illumination",        ico: "o" },
  face_match_result:   { name: "ArcFace face match",  ico: "g" },
  hardware_result:     { name: "Hardware auth",       ico: "v" },
};

function showResult(r) {
  goTo(4);
  addLog(`Analysis complete — verdict: ${r.verdict}`, r.verdict === "APPROVED" ? "pass" : "fail");

  const score   = Math.round(r.risk_score);
  const verdict = r.verdict;

  // Ring animation
  setTimeout(() => {
    const ring = $("ringPath");
    const circumference = 326.7;
    const offset = circumference - (score / 100) * circumference;
    ring.style.transition = "stroke-dashoffset 1.5s cubic-bezier(0.34,1.56,0.64,1)";
    ring.style.strokeDashoffset = offset;
  }, 200);

  $("ringNum").textContent = score;

  // Ring gradient color by verdict
  const gradColors = {
    APPROVED: ["#00D4FF", "#00FF8C"],
    BLOCKED:  ["#FF3B5C", "#FF6B35"],
    REVIEW:   ["#FFB830", "#FF6B35"],
  };
  const [c1, c2] = gradColors[verdict] || gradColors.APPROVED;
  $("rg1").setAttribute("stop-color", c1);
  $("rg2").setAttribute("stop-color", c2);

  // Verdict card
  const vc = $("verdCard");
  vc.className = "verd-card " + verdict.toLowerCase();

  const icons   = { APPROVED: "✓", BLOCKED: "✕", REVIEW: "!" };
  const titles  = { APPROVED: "Identity Verified", BLOCKED: "Fraud Detected", REVIEW: "Manual Review" };
  const subs    = { APPROVED: "All verification layers passed.", BLOCKED: "Deepfake or identity fraud detected.", REVIEW: "One or more layers flagged for review." };

  $("verdIcon").textContent  = icons[verdict];
  $("verdTitle").textContent = titles[verdict];
  $("verdSub").textContent   = subs[verdict];

  // Fraud tags
  const tags = $("verdTags");
  tags.innerHTML = "";
  (r.fraud_types || []).filter(f => f !== "NONE").forEach(f => {
    const t = document.createElement("span");
    t.className = "vtag fraud";
    t.textContent = f.replace(/_/g, " ");
    tags.appendChild(t);
  });
  if (!tags.children.length) {
    const t = document.createElement("span");
    t.className = "vtag clean";
    t.textContent = "No fraud detected";
    tags.appendChild(t);
  }

  // Meta
  $("verdMeta").innerHTML =
    `Session: ${r.session_id.slice(0,16)}... &nbsp;·&nbsp; Processed in ${Math.round(r.total_latency_ms)}ms`;

  // Layer breakdown
  const grid = $("rlayGrid");
  grid.innerHTML = "";
  const icoColors = { c: "#00D4FF", p: "#E040FB", a: "#FFB830", o: "#FF6B35", g: "#00FF8C", v: "#9B59B6" };

  Object.entries(LAYER_META).forEach(([key, meta]) => {
    const lr   = r[key];
    if (!lr) return;
    const pass = lr.label === "REAL";
    const conf = Math.round(lr.confidence * 100);

    const row  = document.createElement("div");
    row.className = `rlay-card ${pass ? "pass" : "fail"}`;
    row.innerHTML = `
      <div class="rlay-icon lcard-ico ${meta.ico}" style="background:${icoColors[meta.ico]}18">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="${pass ? "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" : "M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"}"
            stroke="${pass ? 'var(--green)' : 'var(--red)'}" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </div>
      <div>
        <p class="rlay-name">${meta.name}</p>
        <p class="rlay-detail">${(lr.detail || "").slice(0,60)}</p>
      </div>
      <div class="rlay-verdict" style="color:${pass ? 'var(--green)' : 'var(--red)'}">
        ${lr.label} &nbsp;<span style="color:var(--t3);font-size:10px">${conf}%</span>
      </div>`;
    grid.appendChild(row);
  });
}
