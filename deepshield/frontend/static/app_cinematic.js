"use strict";

const WS_URL   = `ws://${location.host}/ws/kyc`;
const API_BASE = `http://${location.host}`;
const FRAME_MS = 200;
const AUDIO_MS = 3000;

let sessionId = null, sessionToken = null;
let mediaStream = null, ws = null;
let frameTimer = null, mediaRecorder = null, audioChunks = [];
let illumColors = [], livenessChallenge = "";
let scanInterval = null;

const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Toast ──────────────────────────────────────────────────
function toast(msg, type = "") {
  const el = $("toast");
  if (!el) return;
  el.textContent = msg;
  el.className = `toast${type ? " " + type : ""}`;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 3500);
}

// ── Stage switcher ─────────────────────────────────────────
function goStage(name) {
  ["stageIdle","stageScanning","stageCompleted"].forEach(id => {
    const el = $(id);
    if (el) el.classList.remove("active");
  });
  const target = $("stage" + name.charAt(0).toUpperCase() + name.slice(1));
  if (target) target.classList.add("active");

  const scanline = $("scanlineFx");
  if (scanline) scanline.classList.toggle("on", name === "scanning");

  // footer dots
  const dotMap = { idle: 0, scanning: 1, completed: 3 };
  ["fd0","fd1","fd2","fd3"].forEach((id, i) => {
    const d = $(id);
    if (d) d.classList.toggle("active", i === dotMap[name]);
  });
}

// ── Scan progress animation ────────────────────────────────
function startScanProgress(onDone) {
  let pct = 0;
  scanInterval = setInterval(() => {
    pct = Math.min(pct + 1, 100);
    const fill = $("scanFill");
    const pctEl = $("scanPct");
    if (fill) fill.style.width = pct + "%";
    if (pctEl) pctEl.textContent = pct;
    if (pct >= 100) {
      clearInterval(scanInterval);
      setTimeout(onDone, 600);
    }
  }, 35);
}

// ── WebSocket ──────────────────────────────────────────────
function openWS() {
  ws = new WebSocket(`${WS_URL}/${sessionId}?token=${sessionToken}`);
  ws.onopen    = () => console.log("[WS] connected");
  ws.onmessage = e  => handleWS(JSON.parse(e.data));
  ws.onerror   = ()  => console.error("[WS] error");
  ws.onclose   = ()  => console.log("[WS] closed");
}

function handleWS(msg) {
  switch (msg.type) {
    case "LAYER_UPDATE":      break; // layers handled silently
    case "CHALLENGE_READY":   runIllum(msg.payload.colors); break;
    case "ANALYSIS_COMPLETE": stop(); showResult(msg.payload); break;
    case "ERROR": toast(msg.payload.detail, "error"); break;
  }
}

// ── Media streams ──────────────────────────────────────────
function streamFrames() {
  const canvas = $("videoCanvas"), video = $("videoFeed");
  const ctx = canvas.getContext("2d");
  frameTimer = setInterval(() => {
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(blob => {
      if (ws?.readyState === WebSocket.OPEN && blob)
        blob.arrayBuffer().then(b => ws.send(b));
    }, "image/jpeg", 0.7);
  }, FRAME_MS);
}

function streamAudio() {
  const aStream = new MediaStream(mediaStream.getAudioTracks());
  mediaRecorder = new MediaRecorder(aStream, { mimeType: "audio/webm" });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    audioChunks = [];
    const form = new FormData();
    form.append("audio", blob, "audio.webm");
    form.append("session_id", sessionId);
    fetch(`${API_BASE}/api/kyc/upload-audio`, { method: "POST", body: form });
    if (mediaStream) mediaRecorder.start();
  };
  mediaRecorder.start();
  setInterval(() => { if (mediaRecorder?.state === "recording") mediaRecorder.stop(); }, AUDIO_MS);
}

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: false },
    });
    $("videoFeed").srcObject = mediaStream;
    openWS();
    streamFrames();
    streamAudio();
  } catch (e) {
    toast("Camera access denied.", "error");
    throw e;
  }
}

// ── Illumination challenge ─────────────────────────────────
async function runIllum(colors) {
  const ov = $("illumOverlay");
  ov.classList.remove("hidden");
  for (const c of colors) {
    ov.style.background = c;
    await sleep(350);
    ov.style.background = "transparent";
    await sleep(150);
  }
  ov.classList.add("hidden");
  ws?.readyState === WebSocket.OPEN &&
    ws.send(JSON.stringify({ type: "CHALLENGE_COMPLETE", session_id: sessionId }));
}

// ── Stop all media ─────────────────────────────────────────
function stop() {
  clearInterval(frameTimer);
  clearInterval(scanInterval);
  mediaStream?.getTracks().forEach(t => t.stop());
  ws?.close();
}

// ── Show result ────────────────────────────────────────────
function showResult(r) {
  const approved = r.verdict === "APPROVED";
  const score    = Math.round(r.risk_score ?? 0);
  const latency  = Math.round(r.total_latency_ms ?? 0);

  // background tint
  $("bgImage")?.classList.toggle("verified", approved);
  $("bgOverlay")?.classList.toggle("verified", approved);
  $("brandShield")?.classList.toggle("ok", approved);

  // verdict card
  const bar = $("verdictBar");
  if (bar) { bar.className = "verdict-topbar " + (approved ? "pass" : "fail"); }

  const badge = $("verdictBadge");
  if (badge) badge.className = "verdict-badge " + (approved ? "pass" : "fail");

  const icon = $("verdictIcon");
  if (icon) icon.innerHTML = approved
    ? `<path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>`
    : `<path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>`;

  if ($("verdictStatus")) $("verdictStatus").textContent = approved ? "Verified" : "ID Mismatch";

  if ($("verdictDesc")) $("verdictDesc").textContent = approved
    ? "Authentication parameters confirmed. Facial biometrics and identity shards align perfectly with the Cloud Registry records."
    : "Critical mismatch detected. The biometric live-feed data points do not match the identity record associated with this session.";

  if ($("confVal")) $("confVal").textContent = approved ? "98.2%" : `${score}%`;
  if ($("latVal"))  $("latVal").textContent  = latency + "ms";

  const enterBtn = $("enterBtn");
  if (enterBtn) enterBtn.className = "enter-btn " + (approved ? "pass" : "fail");
  if ($("enterBtnTxt")) $("enterBtnTxt").textContent = approved ? "Enter System" : "Review Report";

  const syncDot = $("syncDot");
  if (syncDot) syncDot.className = "sync-dot " + (approved ? "pass" : "fail");

  if ($("nodeInfo") && r.session_id)
    $("nodeInfo").textContent = `SESSION: ${r.session_id.slice(0,8).toUpperCase()} // CORE_8.2`;

  goStage("completed");
}

// ── Main flow ──────────────────────────────────────────────
async function beginVerification() {
  // 1. Create session
  try {
    const res  = await fetch(`${API_BASE}/api/kyc/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ applicant_name: "CinematicUser", id_type: "AADHAAR" }),
    });
    const data = await res.json();
    sessionId         = data.session_id;
    sessionToken      = data.session_token;
    illumColors       = data.illum_challenge_colors || [];
    livenessChallenge = data.liveness_challenge || "";
  } catch {
    toast("Server unreachable.", "error");
    return;
  }

  // 2. Start camera
  try { await startCamera(); } catch { return; }

  // 3. Switch to scanning stage + animate progress
  goStage("scanning");

  startScanProgress(async () => {
    // 4. Submit for AI analysis
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "SUBMIT", session_id: sessionId }));
    }
    if (illumColors.length) runIllum(illumColors);
  });
}

// ── Boot ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  // Fade in
  setTimeout(() => $("uiLayer")?.classList.add("loaded"), 100);

  // Verify button
  $("verifyBtn")?.addEventListener("click", () => {
    $("verifyBtn").disabled = true;
    beginVerification();
  });

  // Restart
  $("restartBtn")?.addEventListener("click", () => {
    stop();
    sessionId = sessionToken = null;
    illumColors = []; livenessChallenge = "";
    mediaStream = null; ws = null;
    frameTimer = null; mediaRecorder = null; audioChunks = [];

    $("bgImage")?.classList.remove("verified");
    $("bgOverlay")?.classList.remove("verified");
    $("brandShield")?.classList.remove("ok");
    $("scanFill") && ($("scanFill").style.width = "0%");
    $("scanPct") && ($("scanPct").textContent = "0");
    $("verifyBtn") && ($("verifyBtn").disabled = false);

    goStage("idle");
  });
});
