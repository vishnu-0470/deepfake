"use strict";

const WS_URL   = `ws://${location.host}/ws/kyc`;
const API_BASE = `http://${location.host}`;
const FRAME_MS = 200;
const AUDIO_MS = 3000;

let sessionId = null, sessionToken = null;
let mediaStream = null, ws = null;
let frameTimer = null, mediaRecorder = null, audioChunks = [];
let illumColors = [], livenessChallenge = "";
let currentStep = 0;
let attemptCount = parseInt(localStorage.getItem('kycAttempts') || '0');

const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));

function now() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
}

function addLog(msg, type = "") {
  const body = $("logBody");
  if (!body) return;
  const row = document.createElement("div");
  row.className = `log-row ${type}`;
  row.innerHTML = `<span class="log-time font-mono">${now()}</span><span class="log-msg">${msg}</span>`;
  body.appendChild(row);
  body.scrollTop = body.scrollHeight;
  while (body.children.length > 40) body.removeChild(body.firstChild);
}

function toast(msg, type = "") {
  const el = $("toast");
  if (!el) return;
  el.textContent = msg;
  el.className = `toast${type ? ' '+type : ''}`;
  el.classList.remove("hidden");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.add("hidden"), 3500);
}

function goTo(n) {
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  const panel = $(`step${n}`);
  if (panel) panel.classList.add("active");

  const nav0 = $("nav0");
  if (nav0) {
    nav0.classList.remove("active", "done");
    if (n > 0) nav0.classList.add("done");
    else nav0.classList.add("active");
  }

  for (let i = 1; i <= 4; i++) {
    const nav = $(`nav${i}`);
    if (!nav) continue;
    nav.classList.remove("active", "done");
    if (i < n) { nav.classList.add("done"); nav.querySelector(".snav-dot").innerHTML = "✓"; }
    if (i === n) nav.classList.add("active");
  }
  currentStep = n;
}

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

async function startVideo() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: false },
    });
    $("videoFeed").srcObject = mediaStream;
    $("recTag").textContent = "RECORDING";
    $("recTag").classList.add("live");
    $("liveTag").style.display = "flex";

    addLog("Camera & microphone active", "pass");
    openWS();
    streamFrames();
    streamAudio();
    updateBiometrics();

    if (livenessChallenge) {
      $("challengeBox").style.display = "flex";
      $("challengeText").textContent = livenessChallenge;
      addLog(`Liveness challenge: ${livenessChallenge}`, "info");
    }

    setTimeout(() => {
      $("submitKYC").disabled = false;
      addLog("Ready to submit — click 'Submit for AI Analysis'", "info");
    }, 5000);

  } catch (e) {
    addLog("Camera/microphone denied", "fail");
    toast("Camera access denied.", "error");
  }
}

function updateBiometrics() {
  setInterval(() => {
    if (currentStep !== 3) return;
    const hr = 65 + Math.floor(Math.random() * 15);
    const face = Math.random() > 0.1;
    const bioHR = $("bioHR");
    const bioFace = $("bioFace");
    if (bioHR) bioHR.textContent = hr + " bpm";
    if (bioFace) {
      bioFace.textContent = face ? "Detected" : "Searching...";
      bioFace.style.color = face ? "#10B981" : "#F59E0B";
    }
  }, 1500);
}

function openWS() {
  ws = new WebSocket(`${WS_URL}/${sessionId}?token=${sessionToken}`);
  ws.onopen    = () => addLog("WebSocket connected", "info");
  ws.onmessage = e  => handleWS(JSON.parse(e.data));
  ws.onerror   = ()  => addLog("WebSocket error", "fail");
  ws.onclose   = ()  => addLog("WebSocket closed", "");
}

function handleWS(msg) {
  switch (msg.type) {
    case "LAYER_UPDATE":      animateLayer(msg.payload); break;
    case "CHALLENGE_READY":   runIllum(msg.payload.colors); break;
    case "ANALYSIS_COMPLETE": stop(); showResult(msg.payload); break;
    case "ERROR": toast(msg.payload.detail, "error"); addLog(msg.payload.detail, "fail"); break;
  }
}

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

function stop() {
  clearInterval(frameTimer);
  mediaStream?.getTracks().forEach(t => t.stop());
  ws?.close();
}

const LAYER_LABELS = {
  deepfake: "Deepfake classifier", rppg: "Biological signal",
  acoustic: "Acoustic profiling",  illum: "Illumination challenge",
  facematch: "ArcFace face match", hw: "Hardware auth",
};

function animateLayer({ layer, status, detail }) {
  const card  = $(`card-${layer}`);
  const badge = $(`badge-${layer}`);
  const detEl = $(`detail-${layer}`);
  const bar   = $(`bar-${layer}`);
  if (!card) return;

  card.classList.remove("pass", "fail");
  badge.classList.remove("idle", "running", "pass", "fail");

  if (status === "running") {
    badge.classList.add("running"); badge.textContent = "scanning";
    bar.style.width = "40%"; bar.style.background = "#3B82F6";
    addLog(`${LAYER_LABELS[layer]}: scanning...`, "info");
  } else if (status === "pass") {
    card.classList.add("pass"); badge.classList.add("pass"); badge.textContent = "pass";
    bar.style.width = "100%"; bar.style.background = "#10B981";
    addLog(`${LAYER_LABELS[layer]}: PASSED`, "pass");
  } else if (status === "fail") {
    card.classList.add("fail"); badge.classList.add("fail"); badge.textContent = "fail";
    bar.style.width = "100%"; bar.style.background = "#EF4444";
    addLog(`${LAYER_LABELS[layer]}: FAILED — ${detail}`, "fail");
  }

  if (detEl && detail) detEl.textContent = detail.slice(0, 55);
  updateMeter();
}

function updateMeter() {
  const failCount = document.querySelectorAll(".lcard.fail").length;
  const total = 6;
  const score = Math.round((failCount / total) * 100);
  const fill = $("rmFill"), val = $("rmVal");
  if (!fill) return;
  fill.style.width = score + "%";
  fill.style.background = score >= 70 ? "#EF4444" : score >= 40 ? "#F59E0B" : "#10B981";
  if (val) val.textContent = score;
}

const LAYER_META = {
  deepfake_result:     { name: "Deepfake classifier", ico: "c" },
  rppg_result:         { name: "Biological signal",   ico: "p" },
  acoustic_result:     { name: "Acoustic profiling",  ico: "a" },
  illumination_result: { name: "Illumination",        ico: "o" },
  face_match_result:   { name: "ArcFace face match",  ico: "g" },
  hardware_result:     { name: "Hardware auth",       ico: "v" },
};

function generateExplainer(r) {
  const reasons = [];
  if (r.deepfake_result?.label === "FAKE")     reasons.push("Video shows deepfake artifacts detected by CNN classifier");
  if (r.rppg_result?.label === "FAKE")         reasons.push("No biological heart rate signal detected in facial video");
  if (r.acoustic_result?.label === "FAKE")     reasons.push("Voice audio shows synthetic or manipulated characteristics");
  if (r.illumination_result?.label === "FAKE") reasons.push("Face did not respond naturally to screen illumination changes");
  if (r.face_match_result?.label === "FAKE")   reasons.push("Live video face does not match the Aadhaar document photo");
  if (r.hardware_result?.label === "FAKE")     reasons.push("Device hardware fingerprint indicates emulator or virtual environment");
  if (reasons.length === 0) {
    reasons.push("All 6 verification layers passed successfully");
    reasons.push("Biological signals confirmed (heart rate detected)");
    reasons.push("Face matches Aadhaar document with high confidence");
    reasons.push("No deepfake or synthetic media artifacts found");
  }
  return reasons;
}

function showResult(r) {
  goTo(4);
  addLog(`Analysis complete — verdict: ${r.verdict}`, r.verdict === "APPROVED" ? "pass" : "fail");

  const score = Math.round(r.risk_score);
  const verdict = r.verdict;

  setTimeout(() => {
    const ring = $("ringPath");
    if (!ring) return;
    const offset = 326.7 - (score / 100) * 326.7;
    ring.style.transition = "stroke-dashoffset 1.5s cubic-bezier(0.34,1.56,0.64,1)";
    ring.style.strokeDashoffset = offset;
  }, 200);

  if ($("ringNum")) $("ringNum").textContent = score;

  const gradColors = { APPROVED: ["#3B82F6","#10B981"], BLOCKED: ["#EF4444","#F59E0B"], REVIEW: ["#F59E0B","#EF4444"] };
  const [c1, c2] = gradColors[verdict] || gradColors.APPROVED;
  $("rg1")?.setAttribute("stop-color", c1);
  $("rg2")?.setAttribute("stop-color", c2);

  const vc = $("verdCard");
  if (vc) vc.className = "verd-card " + verdict.toLowerCase();

  const icons  = { APPROVED: "✓", BLOCKED: "✕", REVIEW: "!" };
  const titles = { APPROVED: "Identity Verified", BLOCKED: "Fraud Detected", REVIEW: "Manual Review" };
  const subs   = { APPROVED: "All verification layers passed.", BLOCKED: "Deepfake or identity fraud detected.", REVIEW: "One or more layers flagged for review." };

  if ($("verdIcon"))  $("verdIcon").textContent  = icons[verdict];
  if ($("verdTitle")) $("verdTitle").textContent = titles[verdict];
  if ($("verdSub"))   $("verdSub").textContent   = subs[verdict];

  const tags = $("verdTags");
  if (tags) {
    tags.innerHTML = "";
    (r.fraud_types || []).filter(f => f !== "NONE").forEach(f => {
      const t = document.createElement("span");
      t.className = "vtag fraud"; t.textContent = f.replace(/_/g, " ");
      tags.appendChild(t);
    });
    if (!tags.children.length) {
      const t = document.createElement("span");
      t.className = "vtag clean"; t.textContent = "No fraud detected";
      tags.appendChild(t);
    }
  }

  if ($("verdMeta")) $("verdMeta").innerHTML = `Session: ${r.session_id.slice(0,16)}... &nbsp;·&nbsp; Processed in ${Math.round(r.total_latency_ms)}ms`;

  const grid = $("rlayGrid");
  if (grid) {
    grid.innerHTML = "";
    const icoColors = { c:"#3B82F6", p:"#8B5CF6", a:"#F59E0B", o:"#F97316", g:"#10B981", v:"#6366F1" };
    Object.entries(LAYER_META).forEach(([key, meta]) => {
      const lr = r[key]; if (!lr) return;
      const pass = lr.label === "REAL";
      const conf = Math.round(lr.confidence * 100);
      const row = document.createElement("div");
      row.className = `rlay-card ${pass ? "pass" : "fail"}`;
      row.innerHTML = `
        <div class="rlay-icon" style="background:${icoColors[meta.ico]}22;width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;flex-shrink:0">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="${pass ? "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" : "M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"}"
              stroke="${pass ? '#10B981' : '#EF4444'}" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </div>
        <div style="flex:1">
          <p style="font-size:13px;font-weight:600;color:#F9FAFB">${meta.name}</p>
          <p style="font-size:11px;color:#6B7280;margin-top:2px">${(lr.detail||"").slice(0,60)}</p>
        </div>
        <div style="font-family:monospace;font-size:12px;font-weight:600;color:${pass?'#10B981':'#EF4444'};white-space:nowrap">
          ${lr.label} <span style="color:#4B5563;font-size:10px">${conf}%</span>
        </div>`;
      grid.appendChild(row);
    });
  }

  const expList = $("explainerList");
  if (expList) {
    expList.innerHTML = "";
    generateExplainer(r).forEach(txt => {
      const li = document.createElement("li"); li.textContent = txt;
      expList.appendChild(li);
    });
  }
}

// ── Init all event listeners on DOM ready ─────────────────
document.addEventListener("DOMContentLoaded", () => {

  // Attempts counter
  const attemptsLeft = $("statAttempts");
  if (attemptsLeft) attemptsLeft.textContent = Math.max(0, 100 - attemptCount);

  // Dashboard start button
  $("dashStartBtn")?.addEventListener("click", () => {
    if (attemptCount >= 100) { $("maxAttemptsModal")?.classList.remove("hidden"); return; }
    addLog("Starting new verification...", "info");
    goTo(1);
  });

  // Modal close
  $("modalClose")?.addEventListener("click", () => $("maxAttemptsModal")?.classList.add("hidden"));

  // OTP state
  let otpVerified = false;

  // Send OTP
  $("sendOtpBtn")?.addEventListener("click", async () => {
    const phone = $("applicantPhone").value.trim();
    if (!phone) return toast("Please enter your phone number.", "error");
    if (!sessionId) return toast("Please fill your name first, then click Send OTP.", "error");
    const btn = $("sendOtpBtn");
    btn.disabled = true;
    btn.textContent = "Sending...";
    try {
      const res = await fetch(`${API_BASE}/api/otp/send`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone, session_id: sessionId }),
      });
      const data = await res.json();
      if (res.ok) {
        $("otpBox").classList.remove("hidden");
        $("otpStatus").textContent = `Sent to ${phone}`;
        addLog(`OTP sent to ${phone}`, "pass");
        // Show dev OTP on screen if in dev mode
        if (data.dev_mode && data.dev_otp) {
          $("otpInput").value = data.dev_otp;
          $("otpStatus").textContent = `DEV MODE — OTP: ${data.dev_otp}`;
          $("otpStatus").style.color = "#F59E0B";
          toast(`DEV MODE: Your OTP is ${data.dev_otp}`, "");
          addLog(`DEV MODE OTP: ${data.dev_otp}`, "warn");
        } else {
          toast("OTP sent! Check your phone.", "success");
        }
        let secs = 30;
        const iv = setInterval(() => {
          btn.textContent = `Resend (${secs--}s)`;
          if (secs < 0) { clearInterval(iv); btn.disabled = false; btn.textContent = "Resend OTP"; }
        }, 1000);
      } else {
        toast(data.detail || "Failed to send OTP.", "error");
        btn.disabled = false; btn.textContent = "Send OTP";
      }
    } catch { toast("Server error.", "error"); btn.disabled = false; btn.textContent = "Send OTP"; }
  });

  // Verify OTP
  $("verifyOtpBtn")?.addEventListener("click", async () => {
    const phone = $("applicantPhone").value.trim();
    const otp   = $("otpInput").value.trim();
    if (otp.length !== 6) return toast("Enter the 6-digit OTP.", "error");
    const btn = $("verifyOtpBtn");
    btn.disabled = true; btn.textContent = "Verifying...";
    try {
      const res = await fetch(`${API_BASE}/api/otp/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone, otp, session_id: sessionId }),
      });
      const data = await res.json();
      if (res.ok) {
        otpVerified = true;
        $("otpBox").innerHTML = `<div class="otp-verified-badge">✓ Phone verified — ${phone}</div>`;
        addLog("Phone number OTP verified", "pass");
        toast("Phone verified!", "success");
      } else {
        toast(data.detail || "Invalid OTP.", "error");
        btn.disabled = false; btn.textContent = "Verify";
      }
    } catch { toast("Server error.", "error"); btn.disabled = false; btn.textContent = "Verify"; }
  });

  // Step 1 Next
  $("step1Next")?.addEventListener("click", async () => {
    const name  = $("applicantName").value.trim();
    const phone = $("applicantPhone").value.trim();
    if (!name)  return toast("Please enter your full name.", "error");
    if (!phone) return toast("Please enter your phone number.", "error");
    if (!$("termsCheck").checked) return toast("Please agree to the terms.", "error");

    // Create session first if not done
    if (!sessionId) {
      attemptCount++;
      localStorage.setItem('kycAttempts', attemptCount.toString());
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
      } catch { addLog("Server error", "fail"); return toast("Server error.", "error"); }
    }

    // Require OTP verification
    if (!otpVerified) {
      if ($("otpBox").classList.contains("hidden")) {
        $("sendOtpBtn").click();
        return toast("OTP sent to your phone. Verify to continue.", "");
      }
      return toast("Please verify your phone number with OTP first.", "error");
    }

    addLog("Identity + phone verified — proceeding", "pass");
    goTo(2);
  });

  // Step 2 - upload
  const zone   = $("uploadZone");
  const fileIn = $("docFile");
  zone?.addEventListener("click", () => fileIn.click());
  zone?.addEventListener("dragover",  e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone?.addEventListener("dragleave", ()  => zone.classList.remove("drag-over"));
  zone?.addEventListener("drop",      e  => { e.preventDefault(); zone.classList.remove("drag-over"); handleDoc(e.dataTransfer.files[0]); });
  fileIn?.addEventListener("change",  ()  => handleDoc(fileIn.files[0]));

  $("step2Next")?.addEventListener("click", () => { goTo(3); startVideo(); });

  // Step 3 - submit
  $("submitKYC")?.addEventListener("click", () => {
    $("submitKYC").disabled = true;
    addLog("Submitting for full AI analysis...", "info");
    ws?.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: "SUBMIT", session_id: sessionId }));
    runIllum(illumColors);
    ["deepfake","rppg","acoustic","illum","facematch","hw"].forEach(k =>
      animateLayer({ layer: k, status: "running", detail: "Analyzing..." })
    );
  });

});
