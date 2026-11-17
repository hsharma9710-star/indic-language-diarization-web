"use strict";
// In-browser language diarization: audio -> RTTM + coloured waveform + pooled audio per language.
// Model: CLSRIL-23 encoder + statistical-pooling / self-attention head, exported to ONNX (int8 MatMul weights).

const SR = 16000;
const BIN = 0.2;
const CHUNK = 30 * SR;
const MAX_SECONDS = 300;
const MODEL_PARTS = ["model/model_int8.part0", "model/model_int8.part1", "model/model_int8.part2"];
const CACHE = "ildiar-model-v1";

// class order of the network output
const CLASSES = ["sil", "en", "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "pa", "sa", "ta", "te", "ur", "od"];   // the 15 outputs of the CLSRIL network + "od" (Odia), which only the Whisper-Linear-LD tab produces
const N_NET = 15;
const NAMES = { en: "English", as: "Assamese", bn: "Bengali", gu: "Gujarati", hi: "Hindi", kn: "Kannada", ml: "Malayalam",
                mr: "Marathi", ne: "Nepali", pa: "Punjabi", sa: "Sanskrit", ta: "Tamil", te: "Telugu", ur: "Urdu", od: "Odia", sil: "silence / no speech" };
const SIL = 0, ENG = 1;
const COLOURS = { en: "#3b82f6", sil: "#9ca3af", order: ["#ec4899", "#22c55e", "#f59e0b", "#a855f7", "#14b8a6", "#ef4444", "#84cc16", "#06b6d4", "#f97316", "#6366f1"] };

const $ = (id) => document.getElementById(id);
let session = null;
let lastRttm = "", lastName = "audio";

function fillSelects() {
  for (const id of ["l1", "l2"]) {
    const s = $(id);
    s.innerHTML = '<option value="auto">auto-detect</option>' +
      CLASSES.slice(2, N_NET).map((c) => `<option value="${c}">${NAMES[c]}</option>`).join("");
  }
}

function wasmThreads() { return self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 2, 8) : 1; }

function setStatus(msg, p) {
  $("status").textContent = msg;
  const bar = $("bar");
  if (p === undefined) { bar.hidden = true; } else { bar.hidden = false; bar.value = p; }
}

async function fetchParts(parts, cacheName, label, totalMB) {
  const cache = await caches.open(cacheName).catch(() => null);
  const bufs = [];
  let done = 0;
  for (let i = 0; i < parts.length; i++) {
    const url = parts[i];
    let resp = cache ? await cache.match(url) : null;
    if (!resp) {
      resp = await fetch(url);
      if (!resp.ok) throw new Error("could not download the model (" + url + ")");
      if (cache) await cache.put(url, resp.clone());
    }
    const reader = resp.body.getReader();
    const chunks = [];
    for (;;) {
      const { done: d, value } = await reader.read();
      if (d) break;
      chunks.push(value);
      done += value.length;
      setStatus(label + "... " + Math.round(done / 1e6) + " MB", Math.min(done / (totalMB * 1e6), 1));
    }
    bufs.push(new Blob(chunks));
  }
  const all = await new Blob(bufs).arrayBuffer();
  return new Uint8Array(all);
}

async function fetchModel() {
  return fetchParts(MODEL_PARTS, CACHE, "downloading the model (first use only)", 137);
}

async function getSession() {
  if (session) return session;
  const bytes = await fetchModel();
  setStatus("preparing the model...");
  ort.env.wasm.wasmPaths = "./";
  ort.env.wasm.numThreads = wasmThreads();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return session;
}

async function decodeAudio(file) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  const arr = await file.arrayBuffer();
  const buf = await ctx.decodeAudioData(arr);
  const n = buf.length;
  const mono = new Float32Array(n);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += d[i] / buf.numberOfChannels;
  }
  ctx.close();
  if (buf.sampleRate !== SR) throw new Error("this browser could not resample the audio to 16 kHz");
  return mono;
}

async function posteriors(wav) {
  const sess = await getSession();
  const n = wav.length;
  const starts = [];
  for (let a = 0; a < Math.max(n, 1); a += CHUNK) starts.push(a);
  if (starts.length > 1 && n - starts[starts.length - 1] < 2 * SR) starts.pop();
  const out = [];
  for (let k = 0; k < starts.length; k++) {
    const a = starts[k];
    const b = k === starts.length - 1 ? n : a + CHUNK;
    const want = Math.ceil((b - a) / (BIN * SR));
    setStatus(`analysing the speech: part ${k + 1} of ${starts.length}...`, k / starts.length);
    await new Promise((r) => setTimeout(r, 20));              // let the page repaint
    const input = new ort.Tensor("float32", wav.slice(a, b), [1, b - a]);
    const res = await sess.run({ wav: input });
    const p = res[Object.keys(res)[0]];
    const S = p.dims[1], C = p.dims[2];
    for (let i = 0; i < want; i++) {
      const row = new Float32Array(C);
      const src = Math.min(i, S - 1) * C;                      // repeat the last bin if the chunk is shorter than expected
      for (let c = 0; c < C; c++) row[c] = p.data[src + c];
      out.push(row);
    }
  }
  return out;
}

// ---- decoding (port of ildiar/decode.py) ----
function pickLanguages(post, maxIndic = 2, ratio = 0.1, minShare = 0.05) {
  const C = post[0].length;
  const mass = new Float64Array(C);
  let speech = 0;
  for (const p of post) {
    if (p[SIL] < 0.5) { speech++; for (let c = 0; c < C; c++) mass[c] += p[c]; }
  }
  if (!speech) return [];
  let total = mass[ENG];
  for (let c = 2; c < C; c++) total += mass[c];
  const order = [];
  for (let c = 2; c < C; c++) order.push(c);
  order.sort((x, y) => mass[y] - mass[x]);
  const ids = [];
  const first = order[0];
  if (mass[first] >= minShare * total) ids.push(first);
  for (let i = 1; i < maxIndic && i < order.length; i++) {
    const c = order[i];
    if (ids.length && mass[c] >= ratio * mass[first] && mass[c] >= minShare * total) ids.push(c);
  }
  return ids;
}

function decode(post, given) {
  const ids = given.length ? given : pickLanguages(post);
  const allowed = [SIL, ENG, ...ids];
  const labels = post.map((p) => {
    let best = allowed[0];
    for (const c of allowed) if (p[c] > p[best]) best = c;
    return best;
  });
  return { labels, ids };
}

function toSegments(labels) {
  const segs = [];
  let s = 0;
  for (let i = 1; i <= labels.length; i++) {
    if (i === labels.length || labels[i] !== labels[s]) {
      if (labels[s] !== SIL) segs.push([s * BIN, (i - s) * BIN, labels[s]]);
      s = i;
    }
  }
  return segs;
}

function makeRttm(segs, name, type) {
  return segs.map(([st, du, c]) =>
    `${type || "LANGUAGE"} ${name} 1 ${st.toFixed(3)} ${du.toFixed(3)} <NA> <NA> ${CLASSES[c].toUpperCase()} <NA> <NA>`).join("\n") + (segs.length ? "\n" : "");
}

function colourFor(c, indicOrder) {
  if (c === ENG) return COLOURS.en;
  const i = indicOrder.indexOf(c);
  return COLOURS.order[Math.min(i, COLOURS.order.length - 1)];
}

function drawPlot(wav, segs, t0, len) {
  const cv = $("plot");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, padB = 54, padT = 16;
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
  const t1 = Math.min(t0 + len, wav.length / SR);
  const a = Math.floor(t0 * SR), b = Math.floor(t1 * SR);
  const mid = padT + (H - padT - padB) / 2, amp = (H - padT - padB) / 2 * 0.95;
  const indicOrder = [];
  for (const [, , c] of segs) if (c !== ENG && !indicOrder.includes(c)) indicOrder.push(c);
  const colAt = (t) => {
    for (const [s, d, c] of segs) if (t >= s && t < s + d) return colourFor(c, indicOrder);
    return COLOURS.sil;
  };
  const step = Math.max(1, Math.floor((b - a) / W));
  let peak = 1e-6;
  for (let i = a; i < b; i++) peak = Math.max(peak, Math.abs(wav[i]));
  for (let x = 0; x < W; x++) {
    const i0 = a + Math.floor(((b - a) * x) / W), i1 = Math.min(b, i0 + step);
    let lo = 0, hi = 0;
    for (let i = i0; i < i1; i++) { const v = wav[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    ctx.fillStyle = colAt(t0 + ((t1 - t0) * x) / W);
    ctx.fillRect(x, mid - (hi / peak) * amp, 1, Math.max(1, ((hi - lo) / peak) * amp));
  }
  ctx.fillStyle = "#111827"; ctx.font = "22px sans-serif"; ctx.strokeStyle = "#111827";
  ctx.beginPath(); ctx.moveTo(0, H - padB + 6); ctx.lineTo(W, H - padB + 6); ctx.stroke();
  const tick = (t1 - t0) > 60 ? 10 : (t1 - t0) > 25 ? 5 : 2;
  for (let t = Math.ceil(t0 / tick) * tick; t <= t1; t += tick) {
    const x = ((t - t0) / (t1 - t0)) * W;
    ctx.beginPath(); ctx.moveTo(x, H - padB + 6); ctx.lineTo(x, H - padB + 16); ctx.stroke();
    ctx.fillText(t + " s", Math.min(x + 4, W - 70), H - padB + 38);
  }
  $("legend").innerHTML = [ENG, ...indicOrder].map((c) =>
    `<span><i class="dot" style="background:${colourFor(c, indicOrder)}"></i>${NAMES[CLASSES[c]]}</span>`).join("") +
    `<span><i class="dot" style="background:${COLOURS.sil}"></i>${NAMES.sil}</span>`;
}

function wavBlob(samples) {
  const n = samples.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  return new Blob([buf], { type: "audio/wav" });
}

function pooledAudio(wav, segs, name) {
  const by = new Map();
  for (const [s, d, c] of segs) {
    const part = wav.slice(Math.floor(s * SR), Math.floor((s + d) * SR));
    const f = Math.min(Math.floor(0.01 * SR), part.length >> 1);
    for (let i = 0; i < f; i++) { const g = i / f; part[i] *= g; part[part.length - 1 - i] *= g; }
    if (!by.has(c)) by.set(c, []);
    by.get(c).push(part);
  }
  const box = $("pooled");
  box.innerHTML = "";
  for (const [c, parts] of by) {
    const total = parts.reduce((a, p) => a + p.length, 0);
    const all = new Float32Array(total);
    let o = 0;
    for (const p of parts) { all.set(p, o); o += p.length; }
    const url = URL.createObjectURL(wavBlob(all));
    const div = document.createElement("div");
    div.innerHTML = `<p><b>${NAMES[CLASSES[c]]}</b> <small>(${(total / SR).toFixed(1)} s)</small></p>` +
      `<audio controls src="${url}"></audio><p><a download="${name}_${CLASSES[c].toUpperCase()}.wav" href="${url}">Download ${name}_${CLASSES[c].toUpperCase()}.wav</a></p>`;
    box.appendChild(div);
  }
}

// ---- English check: Whisper-tiny language detector on sliding 4 s windows (ONNX, fp32) ----
// For every 4 s window (one every 2 s) the detector gives the probability that the window is English. A speech bin that the main model
// calls an Indian language is switched to English when the English probability around it is at least 0.3. Settings were chosen on
// 6 real recordings (English + Malayalam / Kannada) and checked on 6 others.
const LID_PARTS = ["model/lid_fp32.part0", "model/lid_fp32.part1"];
const LID_CACHE = "ildiar-lid-v1";
const LID_MB = 72;
const LID_EN = 18;                      // position of <|en|> among Whisper's 99 language tokens
const LID_WIN = 4 * SR, LID_HOP = 2 * SR, LID_THETA = 0.3;
const N_FFT = 400, FHOP = 160, N_MELS = 80, N_FRAMES = 3000;
let lidSession = null, melFilters = null, dftCos = null, dftSin = null, hannWin = null, lastEngWindows = 0;

async function getLidSession() {
  if (lidSession) return lidSession;
  const bytes = await fetchParts(LID_PARTS, LID_CACHE, "downloading the English check (first use only)", LID_MB);
  setStatus("preparing the English check...");
  ort.env.wasm.wasmPaths = "./";
  ort.env.wasm.numThreads = wasmThreads();
  lidSession = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return lidSession;
}

async function loadMelTables() {
  if (melFilters) return;
  const r = await fetch("mel_filters.bin");
  if (!r.ok) throw new Error("could not load the mel filter bank");
  melFilters = new Float32Array(await r.arrayBuffer());            // [201][80]
  dftCos = new Float32Array(201 * N_FFT);
  dftSin = new Float32Array(201 * N_FFT);
  for (let k = 0; k < 201; k++) for (let n = 0; n < N_FFT; n++) {
    const a = (2 * Math.PI * ((k * n) % N_FFT)) / N_FFT;
    dftCos[k * N_FFT + n] = Math.cos(a); dftSin[k * N_FFT + n] = Math.sin(a);
  }
  hannWin = new Float32Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) hannWin[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
}

// Whisper log-Mel input [80][3000] for one window (zero-padded to 30 s), same recipe as the reference feature extractor
function melWindow(x) {
  const L = x.length;
  const F = Math.min(N_FRAMES, Math.ceil((L + 200) / FHOP));
  const padded = new Float32Array(200 + L + 400);
  for (let i = 0; i < 200; i++) padded[i] = x[Math.min(200 - i, L - 1)];     // reflect on the left, zeros on the right
  padded.set(x, 200);
  const out = new Float32Array(N_MELS * N_FRAMES).fill(-10);                 // log10(1e-10) for the all-zero padding frames
  const frame = new Float32Array(N_FFT), pw = new Float32Array(201);
  for (let f = 0; f < F; f++) {
    const off = f * FHOP;
    for (let n = 0; n < N_FFT; n++) frame[n] = padded[off + n] * hannWin[n];
    for (let k = 0; k < 201; k++) {
      let re = 0, im = 0;
      const o = k * N_FFT;
      for (let n = 0; n < N_FFT; n++) { const v = frame[n]; re += v * dftCos[o + n]; im += v * dftSin[o + n]; }
      pw[k] = re * re + im * im;
    }
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      for (let k = 0; k < 201; k++) s += pw[k] * melFilters[k * N_MELS + m];
      out[m * N_FRAMES + f] = Math.log10(Math.max(s, 1e-10));
    }
  }
  let mx = -Infinity;
  for (let i = 0; i < out.length; i++) if (out[i] > mx) mx = out[i];
  const floor = mx - 8;
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], floor) + 4) / 4;
  return out;
}

// probability of English at each 200 ms bin, interpolated between the window centres
async function englishProbability(wav, nBins) {
  const sess = await getLidSession();
  await loadMelTables();
  const starts = [];
  for (let s = 0; s + LID_WIN <= wav.length; s += LID_HOP) starts.push(s);
  if (!starts.length) starts.push(0);
  const centres = [], pen = [];
  for (let k = 0; k < starts.length; k++) {
    setStatus(`checking for English: part ${k + 1} of ${starts.length}...`, k / starts.length);
    await new Promise((r) => setTimeout(r, 10));
    const x = wav.slice(starts[k], starts[k] + LID_WIN);
    const res = await sess.run({ mel: new ort.Tensor("float32", melWindow(x), [1, N_MELS, N_FRAMES]) });
    pen.push(res[Object.keys(res)[0]].data[LID_EN]);
    centres.push((starts[k] + LID_WIN / 2) / SR);
  }
  lastEngWindows = starts.length;
  const out = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) {
    const t = (i + 0.5) * BIN;
    if (t <= centres[0]) { out[i] = pen[0]; continue; }
    if (t >= centres[centres.length - 1]) { out[i] = pen[pen.length - 1]; continue; }
    let j = 0;
    while (centres[j + 1] < t) j++;
    const a = (t - centres[j]) / (centres[j + 1] - centres[j]);
    out[i] = pen[j] * (1 - a) + pen[j + 1] * a;
  }
  return out;
}

// switch speech bins that the main model calls an Indian language to English where the English probability is high
async function applyEnglishCheck(wav, labels, theta = LID_THETA) {
  const pen = await englishProbability(wav, labels.length);
  let changed = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== SIL && labels[i] !== ENG && pen[i] >= theta) { labels[i] = ENG; changed++; }
  }
  return changed;
}

function audioName(file) {
  return (file.name || "audio").replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 60) || "audio";
}

async function loadAudio(file, maxSeconds) {
  setStatus("reading the audio...");
  const wav = await decodeAudio(file);
  if (wav.length / SR > maxSeconds) throw new Error(`This page accepts up to ${maxSeconds / 60} minutes of audio${maxSeconds === MAX_SECONDS ? "" : " with this model"}; yours is ${(wav.length / SR / 60).toFixed(1)} minutes.`);
  if (wav.length / SR < 1) throw new Error("The audio is shorter than one second.");
  return wav;
}

// show the result cards (both models use them): picture, RTTM, pooled audio and, when a reference RTTM is loaded, the DER
function showResult(wav, segs, rttm, name, summary) {
  lastName = name;
  lastRttm = rttm;
  $("rttm").textContent = rttm || "(no speech found)";
  $("summary").textContent = summary;
  const wlv = +$("wl").value;
  drawPlot(wav, segs, Math.max(0, +$("ws").value || 0), wlv > 0 ? Math.max(2, wlv) : Math.min(wav.length / SR, 60));
  pooledAudio(wav, segs, name);
  $("results").hidden = false;
  updateDer();
}

async function run(file, opts) {
  $("err").textContent = "";
  $("results").hidden = true;
  const l1 = (opts && opts.l1) || $("l1").value, l2 = (opts && opts.l2) || $("l2").value;
  const given = [l1, l2].filter((l) => l && l !== "auto").map((l) => CLASSES.indexOf(l));
  if (new Set(given).size !== given.length) throw new Error("Choose two different languages (or leave them on auto-detect).");
  const wav = await loadAudio(file, MAX_SECONDS);
  await getSession();                       // first use downloads the model; not counted in the processing time
  const t = performance.now();
  const post = await posteriors(wav);
  const { labels, ids } = decode(post, given);
  const useEng = opts && opts.eng !== undefined ? opts.eng : $("engchk").checked;
  let engBins = 0;
  if (useEng) {
    try { engBins = await applyEnglishCheck(wav, labels); }
    catch (e) { $("err").textContent = "The English check could not run (" + (e.message || e) + "); showing the result without it."; }
  }
  const segs = toSegments(labels);
  const name = audioName(file);
  const engNote = useEng ? ` English check (v4) ran on ${lastEngWindows} windows: ${(engBins * BIN).toFixed(1)} s switched to English.` : "  (English check is off.)";
  const summary = (ids.length
    ? "Detected: English, " + ids.map((c) => NAMES[CLASSES[c]]).join(", ") + `  (processed in ${((performance.now() - t) / 1000).toFixed(0)} s)`
    : "No Indian language detected.") + engNote;
  showResult(wav, segs, makeRttm(segs, name), name, summary);
  setStatus("done.");
  return { segs, ids, post };
}

// ---- Whisper-Linear-LD (model by Mohit, github.com/MohitAiry/Whisper-Linear-LD) ----
// Frozen Whisper-large-v3 encoder (blocks 1-28 as four int8 ONNX shards) + linear probes + Viterbi; see wlld.js for the pipeline.
const WL_MAX_SECONDS = 180;                                       // one 30 s window takes about 1-2 minutes in a browser
const WL_SHARDS = [{ parts: 4, mb: 168 }, { parts: 3, mb: 139 }, { parts: 3, mb: 139 }, { parts: 3, mb: 139 }];
const WL_CACHE = "wlld-model-v1";
let wlSessions = null, wlProbes = null;

async function getWlSessions() {
  if (wlSessions) return wlSessions;
  ort.env.wasm.wasmPaths = "./";
  ort.env.wasm.numThreads = wasmThreads();
  const sessions = [];
  for (let i = 0; i < WL_SHARDS.length; i++) {
    const parts = Array.from({ length: WL_SHARDS[i].parts }, (_, j) => `model/wl_shard${i}.part${j}`);
    const bytes = await fetchParts(parts, WL_CACHE, `downloading the Whisper-Linear-LD model (first use only), part ${i + 1} of ${WL_SHARDS.length}`, WL_SHARDS[i].mb);
    setStatus(`preparing the model, part ${i + 1} of ${WL_SHARDS.length}...`);
    sessions.push(await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "disabled" }));
  }
  const [pb, mb] = await Promise.all(["wlld/probe.bin", "wlld/mel128.bin"].map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error("could not load " + u); return r.arrayBuffer(); })));
  wlld.initMel(new Float32Array(mb));
  wlProbes = wlld.loadProbes(new Float32Array(pb));
  wlSessions = sessions;
  return sessions;
}

async function wlEncode(sessions, mel) {
  let t = new ort.Tensor("float32", mel, [1, 128, 3000]);
  for (const s of sessions) {
    const r = await s.run({ x: t });
    t = r[Object.keys(r)[0]];
  }
  return t.data;                                                  // block-28 output, [1500][1280]
}

async function runWl(file) {
  $("err").textContent = "";
  $("results").hidden = true;
  const wav = await loadAudio(file, WL_MAX_SECONDS);
  const sessions = await getWlSessions();                         // first use downloads the model; not counted in the processing time
  const t = performance.now();
  const { x, T } = await wlld.features(wav, (mel) => wlEncode(sessions, mel), async (k, n) => {
    const left = k > 0 ? ` about ${Math.max(1, Math.round(((performance.now() - t) / k) * (n - k) / 60000))} min left` : "";
    setStatus(`analysing with Whisper-large-v3: window ${k + 1} of ${n} (30 s each, 1-2 minutes per window)...${left}`, k / n);
    await new Promise((r) => setTimeout(r, 20));                  // let the page repaint
  });
  setStatus("decoding...");
  await new Promise((r) => setTimeout(r, 20));
  const name = audioName(file);
  const r = wlld.decode(x, T, wlProbes);
  const segs = r.segs.map(([s, e, l]) => [s, e - s, CLASSES.indexOf(l.toLowerCase())]);
  const summary = (r.langs.length
    ? "Detected: " + r.langs.map((l) => NAMES[l.toLowerCase()]).join(", ") + `  (processed in ${((performance.now() - t) / 1000).toFixed(0)} s)`
    : "No speech found.") + "  Model: Whisper-Linear-LD (5-seed ensemble). English is one of its ten languages, so no separate English check is used.";
  showResult(wav, segs, wlld.toRttm(r.segs, name), name, summary);
  setStatus("done.");
  return { segs, langs: r.langs };
}

// ---- reference RTTM and DER ----
let refTurns = null;

function pct(x) { return Number.isFinite(x) ? x.toFixed(2) + " %" : "n/a"; }
function sec(x) { return x.toFixed(1) + " s"; }

function refForScoring() {
  if (!refTurns) return null;
  const ids = ldDer.fileIds(refTurns);
  if (ids.length <= 1) return refTurns;
  const sel = $("refid").value || ids[0];
  return refTurns.filter((t) => t.file === sel);
}

function updateDer() {
  const card = $("dercard"), out = $("derout");
  const ref = refForScoring();
  if (!ref || !lastRttm) { card.hidden = true; return; }
  card.hidden = false;
  try {
    const sys = ldDer.parseRttm(lastRttm).map((t) => ({ ...t, file: "sys" }));
    const refOne = ref.map((t) => ({ ...t, file: "ref" }));
    const strict = ldDer.score(refOne, sys, { collar: 0, ignoreOverlaps: false });
    const lenient = ldDer.score(refOne, sys, { collar: 0.25, ignoreOverlaps: true });
    const f = strict.files[0];
    const rows = f.mapping.map((m) => `<tr><td>${m.ref}</td><td>${m.sys || "(none)"}</td><td>${sec(m.scored)}</td><td>${sec(m.miss)}</td><td>${sec(m.conf)}</td></tr>`).join("");
    out.innerHTML =
      `<p style="margin:0 0 4px"><b>DER, strict</b> <small>(no collar, overlapped speech scored: the official DISPLACE setting)</small>: <span style="font-size:1.5rem;font-weight:700">${pct(strict.der)}</span></p>` +
      `<p style="margin:0 0 10px"><b>DER, lenient</b> <small>(0.25 s collar, overlapped speech excluded)</small>: <span style="font-size:1.5rem;font-weight:700">${pct(lenient.der)}</span></p>` +
      `<p>Strict DER is made of <b>${pct(strict.missPct)}</b> missed speech, <b>${pct(strict.faPct)}</b> false alarm and <b>${pct(strict.confPct)}</b> wrong language, ` +
      `over ${sec(strict.scored)} of reference speech (scoring region ${f.region[0].toFixed(1)}&ndash;${f.region[1].toFixed(1)} s).</p>` +
      `<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;font-size:.9rem;min-width:420px"><thead><tr style="text-align:left"><th>Reference label</th><th>Matched system label</th><th>Reference speech</th><th>Missed</th><th>Wrong language</th></tr></thead><tbody>${rows}</tbody></table></div>` +
      `<p><small>Computed in your browser with a port of the NIST md-eval-22 scorer used by dscore (checked against it to within 0.005 DER points). Each reference label is matched to one system label by the best overlap, so anonymous labels such as L1 / L2 also work. Only speech regions are scored: silence is not a class.</small></p>`;
  } catch (e) {
    out.textContent = "Could not compute the DER: " + (e.message || e);
  }
}

async function loadReference(file) {
  const text = await file.text();
  const turns = ldDer.parseRttm(text);
  if (!turns.length) throw new Error("No RTTM lines were found in that file (expected: TYPE file-id channel start duration <NA> <NA> label ...).");
  refTurns = turns;
  const ids = ldDer.fileIds(turns), sel = $("refid");
  sel.innerHTML = ids.map((i) => `<option value="${i}">${i}</option>`).join("");
  sel.value = ids.find((i) => i === lastName || i.startsWith(lastName)) || ids[0];
  sel.hidden = ids.length <= 1;
  $("refinfo").textContent = `${file.name}: ${turns.length} segments, ${ids.length} recording${ids.length > 1 ? "s" : ""}, labels ${[...new Set(turns.map((t) => t.label))].sort().join(", ")}.`;
  updateDer();
}

function download(filename, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

let tab = "ours";
const LEAD = {
  ours: "Upload a recording and get <b>(a)</b> an RTTM file, <b>(b)</b> the waveform coloured by language, and <b>(c)</b> one pooled audio file per language. Everything runs <b>inside your browser</b>: your audio is never uploaded or stored.",
  wl: "Upload a recording and get <b>(a)</b> an RTTM file, <b>(b)</b> the waveform coloured by language, and <b>(c)</b> one pooled audio file per language. Everything runs <b>inside your browser</b>: your audio is never uploaded or stored."
};
function setTab(t) {
  tab = t;
  $("tabOurs").setAttribute("aria-selected", String(t === "ours"));
  $("tabWl").setAttribute("aria-selected", String(t === "wl"));
  document.querySelectorAll(".oursOnly").forEach((e) => { e.hidden = t !== "ours"; });
  document.querySelectorAll(".wlOnly").forEach((e) => { e.hidden = t !== "wl"; });
  $("lead").innerHTML = LEAD[t];
  $("results").hidden = true;
  $("err").textContent = "";
  setStatus("");
}

window.addEventListener("DOMContentLoaded", () => {
  fillSelects();
  setTab("ours");
  $("tabOurs").addEventListener("click", () => setTab("ours"));
  $("tabWl").addEventListener("click", () => setTab("wl"));
  $("go").addEventListener("click", async () => {
    const f = $("file").files[0];
    if (!f) { $("err").textContent = "Please choose an audio file first."; return; }
    $("go").disabled = true;
    try { await (tab === "wl" ? runWl(f) : run(f)); } catch (e) { $("err").textContent = String(e.message || e); setStatus(""); }
    $("go").disabled = false;
  });
  $("refrttm").addEventListener("change", async () => {
    const f = $("refrttm").files[0];
    if (!f) { refTurns = null; $("refinfo").textContent = ""; $("refid").hidden = true; updateDer(); return; }
    try { await loadReference(f); } catch (e) { refTurns = null; $("refinfo").textContent = ""; $("err").textContent = String(e.message || e); updateDer(); }
  });
  $("refid").addEventListener("change", updateDer);
  $("dlrttm").addEventListener("click", () => download(lastName + ".rttm", new Blob([lastRttm], { type: "text/plain" })));
  $("dlpng").addEventListener("click", () => $("plot").toBlob((b) => download(lastName + "_languages.png", b)));
});

window.ildiar = { run, runWl, decode, pickLanguages, toSegments, CLASSES, melWindow, loadMelTables, loadReference, updateDer, setTab };   // used by automated tests
