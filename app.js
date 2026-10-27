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
const CLASSES = ["sil", "en", "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "pa", "sa", "ta", "te", "ur"];
const NAMES = { en: "English", as: "Assamese", bn: "Bengali", gu: "Gujarati", hi: "Hindi", kn: "Kannada", ml: "Malayalam",
                mr: "Marathi", ne: "Nepali", pa: "Punjabi", sa: "Sanskrit", ta: "Tamil", te: "Telugu", ur: "Urdu", sil: "silence / no speech" };
const SIL = 0, ENG = 1;
const COLOURS = { en: "#3b82f6", sil: "#9ca3af", order: ["#ec4899", "#22c55e", "#f59e0b"] };

const $ = (id) => document.getElementById(id);
let session = null;
let lastRttm = "", lastName = "audio";

function fillSelects() {
  for (const id of ["l1", "l2"]) {
    const s = $(id);
    s.innerHTML = '<option value="auto">auto-detect</option>' +
      CLASSES.slice(2).map((c) => `<option value="${c}">${NAMES[c]}</option>`).join("");
  }
}

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
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 2, 4) : 1;
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

function makeRttm(segs, name) {
  return segs.map(([st, du, c]) =>
    `LANGUAGE ${name} 1 ${st.toFixed(3)} ${du.toFixed(3)} <NA> <NA> ${CLASSES[c].toUpperCase()} <NA> <NA>`).join("\n") + (segs.length ? "\n" : "");
}

function colourFor(c, indicOrder) {
  if (c === ENG) return COLOURS.en;
  const i = indicOrder.indexOf(c);
  return COLOURS.order[Math.min(i, 2)];
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

// ---- English check: an English speech recogniser (wav2vec2-base-960h, int8 ONNX) tells us where real English words are spoken ----
// A bin that the main model calls an Indian language is switched to English when at least 25 % of the surrounding ~4 s is made of
// recognised English words of 4+ letters (or common short words). Settings were chosen on 6 real recordings and checked on 6 others.
const ASR_PARTS = ["model/asr_int8.part0", "model/asr_int8.part1", "model/asr_int8.part2"];
const ASR_CACHE = "ildiar-asr-v1";
const ASR_MB = 122;
const ASR_VOCAB = ["<pad>", "<s>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D", "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z"].map((c, i) => c);
const ASR_CHUNK = 20 * SR;
const STOP_WORDS = new Set("to of is in on it as at be by he we so or an do go no my me up if am us the and but for you are was not can had has his her him she who did get got let say see too two one out our all any how why now yes".split(" "));
let asrSession = null, wordSet = null;

async function getAsrSession() {
  if (asrSession) return asrSession;
  const bytes = await fetchParts(ASR_PARTS, ASR_CACHE, "downloading the English check (first use only)", ASR_MB);
  setStatus("preparing the English check...");
  ort.env.wasm.wasmPaths = "./";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 2, 4) : 1;
  asrSession = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return asrSession;
}

async function getWordSet() {
  if (wordSet) return wordSet;
  const r = await fetch("english_words.txt");
  if (!r.ok) throw new Error("could not load the English word list");
  wordSet = new Set((await r.text()).split(/\s+/).filter(Boolean));
  return wordSet;
}

// recognised words with start / end times (seconds), from greedy CTC decoding of the 20 ms frames
async function recognisedWords(wav) {
  const sess = await getAsrSession();
  const words = [];
  const nChunks = Math.ceil(wav.length / ASR_CHUNK);
  for (let k = 0; k < nChunks; k++) {
    const a = k * ASR_CHUNK;
    const x = wav.slice(a, a + ASR_CHUNK);
    if (x.length < 1600) continue;
    setStatus(`checking for English: part ${k + 1} of ${nChunks}...`, k / nChunks);
    await new Promise((r) => setTimeout(r, 20));
    let mean = 0;
    for (let i = 0; i < x.length; i++) mean += x[i];
    mean /= x.length;
    let v = 0;
    for (let i = 0; i < x.length; i++) { const d = x[i] - mean; v += d * d; }
    const sd = Math.sqrt(v / x.length) + 1e-7;
    const nx = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) nx[i] = (x[i] - mean) / sd;
    const res = await sess.run({ wav: new ort.Tensor("float32", nx, [1, nx.length]) });
    const lg = res[Object.keys(res)[0]];
    const T = lg.dims[1], C = lg.dims[2];
    let cur = "", start = 0, last = 0, prev = -1;
    for (let t = 0; t < T; t++) {
      let best = 0, bv = -Infinity;
      for (let c = 0; c < C; c++) { const val = lg.data[t * C + c]; if (val > bv) { bv = val; best = c; } }
      if (best === prev) continue;
      prev = best;
      const ch = ASR_VOCAB[best];
      if (best <= 3) continue;                                   // <pad> (blank), <s>, </s>, <unk>
      if (ch === "|") {
        if (cur) { words.push([a / SR + start * 0.02, a / SR + (last + 1) * 0.02, cur]); cur = ""; }
        continue;
      }
      if (!cur) start = t;
      cur += ch.toLowerCase();
      last = t;
    }
    if (cur) words.push([a / SR + start * 0.02, a / SR + (last + 1) * 0.02, cur]);
  }
  return words;
}

function englishEvidence(words, nBins, ws, minLen = 4, smooth = 20) {
  const v = new Float64Array(nBins);
  for (const [s, e, w] of words) {
    if ((ws.has(w) && w.length >= minLen) || STOP_WORDS.has(w)) {
      const b1 = Math.min(Math.ceil(e / BIN), nBins);
      for (let b = Math.floor(s / BIN); b < b1; b++) v[b] += Math.max(0, Math.min(e, (b + 1) * BIN) - Math.max(s, b * BIN)) / BIN;
    }
  }
  for (let i = 0; i < nBins; i++) v[i] = Math.min(v[i], 1);
  // moving average over `smooth` bins, same alignment as numpy.convolve(..., mode="same")
  const out = new Float64Array(nBins);
  const left = Math.floor((smooth - 1) / 2), right = smooth - 1 - left;
  for (let i = 0; i < nBins; i++) {
    let sum = 0;
    for (let j = i - right; j <= i + left; j++) if (j >= 0 && j < nBins) sum += v[j];
    out[i] = sum / smooth;
  }
  return out;
}

// switch speech bins to English where the English evidence is strong; returns how many bins changed
async function applyEnglishCheck(wav, labels, tau = 0.25) {
  const ws = await getWordSet();
  const words = await recognisedWords(wav);
  const ev = englishEvidence(words, labels.length, ws);
  let changed = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== SIL && labels[i] !== ENG && ev[i] >= tau) { labels[i] = ENG; changed++; }
  }
  return changed;
}

async function run(file, opts) {
  $("err").textContent = "";
  $("results").hidden = true;
  const l1 = (opts && opts.l1) || $("l1").value, l2 = (opts && opts.l2) || $("l2").value;
  const given = [l1, l2].filter((l) => l && l !== "auto").map((l) => CLASSES.indexOf(l));
  if (new Set(given).size !== given.length) throw new Error("Choose two different languages (or leave them on auto-detect).");
  setStatus("reading the audio...");
  const wav = await decodeAudio(file);
  if (wav.length / SR > MAX_SECONDS) throw new Error(`This page accepts up to ${MAX_SECONDS / 60} minutes of audio; yours is ${(wav.length / SR / 60).toFixed(1)} minutes.`);
  if (wav.length / SR < 1) throw new Error("The audio is shorter than one second.");
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
  const name = (file.name || "audio").replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 60) || "audio";
  lastName = name;
  lastRttm = makeRttm(segs, name);
  $("rttm").textContent = lastRttm || "(no speech found)";
  const engNote = useEng ? ` English check: ${(engBins * BIN).toFixed(1)} s switched to English.` : "";
  $("summary").textContent = (ids.length
    ? "Detected: English, " + ids.map((c) => NAMES[CLASSES[c]]).join(", ") + `  (processed in ${((performance.now() - t) / 1000).toFixed(0)} s)`
    : "No Indian language detected.") + engNote;
  drawPlot(wav, segs, Math.max(0, +$("ws").value || 0), Math.max(2, +$("wl").value || 20));
  pooledAudio(wav, segs, name);
  $("results").hidden = false;
  setStatus("done.");
  return { segs, ids, post };
}

function download(filename, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

window.addEventListener("DOMContentLoaded", () => {
  fillSelects();
  $("go").addEventListener("click", async () => {
    const f = $("file").files[0];
    if (!f) { $("err").textContent = "Please choose an audio file first."; return; }
    $("go").disabled = true;
    try { await run(f); } catch (e) { $("err").textContent = String(e.message || e); setStatus(""); }
    $("go").disabled = false;
  });
  $("dlrttm").addEventListener("click", () => download(lastName + ".rttm", new Blob([lastRttm], { type: "text/plain" })));
  $("dlpng").addEventListener("click", () => $("plot").toBlob((b) => download(lastName + "_languages.png", b)));
});

window.ildiar = { run, decode, pickLanguages, toSegments, CLASSES };   // used by automated tests
