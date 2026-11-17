"use strict";
// Whisper-Linear-LD in the browser (model by Mohit, https://github.com/MohitAiry/Whisper-Linear-LD).
// Pipeline (infer.py): frozen Whisper-large-v3 encoder, block 28 of 32, on 30 s windows with a 25 s hop (central frames kept)
//   -> pairs of 50 Hz frames averaged to 25 Hz, rounded to float16 -> LayerNorm + Linear(1280 -> 11) probe, 5-seed posterior ensemble
//   -> session language pruning -> Viterbi with a switch penalty -> RTTM.
// This file holds everything except the encoder itself (which runs as ONNX shards, see app.js).
(function (root) {
  const SR = 16000, WIN = 1500, HOP = 1250, MARGIN = 125;      // in 20 ms encoder frames: 30 s window, 25 s hop, 2.5 s margin
  const D = 1280, NC = 11, SEEDS = 5;
  const LANGS = ["NS", "EN", "HI", "KN", "ML", "TA", "TE", "AS", "BN", "MR", "OD"];
  const N_FFT = 400, FHOP = 160, N_MELS = 128, N_FRAMES = 3000, N_SAMP = 480000;
  const PENALTY = 32, MIN_SHARE = 0.1;                          // the decoder settings every seed selected

  // ---------------------------------------------------------------- windows (infer.py: windows)
  function windows(n50) {
    const out = [];
    for (let w = 0; ; w++) {
      const s = HOP * w;
      const k0 = w === 0 ? 0 : s + MARGIN;
      const last = s + WIN >= n50;
      const k1 = last ? n50 : s + WIN - MARGIN;
      out.push([s, k0, k1]);
      if (last) return out;
    }
  }

  // ---------------------------------------------------------------- log-Mel (WhisperFeatureExtractor, 128 bands)
  let melFilt = null, cosT = null, sinT = null, hann = null;
  function initMel(filters) {                                    // filters: Float32Array [201][128]
    melFilt = filters;
    cosT = new Float32Array(201 * N_FFT); sinT = new Float32Array(201 * N_FFT);
    for (let k = 0; k < 201; k++) for (let n = 0; n < N_FFT; n++) {
      const a = (2 * Math.PI * ((k * n) % N_FFT)) / N_FFT;
      cosT[k * N_FFT + n] = Math.cos(a); sinT[k * N_FFT + n] = Math.sin(a);
    }
    hann = new Float64Array(N_FFT);
    for (let n = 0; n < N_FFT; n++) hann[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  }

  // x: samples of one window (at most 30 s); zero-padded to 30 s like the feature extractor; returns Float32Array [128][3000]
  function mel128(x) {
    const L = N_SAMP, n = Math.min(x.length, L);
    const p = new Float32Array(L + 400);                         // 200 reflected + 30 s + 200 reflected
    for (let i = 0; i < n; i++) p[200 + i] = x[i];
    for (let i = 0; i < 200; i++) { p[i] = p[200 + 200 - i]; p[200 + L + i] = p[200 + L - 2 - i]; }
    const out = new Float32Array(N_MELS * N_FRAMES);
    const frame = new Float64Array(N_FFT), pw = new Float64Array(201);
    const silent = -10;                                          // log10(1e-10) of an all-zero frame
    let mx = -Infinity;
    for (let f = 0; f < N_FRAMES; f++) {
      const off = f * FHOP;
      let any = false;
      for (let i = 0; i < N_FFT; i++) { const v = p[off + i]; frame[i] = v * hann[i]; if (v !== 0) any = true; }
      if (!any) {
        for (let m = 0; m < N_MELS; m++) out[m * N_FRAMES + f] = silent;
        if (silent > mx) mx = silent;
        continue;
      }
      for (let k = 0; k < 201; k++) {
        let re = 0, im = 0;
        const o = k * N_FFT;
        for (let i = 0; i < N_FFT; i++) { const v = frame[i]; re += v * cosT[o + i]; im += v * sinT[o + i]; }
        pw[k] = re * re + im * im;
      }
      for (let m = 0; m < N_MELS; m++) {
        let s = 0;
        for (let k = 0; k < 201; k++) s += pw[k] * melFilt[k * N_MELS + m];
        const l = Math.log10(Math.max(s, 1e-10));
        out[m * N_FRAMES + f] = l;
        if (l > mx) mx = l;
      }
    }
    const floor = mx - 8;
    for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], floor) + 4) / 4;
    return out;
  }

  // ---------------------------------------------------------------- float16 rounding (the stored features were float16)
  const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
  function roundHalf(v) {
    if (v !== v || v === Infinity || v === -Infinity) return v;
    const a = Math.abs(v);
    if (a >= 65520) return v < 0 ? -Infinity : Infinity;
    if (a < 6.103515625e-5) {                                    // subnormal half: multiples of 2^-24, ties to even
      const step = 5.960464477539063e-8, r = a / step, fl = Math.floor(r), d = r - fl;
      const t = d > 0.5 ? fl + 1 : d < 0.5 ? fl : (fl % 2 === 0 ? fl : fl + 1);
      return (v < 0 ? -1 : 1) * t * step;
    }
    _f[0] = a;
    const bits = _u[0];
    const lsb = (bits >>> 13) & 1;
    const rounded = (bits + 0xfff + lsb) & ~0x1fff;              // round to nearest even at 10 mantissa bits
    _u[0] = rounded >>> 0;
    return (v < 0 ? -1 : 1) * _f[0];
  }

  // ---------------------------------------------------------------- probes
  // buf: Float32Array(5 * (1280 + 1280 + 11 * 1280 + 11)); returns fused per-seed weights (LayerNorm affine folded into the linear layer)
  function loadProbes(buf) {
    const per = D + D + NC * D + NC, out = [];
    for (let s = 0; s < SEEDS; s++) {
      const o = s * per;
      const lw = buf.subarray(o, o + D), lb = buf.subarray(o + D, o + 2 * D);
      const W = buf.subarray(o + 2 * D, o + 2 * D + NC * D), b = buf.subarray(o + 2 * D + NC * D, o + per);
      const Wf = new Float32Array(NC * D), bf = new Float32Array(NC);
      for (let c = 0; c < NC; c++) {
        let acc = b[c];
        for (let i = 0; i < D; i++) { Wf[c * D + i] = W[c * D + i] * lw[i]; acc += W[c * D + i] * lb[i]; }
        bf[c] = acc;
      }
      out.push({ Wf, bf });
    }
    return out;
  }

  // x: Float32Array [T][1280] (25 Hz features); returns Float32Array [T][11] log of the mean of the seeds' posteriors
  function frameLogProbs(x, T, probes) {
    const lp = new Float32Array(T * NC), z = new Float32Array(D), lg = new Float64Array(NC), pr = new Float64Array(NC);
    for (let t = 0; t < T; t++) {
      const o = t * D;
      let mean = 0;
      for (let i = 0; i < D; i++) mean += x[o + i];
      mean /= D;
      let v = 0;
      for (let i = 0; i < D; i++) { const d = x[o + i] - mean; v += d * d; }
      const inv = 1 / Math.sqrt(v / D + 1e-5);
      for (let i = 0; i < D; i++) z[i] = (x[o + i] - mean) * inv;
      pr.fill(0);
      for (const { Wf, bf } of probes) {
        let mxl = -Infinity;
        for (let c = 0; c < NC; c++) {
          let s = bf[c];
          const w = c * D;
          for (let i = 0; i < D; i++) s += Wf[w + i] * z[i];
          lg[c] = s; if (s > mxl) mxl = s;
        }
        let se = 0;
        for (let c = 0; c < NC; c++) se += Math.exp(lg[c] - mxl);
        const ls = Math.log(se);
        for (let c = 0; c < NC; c++) pr[c] += Math.exp(lg[c] - mxl - ls);
      }
      for (let c = 0; c < NC; c++) lp[t * NC + c] = Math.log(Math.max(pr[c] / probes.length, 1e-12));
    }
    return lp;
  }

  // ---------------------------------------------------------------- decoder (ldiar/frame_model.py)
  function sessionLanguages(lp, T, minShare) {
    const allowed = new Uint8Array(NC).fill(1);
    const cnt = new Float64Array(NC);
    let sp = 0;
    for (let t = 0; t < T; t++) {
      let b = 0;
      for (let c = 1; c < NC; c++) if (lp[t * NC + c] > lp[t * NC + b]) b = c;
      if (b > 0) { cnt[b]++; sp++; }
    }
    if (minShare > 0 && sp) {
      let best = 1, any = false;
      for (let c = 1; c < NC; c++) {
        allowed[c] = cnt[c] / sp >= minShare ? 1 : 0;
        if (allowed[c]) any = true;
        if (cnt[c] > cnt[best]) best = c;
      }
      if (!any) allowed[best] = 1;
    }
    return allowed;
  }

  function viterbi(lpIn, T, penalty, allowed) {
    const lp = new Float32Array(lpIn);
    if (allowed) for (let t = 0; t < T; t++) for (let c = 0; c < NC; c++) if (!allowed[c]) lp[t * NC + c] = -1e9;
    const back = new Uint8Array(T * NC);
    let score = new Float32Array(NC), next = new Float32Array(NC);
    for (let c = 0; c < NC; c++) score[c] = lp[c];
    const pen = Math.fround(penalty);
    for (let t = 1; t < T; t++) {
      let best = 0;
      for (let c = 1; c < NC; c++) if (score[c] > score[best]) best = c;
      const jump = Math.fround(score[best] - pen);
      for (let c = 0; c < NC; c++) {
        const useJump = jump > score[c];
        back[t * NC + c] = useJump ? best : c;
        next[c] = Math.fround((useJump ? jump : score[c]) + lp[t * NC + c]);
      }
      const tmp = score; score = next; next = tmp;
    }
    const path = new Int32Array(T);
    let b = 0;
    for (let c = 1; c < NC; c++) if (score[c] > score[b]) b = c;
    path[T - 1] = b;
    for (let t = T - 1; t > 0; t--) path[t - 1] = back[t * NC + path[t]];
    return path;
  }

  function pathToSegments(path, hop) {
    hop = hop || 0.04;
    const segs = [];
    let s = 0;
    for (let t = 1; t <= path.length; t++) {
      if (t === path.length || path[t] !== path[s]) {
        if (path[s] > 0) segs.push([s * hop, t * hop, LANGS[path[s]]]);
        s = t;
      }
    }
    return segs;
  }

  function toRttm(segs, id) {
    return segs.map(([s, e, l]) => `SPEAKER ${id} 1 ${s.toFixed(3)} ${(e - s).toFixed(3)} <NA> <NA> ${l} <NA> <NA>`).join("\n") + (segs.length ? "\n" : "");
  }

  // ---------------------------------------------------------------- whole pipeline
  // encode(melWindow) -> Promise<Float32Array [1500][1280]> (block-28 output of the Whisper encoder); progress(k, n)
  async function features(wav, encode, progress) {
    const n50 = Math.ceil(wav.length / 320);
    const ws = windows(n50);
    const h50 = new Float32Array((n50 + (n50 % 2)) * D);
    for (let k = 0; k < ws.length; k++) {
      const [s, k0, k1] = ws[k];
      if (progress) await progress(k, ws.length);
      const H = await encode(mel128(wav.subarray(s * 320, Math.min((s + WIN) * 320, wav.length))));
      for (let r = k0; r < k1; r++) {
        const src = Math.min(r - s, 1499) * D;
        h50.set(H.subarray(src, src + D), r * D);
      }
    }
    if (n50 % 2) h50.copyWithin(n50 * D, (n50 - 1) * D, n50 * D);
    const T = (n50 + (n50 % 2)) / 2, x = new Float32Array(T * D);
    for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) x[t * D + i] = roundHalf(Math.fround(h50[2 * t * D + i] + h50[(2 * t + 1) * D + i]) / 2);
    return { x, T };
  }

  function decode(x, T, probes, penalty, minShare) {
    const lp = frameLogProbs(x, T, probes);
    const allowed = sessionLanguages(lp, T, minShare === undefined ? MIN_SHARE : minShare);
    const path = viterbi(lp, T, penalty === undefined ? PENALTY : penalty, allowed);
    const langs = [];
    for (let c = 1; c < NC; c++) if (allowed[c]) langs.push(LANGS[c]);
    return { path, segs: pathToSegments(path), langs, lp };
  }

  const api = { windows, initMel, mel128, roundHalf, loadProbes, frameLogProbs, sessionLanguages, viterbi, pathToSegments, toRttm, features, decode,
                LANGS, SR, D, NC, PENALTY, MIN_SHARE };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.wlld = api;
})(typeof self !== "undefined" ? self : this);
