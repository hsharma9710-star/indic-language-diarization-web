"use strict";
// RTTM parsing and diarization error rate (DER). Port of the NIST md-eval-22 scoring that dscore calls:
//   * scoring region = from the first onset to the last offset of reference and system together (dscore's generated UEM);
//   * optional forgiveness collar around every reference turn boundary, optional exclusion of overlapped reference speech;
//   * one-to-one optimal mapping of reference labels to system labels (so anonymous labels such as L1 / L2 work);
//   * DER = (missed + false alarm + confusion) / scored reference speech.
(function (root) {
  function parseRttm(text) {
    const turns = [];
    for (const raw of String(text).split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("SPKR-INFO") || line.startsWith(";;")) continue;
      const f = line.split(/\s+/);
      if (f.length < 8) continue;
      const start = parseFloat(f[3]), dur = parseFloat(f[4]);
      if (!(dur > 0) || !(start >= 0)) continue;
      turns.push({ file: f[1], start, end: start + dur, label: f[7].toUpperCase() });
    }
    return turns;
  }

  function fileIds(turns) {
    const s = new Map();
    for (const t of turns) s.set(t.file, (s.get(t.file) || 0) + (t.end - t.start));
    return [...s.keys()];
  }

  // Hungarian algorithm (maximise the total of w[i][j]); returns assign[i] = j or -1
  function assignMax(w, n, m) {
    const N = Math.max(n, m);
    let big = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) big = Math.max(big, w[i][j]);
    const a = [];
    for (let i = 0; i < N; i++) {
      a.push(new Float64Array(N));
      for (let j = 0; j < N; j++) a[i][j] = i < n && j < m ? big - w[i][j] : big;
    }
    const INF = 1e18, u = new Float64Array(N + 1), v = new Float64Array(N + 1), p = new Int32Array(N + 1), way = new Int32Array(N + 1);
    for (let i = 1; i <= N; i++) {
      p[0] = i;
      let j0 = 0;
      const minv = new Float64Array(N + 1).fill(INF), used = new Uint8Array(N + 1);
      do {
        used[j0] = 1;
        const i0 = p[j0];
        let delta = INF, j1 = 0;
        for (let j = 1; j <= N; j++) {
          if (used[j]) continue;
          const cur = a[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (let j = 0; j <= N; j++) {
          if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    const out = new Array(n).fill(-1);
    for (let j = 1; j <= N; j++) {
      const i = p[j] - 1, jj = j - 1;
      if (i >= 0 && i < n && jj < m) out[i] = jj;
    }
    return out;
  }

  // ref, sys: arrays of {start, end, label} of ONE recording
  function scoreOne(ref, sys, opts) {
    const collar = (opts && opts.collar) || 0;
    const ignoreOverlaps = !!(opts && opts.ignoreOverlaps);
    const all = ref.concat(sys);
    if (!all.length) return null;
    let u0 = Infinity, u1 = -Infinity;
    for (const t of all) { if (t.start < u0) u0 = t.start; if (t.end > u1) u1 = t.end; }
    const refLabels = [...new Set(ref.map((t) => t.label))].sort();
    const sysLabels = [...new Set(sys.map((t) => t.label))].sort();
    const ri = new Map(refLabels.map((l, i) => [l, i])), si = new Map(sysLabels.map((l, i) => [l, i]));

    // collar boundaries: edges of reference turns after merging touching / overlapping turns of the same label
    const refBounds = [];
    for (const l of refLabels) {
      const ts = ref.filter((t) => t.label === l).sort((x, y) => x.start - y.start);
      let cs = ts[0].start, ce = ts[0].end;
      for (let i = 1; i < ts.length; i++) {
        if (ts[i].start <= ce + 1e-6) ce = Math.max(ce, ts[i].end);
        else { refBounds.push(cs, ce); cs = ts[i].start; ce = ts[i].end; }
      }
      refBounds.push(cs, ce);
    }
    refBounds.sort((x, y) => x - y);
    const cuts = new Set([u0, u1]);
    for (const t of all) { cuts.add(t.start); cuts.add(t.end); }
    if (collar > 0) for (const b of refBounds) { cuts.add(b - collar); cuts.add(b + collar); }
    const edges = [...cuts].filter((x) => x >= u0 && x <= u1).sort((x, y) => x - y);

    // sweep: events of ref / sys turns
    const events = [];
    ref.forEach((t) => { events.push([t.start, 1, 0, ri.get(t.label)], [t.end, -1, 0, ri.get(t.label)]); });
    sys.forEach((t) => { events.push([t.start, 1, 1, si.get(t.label)], [t.end, -1, 1, si.get(t.label)]); });
    events.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    const rc = new Int32Array(refLabels.length), sc = new Int32Array(sysLabels.length);
    let e = 0;
    const ivs = [];                                              // scored elementary intervals
    for (let k = 0; k + 1 < edges.length; k++) {
      const a = edges[k], b = edges[k + 1];
      if (b <= a) continue;
      const mid = (a + b) / 2;
      while (e < events.length && events[e][0] <= mid) {
        const ev = events[e++];
        (ev[2] === 0 ? rc : sc)[ev[3]] += ev[1];
      }
      let nref = 0;
      for (let i = 0; i < rc.length; i++) if (rc[i] > 0) nref++;
      if (ignoreOverlaps && nref > 1) continue;
      if (collar > 0) {
        // nearest reference boundary to mid
        let lo = 0, hi = refBounds.length;
        while (lo < hi) { const m = (lo + hi) >> 1; if (refBounds[m] < mid) lo = m + 1; else hi = m; }
        let d = Infinity;
        if (lo < refBounds.length) d = Math.min(d, refBounds[lo] - mid);
        if (lo > 0) d = Math.min(d, mid - refBounds[lo - 1]);
        if (d < collar) continue;
      }
      const rs = [], ss = [];
      for (let i = 0; i < rc.length; i++) if (rc[i] > 0) rs.push(i);
      for (let i = 0; i < sc.length; i++) if (sc[i] > 0) ss.push(i);
      ivs.push({ d: b - a, rs, ss });
    }
    // overlap matrix and optimal one-to-one mapping
    const R = refLabels.length, S = sysLabels.length;
    const w = [];
    for (let i = 0; i < R; i++) w.push(new Float64Array(Math.max(S, 1)));
    for (const iv of ivs) for (const r of iv.rs) for (const s of iv.ss) w[r][s] += iv.d;
    const map = S ? assignMax(w, R, S) : new Array(R).fill(-1);
    let scored = 0, miss = 0, fa = 0, conf = 0;
    const perRef = refLabels.map(() => ({ scored: 0, miss: 0, conf: 0 }));
    for (const iv of ivs) {
      const nref = iv.rs.length, nsys = iv.ss.length;
      let ncor = 0;
      const sset = new Set(iv.ss);
      for (const r of iv.rs) if (map[r] >= 0 && sset.has(map[r])) ncor++;
      scored += nref * iv.d;
      miss += Math.max(0, nref - nsys) * iv.d;
      fa += Math.max(0, nsys - nref) * iv.d;
      conf += (Math.min(nref, nsys) - ncor) * iv.d;
      if (nref === 1) {
        const r = iv.rs[0], p = perRef[r];
        p.scored += iv.d;
        if (nsys === 0) p.miss += iv.d; else if (!(map[r] >= 0 && sset.has(map[r]))) p.conf += iv.d;
      }
    }
    const mapping = refLabels.map((l, i) => ({ ref: l, sys: map[i] >= 0 ? sysLabels[map[i]] : null, ...perRef[i] }));
    return { scored, miss, fa, conf, errors: miss + fa + conf, der: scored > 0 ? (100 * (miss + fa + conf)) / scored : NaN, mapping, region: [u0, u1] };
  }

  // refTurns / sysTurns: output of parseRttm (any number of file ids; files are paired by id, or the single ids are paired)
  function score(refTurns, sysTurns, opts) {
    const rf = fileIds(refTurns), sf = fileIds(sysTurns);
    let pairs = [];
    if (rf.length === 1 && sf.length === 0) pairs = [[rf[0], null]];            // empty system output: everything is missed
    else if (rf.length === 1 && sf.length === 1) pairs = [[rf[0], sf[0]]];
    else for (const f of rf) if (sf.includes(f)) pairs.push([f, f]);
    if (!pairs.length) throw new Error("The reference and system RTTM do not share a file id (reference: " + rf.join(", ") + "; system: " + sf.join(", ") + ").");
    let scored = 0, miss = 0, fa = 0, conf = 0;
    const files = [];
    for (const [a, b] of pairs) {
      const r = scoreOne(refTurns.filter((t) => t.file === a), b === null ? [] : sysTurns.filter((t) => t.file === b), opts);
      if (!r) continue;
      scored += r.scored; miss += r.miss; fa += r.fa; conf += r.conf;
      files.push({ ref: a, sys: b, ...r });
    }
    const errors = miss + fa + conf;
    return { der: scored > 0 ? (100 * errors) / scored : NaN, miss, fa, conf, scored, files,
             missPct: scored ? (100 * miss) / scored : NaN, faPct: scored ? (100 * fa) / scored : NaN, confPct: scored ? (100 * conf) / scored : NaN };
  }

  const api = { parseRttm, score, scoreOne, fileIds, assignMax };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.ldDer = api;
})(typeof self !== "undefined" ? self : this);
