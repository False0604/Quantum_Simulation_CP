/* ===== Engine: data, preprocessing, detectors, quantum simulator, metrics ===== */
const Engine = (() => {
  const PI = Math.PI;

  // ---------- seeded randomness ----------
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0; while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * PI * r());
  }
  function shuffle(arr, r) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }
  const range = n => Array.from({ length: n }, (_, i) => i);

  // ---------- synthetic regimes (as described in the brief, section 7) ----------
  function makeGaussian(dataSeed) {
    const r = rng(1000 + dataSeed), n = 350, d = 15;
    const M = range(d).map(() => range(d).map(() => 0.6 * gauss(r)));
    const Z = range(n).map(() => range(d).map(() => gauss(r)));
    const X = Z.map(z => range(d).map(j => { let s = 0; for (let t = 0; t < d; t++) s += z[t] * M[t][j]; return s; }));
    const mu = range(d).map(j => X.reduce((s, x) => s + x[j], 0) / n);
    const sig = range(d).map(j => Math.sqrt(X.reduce((s, x) => s + (x[j] - mu[j]) ** 2, 0) / (n - 1)));
    return { kind: 'gaussian', benign: X, sig, d };
  }
  function makeRing(dataSeed) {
    const r = rng(2000 + dataSeed), d = 10;
    const R = range(2).map(() => range(d).map(() => gauss(r)));
    const emb = rad => { const th = 2 * PI * r(); const p = [rad * Math.cos(th), rad * Math.sin(th)];
      return range(d).map(j => p[0] * R[0][j] + p[1] * R[1][j] + 0.05 * gauss(r)); };
    return { kind: 'ring', benign: range(350).map(() => emb(1)), attack: range(200).map(() => emb(1.6)), d };
  }

  // Train-on-normal split: N = 80 benign train, M = 50 benign test + 50 attack test
  function split(ds, seed) {
    const r = rng(77 + seed * 7919);
    const idx = shuffle(range(ds.benign.length), r);
    const train = idx.slice(0, 80).map(i => ds.benign[i]);
    const testB = idx.slice(80, 130).map(i => ds.benign[i]);
    let testA;
    if (ds.kind === 'gaussian') {
      testA = testB.map(x => x.map((v, j) => {
        if (r() < 0.6) { const e = -Math.log(1 - r()) * ds.sig[j]; return v + (r() < 0.5 ? -1 : 1) * (3.5 * ds.sig[j] + e); }
        return v;
      }));
    } else {
      testA = shuffle(range(ds.attack.length), r).slice(0, 50).map(i => ds.attack[i]);
    }
    return { train, test: testB.concat(testA), y: range(100).map(i => (i < 50 ? 0 : 1)) };
  }

  // ---------- linear algebra ----------
  function eigSym(A) { // cyclic Jacobi; returns {vals, vecs(columns)} sorted descending
    const n = A.length, a = A.map(r => r.slice());
    const V = range(n).map(i => range(n).map(j => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off < 1e-20) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    const order = range(n).sort((i, j) => a[j][j] - a[i][i]);
    return { vals: order.map(i => a[i][i]), vecs: order.map(i => V.map(row => row[i])) };
  }

  // Preprocessing fitted on benign train only: standardise, PCA to k
  function fitPreprocess(train, k) {
    const n = train.length, d = train[0].length;
    const mu = range(d).map(j => train.reduce((s, x) => s + x[j], 0) / n);
    const sd = range(d).map(j => Math.sqrt(train.reduce((s, x) => s + (x[j] - mu[j]) ** 2, 0) / n));
    const keep = range(d).filter(j => sd[j] > 1e-9);
    const z = x => keep.map(j => (x[j] - mu[j]) / sd[j]);
    const Z = train.map(z), dd = keep.length;
    const C = range(dd).map(i => range(dd).map(j => Z.reduce((s, r) => s + r[i] * r[j], 0) / (n - 1)));
    const { vals, vecs } = eigSym(C);
    const W = vecs.slice(0, k);
    const total = vals.reduce((s, v) => s + Math.max(v, 0), 0);
    const explained = vals.slice(0, k).reduce((s, v) => s + v, 0) / total;
    const transform = X => X.map(x => { const zx = z(x); return W.map(w => w.reduce((s, wi, i) => s + wi * zx[i], 0)); });
    return { transform, explained };
  }

  // ---------- metrics ----------
  function rocAuc(scores, y) {
    const idx = range(scores.length).sort((a, b) => scores[a] - scores[b]);
    const ranks = new Array(scores.length);
    for (let i = 0; i < idx.length;) {
      let j = i; while (j + 1 < idx.length && scores[idx[j + 1]] === scores[idx[i]]) j++;
      const rk = (i + j) / 2 + 1; for (let t = i; t <= j; t++) ranks[idx[t]] = rk; i = j + 1;
    }
    let nP = 0, sum = 0; y.forEach((v, i) => { if (v) { nP++; sum += ranks[i]; } });
    const nN = y.length - nP;
    return (sum - nP * (nP + 1) / 2) / (nP * nN);
  }
  function avgPrecision(scores, y) {
    const idx = range(scores.length).sort((a, b) => scores[b] - scores[a]);
    const nP = y.reduce((s, v) => s + v, 0);
    let tp = 0, ap = 0;
    idx.forEach((i, r) => { if (y[i]) { tp++; ap += tp / (r + 1); } });
    return ap / nP;
  }
  function rocCurve(scores, y) {
    const idx = range(scores.length).sort((a, b) => scores[b] - scores[a]);
    const nP = y.reduce((s, v) => s + v, 0), nN = y.length - nP;
    const pts = [[0, 0]]; let tp = 0, fp = 0;
    idx.forEach((i, r) => { if (y[i]) tp++; else fp++; if (r === idx.length - 1 || scores[idx[r + 1]] !== scores[i]) pts.push([fp / nN, tp / nP]); });
    return pts;
  }
  function f1(pred, y) {
    let tp = 0, fp = 0, fn = 0;
    pred.forEach((p, i) => { if (p && y[i]) tp++; else if (p && !y[i]) fp++; else if (!p && y[i]) fn++; });
    return tp === 0 ? 0 : (2 * tp) / (2 * tp + fp + fn);
  }

  // ---------- one-class SVM (libsvm-style SMO on a precomputed kernel) ----------
  function ocsvm(K, nu) {
    const l = K.length, C = 1, total = nu * l;
    const alpha = new Float64Array(l);
    let rem = total;
    for (let i = 0; i < l && rem > 0; i++) { alpha[i] = Math.min(1, rem); rem -= alpha[i]; }
    const G = new Float64Array(l);
    for (let i = 0; i < l; i++) { let s = 0; for (let j = 0; j < l; j++) s += K[i][j] * alpha[j]; G[i] = s; }
    for (let it = 0; it < 20000; it++) {
      let i = -1, j = -1, gmax = -Infinity, gmin = Infinity;
      for (let t = 0; t < l; t++) {
        if (alpha[t] < C - 1e-12 && -G[t] > gmax) { gmax = -G[t]; i = t; }
        if (alpha[t] > 1e-12 && -G[t] < gmin) { gmin = -G[t]; j = t; }
      }
      if (i < 0 || j < 0 || gmax - gmin < 1e-6) break;
      const eta = Math.max(K[i][i] + K[j][j] - 2 * K[i][j], 1e-12);
      let d = (G[j] - G[i]) / eta;
      d = Math.min(d, C - alpha[i], alpha[j]);
      if (d <= 0) break;
      alpha[i] += d; alpha[j] -= d;
      for (let t = 0; t < l; t++) G[t] += d * (K[t][i] - K[t][j]);
    }
    let sum = 0, cnt = 0, ub = Infinity, lb = -Infinity;
    for (let t = 0; t < l; t++) {
      if (alpha[t] > 1e-9 && alpha[t] < C - 1e-9) { sum += G[t]; cnt++; }
      else if (alpha[t] <= 1e-9) ub = Math.min(ub, G[t]); else lb = Math.max(lb, G[t]);
    }
    const rho = cnt ? sum / cnt : (isFinite(ub) && isFinite(lb) ? (ub + lb) / 2 : 0);
    // score = rho - sum alpha K  (higher = more anomalous); anomaly if score > 0
    return { alpha, rho, score: krow => { let s = 0; for (let t = 0; t < l; t++) if (alpha[t]) s += alpha[t] * krow[t]; return rho - s; } };
  }

  // ---------- classical detectors ----------
  const sqd = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return s; };

  function rbfOCSVM(Xtr) {
    const k = Xtr[0].length, all = Xtr.flat(), m = all.reduce((s, v) => s + v, 0) / all.length;
    const variance = all.reduce((s, v) => s + (v - m) ** 2, 0) / all.length;
    const gamma = 1 / (k * variance);
    const kern = (a, b) => Math.exp(-gamma * sqd(a, b));
    const K = Xtr.map(a => Xtr.map(b => kern(a, b)));
    const m1 = ocsvm(K, 0.1);
    const score = X => X.map(x => m1.score(Xtr.map(t => kern(x, t))));
    return { score, threshold: 0, K, kern };
  }

  function cFactor(n) { return n > 2 ? 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1)) / n : n === 2 ? 1 : 0; }
  function isolationForest(Xtr, seed) {
    const r = rng(500 + seed), psi = Math.min(256, Xtr.length), hlim = Math.ceil(Math.log2(psi)), k = Xtr[0].length;
    const build = (rows, depth) => {
      if (depth >= hlim || rows.length <= 1) return { size: rows.length };
      const q = Math.floor(r() * k);
      let lo = Infinity, hi = -Infinity; rows.forEach(x => { lo = Math.min(lo, x[q]); hi = Math.max(hi, x[q]); });
      if (hi - lo < 1e-12) return { size: rows.length };
      const p = lo + r() * (hi - lo);
      return { q, p, left: build(rows.filter(x => x[q] < p), depth + 1), right: build(rows.filter(x => x[q] >= p), depth + 1) };
    };
    const trees = range(100).map(() => build(shuffle(Xtr, r).slice(0, psi), 0));
    const path = (x, node, d) => (node.size !== undefined ? d + cFactor(node.size) : path(x, x[node.q] < node.p ? node.left : node.right, d + 1));
    const cn = cFactor(psi);
    const score = X => X.map(x => Math.pow(2, -trees.reduce((s, t) => s + path(x, t, 0), 0) / trees.length / cn));
    return { score, threshold: 0.5 };
  }

  function lof(Xtr, kn = 20) {
    const n = Xtr.length;
    const nbrs = x => Xtr.map((t, i) => [Math.sqrt(sqd(x, t)), i]).sort((a, b) => a[0] - b[0]);
    const trN = Xtr.map((x, i) => nbrs(x).filter(p => p[1] !== i).slice(0, kn));
    const kdist = trN.map(nb => nb[kn - 1][0]);
    const lrd = trN.map(nb => 1 / (nb.reduce((s, [d, o]) => s + Math.max(kdist[o], d), 0) / kn + 1e-10));
    const score = X => X.map(x => {
      const nb = nbrs(x).slice(0, kn);
      const l = 1 / (nb.reduce((s, [d, o]) => s + Math.max(kdist[o], d), 0) / kn + 1e-10);
      return nb.reduce((s, [, o]) => s + lrd[o], 0) / kn / l;
    });
    return { score, threshold: 1.5 };
  }

  function autoencoder(Xtr, seed) {
    const r = rng(900 + seed), k = Xtr[0].length, b = Math.min(3, k - 1);
    const sizes = [k, 8, b, 8, k];
    const L = sizes.length - 1;
    const W = [], B = [];
    for (let l = 0; l < L; l++) {
      const lim = Math.sqrt(6 / sizes[l]);
      W.push(range(sizes[l + 1]).map(() => range(sizes[l]).map(() => (r() * 2 - 1) * lim * 0.7)));
      B.push(new Array(sizes[l + 1]).fill(0));
    }
    const forward = x => { const acts = [x]; let h = x;
      for (let l = 0; l < L; l++) { const z = W[l].map((w, o) => w.reduce((s, wi, i) => s + wi * h[i], B[l][o]));
        h = l < L - 1 ? z.map(v => (v > 0 ? v : 0)) : z; acts.push(h); }
      return acts; };
    const err = x => { const o = forward(x)[L]; return o.reduce((s, v, i) => s + (v - x[i]) ** 2, 0) / k; };
    const idx = shuffle(range(Xtr.length), r), nVal = Math.floor(Xtr.length * 0.2);
    const val = idx.slice(0, nVal).map(i => Xtr[i]), tr = idx.slice(nVal).map(i => Xtr[i]);
    const mW = W.map(w => w.map(row => row.map(() => 0))), vW = W.map(w => w.map(row => row.map(() => 0)));
    const mB = B.map(b2 => b2.map(() => 0)), vB = B.map(b2 => b2.map(() => 0));
    const lr = 0.005, b1 = 0.9, b2c = 0.999; let step = 0;
    let best = Infinity, bestW = null, bestB = null, bad = 0;
    for (let ep = 0; ep < 400; ep++) {
      const order = shuffle(range(tr.length), r);
      for (let s0 = 0; s0 < order.length; s0 += 16) {
        const batch = order.slice(s0, s0 + 16).map(i => tr[i]);
        const gW = W.map(w => w.map(row => row.map(() => 0))), gB = B.map(b2 => b2.map(() => 0));
        batch.forEach(x => {
          const acts = forward(x);
          let delta = acts[L].map((v, i) => (2 * (v - x[i])) / k / batch.length);
          for (let l = L - 1; l >= 0; l--) {
            const hin = acts[l];
            delta.forEach((dv, o) => { gB[l][o] += dv; for (let i = 0; i < hin.length; i++) gW[l][o][i] += dv * hin[i]; });
            if (l > 0) delta = hin.map((hv, i) => (hv > 0 ? delta.reduce((s, dv, o) => s + dv * W[l][o][i], 0) : 0));
          }
        });
        step++;
        for (let l = 0; l < L; l++) {
          for (let o = 0; o < W[l].length; o++) {
            for (let i = 0; i < W[l][o].length; i++) {
              const g = gW[l][o][i];
              mW[l][o][i] = b1 * mW[l][o][i] + (1 - b1) * g; vW[l][o][i] = b2c * vW[l][o][i] + (1 - b2c) * g * g;
              W[l][o][i] -= (lr * mW[l][o][i] / (1 - b1 ** step)) / (Math.sqrt(vW[l][o][i] / (1 - b2c ** step)) + 1e-8);
            }
            const g = gB[l][o];
            mB[l][o] = b1 * mB[l][o] + (1 - b1) * g; vB[l][o] = b2c * vB[l][o] + (1 - b2c) * g * g;
            B[l][o] -= (lr * mB[l][o] / (1 - b1 ** step)) / (Math.sqrt(vB[l][o] / (1 - b2c ** step)) + 1e-8);
          }
        }
      }
      const vl = val.reduce((s, x) => s + err(x), 0) / val.length;
      if (vl < best - 1e-6) { best = vl; bestW = JSON.stringify(W); bestB = JSON.stringify(B); bad = 0; }
      else if (++bad >= 40) break;
    }
    if (bestW) { JSON.parse(bestW).forEach((w, l) => (W[l] = w)); JSON.parse(bestB).forEach((b3, l) => (B[l] = b3)); }
    const trErr = Xtr.map(err).sort((a, c) => a - c);
    const threshold = trErr[Math.floor(0.95 * (trErr.length - 1))];
    return { score: X => X.map(err), threshold };
  }

  // ---------- quantum statevector simulator ----------
  function zeroState(nq) { const d = 1 << nq, re = new Float64Array(d), im = new Float64Array(d); re[0] = 1; return { re, im, d }; }
  function H(s, q) { const b = 1 << q, h = Math.SQRT1_2;
    for (let i = 0; i < s.d; i++) if (!(i & b)) { const j = i | b, ar = s.re[i], ai = s.im[i], br = s.re[j], bi = s.im[j];
      s.re[i] = (ar + br) * h; s.im[i] = (ai + bi) * h; s.re[j] = (ar - br) * h; s.im[j] = (ai - bi) * h; } }
  function P(s, q, th) { const b = 1 << q, c = Math.cos(th), sn = Math.sin(th);
    for (let i = 0; i < s.d; i++) if (i & b) { const r0 = s.re[i], i0 = s.im[i]; s.re[i] = r0 * c - i0 * sn; s.im[i] = r0 * sn + i0 * c; } }
  function CX(s, c, t) { const cb = 1 << c, tb = 1 << t;
    for (let i = 0; i < s.d; i++) if ((i & cb) && !(i & tb)) { const j = i | tb;
      let tmp = s.re[i]; s.re[i] = s.re[j]; s.re[j] = tmp; tmp = s.im[i]; s.im[i] = s.im[j]; s.im[j] = tmp; } }
  function RY(s, q, th) { const b = 1 << q, c = Math.cos(th / 2), sn = Math.sin(th / 2);
    for (let i = 0; i < s.d; i++) if (!(i & b)) { const j = i | b, ar = s.re[i], ai = s.im[i], br = s.re[j], bi = s.im[j];
      s.re[i] = c * ar - sn * br; s.im[i] = c * ai - sn * bi; s.re[j] = sn * ar + c * br; s.im[j] = sn * ai + c * bi; } }
  function RZ(s, q, th) { const b = 1 << q;
    for (let i = 0; i < s.d; i++) { const ph = (i & b) ? th / 2 : -th / 2, c = Math.cos(ph), sn = Math.sin(ph), r0 = s.re[i], i0 = s.im[i];
      s.re[i] = r0 * c - i0 * sn; s.im[i] = r0 * sn + i0 * c; } }

  // Qiskit ZZFeatureMap, linear entanglement: H, P(2x_j), [CX, P(2(pi-x_j)(pi-x_j+1)), CX]
  function zzState(x, reps, entangle = true) {
    const nq = x.length, s = zeroState(nq);
    for (let r = 0; r < reps; r++) {
      for (let q = 0; q < nq; q++) H(s, q);
      for (let q = 0; q < nq; q++) P(s, q, 2 * x[q]);
      if (entangle) for (let q = 0; q < nq - 1; q++) { CX(s, q, q + 1); P(s, q + 1, 2 * (PI - x[q]) * (PI - x[q + 1])); CX(s, q, q + 1); }
    }
    return s;
  }
  function trainableState(x, theta) {
    const nq = x.length, s = zeroState(nq);
    for (let q = 0; q < nq; q++) { RY(s, q, x[q]); RZ(s, q, 2 * x[q]); }
    for (let q = 0; q < nq - 1; q++) CX(s, q, q + 1);
    if (nq > 2) CX(s, nq - 1, 0);
    for (let q = 0; q < nq; q++) { RY(s, q, theta[2 * q]); RZ(s, q, theta[2 * q + 1]); }
    return s;
  }
  function fidelity(a, b) { let r = 0, i = 0; for (let t = 0; t < a.d; t++) { r += a.re[t] * b.re[t] + a.im[t] * b.im[t]; i += a.re[t] * b.im[t] - a.im[t] * b.re[t]; } return r * r + i * i; }

  // ---------- quantum encodings ----------
  function percentile(sorted, p) { const pos = (sorted.length - 1) * p, lo = Math.floor(pos), hi = Math.ceil(pos); return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo); }
  function angleScaler(Xtr) {
    const k = Xtr[0].length;
    const q = range(k).map(j => { const col = Xtr.map(x => x[j]).sort((a, b) => a - b); return [percentile(col, 0.01), percentile(col, 0.99)]; });
    return X => X.map(x => x.map((v, j) => Math.min(PI, Math.max(0, (PI * (v - q[j][0])) / (q[j][1] - q[j][0] || 1)))));
  }
  function bandwidthScaler(Xtr, lambda) {
    const k = Xtr[0].length;
    const sd = range(k).map(j => { const m = Xtr.reduce((s, x) => s + x[j], 0) / Xtr.length; return Math.sqrt(Xtr.reduce((s, x) => s + (x[j] - m) ** 2, 0) / Xtr.length) || 1; });
    return X => X.map(x => x.map((v, j) => (lambda * v) / sd[j]));
  }

  const Q_CONFIGS = [
    { id: 'tuned', label: 'ZZ, bandwidth λ, 2 layers', short: 'Bandwidth-scaled', reps: 2, enc: 'bw', nu: 0.1, future: true },
    { id: 'zz2', label: 'ZZ, angle, 2 layers', reps: 2, enc: 'angle', nu: 0.1 },
    { id: 'train', label: 'Trainable, angle', enc: 'angle', nu: 0.1, trainable: true },
    { id: 'zonly', label: 'Z-only, angle', reps: 1, enc: 'angle', nu: 0.1, noEnt: true },
    { id: 'zznoang', label: 'ZZ, no angle', reps: 1, enc: 'raw', nu: 0.1 },
    { id: 'zzang', label: 'ZZ, angle', reps: 1, enc: 'angle', nu: 0.1 },
    { id: 'zznu', label: 'ZZ, angle, ν = 0.3', reps: 1, enc: 'angle', nu: 0.3 },
  ];

  // median heuristic on benign training rows only: choose lambda so the median off-diagonal kernel value is about 0.5
  function autoLambda(Xtr, reps) {
    const sample = Xtr.slice(0, 40);
    const med = lam => { const enc = bandwidthScaler(Xtr, lam)(sample).map(x => zzState(x, reps));
      const v = []; for (let i = 0; i < enc.length; i++) for (let j = i + 1; j < enc.length; j++) v.push(fidelity(enc[i], enc[j]));
      v.sort((a, b) => a - b); return v[Math.floor(v.length / 2)]; };
    let lo = 0.01, hi = 1.5;
    for (let it = 0; it < 22; it++) { const mid = Math.sqrt(lo * hi); if (med(mid) > 0.5) lo = mid; else hi = mid; }
    return Math.sqrt(lo * hi);
  }

  function quantumDetector(cfg, Xtr, opts, seed) {
    const k = Xtr[0].length, dim = 1 << k, p = opts.noise || 0;
    let encode, stateOf, lambdaUsed = null, trainLog = null;
    if (cfg.enc === 'raw') encode = X => X;
    else if (cfg.enc === 'angle') encode = angleScaler(Xtr);
    else { lambdaUsed = opts.lambdaAuto ? autoLambda(Xtr, cfg.reps) : opts.lambda; encode = bandwidthScaler(Xtr, lambdaUsed); }

    if (cfg.trainable) {
      const r = rng(4000 + seed);
      let theta = range(2 * k).map(() => (r() * 2 - 1) * PI);
      const enc20 = encode(Xtr.slice(0, 20));
      const pairs = range(15).map(() => { const a = Math.floor(r() * 20); let b = Math.floor(r() * 19); if (b >= a) b++; return [a, b]; });
      const obj = th => pairs.reduce((s, [a, b]) => s + fidelity(trainableState(enc20[a], th), trainableState(enc20[b], th)), 0) / pairs.length;
      trainLog = [obj(theta)];
      for (let it = 0; it < 15; it++) {
        const grad = theta.map((_, i) => { const tp = theta.slice(), tm = theta.slice(); tp[i] += PI / 2; tm[i] -= PI / 2; return (obj(tp) - obj(tm)) / 2; });
        theta = theta.map((t, i) => t + 0.1 * grad[i]);
        trainLog.push(obj(theta));
      }
      stateOf = x => trainableState(x, theta);
    } else stateOf = x => zzState(x, cfg.reps, !cfg.noEnt);

    const noisy = f => (1 - p) * (1 - p) * f + (2 * (1 - p) * p) / dim + (p * p) / dim;
    const refStates = encode(Xtr).map(stateOf);
    const K = refStates.map(a => refStates.map(b => noisy(fidelity(a, b))));
    const model = ocsvm(K, cfg.nu);
    const score = X => encode(X).map(stateOf).map(s => model.score(refStates.map(rs => noisy(fidelity(s, rs)))));
    return { score, threshold: 0, K, lambdaUsed, trainLog };
  }

  function effectiveRank(K) {
    const vals = eigSym(K).vals.map(v => Math.max(v, 0)), tot = vals.reduce((s, v) => s + v, 0);
    let h = 0; vals.forEach(v => { const q = v / tot; if (q > 1e-15) h -= q * Math.log(q); });
    const cond = vals[0] / Math.max(vals[vals.length - 1], 1e-300);
    return { erank: Math.exp(h), cond, vals };
  }

  const CLASSICAL = [
    { id: 'ae', label: 'Autoencoder', make: (X, s) => autoencoder(X, s) },
    { id: 'lof', label: 'LOF', make: X => lof(X) },
    { id: 'if', label: 'Isolation Forest', make: (X, s) => isolationForest(X, s) },
    { id: 'rbf', label: 'RBF OCSVM', make: X => rbfOCSVM(X) },
  ];

  const dsCache = {};
  function dataset(kind, dataSeed) {
    const key = kind + dataSeed;
    return dsCache[key] || (dsCache[key] = kind === 'gaussian' ? makeGaussian(dataSeed) : makeRing(dataSeed));
  }

  // One full run: every detector, every seed, identical inputs per seed
  function run(opts) {
    const ds = dataset(opts.regime, opts.dataSeed || 0);
    const seeds = range(opts.seeds);
    const res = {}; const keep = {};
    const add = (id, sc, y, thr) => {
      (res[id] = res[id] || { auc: [], ap: [], f1: [] });
      res[id].auc.push(rocAuc(sc, y)); res[id].ap.push(avgPrecision(sc, y)); res[id].f1.push(f1(sc.map(v => v > thr), y));
    };
    let explained = 0;
    seeds.forEach(seed => {
      const sp = split(ds, seed), pre = fitPreprocess(sp.train, opts.k);
      const Xtr = pre.transform(sp.train), Xte = pre.transform(sp.test);
      explained += pre.explained / seeds.length;
      CLASSICAL.forEach(c => { const m = c.make(Xtr, seed), sc = m.score(Xte); add(c.id, sc, sp.y, m.threshold);
        if (seed === 0) keep[c.id] = { model: m, scores: sc }; });
      Q_CONFIGS.forEach(cfg => { const m = quantumDetector(cfg, Xtr, opts, seed), sc = m.score(Xte); add(cfg.id, sc, sp.y, m.threshold);
        if (seed === 0) keep[cfg.id] = { model: m, scores: sc }; else if (m.lambdaUsed) keep[cfg.id].lambdas = (keep[cfg.id].lambdas || []).concat(m.lambdaUsed); });
      if (seed === 0) Object.assign(keep, { Xtr, Xte, y: sp.y });
    });
    const stat = a => { const m = a.reduce((s, v) => s + v, 0) / a.length; return { mean: m, std: Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) }; };
    const summary = {};
    Object.entries(res).forEach(([id, r]) => (summary[id] = { auc: stat(r.auc), ap: stat(r.ap), f1: stat(r.f1) }));
    return { summary, keep, explained };
  }

  return { run, rocCurve, effectiveRank, Q_CONFIGS, CLASSICAL, zzState, fidelity, rng };
})();
if (typeof module !== 'undefined') module.exports = Engine;
