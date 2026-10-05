# Quantum Kernel Lab: live IoT anomaly detection demo

An interactive, single-page demonstration of our Conceptual Project benchmark: **does a quantum fidelity kernel detect IoT anomalies better than classical detectors?** Every live number on the page is computed in the browser, from scratch, each time an input changes.

Open `index.html` directly in a browser. No server, Python or install is needed (fonts load from Google Fonts when online).

## What the page shows

| Section | Content |
|---|---|
| 01 Playground | Four classical detectors and seven quantum kernel configurations, trained on benign rows only and scored on identical test rows. Inputs: data regime, 2 or 4 qubits, 1/3/5 seeds and the first seed, one-class SVM ν, autoencoder weight decay, depolarising noise p, bandwidth λ, focus detector, new data draw. Outputs: ROC-AUC bars (with the paper's reported value as a tick); a seed selector for the ROC curve, score map and kernel matrices, with an option to overlay every seed and the mean ROC; a live ROC cross-check; and a paper-versus-live table. |
| 02 Latency race | Replays the measured median latency per sample (paper Table R4): RBF OCSVM 42.8 µs, autoencoder 43.8 µs, LOF 60.6 µs, Isolation Forest 271.1 µs, quantum 105,164 µs (2 qubits) and 224,468 µs (4 qubits). |
| 03 Classical execution run | Figures from the first end-to-end run of the classical detectors (`run_all.py`, `summarize_results.py`). These came from a **synthetic N-BaIoT-shaped device** and are illustrative only. |
| 04 Method | What is computed live, what is copied from the paper, and the limitations. |

## What is computed live

Implemented in plain JavaScript in `src/engine.js`:

- **Data**, following the research brief: a 15-feature Gaussian-shift regime (attack rows copy benign test rows and shift about 60% of features by ±(3.5σ + e)), and a ring manifold embedded in 10 dimensions (attacks on a circle of radius 1.6).
- **Protocol**: per seed, 80 benign training rows, 50 benign plus 50 attack test rows. Standardisation and PCA (k = number of qubits) are fitted on benign training rows only.
- **Classical detectors**: RBF one-class SVM (ν = 0.1, γ = 'scale'), Isolation Forest (100 trees), LOF (20 neighbours, novelty mode), shallow autoencoder k → 8 → b → 8 → k with Adam and early stopping.
- **Quantum detectors**: exact statevector simulation of Qiskit's ZZFeatureMap (linear entanglement), the Z-only map and the trainable map (parameter-shift training), fidelity kernel |⟨Φ(x)|Φ(y)⟩|², closed-form depolarising noise, and a one-class SVM solved by SMO on the precomputed kernel.

### The bandwidth-scaled kernel (exploratory)

Section 12 of the research brief lists bandwidth-scaled quantum kernels and replacing hard clipping as future work. The page adds that configuration, **"ZZ, bandwidth λ, 2 layers"**: standardised PCA inputs multiplied by λ, no clipping. By default λ is chosen from benign training rows only, so that the median benign kernel value is about 0.5. Attack rows never tune it.

It was **not** part of the paper's experiments and has no reported value. It must be presented as an exploratory, in-browser result.

### Live results with the page defaults (3 seeds, data draw 0)

| Setting | Best classical | Best paper-configuration quantum | Bandwidth-scaled quantum |
|---|---|---|---|
| Ring manifold, 2 qubits | 1.000 (RBF OCSVM) | 0.759 (ZZ, angle, 2 layers) | 0.976 |
| Gaussian shift, 2 qubits | 0.871 (LOF) | 0.680 (Trainable, angle) | 0.841 |
| Gaussian shift, 4 qubits | 0.969 (LOF) | 0.855 (Z-only, angle) | 0.951 |
| Ring manifold, 4 qubits | 1.000 (RBF OCSVM) | 0.607 (ZZ, no angle) | 0.830 |

ROC-AUC, mean over seeds. The bandwidth-scaled kernel comes close to the best classical detector but stays below it in all four settings. Raising λ towards 0.9 drops it to about chance, which shows kernel concentration.

### Different seed sets

The same comparison with other split seeds (3 seeds each, ROC-AUC mean):

| Setting | Seeds | Best classical | Best paper-configuration quantum | Bandwidth-scaled quantum |
|---|---|---|---|---|
| Ring, 2 qubits | 0 to 2 / 10 to 12 / 100 to 102 | 1.000 / 1.000 / 1.000 | 0.759 / 0.805 / 0.770 | 0.976 / 0.993 / 0.999 |
| Gaussian, 2 qubits | 0 to 2 / 10 to 12 / 100 to 102 | 0.871 / 0.826 / 0.846 | 0.680 / 0.670 / 0.597 | 0.841 / 0.795 / 0.836 |
| Ring, 4 qubits | 0 to 2 / 10 to 12 / 100 to 102 | 1.000 / 1.000 / 1.000 | 0.607 / 0.551 / 0.579 | 0.830 / 0.847 / 0.836 |
| Gaussian, 4 qubits | 0 to 2 / 10 to 12 / 100 to 102 | 0.969 / 0.952 / 0.951 | 0.855 / 0.832 / 0.802 | 0.951 / 0.930 / 0.917 |

On single seeds the bandwidth-scaled kernel can edge past a classical detector (for example seed 1, Gaussian 2 qubits: 0.907 against LOF 0.899); averaged over seeds it stays below the best classical detector in every case above.

### Regularisation

The playground exposes two regularisation settings. Defaults match the paper (ν = 0.1, no weight decay); changing them shows a warning next to the paper comparison.

- **One-class SVM ν**, shared by the RBF and quantum SVMs (the "ν = 0.3" variant keeps 0.3). Gaussian, 2 qubits, ν = 0.05 / 0.1 / 0.2 / 0.3 / 0.5: RBF OCSVM 0.780 / 0.783 / 0.857 / 0.878 / 0.875, bandwidth-scaled quantum 0.798 / 0.841 / 0.869 / 0.873 / 0.855. On the 4-qubit ring a larger ν lowers the quantum score (0.839 at ν = 0.05 to 0.799 at ν = 0.5) while RBF stays at 1.000.
- **Autoencoder L2 weight decay**. On the 4-qubit ring it mostly stabilises training: 0.853 ± 0.108 with none, 0.929 ± 0.016 with 1e-2. On the Gaussian regime it changes little (0.804 to 0.782).

### Verification

- **ROC cross-check.** For every detector, seed and setting, the area under the plotted ROC curve (trapezoid rule) equals the rank-based ROC-AUC used in the bar chart: 396 curves checked, 0 mismatches. The page shows this check live under the ROC chart.
- **Normalisation.** Standardisation (population standard deviation, as in scikit-learn) and PCA are fitted on benign training rows only and then applied to test rows. PCA outputs are centred but not whitened, as in scikit-learn; each quantum encoding rescales per component from training rows only (1st/99th percentiles for angle scaling, standard deviation for bandwidth scaling). ROC-AUC depends only on the ranking of scores, so detector scores need no further normalisation.

Two behaviours are worth pointing out in a presentation:

- **Noise does not move ROC-AUC.** The closed-form channel rescales and shifts every kernel value equally, so the SVM's ranking cannot change. Only the kernel matrix loses contrast. This matches the paper's finding that noise was not the bottleneck.
- **Live values land near, not on, the reported ones.** The random draws differ from the Python runs, and the live runs use all 80 training rows as kernel references (the paper's sweep used 60).

## Reported values

The "Reported in the paper" ticks and table are copied unchanged from the research brief:

- Table R3, E3 variant sweep (Gaussian 2 qubits, ring 2 and 4 qubits), 3 seeds, commit `96198ac`.
- Table R2, E2 ablation (Gaussian 4 qubits, p = 0), 2 seeds, commit `2779ca9`.
- Table R4, latency.

All results are on synthetic data. Real N-BaIoT and TON-IoT results are still pending. All quantum results are classical simulations; no quantum hardware was used.

## Repository layout

```
index.html          the demo (built file, self-contained apart from images and fonts)
assets/             figures from the classical execution run (synthetic demo data)
src/engine.js       data generators, preprocessing, detectors, quantum simulator, metrics
src/template.html   page layout, styles and UI code; the engine is inlined at /*__ENGINE__*/
src/build.py        rebuilds index.html from the two source files
.claude/launch.json optional local preview server (python -m http.server 8765)
```

After editing anything in `src/`, run `python src/build.py` to regenerate `index.html`.

## Related repositories

- Quantum benchmark and paper experiments: [False0604/Quantum_Kernel_CP](https://github.com/False0604/Quantum_Kernel_CP), branch `claude/quantum-anomaly-detectors-iot-9n5gn0`.
- Classical detectors: `False0604/CP_Classical_Anomaly`, branches `claude/anomaly-detectors-python-cdwkgw` and `claude/classical-anomaly-detectors-1eudfd`.
