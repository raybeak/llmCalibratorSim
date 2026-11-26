# llmCalibratorSim

> [한국어 버전 (README.ko.md)](./README.ko.md)

A lightweight, pure-Python (`Python 3.12 + NumPy`) simulation testbed evaluating whether a **Sovereignty-Preserving, Clamped LLM Meta-Calibrator**—reading only a sanitized, anonymized timestamped telemetry array (`0 Bytes` of raw text, filenames, or embeddings egressed)—outperforms deterministic **Global PID** and **Per-Group PID** baselines on a local 2-layer Forward-Forward (FF) classifier under controlled drift and structural failure modes.

---

## 1. What It Does
1. **Local Forward-Forward Classifier (`ffModel.py`)**: 2-layer NumPy FF network (`Shared Layer 1` + `Group-Local Layer 2 Head` with class skip-connection) and Jev-like temperature-scaled Softmax, trained locally without backpropagation.
2. **Controlled Failure Environment (`driftEnv.py`)**: Injects three failure modes—`global_decay`, `subgroup_drift` (`doc_hwpx` only), and `structural_label_swap` (`doc_slide` category inversion)—paired with activity-gated implicit feedback ($p_{\text{noise}} = 0.05$).
3. **Strict Outbound Sanitizer (`sanitizer.py`)**: Enforces an 8-key whitelist (`decision_id`, `ext_enum`, `pred_cat`, `actual_cat`, `outcome`, `raw_goodness_G`, `confidence_p`, `timestamp_s`) with zero bypass flags; blocks raw paths, text, or embedding vectors before any external backend call.
4. **3-Way Calibrator Showdown (`calibrators.py`, `runBenchmark.py`)**: Compares `GlobalPidCalibrator`, `PerGroupPidCalibrator`, and `ClampedLlmCalibrator` (offline `DeterministicRuleOracle` or live `PiCliBackend` via `pi-shell`) with hard parameter clamping and automatic `noop` fallback on malformed JSON.

---

## 2. Benchmark Results

### 2.1 Offline 5-Seed Benchmark (`seeds=42..46`, `8 cycles`, `elapsedSec=4.409s`, `10/10 tests OK`)

| Scenario | Calibrator | Pre-Drift Acc | Post-Drift Acc (Cycle 1) | Final Acc | Final ECE | Recovery Cycles |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `global_decay` | `global_pid` | 0.980 | 0.983 | **0.577** | 0.249 | 1.00 |
| `global_decay` | `per_group_pid` | 0.980 | 0.983 | 0.569 | **0.184** | 1.00 |
| `global_decay` | `clamped_llm` (Oracle) | 0.980 | 0.983 | 0.556 | 0.234 | 1.00 |
| `subgroup_drift` | `global_pid` | 0.980 | 0.742 | 0.967 | 0.109 | 7.20 |
| `subgroup_drift` | `per_group_pid` | 0.980 | 0.742 | 0.981 | 0.110 | 6.20 |
| `subgroup_drift` | `clamped_llm` (Oracle) | 0.980 | 0.742 | **0.988** | **0.077** | **4.40** |
| `structural_label_swap` | `global_pid` | 0.980 | 0.737 | 0.540 | 0.387 | 8.00 |
| `structural_label_swap` | `per_group_pid` | 0.980 | 0.737 | 0.738 | 0.206 | 8.00 |
| `structural_label_swap` | `clamped_llm` (Oracle) | 0.980 | 0.737 | **0.929** | **0.141** | **7.00** |

### 2.2 Live Unit 4 Model Showdown via `pi-shell` (`seed=42`, `6 cycles`)

| Model (`OPENGATEWAY/...`) | Elapsed | Parse Rate (A / B / C) | `global_decay` Acc | `subgroup_drift` Acc | `structural_label_swap` Acc | ECE (A / B / C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *Baseline: `per_group_pid`* | — | — | **0.365** | 0.948 | 0.750 | 0.596 / 0.109 / 0.198 |
| `z-ai/glm-5.3-flash-ultrafast` | **26.289s** | **1.00 / 1.00 / 1.00** | 0.354 | **0.958** | **0.948** | 0.601 / 0.058 / 0.117 |
| `deepseek/deepseek-v4.1-flash-eco` | 90.329s | 1.00 / 0.83 / 1.00 | **0.365** | 0.917 | **0.948** | 0.601 / **0.030** / 0.117 |
| `google/gemini-3.5-flash-lite` | 28.104s | **1.00 / 1.00 / 1.00** | 0.125 | 0.854 | **0.948** | 0.788 / 0.059 / 0.117 |
| `openai/gpt-4.1-nano` | 40.444s | 0.83 / 0.67 / 0.83 | 0.208 | 0.760 | 0.771 | 0.638 / 0.141 / 0.168 |

*(Note: `sionic-ai/jev-omni` was also tested and rejected by `OPENGATEWAY` with `400 unsupported_endpoint: use /v1/decisions`, confirming it is a discrete decision endpoint rather than a `/v1/chat/completions` JSON calibrator.)*

---

## 3. Key Findings
1. **Where an LLM Calibrator Actually Wins (`structural_label_swap`)**: Scalar threshold controllers (`Global PID`, `Per-Group PID`) cannot fix structural label inversions inside a subgroup (`0.738`–`0.750`), whereas reading `(pred_cat, actual_cat)` confusion pairs alongside `raw_goodness_G` lets the LLM diagnose the swap and trigger targeted local FF replay (**`0.929`–`0.948`**, **+19.1%p to +19.8%p** over Per-Group PID).
2. **Where PID Is Enough (`global_decay`, `subgroup_drift`)**: On pure scalar threshold decay, PID matches or beats LLMs with zero latency/API cost (`0.577` vs `0.556` offline; `0.365` live).
3. **Group-Local Layer 2 Prevents Catastrophic Forgetting**: When Layer 2 was shared across all groups, replaying `doc_slide` collapsed global accuracy to `0.375`; splitting into `Shared Layer 1 + Group-Local Layer 2 Head` eliminated cross-group interference (`0.929`).
4. **Clear Capability Floor Across Live Models**: `glm-5.3-flash-ultrafast` led on speed (`26.3s`), JSON reliability (`1.00`), and accuracy (`0.958` / `0.948`), while `gpt-4.1-nano` fell below the capability floor (`0.67`–`0.83` parse rate, `0.771` on label swap). Even when `deepseek-eco` failed 1 of 6 JSON parses (`0.83`), `noop` fallback kept the system stable.

---

## 4. Quick Usage & MDMP Docs

```bash
# Run offline unit tests (10/10)
PYTHONPATH=src python3 -m unittest discover -s tests -v

# Run 5-seed offline benchmark (<5s)
PYTHONPATH=src python3 -m llmCalibratorSim.runBenchmark

# Run live Unit 4 benchmark via pi-shell + envrun (zero-disk secret injection)
source ~/.config/myos/env-helpers.sh
envrun timeout -k 5 150 env PYTHONPATH=src python3 -m llmCalibratorSim.runBenchmark --live-pi OPENGATEWAY/z-ai/glm-5.3-flash-ultrafast < /dev/null
```

### MDMP Documentation (`doc/`)
- [01_receiptOfMission.md](./doc/01_receiptOfMission.md) — Step 1: Core Research Question & Scope
- [02_missionAnalysis.md](./doc/02_missionAnalysis.md) — Step 2: Facts vs. Assumptions, FF/Jev Math, 3 Scenarios, MOE/MOP
- [03_coaDevelopment.md](./doc/03_coaDevelopment.md) — Step 3: COA 1 / COA 2 / COA 3
- [04_coaWargaming.md](./doc/04_coaWargaming.md) — Step 4: Adversarial Wargaming
- [05_coaComparison.md](./doc/05_coaComparison.md) — Step 5: Weighted Comparison & Sensitivity Analysis
- [06_coaApproval.md](./doc/06_coaApproval.md) — Step 6: COA 3 Approval & Controls
- [07_ordersProduction.md](./doc/07_ordersProduction.md) — Step 7: OPORD Units 0–4 & Interface Contracts
- [08_redTeamEval.md](./doc/08_redTeamEval.md) — Red-Team Audit Report (`4.83 / 5.00`)
