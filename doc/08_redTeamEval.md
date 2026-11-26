# Red-Team & Evaluation Report (`llmCalibratorSim` MDMP Cycle)

- **Scope**: `README.md`, `.gitignore`, `doc/01` ~ `doc/07`
- **Method**: `mdmp-redteam` Adversarial Audit (실험 공정성, Sovereignty/Secret 보안, 제어 안정성, MDMP 교리 적합성 점검)

---

## 1. Red-Team Findings (`mdmp-redteam` Semicolon Format)
`finding;severity;attackOrFailureVector;mitigation;status`

- `RT-01 (Strawman Baseline Bias)`;`Critical`;Comparing LLM Calibrator only against a Global PID exaggerates LLM value on subgroup drift because a simple Per-Group PID handles scalar per-group shifts cheaply;Mandated `PerGroupPidCalibrator` as a hard control and added `Scenario C (Structural Label-Swap)` where scalar threshold PID fails structurally;`Mitigated in Design`
- `RT-02 (Secret & Raw Data Leakage)`;`Critical`;Accidental logging of file paths, raw text, embeddings in telemetry, or writing plaintext API keys during live LLM testing breaks Sovereignty and Zero-Disk policy;Enforced strict 8-key whitelist `sanitizer.py` gated in `Unit 2`, banned bypass flags, and restricted `Unit 4` keys strictly to Ollama or `envrun` env-injection;`Mitigated in Design / Unit 2 Gate`
- `RT-03 (LLM Hallucination & Divergence)`;`High`;Live LLM returns malformed JSON, markdown wrappers, or extreme parameter jumps ($\theta = 99$, $T \le 0$) crashing or destabilizing the FF classifier;Implemented strict JSON extraction, schema validation, hard per-cycle/absolute clamping, and automatic `noop / PID` fallback tested by fuzzing in `Unit 3`;`Mitigated in Design / Unit 3 Gate`
- `RT-04 (Implicit Feedback False-TP Trap)`;`High`;Marking untouched predictions as `TP` solely after elapsed wall-clock time floods logs with false `TP` during user idle periods, blinding the calibrator;Replaced wall-clock idle rule with activity-gated confirmation ($M$ subsequent actions) plus $p_{\text{noise}} = 0.05$ injected feedback noise in `driftEnv.py`;`Mitigated in Design`
- `RT-05 (Mock Oracle Overfitting)`;`Medium`;A hand-crafted `DeterministicRuleOracle` in `Unit 3` can artificially ace `Scenario B/C` in ways a real noisy LLM prompt cannot reproduce;Separated `CalibratorBackend` interface and gated `Unit 4` live LLM smoke test on identical scenarios to measure parse rate and real-vs-oracle delta;`Open / Unit 4 Gate`
- `RT-06 (FF Pre-Drift Brittleness)`;`Medium`;If the 2-layer NumPy FF baseline starts poorly calibrated or under-trained ($< 0.88$ acc), post-drift recovery deltas become meaningless noise;Enforced `MOE 2` gate in `Unit 1` (pre-drift accuracy $\ge 0.88$, $\text{ECE} \le 0.10$) before running `Unit 3` comparative benchmarks;`Open / Unit 1 Gate`

---

## 2. Evaluation Scorecard (1~5점)
- **Technical Accuracy**: **4.7 / 5.0** (NumPy FF 수식, Group Loss, Hard Clamping, JSON 계약 명확)
- **Scientific Rigor**: **4.8 / 5.0** (`PerGroupPidCalibrator` 대조군 + 3종 시나리오 + 5-seed + Noise 주입으로 공정 비교 설계)
- **Internal Consistency**: **4.9 / 5.0** (`01`~`07`, `README` 간 MOE/MOP, 파일명, 스키마, 점수(`4.415`/`3.660`/`4.710`) 완전 일치)
- **Sovereignty & Secret Safety**: **4.9 / 5.0** (8-key 화이트리스트 Sanitizer + Zero-Disk `.env` 차단 명시)
- **Feasibility & Speed**: **4.9 / 5.0** (기설치 `Python 3.12 + NumPy`만으로 즉시 `Unit 0` 착수 가능)
- **MDMP Doctrine Fit**: **4.8 / 5.0** (Full-Cycle 완료; Facts/Assumptions, 3개 편향 없는 COA, 전 COA 워게임, 민감도 분석, Atomic OPORD Units 완비)
- **Overall Score**: **4.83 / 5.00**

---

## 3. Verdict
- **Verdict: `GO` for `Units 0–3`; `CONDITIONAL GO` for `Unit 4`**
  - **GO Immediately**: `Unit 0` ~ `Unit 3` (패키지 초기화, NumPy FF + 3종 드리프트, Sanitizer 단위 테스트, 3-way 5-seed 벤치마크) 즉시 구현 및 검증 착수 가능.
  - **Conditional Gate (`Unit 4`)**: 실 LLM(`Ollama` 또는 `envrun` API) 연동은 반드시 `Unit 2` 적대적 Sanitizer 100% 통과 및 Zero-Disk 시크릿 주입 확인 후에만 실행.
