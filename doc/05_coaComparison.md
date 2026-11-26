# Step 5: COA Comparison & Sensitivity Analysis (방책 비교 및 민감도 분석)

## 1. Structured COA Comparison (`mdmp-redteam` Format)
`coa;coreApproach;pros;cons;resourceCost;primaryRisk`

- `COA 1`;`Pure Synthetic NumPy Core + Deterministic Rule Oracle Only`;Fastest execution (<2s), 100% reproducible, zero API cost or key exposure;Rule Oracle cannot test real LLM JSON failures or prompt ambiguity;`Python 3.12 + NumPy ($0)`;Mock Oracle overfitting giving false confidence about live LLM stability
- `COA 2`;`Real Local Embeddings + Live LLM API First`;Immediate real-world LLM and embedding telemetry;High risk of premature data/secret leak before sanitizer testing, slow/non-reproducible benchmarks;`API token cost + local embedding setup`;Sovereignty breach or noisy debugging stalling core control experiments
- `COA 3`;`Hybrid Deterministic Core (Units 0-3) + Plug-in Live Adapter (Unit 4)`;Combines <10s deterministic 5-seed benchmarks with Sanitizer-gated real LLM verification;Slightly higher initial interface boilerplate (`DatasetSource`, `CalibratorBackend`);`Python 3.12 + NumPy ($0 core; optional Unit 4 API/Ollama)`;Adapter drift between offline Rule Oracle and live LLM prompt schema

---

## 2. Weighted Evaluation Matrix

### 가중치 (Pinned to `cmd intent`)
- **Sovereignty & Secret Safety Verification**: 30%
- **Scientific Rigor & Reproducibility**: 25%
- **Implementation Speed & Minimal Bloat**: 25%
- **Live LLM / Real-Data Extensibility**: 20%

### 가중치 점수 (1~5점 척도, Semicolon-delimited)
- **COA 1 (`Offline Core + Rule Oracle Only`)**:
  Sovereignty 4.8 (1.440); Scientific Rigor 4.5 (1.125); Speed/Zero-Bloat 5.0 (1.250); Extensibility 3.0 (0.600); **Total = 4.415 / 5.00**
- **COA 2 (`Real Embeddings + Live API First`)**:
  Sovereignty 3.5 (1.050); Scientific Rigor 3.8 (0.950); Speed/Zero-Bloat 2.8 (0.700); Extensibility 4.8 (0.960); **Total = 3.660 / 5.00**
- **COA 3 (`Hybrid Deterministic Core + Plug-in Live Adapter`)**:
  Sovereignty 4.9 (1.470); Scientific Rigor 4.8 (1.200); Speed/Zero-Bloat 4.4 (1.100); Extensibility 4.7 (0.940); **Total = 4.710 / 5.00**

---

## 3. Sensitivity Analysis (민감도 분석)
- **Live LLM / Real-Data Extensibility 가중치를 $0\%$로 제외(나머지 $80\%$ 기준 정규화) 시**:
  - **COA 1**: $3.815 / 0.80 = \mathbf{4.769}$ (오프라인 합성 대조군 실험만 영구 수행할 경우 근소 우위)
  - **COA 3**: $3.770 / 0.80 = 4.713$
- **결론**: 오프라인 재현성과 실 LLM API 안정성 검증을 모두 달성해야 하는 본 PoC 목표상 **COA 3 (4.710)**가 최적이며, `Units 0–3`만 단독 실행해도 `COA 1`의 모든 이점을 그대로 확보할 수 있습니다.
