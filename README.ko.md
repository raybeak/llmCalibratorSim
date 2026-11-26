# llmCalibratorSim (한국어판)

> [English Version (README.md)](./README.md)

원본 텍스트·파일명·임베딩 벡터 유출 **$0\text{ Bytes}$ (Sovereignty & 상용 LLM ToS 준수)** 조건하에서, 오직 익명화된 시계열 성과 배열만 읽는 **Clamped LLM Meta-Calibrator**가 결정론적 **Global PID** 및 **Per-Group PID** 대조군 대비 어떤 드리프트·구조적 고장에서 실제로 우위를 가지는지 검증하는 초경량 순수 Python(`Python 3.12 + NumPy`) 시뮬레이션 테스트베드입니다.

---

## 1. 주요 기능 및 아키텍처
1. **로컬 Forward-Forward 분류기 (`ffModel.py`)**: Backpropagation 없이 동작하는 2-Layer NumPy FF 네트워크(`Shared Layer 1` + `Group-Local Layer 2 Head` with class skip-connection) 및 Jev-like Temperature Softmax 구현.
2. **3대 고장 주입 환경 (`driftEnv.py`)**: `global_decay`(전역 감쇠), `subgroup_drift`(`doc_hwpx` 그룹 특이 이동), `structural_label_swap`(`doc_slide` 내부 카테고리 역전) 및 Activity-Gated Implicit Feedback($p_{\text{noise}} = 0.05$) 시뮬레이션.
3. **엄격한 Outbound Sanitizer (`sanitizer.py`)**: 8개 화이트리스트 키(`decision_id`, `ext_enum`, `pred_cat`, `actual_cat`, `outcome`, `raw_goodness_G`, `confidence_p`, `timestamp_s`) 외 모든 키, 파일 경로, 장문 텍스트, 임베딩 벡터를 100% 차단 (`bypass` 플래그 없음).
4. **3-Way Calibrator 대조군 (`calibrators.py`, `runBenchmark.py`)**: `GlobalPidCalibrator`, `PerGroupPidCalibrator`, `ClampedLlmCalibrator`(오프라인 `DeterministicRuleOracle` 및 실 LLM `PiCliBackend`)를 비교하며 Hard Clamping 및 깨진 JSON에 대한 `noop` 폴백을 강제.

---

## 2. 벤치마크 실측 결과

### 2.1 오프라인 5-Seed 벤치마크 (`seeds=42..46`, `8 cycles`, `elapsedSec=4.409s`, `10/10 tests OK`)

| 시나리오 | Calibrator | 드리프트 전 Acc | 드리프트 직후 Acc (Cycle 1) | 최종 Acc | 최종 ECE | 회복 사이클 |
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

### 2.2 `pi-shell` 기반 실 LLM `Unit 4` 대조 결과 (`seed=42`, `6 cycles`)

| 모델 (`OPENGATEWAY/...`) | 소요 시간 | JSON 파싱률 (A / B / C) | `global_decay` Acc | `subgroup_drift` Acc | `structural_label_swap` Acc | ECE (A / B / C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *대조군: `per_group_pid`* | — | — | **0.365** | 0.948 | 0.750 | 0.596 / 0.109 / 0.198 |
| `z-ai/glm-5.3-flash-ultrafast` | **26.289s** | **1.00 / 1.00 / 1.00** | 0.354 | **0.958** | **0.948** | 0.601 / 0.058 / 0.117 |
| `deepseek/deepseek-v4.1-flash-eco` | 90.329s | 1.00 / 0.83 / 1.00 | **0.365** | 0.917 | **0.948** | 0.601 / **0.030** / 0.117 |
| `google/gemini-3.5-flash-lite` | 28.104s | **1.00 / 1.00 / 1.00** | 0.125 | 0.854 | **0.948** | 0.788 / 0.059 / 0.117 |
| `openai/gpt-4.1-nano` | 40.444s | 0.83 / 0.67 / 0.83 | 0.208 | 0.760 | 0.771 | 0.638 / 0.141 / 0.168 |

*(참고: `sionic-ai/jev-omni`는 `OPENGATEWAY`에서 `400 unsupported_endpoint: use /v1/decisions`를 반환해 `/v1/chat/completions` 기반 JSON Calibrator 역할에는 부적합함이 확인되었습니다.)*

---

## 3. 핵심 발견 (Key Findings)
1. **LLM Calibrator의 실질적 우위 구간 (`structural_label_swap`)**: 스칼라 임계치 제어기(`Global PID`, `Per-Group PID`)는 서브그룹 내 라벨 역전을 해결하지 못해 `0.738`–`0.750`에서 정체되는 반면, LLM은 `(pred_cat, actual_cat)` 혼동 쌍과 높은 `raw_goodness_G`를 근거로 구조적 역전을 진단하고 로컬 FF Replay를 발동해 **`0.929`–`0.948`**(**+19.1%p ~ +19.8%p**)로 복구했습니다.
2. **PID로 충분한 구간 (`global_decay`, `subgroup_drift`)**: 단순 전역 감쇠에서는 API 비용과 지연이 없는 PID가 동등 이상의 성능을 보였습니다(`0.577` vs `0.556` 오프라인; `0.365` 실측).
3. **Group-Local Layer 2 분리를 통한 Catastrophic Forgetting 방지**: Layer 2를 전 그룹이 공유했을 때는 `doc_slide` Replay가 타 그룹 성능을 `0.375`까지 붕괴시켰으나, `Shared Layer 1 + Group-Local Layer 2 Head` 분리 구조로 교차 망각 없이 `0.929` 복구를 달성했습니다.
4. **실 LLM 모델별 지능 하한선 확인**: `glm-5.3-flash-ultrafast`가 속도(`26.3s`), JSON 준수율(`1.00`), 정확도(`0.958` / `0.948`) 모두 1위를 기록한 반면, `gpt-4.1-nano`는 JSON 파싱 실패(`0.67`–`0.83`)와 `label_swap` 미탐(`0.771`)으로 하한선 미달이 확인되었습니다. `deepseek-eco`의 간헐적 파싱 실패(`0.83`)에서도 `noop` 폴백이 안정적으로 시스템을 보호했습니다.

---

## 4. 실행 방법 및 MDMP 문서 (`doc/`)

```bash
# 오프라인 단위 테스트 실행 (10/10)
PYTHONPATH=src python3 -m unittest discover -s tests -v

# 5-Seed 오프라인 벤치마크 실행 (<5s)
PYTHONPATH=src python3 -m llmCalibratorSim.runBenchmark

# pi-shell + envrun 기반 실 LLM Unit 4 벤치마크 실행 (평문 시크릿 디스크 미저장)
source ~/.config/myos/env-helpers.sh
envrun timeout -k 5 150 env PYTHONPATH=src python3 -m llmCalibratorSim.runBenchmark --live-pi OPENGATEWAY/z-ai/glm-5.3-flash-ultrafast < /dev/null
```

- [01_receiptOfMission.md](./doc/01_receiptOfMission.md) — Step 1: 핵심 연구 질문 및 범위
- [02_missionAnalysis.md](./doc/02_missionAnalysis.md) — Step 2: Facts vs. Assumptions, FF/Jev 수식, 3대 시나리오, MOE/MOP
- [03_coaDevelopment.md](./doc/03_coaDevelopment.md) — Step 3: COA 1 / COA 2 / COA 3
- [04_coaWargaming.md](./doc/04_coaWargaming.md) — Step 4: 적대적 워게임
- [05_coaComparison.md](./doc/05_coaComparison.md) — Step 5: 가중치 비교 및 민감도 분석
- [06_coaApproval.md](./doc/06_coaApproval.md) — Step 6: COA 3 채택 결정 및 통제 조건
- [07_ordersProduction.md](./doc/07_ordersProduction.md) — Step 7: OPORD Units 0–4 및 스키마 계약
- [08_redTeamEval.md](./doc/08_redTeamEval.md) — Red-Team 감사 보고서 (`4.83 / 5.00`)
