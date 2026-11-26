# Step 2: Mission Analysis (임무 분석)

## 1. Facts vs. Assumptions Register

### Verified Facts (검증된 사실)
- **로컬 실행 환경**: 호스트에 `Python 3.12.16`, `numpy 2.5.5`, `uv`가 이미 설치되어 있음;
- **연산 규모**: 순수 NumPy 2-Layer Forward-Forward ($d = 32 \to 32 \to 16$, $K = 4$ 카테고리, $4$ `ext_enum` 그룹) 모델은 수천 스텝 시뮬레이션을 $< 2\text{ s}$ 이내에 완료 가능함;
- **대조군 설계 원칙**: 단순 스칼라 드리프트($\theta$ 이동)에서는 결정론적 PID/EMA가 속도·안정성에서 우위임. 따라서 Global PID만 비교하는 Strawman 실험을 피하려면 **Per-Group PID**를 필수 대조군(Hard Control)으로 포함해야 함.

### Assumptions (Unit 1–4에서 검증할 가정)
- **A1 (고장 대표성)**: 3가지 주입 고장(Global Decay, Subgroup Concept Drift, Structural Label-Swap Shift)이 실제 로컬 파일/엣지 분류 환경에서 발생하는 주요 드리프트 패턴을 대표함;
- **A2 (잡음 근사)**: Activity-Gated Implicit Feedback(후속 사용자 조작 $M$회 후 미수정 시 `TP` 확정, 오인 잡음 $p_{\text{noise}} = 0.05$)이 실제 비명시적 피드백 노이즈를 근사함;
- **A3 (실 LLM 검증 경로)**: `Unit 4`에서 로컬 Ollama 또는 `envrun`으로 주입한 외부 OpenAI 호환 LLM API를 호출해 오프라인 Oracle 결과를 교차 검증할 수 있음.

---

## 2. Mathematical Formulation (Group-Aware FF & Calibration)
입력 $x \in \mathbb{R}^{32}$와 선택지 코드 $c_k \in \mathbb{R}^{4}$를 결합한 입력 $x_1 = [x; c_k]$에 대해 Forward-Forward 레이어 $l \in \{1, 2\}$는 다음과 같이 동작함:
- **정규화 및 활성화**: $\hat{x}_l = \frac{x_l}{\|x_l\|_2 + \epsilon}$, $h_l = \text{ReLU}(W_l \hat{x}_l + b_l)$;
- **Goodness & Jev-like Softmax**:
  $$G_l = \frac{1}{N_l} \sum_{j=1}^{N_l} h_{l,j}^2, \quad G(x, c_k) = G_1 + G_2$$
  $$p_k = \frac{\exp(G(x, c_k) / T_g)}{\sum_{m=1}^{K} \exp(G(x, c_m) / T_g)}$$
- **Group-Aware Asymmetric Local Loss** (그룹 $g \in \{\text{doc\_slide}, \text{doc\_hwpx}, \text{doc\_md}, \text{media\_img}\}$):
  $$L_{l,g} = \alpha_g \log\left(1 + \exp\left(-(G_{\text{pos}} - \theta_g)\right)\right) + \beta_g \log\left(1 + \exp\left(G_{\text{neg}} - \theta_g\right)\right)$$
- **Calibration 평가 지표 (Expected Calibration Error)**:
  $$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left|\text{acc}(B_b) - \text{conf}(B_b)\right|$$

---

## 3. 3 Controlled Failure Scenarios
1. **Scenario A (Global Decay — Sanity Check)**: 모든 그룹의 Goodness 스케일이 매 스텝 일정 비율로 감쇠. 목표: Clamped LLM이 Global/Per-Group PID와 동등한 수준으로 발산 없이 안정 추종하는지 확인.
2. **Scenario B (Subgroup Concept Drift)**: 시점 $t_0$ 이후 `doc_hwpx` 그룹만 입력 Feature 중심이 이동하여 `FN`이 급증(타 그룹은 정상). 목표: 소량 샘플($N = 15\text{–}30$) 구간에서 시계열 Changepoint 및 그룹 특이적 이상을 조기 진단해 회복 사이클을 단축하는지 비교.
3. **Scenario C (Structural Label-Swap Shift)**: 시점 $t_1$ 이후 `doc_slide` 그룹 내에서만 `cat_1` $\leftrightarrow$ `cat_2` 라벨 규칙이 역전됨. 임계치 $\theta_g$만 오르내리는 Per-Group PID는 구조적 혼동을 해결하지 못해 `FP`가 폭증하는 반면, LLM Calibrator가 `pred_cat`/`actual_cat` 혼동 행렬 패턴을 읽어 `trigger_replay` 및 $\alpha_g/\beta_g$ 재가중 액션을 발동하는지 검증.

---

## 4. Measures of Effectiveness (MOE) & Performance (MOP)
- **MOE 1 (Sovereignty)**: Outbound Sanitizer 적대적 테스트 100% 차단 (원본 텍스트·파일 경로·임베딩 벡터 유출 $= 0\text{ Bytes}$);
- **MOE 2 (Pre-Drift Baseline)**: 드리프트 주입 전 순수 NumPy FF 모델 정확도 $\ge 0.88$, $\text{ECE} \le 0.10$;
- **MOE 3 (Scenario A Parity)**: Global Decay에서 Clamped LLM이 발산 없이 PID 대비 정확도 $-3\%\text{p}$ 이내 유지;
- **MOE 4 (Scenario B/C Advantage)**: Subgroup Drift 회복 사이클 단축 또는 Scenario C(Label-Swap)에서 Per-Group PID 대비 회복 후 정확도 $\ge +10\%\text{p}$ 우위 입증;
- **MOP 1 (Reproducibility & Speed)**: 5개 고정 시드 기준 오프라인 전체 벤치마크 실행 시간 $\le 10\text{ s}$.
