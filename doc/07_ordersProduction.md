# Step 7: Orders Production (실행 명령서 및 OPORD — `llmCalibratorSim`)

채택된 **COA 3** 구현을 위한 저장소 구조, 입출력 스키마 계약, 클램핑 경계 및 `mdmp-redteam` 원자적 OPORD Unit 명세입니다.

---

## 1. Repository Layout (`camelCase` Convention)
- `pyproject.toml`: 순수 `numpy` 의존성 및 `src/` 패키지 설정;
- `src/llmCalibratorSim/__init__.py`
- `src/llmCalibratorSim/ffModel.py`: NumPy 2-Layer Forward-Forward ($32 \to 32 \to 16$), Jev Softmax, 그룹별 $(\theta_g, \alpha_g, \beta_g, T_g)$ 파라미터 및 Replay Buffer 업데이트;
- `src/llmCalibratorSim/driftEnv.py`: `DatasetSource` 추상 인터페이스, 3대 고장 주입기(`GlobalDecay`, `SubgroupConceptDrift`, `StructuralLabelSwap`), Activity-Gated Implicit Feedback($p_{\text{noise}} = 0.05$);
- `src/llmCalibratorSim/sanitizer.py`: 엄격한 Outbound Telemetry Array 스키마 검증기 및 적대적 페이로드 차단기;
- `src/llmCalibratorSim/calibrators.py`: `CalibratorBackend` 인터페이스, `GlobalPidCalibrator`, `PerGroupPidCalibrator`, `ClampedLlmCalibrator` (`DeterministicRuleOracle` + OpenAI 호환 HTTP / Ollama 어댑터);
- `src/llmCalibratorSim/runBenchmark.py`: 5-seed x 3-scenario x 3-calibrator 비교 실행기 및 MOE 판정 리포트 출력;
- `tests/testSanitizer.py`, `tests/testFfModel.py`, `tests/testBenchmark.py`.

---

## 2. Interface Contracts & Hard Clamping Bounds

### 2.1 Outbound Telemetry Array JSON Schema (Input to Calibrator)
외부 전송 전 `sanitizer.py`가 아래 8개 고정 필드와 타입/범위만 허용하며 그 외 키나 장문/경로/벡터 포함 시 즉시 `SovereigntyViolationError`를 발생시킴:
- `decision_id` (`int >= 0`)
- `ext_enum` (`"doc_slide" | "doc_hwpx" | "doc_md" | "media_img"`)
- `pred_cat` (`"cat_0" | "cat_1" | "cat_2" | "cat_3"`)
- `actual_cat` (`"cat_0" | "cat_1" | "cat_2" | "cat_3"`)
- `outcome` (`"TP" | "FP" | "FN"`)
- `raw_goodness_G` (`float`, finite)
- `confidence_p` (`float` in `[0.0, 1.0]`)
- `timestamp_s` (`int >= 0`)

### 2.2 Clamped Calibrator Output JSON Schema
Calibrator 응답 파싱 후 적용 전 아래 경계로 강제 Clamping 수행 (파싱/스키마 실패 시 `recommended_action = "noop"` 폴백):
- `global_delta`: `{d_theta, d_alpha, d_beta, d_T}`
- `group_overrides`: `{ext_enum: {d_theta, d_alpha, d_beta, d_T}}`
- `recommended_action`: `"adjust_params" | "trigger_replay" | "noop"`
- `diagnosis`: 짧은 상태 태그(최대 200자)
- **Hard Bounds**: $\Delta\theta \in [-0.15, +0.15]$, $\theta_g \in [0.2, 5.0]$; $\Delta\alpha, \Delta\beta \in [-0.2, +0.2]$, $\alpha_g, \beta_g \in [0.2, 3.0]$; $\Delta T \in [-0.2, +0.2]$, $T_g \in [0.2, 5.0]$.

---

## 3. OPORD Atomic Units (`Unit 0` ~ `Unit 4`)

### Unit 0: Package Skeleton & Environment Sanity Check
- **IN State**: `~/workspace/llmCalibratorSim` Git 초기화 및 `.gitignore` 준비 완료; `Python 3.12` + `NumPy` 확인됨.
- **Internal Logic**: `pyproject.toml`, `src/llmCalibratorSim/__init__.py` 및 테스트 디렉토리 생성; editable 또는 `PYTHONPATH=src` 임포트 확인.
  - *Hard Prohibition*: PyTorch/TensorFlow 등 무거운 패키지 설치 금지; `.env` 평문 파일 생성 금지.
- **OUT State (Verification Gate)**: `PYTHONPATH=src python3 -c "import numpy, llmCalibratorSim; print('OK')"` 실행 시 `exit 0`.

### Unit 1: NumPy 2-Layer FF Classifier & 3-Scenario Drift Engine
- **IN State**: Unit 0 통과.
- **Internal Logic**: `ffModel.py` (2-Layer FF, Group Loss, Jev Softmax, $\text{ECE}$ 계산) 및 `driftEnv.py` (`GlobalDecay`, `SubgroupConceptDrift`, `StructuralLabelSwap`, Activity-Gated Feedback $p_{\text{noise}} = 0.05$) 구현; `tests/testFfModel.py` 작성.
  - *Hard Prohibition*: Backpropagation 구현 금지; Global PID나 하드웨어 의존 코드 혼입 금지.
- **OUT State (Verification Gate)**: `pytest`/`unittest`에서 Pre-Drift 정확도 $\ge 0.88$, $\text{ECE} \le 0.10$ (`MOE 2`) 및 3개 드리프트 주입 시 예상 `FN`/`FP` 스파이크 발생 확인 후 `exit 0`.

### Unit 2: Sovereignty-Safe Telemetry Logger & Outbound Schema Sanitizer
- **IN State**: Unit 1 통과.
- **Internal Logic**: `sanitizer.py` 구현 (고정 키 화이트리스트, 문자열 길이/패턴 검사로 파일 경로(`/`, `\`, `.pptx` 등), 원문, 숫자 배열(임베딩) 완전 차단); `tests/testSanitizer.py` 작성.
  - *Hard Prohibition*: Sanitizer 우회 옵션(`bypass=True` 등) 제공 금지.
- **OUT State (Verification Gate)**: 적대적 페이로드(경로·텍스트·임베딩·알 수 없는 키) 테스트 케이스 100% 차단(`MOE 1 = 0 Bytes leak`) 확인 후 `exit 0`.

### Unit 3: 3-Way Calibrator Showdown & 5-Seed Benchmark
- **IN State**: Unit 2 통과.
- **Internal Logic**: `calibrators.py` (`GlobalPidCalibrator`, `PerGroupPidCalibrator`, `ClampedLlmCalibrator` + `DeterministicRuleOracle` + Clamping/Fuzz fallback) 및 `runBenchmark.py` (5 seeds x 3 scenarios) 구현; `tests/testBenchmark.py` 작성.
  - *Hard Prohibition*: Per-Group PID 대조군 누락 금지; `Unit 4` 실 네트워크 호출을 기본 벤치마크에 강제 포함 금지.
- **OUT State (Verification Gate)**: 5-seed 전체 벤치마크 $\le 10\text{ s}$ 완료(`MOP 1`), 깨진 JSON fuzz test 통과, `MOE 3`(Scenario A 동등) 및 `MOE 4`(Scenario B/C에서 Per-Group PID 대비 유의미한 우위, 특히 Scenario C 회복 정확도 $\ge +10\%\text{p}$) 확인 후 `exit 0`.

### Unit 4: `[MANUAL / CONDITIONAL]` Live LLM Adapter Smoke Test
- **IN State**: Units 0–3 전체 `exit 0` 통과; `Unit 2` Sanitizer 100% 검증 완료.
- **Internal Logic**: 로컬 Ollama 또는 `envrun`으로 주입된 OpenAI 호환 API 키를 `ClampedLlmCalibrator` HTTP 어댑터에 연결해 동일 3개 시나리오 스모크 테스트 실행; 실제 JSON 파싱 성공률, Clamping 발동 횟수, Per-Group PID 대비 정확도 델타 로깅.
  - *Hard Prohibition*: 평문 `.env` 파일 생성 또는 `apiKey.txt.gpg` 복호화 결과를 디스크에 저장하는 행위 절대 금지.
- **OUT State (Verification Gate)**: `[MANUAL]` 실 LLM 스모크 리포트 생성 및 Sanitizer 통과 로그 확인(`exit 0`).
