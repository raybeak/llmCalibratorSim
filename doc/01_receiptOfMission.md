# Step 1: Receipt of Mission (임무 수령 — `llmCalibratorSim`)

## 1. Core Research Question
원본 데이터·파일명·임베딩 벡터 유출 **$0\text{ Bytes}$ (Sovereignty & Commercial LLM ToS 준수)** 조건하에서, 오직 익명화된 Timestamped Telemetry Array(`TP/FP/FN + raw_goodness_G + confidence_p + ext_enum + timestamp`)만 읽는 **Clamped LLM Meta-Calibrator**가 단순 **Global PID**뿐 아니라 **Per-Group PID/EMA** 대조군 대비 어떤 드리프트 및 구조적 고장 시나리오에서 유의미한 우위를 가지는가?

---

## 2. Commander's Intent & Scope Boundaries
- **순수 소프트웨어 PoC**: 하드웨어 배선, 브레드보드, MOSFET, `UNO Q` / `ESP32` 의존성을 완전히 배제하고 **제어 루프 및 Sovereignty 검증** 자체에 집중함;
- **초경량 Zero-Bloat 스택**: 기설치된 `Python 3.12` + `NumPy`만으로 구동하며 무거운 DL 프레임워크(PyTorch/TensorFlow)나 불필요한 외부 의존성을 도입하지 않음;
- **Zero-Disk Secret Hygiene**: 평문 `.env` 파일 생성 및 시크릿 하드코딩을 엄격히 금지하며, 실 LLM API 연동 시에도 로컬 Ollama 또는 `.env.gpg`(`envrun`) 경로만 허용함;
- **목표 일정**: 1주 이내 구현 및 다중 시드(5 seeds) 정량 비교 벤치마크 완결.
