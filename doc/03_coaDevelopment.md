# Step 3: COA Development (방책 수립)

`llmCalibratorSim` 구현 및 검증 전략에 대해 상호 배타적인 3개 방책(COA)을 중립적으로 수립함.

---

## COA 1: Pure Synthetic NumPy Core + Offline Mock/Rule Oracle Only
- **구현 방식**: 순수 Python + NumPy 시뮬레이터 위에 3종 고장 시나리오, Sanitizer, 3-way 비교기(`GlobalPidCalibrator`, `PerGroupPidCalibrator`, `DeterministicRuleOracle`)를 구축하고 외부 LLM 호출 없이 오프라인 완결.
- **장점**: 실행 속도 최고($<2\text{ s}$), 100% 결정론적 재현성, API 비용 및 키 관리 부담 $0$.
- **단점**: Rule Oracle이 실제 LLM의 추론 비결정성·JSON 파싱 오류·프롬프트 해석 한계를 완전히 대변하지 못함 ("Mock Oracle Overfitting" 위험).

---

## COA 2: Real Local Embeddings + Live LLM API First
- **구현 방식**: 시작부터 실제 로컬 문서 임베딩 및 외부 LLM API를 직결하여 실 환경 그대로 실험을 구축.
- **장점**: 실데이터 및 실 LLM 거동을 즉시 관찰 가능.
- **단점**: 개발 초기 Sanitizer 미검증 상태에서 민감 데이터/시크릿 유출 위험 존재; API 지연과 비용 때문에 5-seed x 3-scenario 반복 벤치마크가 느리고 비재현적임.

---

## COA 3: Hybrid Deterministic Core + Plug-in Live Adapter
- **구현 방식**:
  1. **Core (`Units 0–3`)**: `DatasetSource` 및 `CalibratorBackend` 추상 인터페이스 아래에서 순수 NumPy FF + 3종 고장 시나리오 + 3-way 대조(Global PID / Per-Group PID / Deterministic Oracle) + Strict Outbound Sanitizer를 먼저 완성해 빠른 5-seed 벤치마크와 MOE 검증을 마침;
  2. **Live Plug-in (`Unit 4`)**: `Unit 2` Sanitizer 100% 통과가 검증된 뒤에만 동일한 Telemetry Array를 로컬 Ollama 또는 `envrun` 주입 외부 LLM API(`OpenAI-compatible` 어댑터)로 흘려 실 LLM 응답 안정성과 정확도를 교차 검증함.
- **장점**: 빠른 결정론적 실험·재현성과 실 LLM/실데이터 확장성을 모두 확보하며, Sanitizer 검증 전 외부 유출 가능성을 원천 차단함.
- **단점**: `CalibratorBackend` 및 `DatasetSource` 추상화 계층 구현에 소폭의 초기 코드량이 추가됨.
