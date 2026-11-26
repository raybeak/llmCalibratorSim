# Step 6: COA Approval (방책 승인 결정문)

## 1. Autonomous Full-Cycle Decision
- **최종 채택 방책**: **COA 3 — Hybrid Deterministic Core (`Units 0–3`) + Plug-in Live Adapter (`Unit 4`)**
- **채택 사유**: 10초 이내의 5-seed 결정론적 벤치마크(`Global PID` vs. `Per-Group PID` vs. `Clamped LLM/Oracle`)를 먼저 완결하고, `Unit 2` Outbound Sanitizer 100% 차단 검증을 통과한 경우에만 `Unit 4` 실 LLM(Ollama 또는 `envrun` API) 어댑터를 연결할 수 있어 과학적 엄밀성과 Sovereignty 안전성을 동시에 만족함.

---

## 2. Autonomous Assumptions & Mandatory Controls
1. **Hard Control 필수화**: 모든 시나리오에서 `GlobalPidCalibrator`뿐 아니라 **`PerGroupPidCalibrator`**를 동시 평가해 LLM Calibrator의 순수 추가 가치를 정직하게 측정함.
2. **Sovereignty & Secret Gate**: `Unit 2` 적대적 Sanitizer 테스트(파일명·경로·장문 텍스트·벡터 배열 차단) 통과 전 `Unit 4` 네트워크 호출 불가; API 키는 `envrun` 환경변수로만 주입하고 평문 `.env` 디스크 생성 절대 금지.
3. **Control Stability & Fallback**: LLM 출력은 반드시 Hard Clamping($\Delta\theta \in [-0.15, +0.15]$ 등)을 거치며, JSON 파싱 실패·타임아웃·스키마 오류 시 즉시 `No-op` 또는 안전한 `Per-Group PID`로 폴백함.
4. **Noisy Feedback Realism**: Activity-Gated Implicit Feedback에 기본 라벨 노이즈($p_{\text{noise}} = 0.05$)를 포함해 현실적인 오인 오염 하에서의 복원력을 검증함.
