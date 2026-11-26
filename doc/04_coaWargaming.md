# Step 4: COA Analysis / Wargaming (전 방책 워게임)

3개 COA(`COA 1`, `COA 2`, `COA 3`) 모두를 4가지 적대적 시나리오에 투입해 **Action $\rightarrow$ Reaction $\rightarrow$ Counteraction**으로 검증함.

## Scenario 1: Strawman Baseline Bias (Global PID만 비교하는 편향)
- **COA 1 & COA 3**:
  - *Action*: Subgroup Drift(`doc_hwpx`만 이동)에서 LLM을 Global PID 한 개하고만 비교함.
  - *Reaction*: 그룹별 독립 추적이 가능한 Per-Group PID만 써도 해결될 문제를 LLM의 공적으로 과대 포장하는 실험 오류 발생.
  - *Counteraction*: **`PerGroupPidCalibrator`를 필수 대조군**으로 고정하고, 단순 임계치 조절로 못 푸는 **Scenario C (Structural Label-Swap Shift)**를 포함시켜 정직하게 비교함.
- **COA 2**: 동일 위험 + 실 API 비용 때문에 대조군 반복 실험 축소 유혹이 더 큼 $\to$ 오프라인 결정론적 러너 부재 시 검증 약화.

## Scenario 2: Sovereignty & Secret Leakage (원본 데이터 및 API 키 유출)
- **COA 1**: 외부 네트워크 호출이 없어 유출 위험 $0$이나, 실 API 연동 시 Sanitizer 동작을 끝까지 검증하지 못함.
- **COA 2**:
  - *Action*: 초기 디버깅 중 테스트 문서 경로, 임베딩, 또는 `.env` 평문 키를 실수로 LLM 페이로드나 디스크에 기록함.
  - *Reaction*: Sovereignty 및 Zero-disk 정책 위반 발생.
  - *Counteraction*: `COA 2` 단독 진행 금지.
- **COA 3**: `Unit 2`에서 파일명·경로·장문 텍스트·수치 배열(벡터) 포함 악성 페이로드를 100% 차단하는 Outbound Sanitizer 단위 테스트를 통과하기 전에는 `Unit 4` Live Backend 활성화를 코드 레벨에서 차단(Gate). API 키는 `envrun` 환경변수로만 수신.

## Scenario 3: LLM Hallucination, Malformed JSON & Control Divergence
- **COA 1**: Deterministic Oracle은 항상 유효 JSON을 뱉으므로 파싱 실패 내성을 검증하지 못함.
- **COA 2 & COA 3**:
  - *Action*: 실 LLM이 Markdown 울타리, 범위 초과 $\theta = 99.0$, 음수 Temperature $T = -1.0$, 혹은 깨진 JSON을 반환함.
  - *Reaction*: 파싱 크래시 또는 FF 분류기 즉각 발산.
  - *Counteraction*: Strict JSON Extractor + 스키마 검증 + **Hard Clamping**($\Delta\theta \in [-0.15, +0.15]$ 등) + 파싱 실패 시 안전하게 `No-op / PID Fallback`으로 전환하는 방어기 및 Fuzz Test 의무화.

## Scenario 4: Noisy Activity-Gated Implicit Feedback
- **COA 1, COA 2, COA 3 (공통)**:
  - *Action*: 유휴 시간 경과만으로 `TP`를 찍거나 피드백 노이즈(오답을 `TP`로 기록)가 5% 섞인 로그가 누적됨.
  - *Reaction*: `TP` 과대 계상으로 Calibrator가 오탐 상태를 정상으로 오판.
  - *Counteraction*: 시간 경과 대신 **후속 활동 이벤트 $M$회 경과 후 미수정 시 `TP` 확정**으로 게이팅하고, 벤치마크 기본 설정에 $p_{\text{noise}} = 0.05$ 라벨 노이즈를 기본 주입해 내성을 검증함.
