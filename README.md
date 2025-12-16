# AI-SS-ASSISTANT-AI

정수선형계획법(ILP)과 강화학습 기반  
사용자 맞춤형 청소 일정 추천 및 시각적 동기부여 AI 서비스

---

## 주요 기능

### 1. 청소 스케줄 자동 생성 (ILP + 강화학습)

**설명**  
사용자의 일정, 청소 작업 특성, 온보딩 선호도 정보를 기반으로  
실행 가능한 주간 청소 루틴을 자동 생성하고,  
실제 사용자 행동 기록을 학습하여 점진적으로 개인화된 스케줄로 개선합니다.

- 초기 단계에서는 ILP를 활용해 일정 충돌 없이 안정적인 기본 루틴 생성
- 이후 강화학습(PPO)을 적용해 선호 요일, 시간대, 미이행 패턴 반영
- Cold-start 상황에서도 즉시 실행 가능한 루틴 제공

---

### 2. 날씨 기반 청소 추천 모듈

**설명**  
주간 청소 일정과는 별도로 동작하는 실시간 추천 기능으로,  
사용자 위치 기반 날씨 및 대기질 데이터를 활용해  
현재 환경에 적합한 청소 작업을 추천합니다.

- 풍속, 구름량, 강수 여부, AQI 임계값 기반 조건 분류
- 환기 가능 여부 및 실내/현관 중심 청소 자동 추천
- 규칙 기반 구조로 빠르고 일관된 결과 제공

---

### 3. 이미지 기반 시각적 동기부여 기능

**설명**  
사용자가 업로드한 방 사진을 기반으로  
청소 완료 후 모습을 시각적으로 생성하여  
청소에 대한 동기와 성취감을 제공합니다.

- Grounded-Segment-Anything으로 정리 대상 객체 마스킹
- SmartEraser 기반 인페인팅으로 자연스러운 정리 결과 생성
- 전체 이미지 재생성 방식 대비 빠른 처리 속도

---

## 기술 스택

| 구분 | 사용 기술 |
|------|-----------|
| **최적화 알고리즘** | [PuLP](https://github.com/coin-or/pulp) (Python Linear Programming) |
| **강화 학습** | [Tianshou](https://github.com/thu-ml/tianshou) (PPO, Reinforcement Learning Framework) |
| **딥러닝 프레임워크** | [PyTorch](https://pytorch.org/) |
| **컴퓨터 비전** | OpenCV, Pillow, scikit-image |
| **비전 모델** | [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) |
| **이미지 생성 / 편집** | [SmartEraser](https://github.com/longtaojiang/SmartEraser) |
| **외부 API** | OpenWeatherMap API (Weather / Air Pollution) |
| **API 서버** | FastAPI |
| **실행 환경** | Docker, NVIDIA CUDA |

---

## 이미지 처리 서비스 실행 안내

이미지 처리 기능은 **독립적인 추론 서버(FastAPI)** 형태로 제공되며,  
백엔드에서 API 호출 방식으로 사용됩니다.

---

## 사전 다운로드 필요 파일

아래 파일들은 **반드시 별도로 다운로드**해야 합니다.

- **SmartEraser Pretrained Weight (Google Drive)**  
  https://drive.google.com/file/d/1D49l9DM6X_s34ISDk0J853z1VdiBzs2N/view

- **SAM Model Checkpoint**  
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

---

## 디렉토리 구조 및 가중치 배치

다운로드한 가중치 파일은 아래 **정확한 경로**에 배치해야 합니다.

```text
image_service/
├─ Grounded-Segment-Anything/
│  └─ models/
│     └─ sam_vit_b_01ec64.pth
│
├─ SmartEraser/
│  └─ Model_framework/
│     └─ ckpts/
│        └─ smarteraser-weights
│
├─ main.py
├─ img_utils.py
└─ Dockerfile