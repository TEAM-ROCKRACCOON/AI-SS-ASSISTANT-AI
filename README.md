# AI-SS-ASSISTANT-AI

## 주요 기능

### 1. 외부 데이터 처리
- **외부 일정 데이터 전처리**: 사용자의 Google 캘린더 데이터를 연산 가능한 형태로 변환한 후 스케줄에 반영.

### 2. Integer Linear Programming (ILP) 기반 스케줄 생성
- **ILP 알고리즘**을 사용해 청소 스케줄의 최적해를 계산.

### 3. 내부 데이터 업데이트
- 사용자의 청소 완료 데이터를 활용해 내부 데이터(마지막 수행 이후 경과 일수)를 업데이트.

### 4. 사용자 행동 벡터 생성
- **24x7 시간 슬롯** 기준으로 청소 수행 확률을 값으로 가지는 사용자 행동 벡터를 생성.

## 기술 스택

| 구분 | 사용 기술 |
|------|-----------|
| **최적화 알고리즘** | [PuLP](https://github.com/coin-or/pulp) (Python Linear Programming) |
| **강화 학습** | [Tianshou](https://github.com/thu-ml/tianshou) (Reinforcement Learning Framework) |
| **딥러닝 프레임워크** | [PyTorch](https://pytorch.org/) |
| **컴퓨터 비전** | OpenCV, Pillow, scikit-image |
| **비전 모델** | [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) |
| **이미지 생성 / 편집** | [SmartEraser](https://github.com/longtaojiang/SmartEraser) |
| **API 서버** | FastAPI |
| **실행 환경** | Docker, NVIDIA CUDA |


### 3. 이미지 처리 서비스 (Grounded-SAM + SmartEraser)
- **Grounded-Segment-Anything**를 이용한 객체 마스크 생성과 **SmartEraser**를 이용한 이미지 객체 제거(inpainting)를 제공하는 GPU 기반 이미지 처리

FastAPI 엔드포인트 형태로 제공되며,  
백엔드에서는 본 서비스를 **독립적인 이미지 추론 서버**로 호출하여 사용합니다.

---
  아래 링크에서 반드시 별도로 다운로드해야 합니다.

- **Pretrained Weight (Google Drive)**  
  https://drive.google.com/file/d/1D49l9DM6X_s34ISDk0J853z1VdiBzs2N/view
  
- **Model checkpoint**  
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth 

---

## 디렉토리 구조 및 가중치 배치

Google Drive에서 다운로드한 SmartEraser 가중치와 ViT-L SAM model를 
아래 **정확한 경로**에 배치해야 합니다.

image_service/  
│  
├─ Grounded-Segment-Anything/  
│  └─ models  
│     └─ **sam_vit_b_01ec64.pth**  
├─ SmartEraser/  
│  └─ Model_framework/ckpts/  
│     └─ **smarteraser-weights**  
├─ main.py  
├─ img_utils.py  
└─ Dockerfile  
