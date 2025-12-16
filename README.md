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
| 구분         | 사용 기술                |
|--------------|--------------------------|
| **최적화 알고리즘** | Pulp (Python Linear Programming) |
| **강화 학습**  | Tianshou (RL Framework) |

https://github.com/IDEA-Research/Grounded-Segment-Anything
https://github.com/longtaojiang/SmartEraser



1. 사용 모델 및 외부 리포지토리
Grounded-Segment-Anything (Grounded SAM)

GitHub
https://github.com/IDEA-Research/Grounded-Segment-Anything

SAM 공식 모델 가중치 (Meta)
https://github.com/facebookresearch/segment-anything#model-checkpoints

본 프로젝트에서는 예시로 다음 SAM 가중치를 사용합니다.

sam_vit_b_01ec64.pth

SmartEraser

GitHub
https://github.com/longtaojiang/SmartEraser

추가 Pretrained Weight (Google Drive, 작성자 제공)
아래 링크에서 반드시 별도로 다운로드해야 합니다.

https://drive.google.com/file/d/1D49l9DM6X_s34ISDk0J853z1VdiBzs2N/view

⚠️ 중요
SmartEraser는 git clone만으로는 동작하지 않으며,
위 Google Drive에서 제공되는 추가 가중치 파일이 반드시 필요합니다.

2. 디렉토리 구조 및 가중치 배치

Google Drive에서 다운로드한 SmartEraser 가중치는
아래 정확한 경로에 배치해야 합니다.

image_service/
│
├─ ckpts/
│  └─ smarteraser-weights/
│     ├─ (Google Drive에서 받은 모든 weight 파일)
│
├─ Grounded-Segment-Anything/
├─ SmartEraser/
├─ main.py
├─ img_utils.py
└─ Dockerfile


폴더명 및 파일명을 임의로 변경하면 정상 동작하지 않습니다.

코드에서는 다음 경로를 기준으로 가중치를 로드합니다.

checkpoint_dir = "./ckpts/smarteraser-weights"
