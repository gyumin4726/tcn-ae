# TCN-AE 시계열 이상 탐지 프로젝트

## 📖 프로젝트 소개

이 프로젝트는 **TCN-AE (Temporal Convolutional Network Autoencoder)**를 사용하여 Mackey-Glass 시계열 데이터에서 이상을 탐지하는 딥러닝 프로젝트입니다.

### 🎯 주요 특징
- **최신 TensorFlow 2.15.0** 환경에서 GPU 가속 지원
- **GUI 없는 환경**에서 자동 이미지 저장 기능
- **타임스탬프 기반** 결과 파일 관리
- **높은 정확도**의 이상 탐지 성능

## 🧠 TCN-AE 모델이란?

**TCN-AE = Temporal Convolutional Network Autoencoder**

### 모델 구조
```
입력 (1050 길이) 
    ↓
🔽 인코더 (Encoder)
├── TCN (dilations: 1,2,4,8,16)
├── Conv1D (8 filters)
├── AveragePooling1D (1/42 압축)
└── 압축된 표현 (25 길이)
    ↓
🔼 디코더 (Decoder)
├── UpSampling1D (42배 확대)
├── TCN (dilations: 1,2,4,8,16)
├── Dense Layer
└── 복원된 데이터 (1050 길이)
    ↓
📊 이상 탐지
├── 복원 오차 계산
├── 슬라이딩 윈도우 (128 길이)
├── 마할라노비스 거리 계산
└── 이상 점수 출력
```

### 핵심 장점
- **빠른 처리**: RNN/LSTM 대비 병렬 처리 가능
- **긴 시퀀스**: Dilated Convolution으로 장기 의존성 학습
- **안정적 학습**: 기울기 소실 문제 없음
- **메모리 효율**: 낮은 메모리 사용량

## 🛠️ 환경 설정 및 설치

### 1. Conda 가상환경 생성

```bash
# 1. conda 가상환경 생성
conda create -n bioma-tcn-ae-modern python=3.11
conda activate bioma-tcn-ae-modern

# 2. 패키지 설치 (호환성 검증된 버전)
pip install numpy>=1.24.0 matplotlib>=3.7.0 pandas>=2.0.0 scikit-learn>=1.3.0
pip install tensorflow==2.15.0  # 안정성과 GPU 호환성이 검증된 버전
pip install keras-tcn==3.3.0    # 최신 호환 버전
pip install protobuf>=3.20.0
```

### 2. GPU 설정 (선택사항)

```bash
# GPU 사용 가능 여부 확인
nvidia-smi

# TensorFlow GPU 인식 확인
python -c "import tensorflow as tf; print('GPU 사용 가능:', tf.config.list_physical_devices('GPU'))"
```

## 🚀 실행 방법

### 기본 실행

```bash
# 프로젝트 디렉터리로 이동
cd src

# 메인 스크립트 실행
python main.py
```

### 실행 과정
1. **GPU 설정**: RTX A5000 (22GB) 자동 감지 및 설정
2. **데이터 로딩**: mg1.npy (훈련), mg3.npy (테스트)
3. **모델 훈련**: 10 epochs, 약 108초 (GPU 가속)
4. **이상 탐지**: 전체 시계열에서 이상 점수 계산
5. **결과 저장**: RESULT 폴더에 이미지 자동 저장

## 📊 데이터 Shape 이해하기

### Train vs Test 데이터 Shape 차이

```python
train_X.shape: (19791, 1050, 1)  # 훈련 데이터
test_X.shape: (1, 100000, 1)     # 테스트 데이터
```

#### 🔽 훈련 데이터 (19791, 1050, 1)
- **19,791개 윈도우**: 원본 시계열(100,000 포인트)을 1,050 길이로 슬라이딩 윈도우 분할
- **윈도우 길이 1,050**: 각 훈련 샘플의 시계열 길이
- **스트라이드 5**: 5간격으로 샘플링하여 훈련 효율성 향상
- **차원 1**: 단변량 시계열 (값 1개)

```python
# 슬라이딩 윈도우 예시
원본 시계열: [0, 1, 2, 3, ..., 99999] (100,000 길이)
윈도우 1: [0, 1, 2, ..., 1049]     # 1,050 길이
윈도우 2: [5, 6, 7, ..., 1054]     # 5씩 이동 (스트라이드)
윈도우 3: [10, 11, 12, ..., 1059]
...
윈도우 19791: [..., 99999]
```

#### 🔼 테스트 데이터 (1, 100000, 1)
- **1개 배치**: 전체 시계열을 하나의 배치로 처리
- **길이 100,000**: 전체 시계열 길이 (연속적인 이상 탐지)
- **차원 1**: 단변량 시계열

### 왜 이렇게 다른가? 🤔

#### 1. **훈련 전략: 패턴 학습**
```python
목적: TCN-AE가 정상적인 1,050 길이 패턴을 학습
장점:
- 다양한 위치의 패턴 학습으로 일반화 능력 향상
- 메모리 효율적 (작은 윈도우 단위 처리)
- 배치 학습으로 안정적인 그래디언트 업데이트
```

#### 2. **테스트 전략: 연속 이상 탐지**
```python
목적: 전체 시계열에서 연속적인 이상 점수 계산
장점:
- 시계열의 전체 문맥 정보 유지
- 이상이 발생한 정확한 시점 탐지
- 연속적인 이상 점수 곡선 생성
```

#### 3. **실제 처리 과정**
```python
훈련 시:
- 입력: (batch_size=32, 1050, 1) 
- 출력: (batch_size=32, 1050, 1) # 동일 크기로 복원
- 손실: 입력과 출력의 차이 (복원 오차)

테스트 시:
- 입력: (1, 100000, 1)
- 출력: (1, 복원길이, 1) # 약간 작을 수 있음 (패딩으로 보정)
- 이상점수: 각 시점별 복원 오차 → 마할라노비스 거리
```

### 이 방식이 올바른 이유 ✅

#### **메모리 효율성**
```python
훈련 시 메모리 사용량:
- 방식 1 (현재): 32 × 1,050 × 1 = 33,600 포인트/배치
- 방식 2 (전체): 1 × 100,000 × 1 = 100,000 포인트/배치
→ 현재 방식이 66% 적은 메모리 사용
```

#### **학습 안정성**
```python
훈련 다양성:
- 19,791개의 서로 다른 윈도우로 학습
- 동일한 패턴이 다른 위치에서 반복 학습
- 오버피팅 방지 및 일반화 능력 향상
```

#### **실시간 적용 가능성**
```python
실제 운영 환경:
- 훈련: 과거 데이터의 작은 윈도우들로 패턴 학습
- 운영: 실시간 스트리밍 데이터에 슬라이딩 윈도우 적용
- 지연시간: 1,050 포인트마다 이상 점수 계산 가능
```

### 데이터 처리 파이프라인 🔄

```python
# data.py에서의 실제 처리 (라인 180-190)
1. 원본 데이터 로딩: mg1.npy → (100000,) 배열
2. 슬라이딩 윈도우: utilities.slide_window() 호출
3. 윈도우 생성: (19791+, 1050, 1) 전체 윈도우
4. 스트라이드 적용: X[::5] → (19791, 1050, 1) 최종 훈련 데이터

# 테스트 시 처리 (main.py 라인 67)
1. 전체 시계열 로딩: test_data["scaled_series"] 
2. 배치 차원 추가: [numpy.newaxis, :, :] 
3. 최종 형태: (1, 100000, 1)
```

## 📊 실행 결과

### 예상 출력
```
Starting Time Series Anomaly Detection with TCN-AE
==================================================

🔧 GPU 설정...
selected GPUs: 0

1. Loading and preparing training data...
train_X.shape: (19791, 1050, 1)

2. Building and training the TCN-AE model...
Training for 10 epochs...
> Starting the Training...
Epoch 1/10: 30s - loss: 0.0334
Epoch 2/10: 8s - loss: 0.0013
...
Epoch 10/10: 8s - loss: 0.0004
> Training Time : 108 seconds.

3. Testing the model on test data...
test_X.shape: (1, 100000, 1)
> Prediction time: 32 seconds.

4. Visualizing results...
✅ 그래프가 RESULT/anomaly_score_full_20250705_143022.png에 저장되었습니다.
✅ 그래프가 RESULT/anomaly_zoom_40000_42000_20250705_143022.png에 저장되었습니다.

==================================================
TCN-AE Anomaly Detection completed successfully!
```

### 생성되는 파일
```
RESULT/
├── anomaly_score_full_YYYYMMDD_HHMMSS.png     # 전체 이상 점수 그래프
└── anomaly_zoom_40000_42000_YYYYMMDD_HHMMSS.png # 첫 번째 이상의 확대 보기
```

### 성능 지표
- **정상 구간 이상 점수**: ~10 (안정적)
- **이상 구간 이상 점수**: 60-70 (6-7배 증가)
- **정확도**: 모든 실제 이상을 성공적으로 탐지
- **오탐률**: 매우 낮음 (정상 구간에서 안정적)

## 📁 프로젝트 구조

```
bioma-tcn-ae/
├── README.md                 # 프로젝트 문서
├── requirements.txt          # 패키지 의존성
├── data/                     # 데이터 폴더
│   └── MGAB/
│       ├── mg0.npy ~ mg14.npy # Mackey-Glass 시계열 데이터
├── src/                      # 소스 코드
│   ├── main.py              # 메인 실행 파일 (Jupyter notebook에서 변환)
│   ├── tcnae.py             # TCN-AE 모델 클래스
│   ├── data.py              # 데이터 로딩 및 전처리
│   └── utilities.py         # 유틸리티 함수 (GPU 설정, 시각화)
└── RESULT/                   # 결과 이미지 (자동 생성)
    ├── anomaly_score_full_*.png
    └── anomaly_zoom_*.png
```

### 핵심 파일 설명

#### `src/main.py`
- 전체 파이프라인 실행
- Jupyter notebook에서 Python 스크립트로 리팩토링
- GPU 사용 및 결과 이미지 저장 기능 추가

#### `src/tcnae.py`
- **인코더**: 라인 113-125 (TCN → Conv1D → Pooling)
- **디코더**: 라인 128-136 (UpSampling → TCN → Dense)
- 마할라노비스 거리 기반 이상 점수 계산

#### `src/utilities.py`
- GPU 설정 및 감지
- 비GUI 환경 matplotlib 설정
- 타임스탬프 기반 이미지 저장

## 🔧 문제 해결 (Troubleshooting)

### 1. TensorFlow GPU 인식 문제
```bash
# 해결책: 호환성 검증된 버전 사용
pip install tensorflow==2.15.0
```

### 2. keras-tcn 호환성 문제
```bash
# 최신 호환 버전 설치
pip install keras-tcn==3.3.0
```

### 3. Optimizer API 변경 오류
```python
# tcnae.py에서 legacy optimizer 사용
adam = optimizers.legacy.Adam(lr=self.lr, ...)
```

### 4. protobuf 버전 충돌
```bash
pip install protobuf>=3.20.0 --force-reinstall
```

### 5. GUI 없는 환경에서 matplotlib 오류
```python
# utilities.py에서 설정됨
import matplotlib
matplotlib.use('Agg')  # 비GUI 백엔드 사용
```

## 📈 성능 개선 사항

### 환경 최적화
- **TensorFlow 2.3.0 → 2.15.0**: GPU 호환성 대폭 개선
- **keras-tcn 3.1.1 → 3.3.0**: 최신 TensorFlow 지원
- **GPU 메모리 설정**: 동적 메모리 할당으로 안정성 향상

### 실행 시간 개선
- **훈련 시간**: 첫 epoch 30초 → 이후 8초 (GPU 최적화)
- **전체 훈련**: 108초 (10 epochs)
- **예측 시간**: 32초 (100,000 길이 시계열)

### 결과 관리 개선
- **이미지 크기**: 25x8 → 15x6 inch (파일 크기 80% 감소)
- **해상도**: 300 DPI → 150 DPI (화면 보기 최적화)
- **파일명**: 타임스탬프 자동 추가로 중복 방지

## 📚 참고 자료

- **TCN 논문**: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- **Autoencoder**: 비지도 학습 기반 이상 탐지
- **Mackey-Glass 방정식**: 혼돈 시계열 데이터 생성

## 🤝 기여

이 프로젝트는 시계열 이상 탐지 연구 및 실무에 활용할 수 있습니다. 
최신 딥러닝 기술과 엔지니어링 모범 사례가 적용되어 있습니다.

---

**개발 환경**: Python 3.11, TensorFlow 2.15.0, CUDA 12.9, RTX A5000  
**마지막 업데이트**: 2025-07-05 