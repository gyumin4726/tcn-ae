#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEP 데이터를 사용한 다변량 TCN-AE 모델 훈련

개선: 52개 변수를 하나의 모델로 동시 학습 (변수 간 상관관계 학습)

중요: 이상 탐지를 위해 정상 데이터(fault_0)만 사용하여 학습
- fault_0: 정상 상태 (Normal Operation)
- fault_1~12: 고장 상태 (테스트 시에만 사용)

오토인코더 기반 이상 탐지 원리:
1. 정상 데이터만으로 학습하여 정상 패턴 재구성 능력 학습
2. 이상 데이터 입력 시 재구성 오차가 크게 나타남
3. 재구성 오차를 기반으로 이상 여부 판단
"""

import numpy as np
import pandas as pd
import time
import os
import glob
from datetime import datetime
from tqdm import tqdm
import utilities
from tcnae import TCNAE


def load_tep_training_data():
    """TEP 훈련 데이터 로드 및 전처리 (정상 데이터만)"""
    print("TEP 훈련 데이터 로딩 중...")
    
    # 정상 데이터만 로드 (fault_0)
    normal_file = "../data/train/train_faults/train_fault_0.csv"
    
    print(f"정상 데이터 파일: {normal_file}")
    
    if not os.path.exists(normal_file):
        raise FileNotFoundError(f"정상 데이터 파일을 찾을 수 없습니다: {normal_file}")
    
    # 정상 데이터 로드
    print("정상 데이터(fault_0) 로딩 중...")
    data = pd.read_csv(normal_file)
    print(f"정상 데이터 크기: {data.shape}")
    
    # 변수 분리 (52개 변수: xmeas_1~41, xmv_1~11)
    variable_cols = [col for col in data.columns if col.startswith(('xmeas_', 'xmv_'))]
    
    print(f"총 변수: {len(variable_cols)}개")
    
    # 시뮬레이션별로 데이터 재구성
    print("\n시뮬레이션별 데이터 재구성 중...")
    
    # 시뮬레이션 ID로만 그룹화 (fault_0만 있으니까)
    grouped = data.groupby('simulationRun')
    
    sequences = []
    
    for sim_run, group in tqdm(grouped, desc="시뮬레이션 처리"):
        # 시간 순서로 정렬
        group_sorted = group.sort_values('sample')
        
        # 모든 변수 데이터 추출 (다변량)
        sequence = group_sorted[variable_cols].values  # (500, 52)
        
        sequences.append(sequence)
    
    sequences = np.array(sequences)  # (num_sequences, 500, 52)
    print(f"최종 다변량 시퀀스 데이터 크기: {sequences.shape}")
    
    return sequences


def main():
    """메인 함수"""
    print("TEP 다변량 TCN-AE 모델 훈련 시작")
    print("=" * 60)
    print("개선: 52개 변수를 하나의 모델로 동시 학습")
    print("변수 간 상관관계 학습")
    print("더 강력한 패턴 인식")
    print("빠른 훈련 시간")
    print("h5 파일 하나만 사용")
    print("=" * 60)
    
    # GPU 설정
    print("\nGPU 설정...")
    utilities.select_gpus(0)
    
    # 1. 데이터 로드
    sequences = load_tep_training_data()
    
    # 2. 모델 생성
    print(f"\n다변량 TCN-AE 모델 생성...")
    model = TCNAE(ts_dimension=52, verbose=1)
    
    print(f"모델 파라미터 수: {model.model.count_params():,}")
    
    # 3. 모델 출력 길이 확인 및 데이터 조정
    test_output = model.model.predict(sequences[:1], verbose=0)
    output_length = test_output.shape[1]
    input_length = sequences.shape[1]
    
    print(f"모델 입력 길이: {input_length}")
    print(f"모델 출력 길이: {output_length}")
    
    if output_length < input_length:
        start_idx = input_length - output_length
        sequences = sequences[:, start_idx:, :]
        print(f"데이터 길이 조정: {input_length} -> {sequences.shape[1]}")
    
    # 4. 모델 훈련
    print(f"\n다변량 TCN-AE 훈련 시작...")
    print(f"입력 형태: {sequences.shape} (시퀀스, 시간, 변수)")
    
    epochs = 40
    batch_size = 32
    
    start_time = time.time()
    
    print(f"훈련 설정: epochs={epochs}, batch_size={batch_size}")
    model.fit(sequences, sequences, 
              batch_size=batch_size, 
              epochs=epochs, 
              verbose=1)
    
    training_time = time.time() - start_time
    print(f"다변량 TCN-AE 훈련 완료 - 시간: {training_time:.2f}초")
    
    # 5. 모델 저장 (h5 파일 하나만)
    print("\n모델 저장 중...")
    
    # 타임스탬프 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 폴더 생성
    model_dir = "checkpoint"
    os.makedirs(model_dir, exist_ok=True)
    
    # 모델 저장
    model_filename = f"{model_dir}/tep_tcnae_model_{current_time}.h5"
    model.model.save(model_filename)
    
    print(f"모델 저장 완료: {model_filename}")
    print(f"변수 수: 52")
    print(f"모델 크기: {os.path.getsize(model_filename) / (1024*1024):.1f}MB")
    
    # 6. 훈련 결과 요약
    print("\n" + "=" * 60)
    print("TEP 다변량 TCN-AE 모델 훈련 완료!")
    print(f"총 훈련 시간: {training_time:.2f}초")
    print(f"훈련 데이터 크기: {sequences.shape}")
    print(f"모델 타입: 다변량 TCN-AE (52차원 동시 처리)")
    
    print(f"\n저장된 파일:")
    print(f"  모델: {model_filename}")
    
    print(f"\n주요 개선사항:")
    print(f"  - 52개 개별 모델 -> 1개 통합 모델")
    print(f"  - 변수 간 상관관계 학습")
    print(f"  - 훨씬 빠른 훈련 속도")
    print(f"  - 더 강력한 이상 탐지 성능")
    print(f"  - 단순한 h5 파일 하나만 사용")
    
    print("\n다음 단계:")
    print("  test_tep_model.py를 실행하여 성능 테스트")


if __name__ == "__main__":
    main() 