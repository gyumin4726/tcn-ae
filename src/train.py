#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCN-AE Model Training for Time Series Anomaly Detection
"""

import numpy
import time
import os
from datetime import datetime
from utilities import select_gpus  # utilities.py: Contains a few miscellaneous functions 
from tcnae import TCNAE  # tcnae.py: Specification of the TCN-AE model
import data  # data.py: Allows to generate anomalous Mackey-Glass (MG) time series 


def main():
    """Main function to train the TCN-AE model."""
    
    print("TCN-AE 모델 훈련 시작")
    print("=" * 50)
    
    # GPU 설정: 0번 GPU 사용 (여러 GPU가 있는 경우 리스트로 지정 가능)
    print("\nGPU 설정...")
    select_gpus(0)  # 0번 GPU 사용
    
    # ==========================================
    # 1. Data Loading and Preparation
    # ==========================================
    print("\n1. Loading and preparing training data...")
    
    train_ts_id = 1  # [0-14]. Train the model on Mackey-Glass time series 1
    data_gen = data.Data()
    train_data = data_gen.build_data(train_ts_id, verbose=0)  # Returns a dictionary
    train_X = train_data["train_X"]  # We only need train_X (input = output) for the training process
    print(f"train_X.shape: {train_X.shape}")  # A lot of training sequences of length 1050 and dimension 1
    
    # ==========================================
    # 2. Model Building and Training
    # ==========================================
    print("\n2. Building and training the TCN-AE model...")
    
    # Build and compile the model
    tcn_ae = TCNAE()  # Use the parameters specified in the paper
    
    # 모델 파라미터 수 출력
    print(f"\n모델 파라미터 수: {tcn_ae.model.count_params():,}")
    print("모델 구조:")
    tcn_ae.model.summary()
    
    # Train TCN-AE for 10 epochs. For a better accuracy 
    # on the test case, increase the epochs to epochs=40 
    # The training takes about 3-4 minutes for 10 epochs, 
    # and 15 minutes for 40 epochs (on Google CoLab, with GPU enabled)
    epochs = 10  # You can change this to 40 for better accuracy
    print(f"\nTraining for {epochs} epochs...")
    
    training_start_time = time.time()
    tcn_ae.fit(train_X, train_X, batch_size=32, epochs=epochs, verbose=1)
    training_time = time.time() - training_start_time
    
    # ==========================================
    # 3. Model Saving
    # ==========================================
    print("\n3. Saving the trained model...")
    
    # 모델 저장 (현재 날짜와 시간 포함)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 폴더 생성 (없는 경우)
    os.makedirs("checkpoint", exist_ok=True)
    
    # 모델 저장 (전체 모델 - 구조 + 가중치 + 컴파일 정보 포함)
    model_filename = f"checkpoint/tcn_ae_model_{current_time}.h5"
    tcn_ae.model.save(model_filename)
    
    print("\n" + "=" * 50)
    print("TCN-AE 모델 훈련 완료!")
    print(f"훈련 시간: {training_time:.2f}초")
    print(f"모델 저장 위치: {model_filename}")


if __name__ == "__main__":
    main() 