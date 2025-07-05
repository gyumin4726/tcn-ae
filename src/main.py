#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Anomaly Detection in Mackey-Glass Time Series with TCN-AE
"""

import numpy
import time
import os
from utilities import select_gpus, plot_results  # utilities.py: Contains a few miscellaneous functions 
from tcnae import TCNAE  # tcnae.py: Specification of the TCN-AE model
import data  # data.py: Allows to generate anomalous Mackey-Glass (MG) time series 


def main():
    """Main function to execute the TCN-AE anomaly detection pipeline."""
    
    print("Starting Time Series Anomaly Detection with TCN-AE")
    print("=" * 50)
    
    # GPU 설정: 0번 GPU 사용 (여러 GPU가 있는 경우 리스트로 지정 가능)
    print("\n🔧 GPU 설정...")
    select_gpus(0)  # 0번 GPU 사용
    
    # ==========================================
    # 1. Data Loading and Preparation
    # ==========================================
    print("\n1. Loading and preparing training data...")
    
    train_ts_id = 1  # [1-10]. Train the model on Mackey-Glass time series 1
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
    
    # Train TCN-AE for 10 epochs. For a better accuracy 
    # on the test case, increase the epochs to epochs=40 
    # The training takes about 3-4 minutes for 10 epochs, 
    # and 15 minutes for 40 epochs (on Google CoLab, with GPU enabled)
    epochs = 10  # You can change this to 40 for better accuracy
    print(f"Training for {epochs} epochs...")
    
    tcn_ae.fit(train_X, train_X, batch_size=32, epochs=epochs, verbose=1)
    
    # ==========================================
    # 3. Model Testing and Evaluation
    # ==========================================
    print("\n3. Testing the model on test data...")
    
    # Test the model on another Mackey-Glass time series
    # Might take a few minutes...
    start_time = time.time()
    test_ts_id = 3  # Test the model on Mackey-Glass time series 3
    test_data = data_gen.build_data(test_ts_id, verbose=0)  # Returns a dictionary
    
    # Take the whole time series... Like the training data, the test data is standardized (zero mean and unit variance)
    test_X = test_data["scaled_series"].values[numpy.newaxis, :, :]  # We need an extra dimension for the batch-dimension
    print(f"test_X.shape: {test_X.shape}")  # This is one long time series
    
    anomaly_score = tcn_ae.predict(test_X)
    print(f"> Prediction time: {round(time.time() - start_time)} seconds.")
    
    # ==========================================
    # 4. Results Visualization
    # ==========================================
    print("\n4. Visualizing results...")
    
    # Make a plot of the anomaly-score and see how it matches the real anomaly windows
    # Vertical red bars show the actual anomalies.
    # Vertical yellow bars show regions which can be ignored (usually start and 
    # end of a time series, which lead to transient behavior for some algorithms).
    # The blue curve is the anomaly score.
    # The red horizontal line indicates a simple threshold, which is the smallest possible value that would not produce a false positive
    print("Plotting anomaly score...")
    plot_results(test_data, anomaly_score, pl_range=None, plot_signal=False, plot_anomaly_score=True)
    
    # Take a look at the MG time series: zoom into the first anomaly
    print("Plotting zoomed view of first anomaly...")
    plot_results(test_data, anomaly_score, pl_range=(40000, 42000), plot_signal=True, plot_anomaly_score=False)
    
    print("\n" + "=" * 50)
    print("TCN-AE Anomaly Detection completed successfully!")


if __name__ == "__main__":
    main() 