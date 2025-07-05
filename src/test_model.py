#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saved TCN-AE Model Testing for Time Series Anomaly Detection
"""

import numpy as np
import tensorflow as tf
import time
import os
import glob
from datetime import datetime
import utilities  # 전체 utilities 모듈 import
import data
from tcn import TCN  # TCN 레이어 import 추가


def list_saved_models():
    """checkpoint 폴더에서 저장된 모델 목록을 반환"""
    model_files = glob.glob("checkpoint/tcn_ae_model_*.h5")
    if not model_files:
        print("checkpoint 폴더에 저장된 모델이 없습니다.")
        print("   먼저 main.py를 실행하여 모델을 학습하고 저장하세요.")
        return []
    
    # 파일명을 최신순으로 정렬
    model_files.sort(reverse=True)
    
    print("저장된 모델 목록:")
    for i, model_file in enumerate(model_files, 1):
        # 파일 크기와 수정 시간 정보 추가
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
        print(f"   {i}. {model_file} ({file_size:.1f}MB, {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return model_files


def load_model(model_path):
    """저장된 모델을 불러오기"""
    try:
        print(f"\n모델 로딩 중: {model_path}")
        
        # TCN 레이어를 custom_object_scope에 포함시켜 모델 로드
        with tf.keras.utils.custom_object_scope({'TCN': TCN}):
            model = tf.keras.models.load_model(model_path)
        
        print("모델 로딩 완료!")
        
        # 모델 정보 출력
        print(f"모델 파라미터 수: {model.count_params():,}")
        print("모델 구조:")
        model.summary()
        
        return model
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return None


def test_on_different_series(model, test_series_ids=[3], verbose=True):
    """다른 시계열 데이터들에 대해 테스트"""
    print(f"\n시계열 데이터 테스트 시작...")
    print(f"   테스트할 시계열 ID: {test_series_ids}")
    
    data_gen = data.Data()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 결과 저장용 폴더 생성
    test_result_dir = f"results/test_results_{current_time}"
    os.makedirs(test_result_dir, exist_ok=True)
    
    results = {}
    
    for ts_id in test_series_ids:
        print(f"\n--- 시계열 {ts_id} 테스트 ---")
        
        try:
            # 테스트 데이터 로드
            test_data = data_gen.build_data(ts_id, verbose=0)
            test_X = test_data["scaled_series"].values[np.newaxis, :, :]
            
            # 실제 이상치 개수 정보 로깅
            num_anomalies = test_data.get("num_anomalies", "Unknown")
            print(f"   실제 이상치 개수: {num_anomalies}개")
            print(f"   데이터 shape: {test_X.shape}")
            
            # 예측 수행
            start_time = time.time()
            
            # 모델 예측
            reconstructed = model.predict(test_X, verbose=0)
            
            # shape 차이를 해결하기 위한 padding (TCNAE.predict 메서드와 동일한 처리)
            if reconstructed.shape[1] != test_X.shape[1]:
                # 끝에 padding 추가
                pad_width = ((0, 0), (0, test_X.shape[1] - reconstructed.shape[1]), (0, 0))
                reconstructed = np.pad(reconstructed, pad_width, 'constant')
                print(f"   Shape 조정: {reconstructed.shape}")
            
            # 복원 오차 계산 (MSE)
            reconstruction_error = np.mean((test_X - reconstructed) ** 2, axis=2).flatten()
            
            # 마할라노비스 거리 계산을 위한 슬라이딩 윈도우 처리
            error_window_length = 128
            
            # utilities.slide_window와 동일한 처리
            import pandas as pd
            E_rec = reconstruction_error
            Err = utilities.slide_window(pd.DataFrame(E_rec), error_window_length, verbose=0)
            Err = Err.reshape(-1, Err.shape[-1] * Err.shape[-2])
            
            # 마할라노비스 거리 계산
            sel = np.random.choice(range(Err.shape[0]), int(Err.shape[0] * 0.98))
            mu = np.mean(Err[sel], axis=0)
            cov = np.cov(Err[sel], rowvar=False)
            sq_mahalanobis = utilities.mahalanobis_distance(X=Err[:], cov=cov, mu=mu)
            
            # 이동 평균으로 스무딩
            anomaly_score = np.convolve(sq_mahalanobis, np.ones((50,))/50, mode='same')
            anomaly_score = np.sqrt(anomaly_score)
            
            prediction_time = time.time() - start_time
            
            print(f"   예측 시간: {prediction_time:.2f}초")
            print(f"   이상 점수 범위: {np.min(anomaly_score):.4f} ~ {np.max(anomaly_score):.4f}")
            
            # 결과 저장
            results[ts_id] = {
                'anomaly_score': anomaly_score,
                'test_data': test_data,
                'prediction_time': prediction_time,
                'data_shape': test_X.shape,
                'num_anomalies': num_anomalies
            }
            
            if verbose:
                # 시각화 결과 저장
                utilities.plot_results(test_data, anomaly_score, pl_range=None, plot_signal=False, 
                           plot_anomaly_score=True, filename=f'{test_result_dir}/series_{ts_id}_anomaly_score.png')
                
                # 첫 번째 이상 구간 확대
                utilities.plot_results(test_data, anomaly_score, pl_range=(40000, 42000), plot_signal=True, 
                           plot_anomaly_score=False, filename=f'{test_result_dir}/series_{ts_id}_anomaly_zoom.png')
            
        except Exception as e:
            print(f"   시계열 {ts_id} 테스트 실패: {e}")
            results[ts_id] = None
    
    print(f"\n테스트 결과가 저장되었습니다: {test_result_dir}/")
    return results


def interactive_test():
    """대화형 테스트 인터페이스"""
    print("TCN-AE 모델 테스트 도구")
    print("=" * 50)
    
    # GPU 설정
    print("\nGPU 설정...")
    utilities.select_gpus(0)
    
    # 저장된 모델 목록 표시
    model_files = list_saved_models()
    if not model_files:
        return
    
    # 모델 선택
    print(f"\n테스트할 모델을 선택하세요 (1-{len(model_files)}):")
    try:
        choice = int(input("선택: ")) - 1
        if choice < 0 or choice >= len(model_files):
            print("잘못된 선택입니다.")
            return
        
        selected_model = model_files[choice]
    except ValueError:
        print("숫자를 입력해주세요.")
        return
    
    # 모델 로드
    model = load_model(selected_model)
    if model is None:
        return
    
    # 시계열 ID 입력
    print("\n테스트할 시계열 ID를 입력하세요 (0-14):")
    try:
        ts_id = int(input("시계열 ID: "))
        if not (0 <= ts_id <= 14):
            print("시계열 ID는 0-14 범위여야 합니다.")
            return
    except ValueError:
        print("숫자를 입력해주세요.")
        return
    
    # 선택한 시계열로 테스트
    print(f"\n시계열 {ts_id}번으로 테스트 시작...")
    results = test_on_different_series(model, [ts_id])
    
    # 결과 요약
    print("\n테스트 결과 요약:")
    print("-" * 50)
    for ts_id, result in results.items():
        if result is not None:
            num_anomalies = result.get('num_anomalies', 'Unknown')
            print(f"시계열 {ts_id}: 실제 이상치 {num_anomalies}개, 예측 시간 {result['prediction_time']:.2f}초, "
                  f"이상 점수 범위 {np.min(result['anomaly_score']):.4f}~{np.max(result['anomaly_score']):.4f}")
        else:
            print(f"시계열 {ts_id}: 테스트 실패")


def main():
    """메인 함수"""
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\n\n테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main() 