#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEP 데이터셋을 위한 TCN-AE 모델 테스트 코드
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import glob
import json
from datetime import datetime
import utilities
from tcn import TCN
import matplotlib.pyplot as plt
import seaborn as sns


def load_tep_configs(data_folder="../data/"):
    """TEP 데이터셋 설정 파일들을 로드"""
    config_files = {
        'variable_config': os.path.join(data_folder, 'variable_config.json'),
        'label_map': os.path.join(data_folder, 'label_map.json'),
        'normalization_params': os.path.join(data_folder, 'normalization_params.json')
    }
    
    configs = {}
    for name, filepath in config_files.items():
        with open(filepath, 'r') as f:
            configs[name] = json.load(f)
    
    return configs


def load_tep_test_data(fault_id, data_folder="../data/"):
    """TEP 테스트 데이터를 로드"""
    filepath = os.path.join(data_folder, 'test', 'test_faults', f'test_fault_{fault_id}.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {filepath}")
    
    print(f"로딩 중: {filepath}")
    data = pd.read_csv(filepath)
    print(f"데이터 크기: {data.shape}")
    
    return data


def prepare_tep_sequences(data, configs, window_length=1050, input_type='x_input'):
    """TEP 데이터를 시퀀스로 변환"""
    # 입력 변수 선택
    if input_type == 'x_input':
        input_columns = configs['variable_config']['x_input']
    elif input_type == 'x_target':
        input_columns = configs['variable_config']['x_target']
    elif input_type == 'all':
        input_columns = (configs['variable_config']['x_input'] + 
                        configs['variable_config']['x_target'] + 
                        configs['variable_config']['m_input'])
    else:
        raise ValueError(f"지원하지 않는 input_type: {input_type}")
    
    # 사용 가능한 컬럼 확인
    available_columns = [col for col in input_columns if col in data.columns]
    if len(available_columns) != len(input_columns):
        missing_columns = set(input_columns) - set(available_columns)
        print(f"경고: 누락된 컬럼들: {missing_columns}")
    
    print(f"사용할 변수: {len(available_columns)}개")
    
    # 시뮬레이션별로 시퀀스 생성
    all_sequences = []
    all_labels = []
    simulation_info = []
    
    unique_runs = data['simulationRun'].unique()
    print(f"총 시뮬레이션 개수: {len(unique_runs)}")
    
    for run_id in unique_runs:
        run_data = data[data['simulationRun'] == run_id].copy()
        run_data = run_data.sort_values('sample')
        
        # 변수 데이터 추출
        multivar_data = run_data[available_columns].values
        
        # 시퀀스 길이 확인
        if len(multivar_data) >= window_length:
            # 슬라이딩 윈도우 적용
            sequences = utilities.slide_window(multivar_data, window_length, verbose=0)
            
            # 각 시퀀스에 대해 라벨 정보 저장
            fault_number = run_data['faultNumber'].iloc[0]
            labels = np.full(len(sequences), fault_number)
            
            all_sequences.append(sequences)
            all_labels.append(labels)
            
            # 시뮬레이션 정보 저장
            simulation_info.extend([{
                'simulationRun': run_id,
                'faultNumber': fault_number,
                'sequence_idx': i
            } for i in range(len(sequences))])
    
    # 모든 시퀀스 합치기
    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"생성된 시퀀스 개수: {len(X)}")
    print(f"시퀀스 크기: {X.shape}")
    print(f"라벨 분포: {np.bincount(y)}")
    
    return X, y, simulation_info


def test_tep_model(model, fault_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                   data_folder="../data/", input_type='x_input', verbose=True):
    """TEP 데이터셋에 대해 모델 테스트 수행"""
    
    # 설정 파일 로드
    configs = load_tep_configs(data_folder)
    
    # 결과 저장용 폴더 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_result_dir = f"results/tep_test_results_{current_time}"
    os.makedirs(test_result_dir, exist_ok=True)
    
    # 전체 결과 저장
    all_results = {}
    
    for fault_id in fault_ids:
        print(f"\n--- 고장 유형 {fault_id} 테스트 ---")
        print(f"고장 이름: {configs['label_map'][str(fault_id)]}")
        
        try:
            # 데이터 로드
            data = load_tep_test_data(fault_id, data_folder)
            
            # 시퀀스 생성
            X, y, simulation_info = prepare_tep_sequences(data, configs, 
                                                         window_length=1050, 
                                                         input_type=input_type)
            
            # 모델 예측
            print("모델 예측 수행 중...")
            start_time = time.time()
            
            # 배치 단위로 예측 (메모리 절약)
            batch_size = 32
            predictions = []
            
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_pred = model.predict(batch_X, verbose=0)
                predictions.append(batch_pred)
            
            reconstructed = np.concatenate(predictions, axis=0)
            
            # Shape 조정 (필요시)
            if reconstructed.shape[1] != X.shape[1]:
                pad_width = ((0, 0), (0, X.shape[1] - reconstructed.shape[1]), (0, 0))
                reconstructed = np.pad(reconstructed, pad_width, 'constant')
                print(f"Shape 조정됨: {reconstructed.shape}")
            
            # 복원 오차 계산
            reconstruction_error = np.mean((X - reconstructed) ** 2, axis=2)
            
            # 마할라노비스 거리 계산
            print("이상 점수 계산 중...")
            error_window_length = 128
            
            # 시퀀스별 이상 점수 계산
            anomaly_scores = []
            
            for i in range(len(reconstruction_error)):
                E_rec = reconstruction_error[i]
                
                # 슬라이딩 윈도우
                if len(E_rec) >= error_window_length:
                    Err = utilities.slide_window(pd.DataFrame(E_rec), error_window_length, verbose=0)
                    Err = Err.reshape(-1, Err.shape[-1] * Err.shape[-2])
                    
                    # 마할라노비스 거리
                    if len(Err) > 0:
                        sel = np.random.choice(range(Err.shape[0]), 
                                             min(int(Err.shape[0] * 0.98), len(Err)), 
                                             replace=False)
                        mu = np.mean(Err[sel], axis=0)
                        cov = np.cov(Err[sel], rowvar=False)
                        
                        if np.linalg.det(cov) != 0:
                            sq_mahalanobis = utilities.mahalanobis_distance(X=Err, cov=cov, mu=mu)
                            anomaly_score = np.convolve(sq_mahalanobis, np.ones((50,))/50, mode='same')
                            anomaly_score = np.sqrt(anomaly_score)
                        else:
                            anomaly_score = np.mean(reconstruction_error[i])
                    else:
                        anomaly_score = np.mean(reconstruction_error[i])
                else:
                    anomaly_score = np.mean(reconstruction_error[i])
                
                anomaly_scores.append(anomaly_score)
            
            prediction_time = time.time() - start_time
            
            # 결과 저장
            result = {
                'fault_id': fault_id,
                'fault_name': configs['label_map'][str(fault_id)],
                'num_sequences': len(X),
                'anomaly_scores': anomaly_scores,
                'labels': y,
                'reconstruction_error': reconstruction_error,
                'prediction_time': prediction_time,
                'simulation_info': simulation_info
            }
            
            all_results[fault_id] = result
            
            print(f"테스트 완료: {prediction_time:.2f}초")
            print(f"시퀀스 개수: {len(X)}")
            
            # 시각화 (각 고장별 대표 1개)
            if verbose:
                plot_tep_results(result, test_result_dir, fault_id)
            
        except Exception as e:
            print(f"고장 {fault_id} 테스트 실패: {e}")
            all_results[fault_id] = None
    
    # 전체 결과 요약
    print_tep_summary(all_results, test_result_dir)
    
    return all_results


def plot_tep_results(result, save_dir, fault_id):
    """TEP 테스트 결과 시각화"""
    fault_name = result['fault_name']
    anomaly_scores = result['anomaly_scores']
    
    # 대표 시뮬레이션 선택 (첫 번째)
    if len(anomaly_scores) > 0:
        if isinstance(anomaly_scores[0], np.ndarray):
            # 첫 번째 시뮬레이션의 이상 점수
            sample_score = anomaly_scores[0]
            
            plt.figure(figsize=(15, 6))
            plt.plot(sample_score, label=f'Fault {fault_id} Anomaly Score')
            plt.title(f'TEP Fault {fault_id}: {fault_name}')
            plt.xlabel('Time Steps')
            plt.ylabel('Anomaly Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'fault_{fault_id}_anomaly_score.png'), dpi=300)
            plt.close()
        else:
            # 스칼라 점수들의 분포
            plt.figure(figsize=(10, 6))
            plt.hist(anomaly_scores, bins=50, alpha=0.7, label=f'Fault {fault_id}')
            plt.title(f'TEP Fault {fault_id} Anomaly Score Distribution: {fault_name}')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'fault_{fault_id}_score_distribution.png'), dpi=300)
            plt.close()


def print_tep_summary(all_results, save_dir):
    """TEP 테스트 결과 요약 출력"""
    print("\n" + "="*80)
    print("TEP 데이터셋 테스트 결과 요약")
    print("="*80)
    
    summary_data = []
    
    for fault_id, result in all_results.items():
        if result is not None:
            fault_name = result['fault_name']
            num_sequences = result['num_sequences']
            prediction_time = result['prediction_time']
            
            # 이상 점수 통계
            anomaly_scores = result['anomaly_scores']
            if len(anomaly_scores) > 0:
                if isinstance(anomaly_scores[0], np.ndarray):
                    # 각 시퀀스의 평균 이상 점수
                    avg_scores = [np.mean(score) for score in anomaly_scores]
                    mean_score = np.mean(avg_scores)
                    std_score = np.std(avg_scores)
                else:
                    # 스칼라 점수들
                    mean_score = np.mean(anomaly_scores)
                    std_score = np.std(anomaly_scores)
            else:
                mean_score = std_score = 0
            
            summary_data.append({
                'Fault ID': fault_id,
                'Fault Name': fault_name,
                'Sequences': num_sequences,
                'Mean Score': f"{mean_score:.4f}",
                'Std Score': f"{std_score:.4f}",
                'Time (s)': f"{prediction_time:.2f}"
            })
            
            print(f"고장 {fault_id:2d}: {fault_name:50s} | "
                  f"시퀀스: {num_sequences:4d} | "
                  f"점수: {mean_score:.4f}±{std_score:.4f} | "
                  f"시간: {prediction_time:.2f}s")
        else:
            print(f"고장 {fault_id:2d}: 테스트 실패")
    
    # 결과를 CSV로 저장
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(save_dir, 'tep_test_summary.csv'), index=False)
        print(f"\n결과가 저장되었습니다: {save_dir}/")


def list_saved_models():
    """저장된 모델 목록 반환"""
    model_files = glob.glob("checkpoint/tcn_ae_model_*.h5")
    if not model_files:
        print("checkpoint 폴더에 저장된 모델이 없습니다.")
        return []
    
    model_files.sort(reverse=True)
    print("저장된 모델 목록:")
    for i, model_file in enumerate(model_files, 1):
        file_size = os.path.getsize(model_file) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_file))
        print(f"   {i}. {model_file} ({file_size:.1f}MB, {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return model_files


def load_model(model_path):
    """저장된 모델 불러오기"""
    try:
        print(f"\n모델 로딩 중: {model_path}")
        with tf.keras.utils.custom_object_scope({'TCN': TCN}):
            model = tf.keras.models.load_model(model_path)
        print("모델 로딩 완료!")
        return model
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return None


def main():
    """메인 함수"""
    print("TEP 데이터셋 TCN-AE 모델 테스트 도구")
    print("=" * 60)
    
    # GPU 설정
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
    
    # 테스트할 고장 유형 선택
    print("\n테스트할 고장 유형을 선택하세요:")
    print("1. 모든 고장 유형 (0-12)")
    print("2. 특정 고장 유형 선택")
    
    try:
        test_choice = int(input("선택: "))
        if test_choice == 1:
            fault_ids = list(range(13))
        elif test_choice == 2:
            fault_input = input("고장 ID를 입력하세요 (0-12, 쉼표로 구분): ")
            fault_ids = [int(x.strip()) for x in fault_input.split(',')]
            fault_ids = [f for f in fault_ids if 0 <= f <= 12]
        else:
            print("잘못된 선택입니다.")
            return
    except ValueError:
        print("올바른 숫자를 입력해주세요.")
        return
    
    print(f"\n테스트할 고장 유형: {fault_ids}")
    
    # 테스트 실행
    results = test_tep_model(model, fault_ids=fault_ids, verbose=True)
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    main() 