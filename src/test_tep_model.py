#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다변량 TEP 모델을 사용한 테스트 스크립트

개선: 52개 변수를 하나의 모델로 동시 처리 (변수 간 상관관계 고려)
"""

import numpy as np
import pandas as pd
import os
import glob
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용하는 backend
import matplotlib.pyplot as plt
import utilities
from tcnae import TCNAE
import tensorflow as tf


def load_tep_model():
    """TEP 모델을 h5 파일에서 불러오기"""
    try:
        # h5 모델 파일 찾기
        model_files = glob.glob("checkpoint/tep_tcnae_model_*.h5")
        if not model_files:
            print("checkpoint 폴더에 저장된 TEP 모델이 없습니다.")
            return None
        
        # 최신 모델 파일 사용
        model_files.sort(reverse=True)
        latest_model = model_files[0]
        
        print(f"\nTEP 모델 로딩 중: {latest_model}")
        
        # 모델 로드 (TCN은 tcnae에서 자동으로 처리)
        model = tf.keras.models.load_model(latest_model)
        
        print("h5 모델 로딩 완료!")
        print(f"모델 입력 크기: {model.input_shape}")
        print(f"모델 출력 크기: {model.output_shape}")
        
        return model
        
    except Exception as e:
        print(f"TEP 모델 로딩 실패: {e}")
        print("모델 로드 중 오류 발생. tcnae.py를 통해 다시 시도합니다.")
        
        # tcnae.py를 통한 모델 로드 시도
        try:
            tcnae = TCNAE(ts_dimension=52, verbose=0)
            tcnae.model = tf.keras.models.load_model(latest_model)
            print("tcnae를 통한 모델 로딩 완료!")
            return tcnae.model
        except Exception as e2:
            print(f"tcnae를 통한 모델 로딩도 실패: {e2}")
            return None


def load_tep_test_data(fault_id, data_folder="../data/"):
    """TEP 테스트 데이터 로드 (다변량 형태)"""
    filepath = os.path.join(data_folder, 'test', 'test_faults', f'test_fault_{fault_id}.csv')
    
    print(f"테스트 데이터 로딩: {filepath}")
    df = pd.read_csv(filepath)
    
    # 변수 분리 (52개 변수: xmeas_1~41, xmv_1~11)
    variable_cols = [col for col in df.columns if col.startswith(('xmeas_', 'xmv_'))]
    
    print(f"데이터 크기: {df.shape}")
    print(f"변수 수: {len(variable_cols)}")
    
    # 시뮬레이션별로 데이터 재구성 (다변량)
    sequences = []
    
    grouped = df.groupby('simulationRun')
    
    for sim_run, group in grouped:
        # 시간 순서로 정렬
        group_sorted = group.sort_values('sample')
        
        # 모든 변수 데이터 추출 (다변량)
        sequence = group_sorted[variable_cols].values  # (960, 52)
        
        sequences.append(sequence)
    
    sequences = np.array(sequences)  # (num_sequences, 960, 52)
    
    return sequences


def test_fault_with_model(model, test_data, fault_id):
    """모델로 특정 고장 유형에 대해 테스트 수행"""
    print(f"\n고장 {fault_id} 테스트 시작...")
    
    num_sequences, sequence_length, num_variables = test_data.shape
    print(f"테스트 데이터 크기: {test_data.shape}")
    
    # 모델 출력 길이 확인 및 데이터 조정
    test_output = model.predict(test_data[:1], verbose=0)
    output_length = test_output.shape[1]
    input_length = test_data.shape[1]
    
    print(f"모델 입력 길이: {input_length}")
    print(f"모델 출력 길이: {output_length}")
    
    if output_length < input_length:
        start_idx = input_length - output_length
        test_data = test_data[:, start_idx:, :]
        print(f"데이터 길이 조정: {input_length} -> {test_data.shape[1]}")
    
    # 모델 예측 수행
    print("모델 예측 중...")
    start_time = time.time()
    
    # TCNAE 객체 생성하여 predict 메서드 사용 (이상 점수 계산)
    tcn_ae = TCNAE(ts_dimension=52, verbose=0)
    tcn_ae.model = model  # 로드된 모델 사용
    
    # 각 시퀀스별로 예측 수행
    anomaly_scores = []
    
    for seq_idx in tqdm(range(num_sequences), desc="시퀀스 처리"):
        seq_data = test_data[seq_idx:seq_idx+1]  # (1, length, 52)
        
        # TCNAE predict 메서드 사용 (이상 점수 직접 계산)
        seq_anomaly_score = tcn_ae.predict(seq_data)
        
        anomaly_scores.append(seq_anomaly_score)
    
    anomaly_scores = np.array(anomaly_scores)  # (num_sequences, length)
    
    prediction_time = time.time() - start_time
    
    print(f"예측 완료: {anomaly_scores.shape}")
    print(f"예측 시간: {prediction_time:.2f}초")
    print(f"이상 점수 범위: {np.min(anomaly_scores):.4f} ~ {np.max(anomaly_scores):.4f}")
    
    return anomaly_scores


def visualize_results(anomaly_scores, fault_id, save_dir="results", global_y_range=None):
    """결과 시각화 (스케일 통일)"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n고장 {fault_id} 결과 시각화 중...")
    
    # 1. 대표 시뮬레이션의 시간별 변화
    plt.figure(figsize=(10, 5))
    
    # 첫 번째 시뮬레이션 시각화
    representative_scores = anomaly_scores[0]  # (length,)
    time_steps = np.arange(len(representative_scores))
    
    plt.plot(time_steps, representative_scores, 'b-', linewidth=1, alpha=0.8, label='Representative Simulation')
    
    # 모든 시뮬레이션의 평균 추가
    mean_scores = np.mean(anomaly_scores, axis=0)  # (length,)
    plt.plot(time_steps, mean_scores, 'r-', linewidth=2, label='Average of All Simulations')
    
    # 정상과 고장 구분하여 제목 설정
    if fault_id == 0:
        title = f'TEP Fault {fault_id} (Normal): Time-series Anomaly Scores'
    else:
        title = f'TEP Fault {fault_id} (Abnormal): Time-series Anomaly Scores'
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Anomaly Score (Lower=Normal, Higher=Abnormal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 전체 범위 통일 (제공된 경우)
    if global_y_range is not None:
        plt.ylim(global_y_range[0], global_y_range[1])
    else:
        # 동적 Y축 범위 설정
        y_min = np.min(anomaly_scores) * 0.95
        y_max = np.max(anomaly_scores) * 1.05
        plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'tep_fault_{fault_id}_time_series.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 평균 점수 분포 히스토그램
    plt.figure(figsize=(10, 5))
    
    # 각 시뮬레이션의 평균 점수 계산
    mean_scores_per_sim = np.mean(anomaly_scores, axis=1)  # (num_sequences,)
    
    plt.hist(mean_scores_per_sim, bins=30, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=0.5)
    
    # 정상과 고장 구분하여 제목 설정
    if fault_id == 0:
        title = f'TEP Fault {fault_id} (Normal): Mean Anomaly Score Distribution'
    else:
        title = f'TEP Fault {fault_id} (Abnormal): Mean Anomaly Score Distribution'
    
    plt.title(title)
    plt.xlabel('Mean Anomaly Score (Lower=Normal, Higher=Abnormal)')
    plt.ylabel('Frequency')
    
    # 통계 정보 표시
    overall_mean = np.mean(mean_scores_per_sim)
    overall_std = np.std(mean_scores_per_sim)
    
    plt.axvline(overall_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {overall_mean:.3e}')
    plt.axvline(overall_mean + overall_std, color='orange', linestyle=':', 
                linewidth=1, alpha=0.7, label=f'Mean+Std: {overall_mean+overall_std:.3e}')
    plt.axvline(overall_mean - overall_std, color='orange', linestyle=':', 
                linewidth=1, alpha=0.7, label=f'Mean-Std: {overall_mean-overall_std:.3e}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'tep_fault_{fault_id}_distribution.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"시각화 완료: {save_dir}/tep_fault_{fault_id}_*.png")


def create_unified_visualization(all_results, save_dir="results"):
    """모든 고장 유형의 통합 시각화"""
    print(f"\n통합 시각화 생성 중...")
    
    # 색상 맵 (13개 고장 유형용)
    colors = plt.cm.tab20(np.linspace(0, 1, 13))
    
    # 1. 모든 고장 유형의 평균 그래프를 한 이미지에
    plt.figure(figsize=(10, 5))
    
    for fault_id, result in all_results.items():
        if result and 'anomaly_scores' in result:
            # 각 고장의 평균 점수 계산
            mean_scores = np.mean(result['anomaly_scores'], axis=0)  # (length,)
            time_steps = np.arange(len(mean_scores))
            
            # 정상과 고장 구분
            if fault_id == 0:
                color = 'green'
                linewidth = 3
                label = f'Fault {fault_id} (Normal)'
                alpha = 1.0
            else:
                color = colors[fault_id]
                linewidth = 2
                label = f'Fault {fault_id} (Abnormal)'
                alpha = 0.8
            
            plt.plot(time_steps, mean_scores, 
                    color=color, 
                    linewidth=linewidth, 
                    label=label,
                    alpha=alpha)
    
    plt.title('TEP All Faults: Time-series Anomaly Scores Comparison\n(Lower Score = Normal, Higher Score = Abnormal)')
    plt.xlabel('Time Step')
    plt.ylabel('Anomaly Score (Lower=Normal, Higher=Abnormal)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tep_all_faults_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 고장별 평균 이상 점수 바 차트
    plt.figure(figsize=(10, 5))
    
    fault_ids = []
    mean_anomaly_scores = []
    std_anomaly_scores = []
    
    for fault_id, result in all_results.items():
        if result and 'anomaly_scores' in result:
            # 각 고장의 전체 평균 점수
            fault_mean = np.mean(result['anomaly_scores'])
            fault_std = np.std(result['anomaly_scores'])
            
            fault_ids.append(fault_id)
            mean_anomaly_scores.append(fault_mean)
            std_anomaly_scores.append(fault_std)
    
    # 정상(0)과 고장(1~12) 구분하여 색상 설정
    bar_colors = []
    for fault_id in fault_ids:
        if fault_id == 0:
            bar_colors.append('green')  # 정상 - 초록색
        else:
            bar_colors.append('red')    # 고장 - 빨간색
    
    bars = plt.bar(fault_ids, mean_anomaly_scores, 
                   yerr=std_anomaly_scores, 
                   capsize=5, 
                   alpha=0.7,
                   color=bar_colors)
    
    plt.title('TEP Faults: Mean Anomaly Score Comparison\n(Green=Normal, Red=Fault)')
    plt.xlabel('Fault ID (0=Normal, 1-12=Faults)')
    plt.ylabel('Mean Anomaly Score (Lower=Normal, Higher=Abnormal)')
    plt.xticks(fault_ids)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar, mean_val in zip(bars, mean_anomaly_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                f'{mean_val:.2e}', ha='center', va='bottom', fontsize=8)
    
    # 범례 추가
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Normal (Fault 0)'),
        Patch(facecolor='red', alpha=0.7, label='Faults (Fault 1-12)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tep_faults_mean_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"통합 시각화 완료:")
    print(f"  - {save_dir}/tep_all_faults_comparison.png")
    print(f"  - {save_dir}/tep_faults_mean_comparison.png")


def main():
    """메인 함수"""
    print("TEP 모델 테스트 도구")
    print("=" * 60)
    print("개선: 52개 변수를 하나의 모델로 동시 처리")
    print("h5 파일 하나만 사용")
    print("=" * 60)
    
    # GPU 설정
    utilities.select_gpus(0)
    
    # 모델 로드
    model = load_tep_model()
    if model is None:
        print("모델 로딩 실패!")
        print("먼저 train_tep_model.py를 실행하여 모델을 훈련하세요.")
        return
    
    print(f"모델 크기: {model.count_params():,} 파라미터")
    
    # 테스트할 고장 유형 선택
    print("\n테스트할 고장 유형을 선택하세요:")
    print("1. 특정 고장 (예: 0)")
    print("2. 여러 고장 (예: 0,1,2)")
    print("3. 모든 고장 (0-12)")
    
    choice = input("선택 (1-3): ").strip()
    
    if choice == "1":
        fault_id = int(input("고장 ID (0-12): "))
        fault_ids = [fault_id]
    elif choice == "2":
        fault_input = input("고장 IDs (쉼표로 구분, 예: 0,1,2): ")
        fault_ids = [int(x.strip()) for x in fault_input.split(',')]
    elif choice == "3":
        fault_ids = list(range(13))  # 0-12
    else:
        print("잘못된 선택입니다.")
        return
    
    print(f"\n테스트할 고장 유형: {fault_ids}")
    
    # 결과 저장 폴더 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/tep_test_results_{current_time}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 각 고장 유형별 테스트 수행
    all_results = {}
    
    # 1단계: 모든 고장 유형 테스트 수행
    for fault_id in fault_ids:
        print(f"\n{'='*60}")
        print(f"고장 {fault_id} 테스트 시작")
        print(f"{'='*60}")
        
        try:
            # 테스트 데이터 로드
            test_data = load_tep_test_data(fault_id)
            
            # 시간 측정
            start_time = time.time()
            
            # 테스트 수행
            anomaly_scores = test_fault_with_model(model, test_data, fault_id)
            
            test_time = time.time() - start_time
            
            # 결과 저장
            all_results[fault_id] = {
                'anomaly_scores': anomaly_scores,
                'test_time': test_time,
                'num_sequences': len(test_data)
            }
            
            print(f"고장 {fault_id} 테스트 완료!")
            print(f"  시뮬레이션 수: {len(test_data)}")
            print(f"  처리 시간: {test_time:.2f}초")
            print(f"  이상 점수 범위: {np.min(anomaly_scores):.4f} ~ {np.max(anomaly_scores):.4f}")
            
        except Exception as e:
            print(f"고장 {fault_id} 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 2단계: 전체 Y축 범위 계산 (스케일 통일)
    print(f"\n{'='*60}")
    print("전체 Y축 범위 계산 중...")
    print(f"{'='*60}")
    
    global_y_min = float('inf')
    global_y_max = float('-inf')
    
    for fault_id, result in all_results.items():
        if result and 'anomaly_scores' in result:
            scores = result['anomaly_scores']
            y_min = np.min(scores)
            y_max = np.max(scores)
            
            global_y_min = min(global_y_min, y_min)
            global_y_max = max(global_y_max, y_max)
    
    # 약간의 여백 추가
    global_y_range = (global_y_min * 0.95, global_y_max * 1.05)
    print(f"전체 Y축 범위: {global_y_range[0]:.4f} ~ {global_y_range[1]:.4f}")
    
    # 3단계: 통일된 스케일로 개별 시각화
    print(f"\n{'='*60}")
    print("통일된 스케일로 개별 시각화 중...")
    print(f"{'='*60}")
    
    for fault_id, result in all_results.items():
        if result and 'anomaly_scores' in result:
            visualize_results(result['anomaly_scores'], fault_id, save_dir, global_y_range)
    
    # 4단계: 통합 시각화
    print(f"\n{'='*60}")
    print("통합 시각화 생성 중...")
    print(f"{'='*60}")
    
    create_unified_visualization(all_results, save_dir)
    
    # 전체 결과 요약
    print(f"\n{'='*60}")
    print("TEP 테스트 결과 요약")
    print(f"{'='*60}")
    print("이상 점수 해석: 낮은 점수=정상, 높은 점수=이상")
    print("기대 결과: Fault 0(정상) < Fault 1~12(고장)")
    print(f"{'='*60}")
    
    for fault_id, result in all_results.items():
        if result:
            fault_type = "정상" if fault_id == 0 else "고장"
            mean_score = np.mean(result['anomaly_scores'])
            
            print(f"고장 {fault_id:2d} ({fault_type}): {result['num_sequences']:3d}개 시뮬레이션, "
                  f"{result['test_time']:6.2f}초, "
                  f"평균 점수: {mean_score:.4f}, "
                  f"점수 범위: {np.min(result['anomaly_scores']):.4f}~{np.max(result['anomaly_scores']):.4f}")
    
    print(f"\n결과 저장 위치: {save_dir}/")
    print("생성된 파일:")
    print("  - 개별 시각화: tep_fault_X_time_series.png, tep_fault_X_distribution.png")
    print("  - 통합 시각화: tep_all_faults_comparison.png, tep_faults_mean_comparison.png")
    print("TEP 테스트 완료!")


if __name__ == "__main__":
    main() 