#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEP 데이터 xmv-xmeas 상관관계 분석 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import os

def analyze_tep_correlation(file_path='../data/test/test_faults/test_fault_0.csv'):
    """TEP 데이터에서 xmv와 xmeas 간의 상관관계 분석"""
    
    print("TEP 데이터 로딩 중...")
    # 전체 데이터 로드 (크기가 크므로 청크 단위로 처리)
    df = pd.read_csv(file_path)
    print(f"데이터 크기: {df.shape}")
    
    # xmv와 xmeas 변수 분리
    xmv_cols = [col for col in df.columns if col.startswith('xmv_')]
    xmeas_cols = [col for col in df.columns if col.startswith('xmeas_')]
    
    print(f"조작 변수 (xmv): {len(xmv_cols)}개")
    print(f"측정 변수 (xmeas): {len(xmeas_cols)}개")
    
    # 상관관계 계산용 데이터 준비
    correlation_data = df[xmv_cols + xmeas_cols].copy()
    
    # 전체 상관관계 매트릭스 계산
    print("\n상관관계 매트릭스 계산 중...")
    corr_matrix = correlation_data.corr()
    
    # xmv와 xmeas 간의 상관관계만 추출
    xmv_xmeas_corr = corr_matrix.loc[xmv_cols, xmeas_cols]
    
    print(f"xmv-xmeas 상관관계 매트릭스 크기: {xmv_xmeas_corr.shape}")
    
    return xmv_xmeas_corr, correlation_data


def visualize_correlation(xmv_xmeas_corr, save_dir='results'):
    """상관관계 시각화"""
    
    # 결과 폴더 생성
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('default')
    
    # 1. 전체 상관관계 히트맵
    plt.figure(figsize=(16, 10))
    
    # 색상 맵 설정 (상관관계가 강한 것을 더 진하게)
    mask = np.abs(xmv_xmeas_corr) < 0.1  # 약한 상관관계는 마스킹
    
    sns.heatmap(xmv_xmeas_corr, 
                annot=True, 
                cmap='RdBu_r',
                center=0,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'},
                mask=mask,
                square=False,
                linewidths=0.5)
    
    plt.title('TEP Data: Correlation between Manipulated Variables (xmv) and Measured Variables (xmeas)', 
              fontsize=16, pad=20)
    plt.xlabel('Measured Variables (xmeas)', fontsize=12)
    plt.ylabel('Manipulated Variables (xmv)', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tep_xmv_xmeas_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 강한 상관관계 (|r| > 0.3) 히트맵
    strong_corr = xmv_xmeas_corr.copy()
    strong_corr[np.abs(strong_corr) < 0.3] = 0
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(strong_corr, 
                annot=True, 
                cmap='RdBu_r',
                center=0,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'},
                square=False,
                linewidths=0.5)
    
    plt.title('TEP Data: Strong Correlations (|r| > 0.3) between xmv and xmeas', 
              fontsize=16, pad=20)
    plt.xlabel('Measured Variables (xmeas)', fontsize=12)
    plt.ylabel('Manipulated Variables (xmv)', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tep_strong_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 각 xmv별 가장 강한 상관관계 상위 5개 시각화
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, xmv in enumerate(xmv_xmeas_corr.index):
        if i < len(axes):
            # 해당 xmv의 상관관계 절댓값 기준으로 상위 5개 추출
            top_corr = xmv_xmeas_corr.loc[xmv].abs().nlargest(5)
            colors = ['red' if xmv_xmeas_corr.loc[xmv, var] > 0 else 'blue' for var in top_corr.index]
            
            axes[i].barh(range(len(top_corr)), 
                        [xmv_xmeas_corr.loc[xmv, var] for var in top_corr.index],
                        color=colors, alpha=0.7)
            axes[i].set_yticks(range(len(top_corr)))
            axes[i].set_yticklabels(top_corr.index, fontsize=10)
            axes[i].set_xlabel('Correlation Coefficient', fontsize=10)
            axes[i].set_title(f'{xmv} - Top 5 Correlations', fontsize=12)
            axes[i].grid(axis='x', alpha=0.3)
            axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 빈 서브플롯 제거
    for i in range(len(xmv_xmeas_corr.index), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tep_xmv_top_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 상관관계 분포 히스토그램
    plt.figure(figsize=(12, 6))
    
    # 모든 상관관계 값을 flatten
    all_corr_values = xmv_xmeas_corr.values.flatten()
    
    plt.subplot(1, 2, 1)
    plt.hist(all_corr_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Distribution of All xmv-xmeas Correlations')
    plt.grid(alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    
    plt.subplot(1, 2, 2)
    abs_corr_values = np.abs(all_corr_values)
    plt.hist(abs_corr_values, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('|Correlation Coefficient|')
    plt.ylabel('Frequency')
    plt.title('Distribution of Absolute Correlations')
    plt.grid(alpha=0.3)
    plt.axvline(x=0.3, color='red', linestyle='--', linewidth=1, label='|r| = 0.3')
    plt.axvline(x=0.5, color='orange', linestyle='--', linewidth=1, label='|r| = 0.5')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/tep_correlation_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_strongest_relationships(xmv_xmeas_corr, top_n=10):
    """가장 강한 상관관계 분석"""
    
    print(f"\n=== 가장 강한 상관관계 Top {top_n} ===")
    
    # 모든 상관관계를 (xmv, xmeas, correlation) 형태로 변환
    correlations = []
    for xmv in xmv_xmeas_corr.index:
        for xmeas in xmv_xmeas_corr.columns:
            corr_val = xmv_xmeas_corr.loc[xmv, xmeas]
            correlations.append((xmv, xmeas, corr_val))
    
    # 절댓값 기준으로 정렬
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("순위 | 조작변수 | 측정변수 | 상관계수 | 관계")
    print("-" * 50)
    for i, (xmv, xmeas, corr) in enumerate(correlations[:top_n], 1):
        relationship = "양의 상관관계" if corr > 0 else "음의 상관관계"
        print(f"{i:2d}   | {xmv:6s} | {xmeas:8s} | {corr:8.3f} | {relationship}")
    
    return correlations[:top_n]


def main():
    """메인 함수"""
    print("TEP 데이터 xmv-xmeas 상관관계 분석 시작")
    print("=" * 50)
    
    try:
        # 상관관계 분석
        xmv_xmeas_corr, correlation_data = analyze_tep_correlation()
        
        # 시각화
        print("\n시각화 생성 중...")
        visualize_correlation(xmv_xmeas_corr)
        
        # 가장 강한 상관관계 분석
        strongest_relationships = analyze_strongest_relationships(xmv_xmeas_corr)
        
        # 기본 통계 정보
        print(f"\n=== 상관관계 기본 통계 ===")
        all_corr = xmv_xmeas_corr.values.flatten()
        print(f"전체 상관관계 개수: {len(all_corr)}")
        print(f"평균 상관관계: {np.mean(all_corr):.3f}")
        print(f"표준편차: {np.std(all_corr):.3f}")
        print(f"최대 상관관계: {np.max(all_corr):.3f}")
        print(f"최소 상관관계: {np.min(all_corr):.3f}")
        print(f"강한 상관관계 (|r| > 0.3) 개수: {np.sum(np.abs(all_corr) > 0.3)}")
        print(f"매우 강한 상관관계 (|r| > 0.5) 개수: {np.sum(np.abs(all_corr) > 0.5)}")
        
        print(f"\n결과가 'results' 폴더에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 