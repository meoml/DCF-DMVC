"""
损失函数消融实验脚本
=================================
对以下5组配置进行对比实验：
  1. Full          - 完整损失函数 (L_recon + L_CL + L_diff)
  2. w/o Recon     - 去掉重构损失
  3. w/o CL        - 去掉对比损失
  4. w/o Diff      - 去掉差异性损失
  5. w/o both      - 只剩重构损失 (退化为普通自编码器)

使用方法：
  python run_ablation.py

"""

import os
import sys
import copy
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from sklearn.cluster import KMeans

from dataloader import load_data, MultiViewDataset
from model import DisentangleModel
from loss import TotalLoss
from metric import evaluate, cluster_acc, purity

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 实验配置
# ============================================================

# 数据集列表（可自行添加）
DATASETS = ['ORL', 'Yale', 'BBCSport', 'MSRCV1','CiteSeer', 'ACM', 'WebKB','Animal']
# DATASETS = ['Animal', 'CiteSeer', 'ACM', 'WebKB', 'ORL']

# 消融实验配置
ABLATION_CONFIGS = {
    'Full':        {'alpha': 0.1,  'beta': 0.05, 'use_recon': True,  'use_cl': True,  'use_diff': True},
    'w/o Recon':   {'alpha': 0.1,  'beta': 0.05, 'use_recon': False, 'use_cl': True,  'use_diff': True},
    'w/o CL':      {'alpha': 0.0,  'beta': 0.05, 'use_recon': True,  'use_cl': False, 'use_diff': True},
    'w/o Diff':    {'alpha': 0.1,  'beta': 0.0,  'use_recon': True,  'use_cl': True,  'use_diff': False},
    'w/o both':    {'alpha': 0.0,  'beta': 0.0,  'use_recon': True,  'use_cl': False, 'use_diff': False},
}

# 其他超参数（使用网格搜索得到的最优值，可根据需要调整）
DEFAULT_ARGS = {
    'lr': 0.001,
    'epochs': 200,
    'feature_dim': 64,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.0,
    'alpha': 0.1,
    'beta': 0.05,
    'temperature': 0.5,
    'k_neighbors': 5,
    'num_clusters': None,  # 由数据集决定
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'verbose': False,
}

# 重复实验次数（每次训练后KMeans评估多次取平均）
TRAIN_RUNS = 5
KMEANS_RUNS = 5

# 实验结果保存路径
RESULTS_DIR = './ablation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 工具函数
# ============================================================

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_data_loader(dataset_name):
    """获取数据加载器"""
    dataset, dims, num_views, num_samples, num_classes = load_data(dataset_name, './datasets/')
    dataset.num_views = num_views
    dataset.feature_dims = dims
    dataset.num_classes = num_classes
    return dataset


def train_one_epoch(model, view_list, y, optimizer, criterion, config, args):
    """训练一个epoch"""
    model.train()
    optimizer.zero_grad()
    
    xrs, zs, h_common, h_privates, att = model(view_list)
    
    # 构建邻接矩阵用于对比损失
    h_norm = F.normalize(h_common, dim=1)
    S = build_knn_graph(h_norm, args.k_neighbors)
    
    alpha_val = config['alpha'] if config['use_cl'] else 0.0
    beta_val = config['beta'] if config['use_diff'] else 0.0
    
    loss, loss_recon, loss_cl, loss_diff = criterion(
        xrs, view_list, zs, h_common, h_privates, S, alpha_val, beta_val
    )
    
    # 如果不使用重构损失，只保留其他损失
    if not config['use_recon']:
        loss = loss_cl + loss_diff
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def build_knn_graph(features, k):
    """构建KNN图"""
    features_norm = F.normalize(features, dim=1)
    dist = torch.mm(features_norm, features_norm.t())
    _, indices = torch.topk(dist, k=k+1, dim=1)
    N = features.shape[0]
    adj = torch.zeros(N, N).to(features.device)
    adj = adj.scatter_(1, indices[:, 1:], 1)  # 排除自身
    adj = (adj + adj.t()).clamp(max=1)
    return adj


def evaluate_model(model, view_list, y_true, args):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        xrs, zs, h_common, h_privates, att = model(view_list)
        
        # 归一化
        h_common = F.normalize(h_common, p=2, dim=1)
        h_np = h_common.cpu().numpy()
        y_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        
        # 聚类
        kmeans = KMeans(n_clusters=args.num_clusters, n_init=100, random_state=42)
        cluster_labels = kmeans.fit_predict(h_np)
        
        # 评估指标
        nmi, ari, acc, pur = evaluate(y_np, cluster_labels)
        
        return acc, nmi, ari, pur, cluster_labels


# ============================================================
# 单组消融实验
# ============================================================

def run_single_ablation(dataset_name, config_name, config, args, save_model=False):
    """运行单组消融实验"""
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name} | Config: {config_name}")
    print(f"{'='*60}")
    
    dataset = get_data_loader(dataset_name)
    num_views = dataset.num_views
    num_samples = dataset.num_samples
    num_clusters = dataset.num_classes
    input_dims = dataset.feature_dims
    
    args.num_clusters = num_clusters
    
    # 存储所有训练run的结果
    all_runs_results = []
    
    for train_run in range(TRAIN_RUNS):
        seed = args.seed + train_run
        set_seed(seed)
        
        # 准备数据
        view_list = [torch.FloatTensor(dataset.views[i]).to(args.device) for i in range(num_views)]
        y_true = torch.LongTensor(dataset.labels).to(args.device)
        
        # 初始化模型
        model = DisentangleModel(
            view=num_views,
            input_dims=input_dims,
            hidden_dim=args.hidden_dim,
            feature_dim=args.feature_dim,
            device=args.device,
            k=args.k_neighbors
        ).to(args.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = TotalLoss(args.temperature, args.device).to(args.device)
        
        # 训练
        best_loss = float('inf')
        best_acc = 0
        patience = 30
        no_improve = 0
        
        for epoch in range(args.epochs):
            loss = train_one_epoch(model, view_list, y_true, optimizer, criterion, config, args)
            
            # 评估
            if (epoch + 1) % 10 == 0:
                # KMeans多次评估取平均
                acc_list, nmi_list, ari_list, pur_list = [], [], [], []
                for km_run in range(KMEANS_RUNS):
                    acc, nmi, ari, pur, _ = evaluate_model(model, view_list, y_true, args)
                    acc_list.append(acc)
                    nmi_list.append(nmi)
                    ari_list.append(ari)
                    pur_list.append(pur)
                
                acc = np.mean(acc_list)
                nmi = np.mean(nmi_list)
                ari = np.mean(ari_list)
                pur = np.mean(pur_list)
                
                if acc > best_acc:
                    best_acc = acc
                    best_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                
                if args.verbose:
                    print(f"  Epoch {epoch+1:3d}: Loss={loss:.4f}, ACC={acc:.4f}, NMI={nmi:.4f}, PUR={pur:.4f}")
                
                if no_improve >= patience:
                    if args.verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # 训练结束后最终评估（KMeans多次取平均）
        acc_list, nmi_list, ari_list, pur_list = [], [], [], []
        for km_run in range(KMEANS_RUNS):
            acc, nmi, ari, pur, _ = evaluate_model(model, view_list, y_true, args)
            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            pur_list.append(pur)
        
        final_metrics = (np.mean(acc_list), np.mean(nmi_list), np.mean(ari_list), np.mean(pur_list))
        all_runs_results.append(final_metrics)
        print(f"  Train Run {train_run+1:2d}/{TRAIN_RUNS}: ACC={final_metrics[0]:.4f}, NMI={final_metrics[1]:.4f}, PUR={final_metrics[2]:.4f}")
    
    # 汇总结果
    results = np.array(all_runs_results)
    mean_results = results.mean(axis=0)
    std_results = results.std(axis=0)
    
    print(f"\n  >>> Mean: ACC={mean_results[0]:.4f}±{std_results[0]:.4f}, "
          f"NMI={mean_results[1]:.4f}±{std_results[1]:.4f}, "
          f"PUR={mean_results[2]:.4f}±{std_results[2]:.4f}")
    
    return {
        'dataset': dataset_name,
        'config': config_name,
        'ACC_mean': mean_results[0],
        'ACC_std': std_results[0],
        'NMI_mean': mean_results[1],
        'NMI_std': std_results[1],
        'PUR_mean': mean_results[2],
        'PUR_std': std_results[2],
    }


# ============================================================
# 主实验流程
# ============================================================

def run_all_ablation_experiments():
    """运行所有消融实验"""
    print("\n" + "="*70)
    print("         损失函数消融实验 (Ablation Study)")
    print("="*70)
    print(f"  数据集: {DATASETS}")
    print(f"  配置数量: {len(ABLATION_CONFIGS)}")
    print(f"  训练次数: {TRAIN_RUNS}")
    print(f"  KMeans评估次数: {KMEANS_RUNS}")
    print(f"  设备: {DEFAULT_ARGS['device']}")
    print("="*70)
    
    all_results = []
    start_time = time.time()
    
    for dataset_name in DATASETS:
        for config_name, config in ABLATION_CONFIGS.items():
            args = argparse.Namespace(**DEFAULT_ARGS)
            result = run_single_ablation(dataset_name, config_name, config, args)
            all_results.append(result)
    
    total_time = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"  所有实验完成！总耗时: {total_time/60:.1f} 分钟")
    print(f"{'='*70}")
    
    # 保存结果到CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, 'ablation_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n  结果已保存到: {csv_path}")
    
    return df


# ============================================================
# 可视化
# ============================================================

def plot_ablation_results(df):
    """绘制消融实验结果柱状图"""
    
    # 设置高级配色方案
    colors = {
        'Full':      '#2E86AB',  # 深蓝色
        'w/o Recon': '#A23B72',  # 玫红色
        'w/o CL':    '#F18F01',  # 橙色
        'w/o Diff':  '#C73E1D',  # 红色
        'w/o both':  '#6B7B8C',  # 灰色
    }
    
    configs = list(ABLATION_CONFIGS.keys())
    datasets = DATASETS
    metrics = ['ACC', 'NMI', 'PUR']
    metric_labels = {'ACC': 'Accuracy', 'NMI': 'NMI', 'PUR': 'Purity'}
    
    # 为每个数据集创建一张图
    for dataset_name in datasets:
        df_dataset = df[df['dataset'] == dataset_name]
        
        if df_dataset.empty:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f'Ablation Study: {dataset_name}', fontsize=16, fontweight='bold', y=1.02)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            means = [df_dataset[df_dataset['config'] == cfg][f'{metric}_mean'].values[0] for cfg in configs]
            stds = [df_dataset[df_dataset['config'] == cfg][f'{metric}_std'].values[0] for cfg in configs]
            
            color_list = [colors[cfg] for cfg in configs]
            
            bars = ax.bar(range(len(configs)), means, yerr=stds, capsize=5,
                         color=color_list, edgecolor='white', linewidth=1.5,
                         error_kw={'elinewidth': 2, 'capthick': 2, 'ecolor': '#333333'})
            
            # 添加数值标签
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel(metric_labels[metric], fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=9)
            ax.set_ylim(0, max(means) * 1.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', labelsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = os.path.join(RESULTS_DIR, f'ablation_{dataset_name}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"  图片已保存: {fig_path}")
        plt.close()
    
    # 创建汇总对比图（所有数据集堆叠）
    if len(datasets) > 1:
        plot_summary_heatmap(df)


def plot_summary_heatmap(df):
    """绘制汇总热力图"""
    
    metrics = ['ACC', 'NMI', 'PUR']
    configs = list(ABLATION_CONFIGS.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle('Ablation Study Summary', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # 构建热力图数据
        heatmap_data = []
        for dataset in DATASETS:
            row = []
            for config in configs:
                val = df[(df['dataset'] == dataset) & (df['config'] == config)][f'{metric}_mean'].values
                row.append(val[0] if len(val) > 0 else 0)
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        # 绘制热力图
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)
        
        # 添加数值标签
        for i in range(len(DATASETS)):
            for j in range(len(configs)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                             ha='center', va='center', color='black', fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=9)
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels(DATASETS, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    
    fig_path = os.path.join(RESULTS_DIR, 'ablation_summary_heatmap.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  汇总热力图已保存: {fig_path}")
    plt.close()


def plot_grouped_bar_chart(df):
    """绘制分组柱状图（更直观的对比）"""
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B7B8C']
    
    datasets = DATASETS
    configs = list(ABLATION_CONFIGS.keys())
    metrics = ['ACC', 'NMI', 'PUR']
    
    x = np.arange(len(configs))
    width = 0.25
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Loss Function Ablation Study', fontsize=18, fontweight='bold', y=1.02)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for d_idx, dataset in enumerate(datasets):
            values = []
            stds = []
            for config in configs:
                row = df[(df['dataset'] == dataset) & (df['config'] == config)]
                if not row.empty:
                    values.append(row[f'{metric}_mean'].values[0])
                    stds.append(row[f'{metric}_std'].values[0])
                else:
                    values.append(0)
                    stds.append(0)
            
            offset = (d_idx - len(datasets)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=dataset,
                         yerr=stds, capsize=3, alpha=0.85,
                         color=colors[d_idx % len(colors)])
        
        ax.set_ylabel(metric, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace(' ', '\n') for c in configs], fontsize=10)
        ax.legend(loc='lower right', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    
    fig_path = os.path.join(RESULTS_DIR, 'ablation_grouped_bar.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  分组柱状图已保存: {fig_path}")
    plt.close()


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loss Function Ablation Study')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                       help='数据集列表')
    parser.add_argument('--train_runs', type=int, default=TRAIN_RUNS,
                       help='训练次数')
    parser.add_argument('--kmeans_runs', type=int, default=KMEANS_RUNS,
                       help='KMeans评估次数')
    cmd_args = parser.parse_args()
    
    # 更新全局配置
    if cmd_args.datasets:
        DATASETS = cmd_args.datasets
    if cmd_args.train_runs:
        TRAIN_RUNS = cmd_args.train_runs
    if cmd_args.kmeans_runs:
        KMEANS_RUNS = cmd_args.kmeans_runs
    
    print(f"\n{'#'*70}")
    print(f"#  损失函数消融实验")
    print(f"#  数据集: {DATASETS}")
    print(f"#  配置: {list(ABLATION_CONFIGS.keys())}")
    print(f"#  训练次数: {TRAIN_RUNS}, KMeans评估次数: {KMEANS_RUNS}")
    print(f"{'#'*70}\n")
    
    # 运行所有实验
    df = run_all_ablation_experiments()
    
    # 可视化结果
    print("\n正在生成可视化图表...")
    plot_ablation_results(df)
    if len(DATASETS) > 1:
        plot_summary_heatmap(df)
    plot_grouped_bar_chart(df)
    
    print(f"\n{'='*70}")
    print(f"  实验完成！所有结果保存在: {RESULTS_DIR}")
    print(f"{'='*70}\n")
