"""
收敛性分析实验脚本
=================================
绘制训练过程中 ACC、NMI、PUR 及目标函数值的收敛曲线

使用方法：
  python run_convergence.py
  python run_convergence.py --datasets ORL MSRCV1  # 指定数据集
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

# 数据集列表
DATASETS = ['ORL', 'Yale', 'BBCSport', 'MSRCV1', 'CiteSeer', 'ACM', 'WebKB', 'Animal']

# 训练参数
DEFAULT_ARGS = {
    'lr': 0.001,
    'epochs': 500,           # 收敛性分析使用500个epoch
    'feature_dim': 64,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.0,
    'alpha': 0.1,            # 对比损失权重
    'beta': 0.05,            # 差异性损失权重
    'temperature': 0.5,
    'k_neighbors': 5,
    'num_clusters': None,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'verbose': True,
}

# 重复实验次数（KMeans评估，用于平滑曲线）
KMEANS_RUNS = 5

# 结果保存路径
RESULTS_DIR = './convergence_results'
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


def train_one_epoch(model, view_list, y, optimizer, criterion, args):
    """训练一个epoch，返回总损失及各分损失"""
    model.train()
    optimizer.zero_grad()
    
    xrs, zs, h_common, h_privates, att = model(view_list)
    
    # 构建邻接矩阵用于对比损失
    h_norm = F.normalize(h_common, dim=1)
    S = build_knn_graph(h_norm, args.k_neighbors)
    
    loss, loss_recon, loss_cl, loss_diff = criterion(
        xrs, view_list, zs, h_common, h_privates, S, args.alpha, args.beta
    )
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), loss_recon.item(), loss_cl.item(), loss_diff.item()


def build_knn_graph(features, k):
    """构建KNN图"""
    features_norm = F.normalize(features, dim=1)
    dist = torch.mm(features_norm, features_norm.t())
    _, indices = torch.topk(dist, k=k+1, dim=1)
    N = features.shape[0]
    adj = torch.zeros(N, N).to(features.device)
    adj = adj.scatter_(1, indices[:, 1:], 1)
    adj = (adj + adj.t()).clamp(max=1)
    return adj


def evaluate_model(model, view_list, y_true, args):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        xrs, zs, h_common, h_privates, att = model(view_list)
        
        h_common = F.normalize(h_common, p=2, dim=1)
        h_np = h_common.cpu().numpy()
        y_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        
        kmeans = KMeans(n_clusters=args.num_clusters, n_init=100, random_state=42)
        cluster_labels = kmeans.fit_predict(h_np)
        
        nmi, ari, acc, pur = evaluate(y_np, cluster_labels)
        
        return acc, nmi, pur, cluster_labels


# ============================================================
# 收敛性分析主函数
# ============================================================

def run_convergence_analysis(dataset_name, args, save_model=False):
    """运行单个数据集的收敛性分析"""
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    dataset = get_data_loader(dataset_name)
    num_views = dataset.num_views
    num_samples = dataset.num_samples
    num_clusters = dataset.num_classes
    input_dims = dataset.feature_dims
    
    args.num_clusters = num_clusters
    
    # 只训练一次
    set_seed(args.seed)
    
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
    
    # 记录训练历史（loss）和评估历史（metrics）
    history = {
        'epoch': [],
        'loss': [],
        'loss_recon': [],
        'loss_cl': [],
        'loss_diff': [],
        'acc': [],
        'nmi': [],
        'pur': []
    }
    
    # 存储KMeans评估结果用于平均
    kmeans_acc_history = []
    kmeans_nmi_history = []
    kmeans_pur_history = []
    
    for epoch in range(args.epochs):
        # 训练
        loss, loss_recon, loss_cl, loss_diff = train_one_epoch(
            model, view_list, y_true, optimizer, criterion, args
        )
        
        # 记录训练损失
        history['epoch'].append(epoch + 1)
        history['loss'].append(loss)
        history['loss_recon'].append(loss_recon)
        history['loss_cl'].append(loss_cl)
        history['loss_diff'].append(loss_diff)
        
        # 每个epoch进行KMeans评估（多次取平均）
        acc_list, nmi_list, pur_list = [], [], []
        for km_run in range(KMEANS_RUNS):
            acc, nmi, pur, _ = evaluate_model(model, view_list, y_true, args)
            acc_list.append(acc)
            nmi_list.append(nmi)
            pur_list.append(pur)
        
        history['acc'].append(np.mean(acc_list))
        history['nmi'].append(np.mean(nmi_list))
        history['pur'].append(np.mean(pur_list))
        
        # 存储KMeans评估的标准差（用于显示）
        kmeans_acc_history.append(acc_list)
        kmeans_nmi_history.append(nmi_list)
        kmeans_pur_history.append(pur_list)
        
        # 打印进度
        if (epoch + 1) % 50 == 0 and args.verbose:
            acc_mean = np.mean(acc_list)
            nmi_mean = np.mean(nmi_list)
            pur_mean = np.mean(pur_list)
            print(f"    Epoch {epoch+1:3d}: Loss={loss:.4f}, ACC={acc_mean:.4f}, NMI={nmi_mean:.4f}, PUR={pur_mean:.4f}")
    
    # 计算KMeans评估的标准差
    std_history = {
        'acc_std': np.std(kmeans_acc_history, axis=1),
        'nmi_std': np.std(kmeans_nmi_history, axis=1),
        'pur_std': np.std(kmeans_pur_history, axis=1),
        'loss_std': np.zeros(args.epochs),  # 训练只跑一次，loss无标准差
    }
    
    print(f"    Final: ACC={history['acc'][-1]:.4f}, NMI={history['nmi'][-1]:.4f}, PUR={history['pur'][-1]:.4f}")
    
    return history, std_history


# ============================================================
# 可视化
# ============================================================

def plot_convergence_curve(dataset_name, avg_history, std_history, save_path):
    """绘制收敛曲线（参考图样式）"""
    
    epochs = avg_history['epoch']
    acc = avg_history['acc']
    nmi = avg_history['nmi']
    pur = avg_history['pur']
    loss = avg_history['loss']
    
    acc_std = std_history['acc_std']
    nmi_std = std_history['nmi_std']
    pur_std = std_history['pur_std']
    loss_std = std_history['loss_std']
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左侧Y轴：评估指标
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Evaluation Metrics', fontsize=13, fontweight='bold')
    
    # 绘制评估指标（带置信区间）
    epochs_np = np.array(epochs)
    
    # ACC - 绿色
    ax1.plot(epochs_np, acc, color='#2ca02c', linewidth=2.0, label='ACC', zorder=5)
    ax1.fill_between(epochs_np, acc - acc_std, acc + acc_std, color='#2ca02c', alpha=0.15)
    
    # NMI - 蓝色
    ax1.plot(epochs_np, nmi, color='#1f77b4', linewidth=2.0, label='NMI', zorder=5)
    ax1.fill_between(epochs_np, nmi - nmi_std, nmi + nmi_std, color='#1f77b4', alpha=0.15)
    
    # PUR - 橙色
    ax1.plot(epochs_np, pur, color='#ff7f0e', linewidth=2.0, label='PUR', zorder=5)
    ax1.fill_between(epochs_np, pur - pur_std, pur + pur_std, color='#ff7f0e', alpha=0.15)
    
    ax1.set_ylim([0, 1.05])
    ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
    
    # 右侧Y轴：损失函数
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Loss', fontsize=13, fontweight='bold', color='#9467bd')
    
    # Loss - 紫色
    ax2.plot(epochs_np, loss, color='#9467bd', linewidth=2.0, linestyle='--', label='Total Loss', zorder=4)
    ax2.fill_between(epochs_np, loss - loss_std, loss + loss_std, color='#9467bd', alpha=0.15)
    
    ax2.tick_params(axis='y', labelcolor='#9467bd', labelsize=11)
    
    # 设置坐标轴颜色
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', 
               fontsize=11, framealpha=0.9, edgecolor='gray')
    
    # 标题
    plt.title(f'Convergence Analysis: {dataset_name}', fontsize=15, fontweight='bold', pad=15)
    
    # 网格
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    plt.tight_layout()
    
    # 保存图片
    fig_path_png = os.path.join(save_path, f'convergence_{dataset_name}.png')
    fig_path_pdf = os.path.join(save_path, f'convergence_{dataset_name}.pdf')
    plt.savefig(fig_path_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(fig_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"  图片已保存: {fig_path_png}")
    
    plt.close()


def save_convergence_data(dataset_name, avg_history, std_history, save_path):
    """保存收敛数据到CSV"""
    
    df = pd.DataFrame({
        'epoch': avg_history['epoch'],
        'ACC': avg_history['acc'],
        'ACC_std': std_history['acc_std'],
        'NMI': avg_history['nmi'],
        'NMI_std': std_history['nmi_std'],
        'PUR': avg_history['pur'],
        'PUR_std': std_history['pur_std'],
        'Total_Loss': avg_history['loss'],
        'Loss_std': std_history['loss_std'],
        'Loss_Recon': avg_history['loss_recon'],
        'Loss_CL': avg_history['loss_cl'],
        'Loss_Diff': avg_history['loss_diff'],
    })
    
    csv_path = os.path.join(save_path, f'convergence_data_{dataset_name}.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    
    print(f"  数据已保存: {csv_path}")
    
    return df


# ============================================================
# 主函数
# ============================================================

def run_all_convergence_experiments():
    """运行所有数据集的收敛性分析"""
    
    print("\n" + "="*70)
    print("         收敛性分析实验 (Convergence Analysis)")
    print("="*70)
    print(f"  数据集: {DATASETS}")
    print(f"  Epochs: {DEFAULT_ARGS['epochs']}")
    print(f"  重复次数: {KMEANS_RUNS}")
    print(f"  设备: {DEFAULT_ARGS['device']}")
    print("="*70)
    
    start_time = time.time()
    
    for dataset_name in DATASETS:
        args = argparse.Namespace(**DEFAULT_ARGS)
        
        # 运行收敛性分析
        avg_history, std_history = run_convergence_analysis(dataset_name, args)
        
        # 保存数据
        save_convergence_data(dataset_name, avg_history, std_history, RESULTS_DIR)
        
        # 绘制收敛曲线
        plot_convergence_curve(dataset_name, avg_history, std_history, RESULTS_DIR)
        
        # 打印最终结果
        final_epoch = DEFAULT_ARGS['epochs']
        final_idx = final_epoch - 1
        print(f"\n  >>> {dataset_name} Final Results:")
        print(f"      ACC={avg_history['acc'][final_idx]:.4f}±{std_history['acc_std'][final_idx]:.4f}")
        print(f"      NMI={avg_history['nmi'][final_idx]:.4f}±{std_history['nmi_std'][final_idx]:.4f}")
        print(f"      PUR={avg_history['pur'][final_idx]:.4f}±{std_history['pur_std'][final_idx]:.4f}")
        print(f"      Loss={avg_history['loss'][final_idx]:.4f}±{std_history['loss_std'][final_idx]:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"  所有实验完成！总耗时: {total_time/60:.1f} 分钟")
    print(f"{'='*70}")
    print(f"  结果保存在: {RESULTS_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convergence Analysis')
    parser.add_argument('--datasets', type=str, nargs='+', default=DATASETS,
                       help='数据集列表')
    parser.add_argument('--epochs', type=int, default=DEFAULT_ARGS['epochs'],
                       help='训练轮数')
    parser.add_argument('--runs', type=int, default=KMEANS_RUNS,
                       help='重复实验次数')
    cmd_args = parser.parse_args()
    
    # 更新配置
    if cmd_args.datasets:
        DATASETS = cmd_args.datasets
    if cmd_args.epochs:
        DEFAULT_ARGS['epochs'] = cmd_args.epochs
    if cmd_args.runs:
        KMEANS_RUNS = cmd_args.runs
    
    print(f"\n{'#'*70}")
    print(f"#  收敛性分析实验")
    print(f"#  数据集: {DATASETS}")
    print(f"#  Epochs: {DEFAULT_ARGS['epochs']}")
    print(f"#  Runs: {KMEANS_RUNS}")
    print(f"{'#'*70}\n")
    
    # 运行实验
    run_all_convergence_experiments()
    
    print(f"\n{'='*70}")
    print(f"  实验完成！所有结果保存在: {RESULTS_DIR}")
    print(f"{'='*70}\n")
