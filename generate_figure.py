#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成论文级别的模型流程图
Usage: python generate_figure.py
Output: model_architecture.png / model_architecture.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

def draw_box(ax, x, y, width, height, text, color='#E3F2FD', text_color='black', fontsize=9, bold=False):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor='#1976D2', linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, wrap=True,
            multialignment='center')

def draw_arrow(ax, start, end, color='#757575'):
    """绘制箭头连接"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

def draw_diamond(ax, x, y, size, text, color='#FFF3E0'):
    """绘制菱形（判断节点）"""
    diamond = mpatches.RegularPolygon((x, y), numVertices=4, radius=size,
                                       orientation=np.pi/4, facecolor=color,
                                       edgecolor='#E65100', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, color='#E65100', weight='bold')

def draw_loss_box(ax, x, y, width, height, text, color='#FCE4EC'):
    """绘制损失函数框"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='#C2185B', linewidth=1.5,
                         linestyle='--')
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=8, color='#C2185B', weight='bold')

def create_model_figure():
    """创建模型架构流程图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    # 标题
    ax.text(8, 11.5, 'Multi-View Disentangled Clustering Framework', 
            ha='center', va='center', fontsize=16, weight='bold', color='#1565C0')
    ax.text(8, 11.0, '多视图解耦聚类框架', 
            ha='center', va='center', fontsize=12, color='#424242')

    # ============ 第一层：输入 ============
    input_y = 10.0
    draw_box(ax, 2.5, input_y, 1.5, 0.6, 'View 1\n$X^{(1)}$', '#BBDEFB', fontsize=8)
    draw_box(ax, 5.0, input_y, 1.5, 0.6, 'View 2\n$X^{(2)}$', '#BBDEFB', fontsize=8)
    draw_box(ax, 8.0, input_y, 1.5, 0.6, '...', '#BBDEFB', fontsize=10)
    draw_box(ax, 11.0, input_y, 1.5, 0.6, 'View V\n$X^{(V)}$', '#BBDEFB', fontsize=8)
    
    ax.text(6.5, input_y + 0.5, 'Multi-View Input', ha='center', va='center', 
            fontsize=10, style='italic', color='#1976D2')

    # ============ 第二层：KNN图构建 ============
    knn_y = 8.8
    draw_box(ax, 6.5, knn_y, 2.0, 0.6, 'KNN Graph\nConstruction', '#C8E6C9', fontsize=8)
    draw_arrow(ax, (2.5, input_y - 0.3), (5.5, knn_y + 0.3))
    draw_arrow(ax, (5.0, input_y - 0.3), (6.0, knn_y + 0.3))
    draw_arrow(ax, (11.0, input_y - 0.3), (7.5, knn_y + 0.3))

    # 动态图更新标注
    ax.text(9.0, knn_y, '→', fontsize=10, color='#757575')
    draw_box(ax, 10.5, knn_y, 1.8, 0.6, 'Dynamic Update\n(epoch ≥ 50)', '#FFF9C4', fontsize=7)
    ax.annotate('', xy=(9.7, knn_y), xytext=(9.0, knn_y),
                arrowprops=dict(arrowstyle='->', color='#757575', lw=1))

    # ============ 第三层：GCN编码器 ============
    gcn_y = 7.3
    draw_box(ax, 2.5, gcn_y, 1.8, 0.7, 'GCN\nEncoder', '#E1BEE7', fontsize=8)
    draw_box(ax, 6.5, gcn_y, 1.8, 0.7, 'GCN\nEncoder', '#E1BEE7', fontsize=8)
    draw_box(ax, 10.5, gcn_y, 1.8, 0.7, 'GCN\nEncoder', '#E1BEE7', fontsize=8)
    
    ax.text(6.5, gcn_y + 0.6, 'View-Specific Encoding', ha='center', va='center', 
            fontsize=10, style='italic', color='#7B1FA2')
    
    draw_arrow(ax, (6.5, knn_y - 0.3), (2.5, gcn_y + 0.35))
    draw_arrow(ax, (6.5, knn_y - 0.3), (6.5, gcn_y + 0.35))
    draw_arrow(ax, (6.5, knn_y - 0.3), (10.5, gcn_y + 0.35))

    # ============ 第四层：私有特征 + 公共融合 ============
    private_y = 6.0
    fusion_x = 6.5
    
    # 私有编码器
    draw_box(ax, 2.5, private_y, 1.6, 0.6, 'Private\nEncoder', '#FFCCBC', fontsize=8)
    draw_box(ax, 6.5, private_y, 1.6, 0.6, 'Private\nEncoder', '#FFCCBC', fontsize=8)
    draw_box(ax, 10.5, private_y, 1.6, 0.6, 'Private\nEncoder', '#FFCCBC', fontsize=8)
    
    ax.text(6.5, private_y + 0.5, 'Private Feature Extraction', ha='center', va='center', 
            fontsize=9, style='italic', color='#BF360C')

    draw_arrow(ax, (2.5, gcn_y - 0.35), (2.5, private_y + 0.3))
    draw_arrow(ax, (6.5, gcn_y - 0.35), (6.5, private_y + 0.3))
    draw_arrow(ax, (10.5, gcn_y - 0.35), (10.5, private_y + 0.3))

    # 双通道融合
    fusion_y = 4.5
    draw_box(ax, fusion_x, fusion_y, 3.5, 1.0, 'Dual-Channel\nFeature Fusion', '#B3E5FC', fontsize=10, bold=True)
    
    ax.text(4.0, fusion_y + 0.3, '→', fontsize=14, color='#1565C0')
    ax.text(9.0, fusion_y + 0.3, '←', fontsize=14, color='#1565C0')
    ax.text(fusion_x, fusion_y - 0.7, 'Local Attention + Global Gating', ha='center', va='center', 
            fontsize=8, style='italic', color='#0277BD')

    draw_arrow(ax, (2.5, private_y - 0.3), (5.0, fusion_y + 0.2))
    draw_arrow(ax, (6.5, private_y - 0.3), (6.5, fusion_y + 0.5))
    draw_arrow(ax, (10.5, private_y - 0.3), (8.0, fusion_y + 0.2))

    # ============ 第五层：公共特征 + 重构 ============
    common_y = 3.0
    recon_y = 3.0
    
    draw_box(ax, 4.0, common_y, 2.0, 0.7, '$H_c$\n(Common)', '#90CAF9', fontsize=9, bold=True)
    
    ax.text(6.5, common_y + 0.6, '→', fontsize=14, color='#1565C0')
    ax.text(8.0, common_y + 0.6, '←', fontsize=14, color='#1565C0')

    # 重构解码器
    draw_box(ax, 9.5, recon_y, 2.0, 0.7, 'Decoders\n(Reconstruction)', '#A5D6A7', fontsize=8)
    
    draw_arrow(ax, (fusion_x, fusion_y - 0.5), (4.0, common_y + 0.35))
    draw_arrow(ax, (5.0, common_y), (8.5, recon_y + 0.35))

    # ============ 第六层：损失函数 ============
    loss_y = 1.5
    
    # 重构损失
    draw_loss_box(ax, 2.5, loss_y, 1.6, 0.6, '$L_{recon}$')
    
    # 对比损失
    draw_loss_box(ax, 6.5, loss_y, 1.6, 0.6, '$L_{CL}$')
    
    # 差异损失
    draw_loss_box(ax, 10.5, loss_y, 1.6, 0.6, '$L_{diff}$')
    
    ax.text(6.5, loss_y + 0.55, 'Loss Functions', ha='center', va='center', 
            fontsize=10, style='italic', color='#880E4F')

    draw_arrow(ax, (4.0, common_y - 0.35), (2.5, loss_y + 0.3))
    draw_arrow(ax, (8.5, recon_y - 0.35), (2.5, loss_y + 0.3))
    draw_arrow(ax, (5.0, common_y - 0.35), (6.5, loss_y + 0.3))
    draw_arrow(ax, (6.5, private_y - 0.3), (10.5, loss_y + 0.3))

    # 总损失
    draw_box(ax, 6.5, 0.5, 2.0, 0.5, '$L = L_{recon} + \\alpha L_{CL} + \\beta L_{diff}$', 
             '#E8F5E9', fontsize=8, bold=True)
    
    draw_arrow(ax, (2.5, loss_y - 0.3), (5.5, 0.75))
    draw_arrow(ax, (6.5, loss_y - 0.3), (6.5, 0.75))
    draw_arrow(ax, (10.5, loss_y - 0.3), (7.5, 0.75))

    # 反向传播标注
    ax.annotate('', xy=(6.5, 3.25), xytext=(6.5, 0.75),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=2, ls='--'))
    ax.text(7.0, 2.0, 'Backward\nPropagation', ha='left', va='center', 
            fontsize=8, color='#F44336', weight='bold')

    # ============ 输出 ============
    ax.text(4.0, 2.3, '→', fontsize=14, color='#1565C0')
    draw_box(ax, 4.0, 1.7, 1.5, 0.5, '$K$-Means', '#E8F5E9', fontsize=9)
    draw_arrow(ax, (4.0, common_y - 0.35), (4.0, 1.95))

    # ============ 图例 ============
    legend_y = 10.5
    legend_items = [
        ('#BBDEFB', 'Input Data'),
        ('#C8E6C9', 'Graph Construction'),
        ('#E1BEE7', 'GCN Encoding'),
        ('#FFCCBC', 'Private Feature'),
        ('#B3E5FC', 'Fusion Module'),
        ('#A5D6A7', 'Reconstruction'),
        ('#FCE4EC', 'Loss Function'),
        ('#90CAF9', 'Common Feature'),
    ]
    
    ax.text(13.5, legend_y, 'Legend', ha='center', va='center', fontsize=10, weight='bold')
    for i, (color, label) in enumerate(legend_items):
        y_pos = legend_y - 0.5 - i * 0.35
        rect = FancyBboxPatch((13.0, y_pos - 0.1), 0.3, 0.2,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor=color, edgecolor='gray', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(13.4, y_pos, label, ha='left', va='center', fontsize=8)

    # 添加编号
    ax.text(0.5, input_y, '①', ha='center', va='center', fontsize=14, weight='bold', color='#1565C0')
    ax.text(0.5, knn_y, '②', ha='center', va='center', fontsize=14, weight='bold', color='#1565C0')
    ax.text(0.5, gcn_y, '③', ha='center', va='center', fontsize=14, weight='bold', color='#1565C0')
    ax.text(0.5, private_y, '④', ha='center', va='center', fontsize=14, weight='bold', color='#1565C0')
    ax.text(0.5, fusion_y, '⑤', ha='center', va='center', fontsize=14, weight='bold', color='#1565C0')
    ax.text(0.5, common_y, '⑥', ha='center', va='center', fontsize=14, weight='bold', color='#1565C0')
    ax.text(0.5, loss_y, '⑦', ha='center', va='center', fontsize=14, weight='bold', color='#1565C0')

    # 边框
    fig.patch.set_facecolor('white')
    
    return fig

def create_detail_figure():
    """创建详细模块图（双通道融合+结构对比）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：双通道融合
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a) Dual-Channel Feature Fusion', fontsize=12, weight='bold', pad=10)
    
    # 输入
    draw_box(ax1, 2, 7, 1.5, 0.6, '$Z^{(1)}$', '#E1BEE7', fontsize=9)
    draw_box(ax1, 5, 7, 1.5, 0.6, '$Z^{(2)}$', '#E1BEE7', fontsize=9)
    draw_box(ax1, 8, 7, 1.5, 0.6, '$Z^{(V)}$', '#E1BEE7', fontsize=9)
    
    # 局部注意力分支
    draw_box(ax1, 5, 5.5, 2.0, 0.6, 'Local Attention\n$\\alpha_{local}$', '#B3E5FC', fontsize=8)
    ax1.annotate('', xy=(4.0, 5.8), xytext=(2.8, 6.7),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1))
    ax1.annotate('', xy=(5.0, 5.8), xytext=(5.0, 6.7),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1))
    ax1.annotate('', xy=(6.0, 5.8), xytext=(7.2, 6.7),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1))
    
    # 全局门控分支
    draw_box(ax1, 5, 4.0, 2.5, 0.8, 'Global Gating\nMLP + Softmax', '#C8E6C9', fontsize=8)
    ax1.annotate('', xy=(3.5, 4.4), xytext=(2.8, 6.7),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1, ls='--'))
    ax1.annotate('', xy=(5.0, 4.4), xytext=(5.0, 6.7),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1, ls='--'))
    ax1.annotate('', xy=(6.5, 4.4), xytext=(7.2, 6.7),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1, ls='--'))
    
    # 融合
    draw_box(ax1, 5, 2.5, 2.5, 0.8, 'Weight Fusion\n$\\beta = \\alpha_{local} \\times \\alpha_{global}$', '#FFCCBC', fontsize=8)
    draw_arrow(ax1, (5, 5.2), (5, 2.9))
    draw_arrow(ax1, (5, 3.8), (5, 2.9))
    
    # 输出
    draw_box(ax1, 5, 1.0, 2.0, 0.6, '$H_c = L2Norm(MLP(\\sum \\beta_v Z^{(v)}))$', '#90CAF9', fontsize=7, bold=True)
    draw_arrow(ax1, (5, 2.1), (5, 1.3))
    
    ax1.text(0.5, 5.5, 'Channel 1', ha='center', va='center', fontsize=9, color='#1976D2')
    ax1.text(0.5, 4.0, 'Channel 2', ha='center', va='center', fontsize=9, color='#388E3C')
    
    # 右图：结构对比损失
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('(b) Structure-Guided Contrastive Loss', fontsize=12, weight='bold', pad=10)
    
    # 输入
    draw_box(ax2, 2, 7, 1.5, 0.5, '$Z^{(v)}$', '#E1BEE7', fontsize=9)
    draw_box(ax2, 5, 7, 1.5, 0.5, '$H_c$', '#90CAF9', fontsize=9)
    draw_box(ax2, 8, 7, 1.5, 0.5, 'KNN Graph $S$', '#FFF9C4', fontsize=8)
    
    # 拼接
    draw_box(ax2, 3.5, 5.5, 1.8, 0.5, 'Concat\n$[Z; H_c]$', '#B3E5FC', fontsize=8)
    draw_box(ax2, 6.5, 5.5, 1.8, 0.5, 'Similarity\n$S_{sim}/\\tau$', '#FFCCBC', fontsize=8)
    
    draw_arrow(ax2, (2, 6.7), (2.6, 5.8))
    draw_arrow(ax2, (5, 6.7), (5.7, 5.8))
    
    # 结构掩码
    draw_box(ax2, 6.5, 4.2, 2.0, 0.5, 'Structure Mask\n$1 - S_{rep}$', '#C8E6C9', fontsize=8)
    draw_arrow(ax2, (8, 6.7), (6.5, 4.5))
    
    # 掩蔽相似度
    draw_box(ax2, 6.5, 2.8, 2.2, 0.6, 'Masked Sim\n(remove neighbors)', '#FFCDD2', fontsize=7)
    draw_arrow(ax2, (6.5, 5.2), (6.5, 3.1))
    draw_arrow(ax2, (5.7, 5.2), (6.0, 3.1))
    
    # 正样本提取
    draw_box(ax2, 3.5, 3.5, 1.5, 0.5, 'Positive Pairs', '#A5D6A7', fontsize=8)
    draw_box(ax2, 3.5, 2.2, 1.5, 0.5, 'Negative Pairs', '#FFCDD2', fontsize=8)
    draw_arrow(ax2, (3.5, 5.2), (3.5, 3.8))
    draw_arrow(ax2, (5.7, 5.2), (4.2, 2.5))
    
    # InfoNCE
    draw_box(ax2, 6.5, 1.2, 2.5, 0.6, '$L_{CL} = -\\log \\frac{exp(sim^+/\\tau)}{\\sum exp(sim/\\tau)}$', '#E1BEE7', fontsize=7, bold=True)
    draw_arrow(ax2, (3.5, 2.0), (5.5, 1.5))
    draw_arrow(ax2, (6.5, 2.5), (6.5, 1.5))
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # 生成主架构图
    fig1 = create_model_figure()
    fig1.savefig('model_architecture.png', dpi=300, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
    print("Saved: model_architecture.png")
    
    # 生成详细模块图
    fig2 = create_detail_figure()
    fig2.savefig('model_detail.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("Saved: model_detail.png")
    
    plt.show()
