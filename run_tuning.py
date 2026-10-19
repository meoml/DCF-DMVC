import torch
import numpy as np
import pandas as pd
import itertools
from dataloader import load_data
from model import DisentangleModel
from loss import TotalLoss
from sklearn.cluster import KMeans
from metric import evaluate, resolve_batch_size
import torch.nn.functional as F
import os

LARGE_DATASET_THRESHOLD = 10000
LARGE_DATASET_BATCH_SIZE = 512
# =====================================================
# 设备
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())

# =====================================================
# 1. 数据集 + 参数搜索空间
# =====================================================
# DATASETS = ['Animal', 'CiteSeer','ACM', 'WebKB']
DATASETS = ['ORL']

LR_LIST = [1e-4, 1e-3]
FEATURE_DIM_LIST = [64]
ALPHA_LIST = [0.1, 0.5]
BETA_LIST = [0.005, 0.01, 0.05]
TEMP_LIST = [0.5]
K_LIST = [3, 5, 10]

EPOCHS = 500
HIDDEN_DIM = 512

EVAL_INTERVAL = 10      # 每多少 epoch 评估一次
KMEANS_RUNS = 10        # 最终统计用

# =====================================================
# 2. Excel 初始化
# =====================================================
excel_path = 'result44.xlsx'

if not os.path.exists(excel_path):
    records = []
    exp_id = 0
    for (dataset, lr, fd, alpha, beta, temp, k) in itertools.product(
        DATASETS, LR_LIST, FEATURE_DIM_LIST,
        ALPHA_LIST, BETA_LIST, TEMP_LIST, K_LIST
    ):
        records.append([
            exp_id, dataset, lr, fd, alpha, beta, temp, k,
            '', '', '', '', '', '', '', '', ''
        ])
        exp_id += 1

    columns = [
        'exp_id', 'dataset', 'learning_rate', 'feature_dim',
        'alpha', 'beta', 'temp', 'k',
        'ACC_mean', 'ACC_std',
        'NMI_mean', 'NMI_std',
        'ARI_mean', 'ARI_std',
        'PUR_mean', 'PUR_std',
        'best_epoch'
    ]

    pd.DataFrame(records, columns=columns).to_excel(excel_path, index=False)
    print(f'Excel created: {len(records)} experiments')

df = pd.read_excel(excel_path)

# =====================================================
# 3. KNN 图（用于 loss，动态更新）
# =====================================================
def get_knn_graph(features, k):
    feat = F.normalize(features, dim=1)
    sim = torch.mm(feat, feat.t())
    _, idx = torch.topk(sim, k=k, dim=1)

    n = features.size(0)
    S = torch.zeros(n, n, device=features.device)
    S.scatter_(1, idx, 1)
    S = (S + S.t()) / 2
    S[S > 0] = 1
    S.fill_diagonal_(1)
    return S



def collect_embeddings(model, loader, device):
    model.eval()
    labels = []
    embeddings = []

    with torch.no_grad():
        for xs, y_batch, _ in loader:
            xs = [x.to(device) for x in xs]
            _, _, h_c, _, _ = model(xs)
            embeddings.append(h_c.cpu().numpy())
            labels.append(y_batch.numpy())

    return np.concatenate(labels), np.concatenate(embeddings)

# =====================================================
# 4. 主循环
# =====================================================
for i, row in df.iterrows():

    # 已完成实验直接跳过
    if not pd.isna(row['ACC_mean']):
        continue

    print(f"\n===== Exp {int(row.exp_id)} | {row.dataset} =====")

    args = type('Args', (), {
        'dataset': row.dataset,
        'learning_rate': row.learning_rate,
        'epochs': EPOCHS,
        'hidden_dim': HIDDEN_DIM,
        'feature_dim': int(row.feature_dim),
        'alpha': row.alpha,
        'beta': row.beta,
        'temp': row.temp,
        'k': int(row.k),
        'batch_size': None
    })()

    # ---------- 数据 ----------
    dataset, dims, view, n, class_num = load_data(args.dataset)
    args.batch_size = resolve_batch_size(n, args.batch_size)
    print(f"Resolved batch_size: {args.batch_size} (n={n})")

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    # ---------- 模型 ----------
    model = DisentangleModel(
        view, dims,
        HIDDEN_DIM,
        args.feature_dim,
        device,
        args.k
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = TotalLoss(args.temp, device).to(device)

    y_true, _ = collect_embeddings(model, eval_loader, device)
    best_acc = -1.0
    best_epoch = -1
    best_z = None

    # ---------- 训练 ----------
    for epoch in range(args.epochs):
        for xs, _, _ in train_loader:
            xs = [x.to(device) for x in xs]
            model.train()
            optimizer.zero_grad()

            xrs, zs, h_c, h_p, _ = model(xs)
            S = get_knn_graph(xs[0].reshape(xs[0].size(0), -1), args.k)

            if epoch > 50 and epoch % 10 == 0:
                with torch.no_grad():
                    S = get_knn_graph(h_c, args.k)

            loss, *_ = criterion(xrs, xs, zs, h_c, h_p, S, args.alpha, args.beta)
            loss.backward()
            optimizer.step()

        # ---------- 中途评估（按 ACC 选最优） ----------
        if epoch % EVAL_INTERVAL == 0:
            _, z_np = collect_embeddings(model, eval_loader, device)

            y_pred = KMeans(
                n_clusters=class_num,
                n_init=20
            ).fit_predict(z_np)

            _, _, acc, _ = evaluate(y_true, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_z = z_np.copy()

    print(f"Best epoch: {best_epoch}, Best ACC (single run): {best_acc:.4f}")

    # ---------- 最终统计（best epoch 表示） ----------
    accs, nmis, aris, purs = [], [], [], []

    for _ in range(KMEANS_RUNS):
        y_pred = KMeans(
            n_clusters=class_num,
            n_init=20
        ).fit_predict(best_z)

        nmi, ari, acc, pur = evaluate(y_true, y_pred)
        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)
        purs.append(pur)

    df.loc[i, 'ACC_mean'] = np.mean(accs)
    df.loc[i, 'ACC_std']  = np.std(accs)
    df.loc[i, 'NMI_mean'] = np.mean(nmis)
    df.loc[i, 'NMI_std']  = np.std(nmis)
    df.loc[i, 'ARI_mean'] = np.mean(aris)
    df.loc[i, 'ARI_std']  = np.std(aris)
    df.loc[i, 'PUR_mean'] = np.mean(purs)
    df.loc[i, 'PUR_std']  = np.std(purs)
    df.loc[i, 'best_epoch'] = best_epoch

    df.to_excel(excel_path, index=False)

    print(
        f"ACC {df.loc[i,'ACC_mean']:.4f}±{df.loc[i,'ACC_std']:.4f} | "
        f"NMI {df.loc[i,'NMI_mean']:.4f}±{df.loc[i,'NMI_std']:.4f}"
    )

print("\n===== ALL EXPERIMENTS FINISHED =====")