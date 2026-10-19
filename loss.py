import torch
import torch.nn as nn
import torch.nn.functional as F

class TotalLoss(nn.Module):
    def __init__(self, temperature_f, device):
        super(TotalLoss, self).__init__()
        self.temp = temperature_f
        self.device = device
        self.mse = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        # 创建对比学习的掩码
        # N 这里传入的是 batch_size * 2
        N = batch_size * 2
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def structure_contrastive_loss(self, h_i, h_j, S):
        """
        结构引导的对比损失 (Ego-Graph CL)
        h_i: View Feature, h_j: Common Feature
        S: 指导图 (N x N)
        """
        batch_size = h_i.size(0)
        N = 2 * batch_size
        
        # 1. 拼接特征
        h = torch.cat((h_i, h_j), dim=0) # [2N, D]
        
        # 2. 计算相似度矩阵
        sim = torch.matmul(h, h.T) / self.temp
        
        # 3. 构建结构掩码
        # 如果 S_ij=1 (是邻居)，则 (1-S_ij)=0，这个负样本被 Mask 掉
        S_rep = S.repeat(2, 2)
        all_one = torch.ones(N, N).to(self.device)
        structure_mask = all_one - S_rep
        
        # 剔除邻居作为负样本
        sim_masked = torch.multiply(sim, structure_mask)
        
        # 4. 提取正样本 (对角线 + 对角块)
        # 简单的 InfoNCE: i 和 i+N 是正样本
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # 5. 提取负样本
        # mask_correlated_samples 已经剔除了自身和对应的正样本
        mask = self.mask_correlated_samples(batch_size).to(self.device)
        negative_samples = sim_masked[mask].reshape(N, -1)
        
        # 6. 计算 CrossEntropy
        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        return loss / N

    def difference_loss(self, h_common, h_privates):
        """
        差异性损失：HSIC (去除相关性)
        """
        loss = 0
        batch_size = h_common.size(0)
        h_c = h_common - h_common.mean(dim=0)
        for h_p in h_privates:
            h_p = h_p - h_p.mean(dim=0)
            # 计算互协方差矩阵的 Frobenius 范数
            # 目标是让它趋近于 0
            cov = torch.mm(h_c.t(), h_p) / max(batch_size - 1, 1)
            loss += torch.norm(cov, p='fro')
        return loss

    def forward(self, xrs, xs, zs, h_common, h_privates, S, alpha, beta):
        # 1. 重构损失
        rec_loss = sum([self.mse(xr, x) for xr, x in zip(xrs, xs)])
        
        # 2. 对比损失 (View vs Common)
        con_loss = 0
        for z_v in zs:
            # 归一化后输入
            z_norm = F.normalize(z_v, dim=1)
            h_norm = F.normalize(h_common, dim=1)
            con_loss += self.structure_contrastive_loss(z_norm, h_norm, S)
            
        # 3. 差异性损失
        diff_loss = self.difference_loss(h_common, h_privates)
        
        total_loss = rec_loss + alpha * con_loss + beta * diff_loss
        return total_loss, rec_loss, con_loss, diff_loss