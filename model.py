import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super(GraphConvLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.activation = activation

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.activation is not None:
            output = self.activation(output)
        return output

class GCN_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_Encoder, self).__init__()
        self.gc1 = GraphConvLayer(input_dim, hidden_dim, activation=F.relu)
        self.gc2 = GraphConvLayer(hidden_dim, output_dim, activation=None)

    def forward(self, x, adj):
        h = self.gc1(x, adj)
        z = self.gc2(h, adj)
        return z

class DualChannelFusion(nn.Module):
    def __init__(self, num_views, feature_dim, hidden_dim):
        super(DualChannelFusion, self).__init__()
        self.att_vec = nn.Parameter(torch.zeros(feature_dim, 1)) 
        nn.init.xavier_uniform_(self.att_vec)
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_views * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_views)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, zs):
        z_stack = torch.stack(zs, dim=0)
        # 局部注意力
        local_scores = torch.matmul(torch.tanh(z_stack), self.att_vec).squeeze(-1)
        alpha_local = F.softmax(local_scores, dim=0) 
        
        # 全局门控
        z_cat = torch.cat(zs, dim=1)
        global_scores = self.gate_mlp(z_cat)
        alpha_global = F.softmax(global_scores, dim=1).t()
        
        # 融合权重
        final_weights = alpha_local * alpha_global 
        final_weights = final_weights / (torch.sum(final_weights, dim=0, keepdim=True) + 1e-8)
        
        z_fused = torch.sum(final_weights.unsqueeze(-1) * z_stack, dim=0)
        h_common = self.fusion_mlp(z_fused)
        return h_common, final_weights

class PrivateEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(PrivateEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

# ==========================================
# 修改后的 DisentangleModel
# ==========================================
class DisentangleModel(nn.Module):
    def __init__(self, view, input_dims, hidden_dim, feature_dim, device, k):
        super(DisentangleModel, self).__init__()
        self.view = view
        self.device = device
        self.k = k
        # 不缓存
        # self.cached_adjs = None 
        
        self.encoders = nn.ModuleList([
            GCN_Encoder(input_dims[v], hidden_dim, feature_dim) for v in range(view)
        ])
        
        self.private_encoders = nn.ModuleList([
            PrivateEncoder(feature_dim, feature_dim) for v in range(view)
        ])
        
        self.consensus_module = DualChannelFusion(view, feature_dim, hidden_dim)
        
        self.decoders = nn.ModuleList([
            Decoder(feature_dim, input_dims[v]) for v in range(view)
        ])

    def get_adj(self, x):
        with torch.no_grad():
            x = F.normalize(x, dim=1)
            dist = torch.mm(x, x.t())
            _, indices = torch.topk(dist, k=self.k, dim=1)
            N = x.shape[0]
            adj = torch.zeros(N, N).to(self.device)
            adj = adj.scatter_(1, indices, 1)
            adj = (adj + adj.t()) / 2
            adj[adj > 0] = 1
            adj = adj.fill_diagonal_(1)
            D = torch.sum(adj, dim=1)
            D_inv_sqrt = torch.pow(D, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
            D_mat = torch.diag(D_inv_sqrt)
            return torch.mm(torch.mm(D_mat, adj), D_mat)


    def forward(self, xs):
        # if self.cached_adjs is None:
        #     self.cached_adjs = [self.get_adj(x) for x in xs]
        
        # adjs = self.cached_adjs
        
        adjs = [self.get_adj(x) for x in xs]
        zs = []
        h_privates = []
        xrs = []
        
        # 提取特征
        for v in range(self.view):
            z = self.encoders[v](xs[v], adjs[v])
            zs.append(z)
            
            h_p = self.private_encoders[v](zs[v])
            h_privates.append(h_p)
            
        h_common, att = self.consensus_module(zs)
        h_common = F.normalize(h_common, dim=1)
        
        # 4. 重构 (直接使用 Z 进行重构)
        for v in range(self.view):
            xr = self.decoders[v](zs[v])
            xrs.append(xr)
            
        return xrs, zs, h_common, h_privates, att