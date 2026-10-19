import numpy as np
import scipy.io
import scipy.sparse # 需要用到这个来检查是否为稀疏矩阵
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import os

class MultiViewDataset(Dataset):
    def __init__(self, root_path, dataset_name):
        """
        通用多视图数据加载器
        :param root_path: 数据集文件夹路径 (例如 './datasets/')
        :param dataset_name: 数据集文件名 (不带.mat后缀，例如 'ORL')
        """
        self.root_path = root_path
        self.dataset_name = dataset_name
        
        # 拼接完整路径
        file_path = os.path.join(root_path, dataset_name + '.mat')
        if not os.path.exists(file_path):
            file_path = os.path.join(root_path, dataset_name)
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"Cannot find dataset file at: {file_path}")

        try:
            data = scipy.io.loadmat(file_path)
        except Exception as e:
            print(f"Error loading .mat file: {e}")
            raise

        # ---------------------------------------------------------
        # 1. 智能解析 Labels
        # ---------------------------------------------------------
        label_keys = ['truelabel', 'Y', 'label', 'labels', 'gnd','y']
        raw_label = self._find_value_by_keys(data, label_keys)
        
        if raw_label is None:
            raise ValueError(f"Could not find label in keys: {data.keys()}")

        if raw_label.dtype == 'object': 
             try:
                 self.labels = raw_label[0, 0].flatten()
             except:
                 self.labels = raw_label.flatten()
        else:
             self.labels = raw_label.flatten()
             
        self.labels = self.labels.astype(np.int64)
        if self.labels.min() == 1:
            self.labels = self.labels - 1
            
        self.num_samples = self.labels.shape[0]
        
        # ---------------------------------------------------------
        # 2. 智能解析 Views (数据)
        # ---------------------------------------------------------
        data_keys = ['data', 'X', 'features', 'view']
        raw_data = self._find_value_by_keys(data, data_keys)
        
        if raw_data is None:
            raise ValueError(f"Could not find data in keys: {data.keys()}")

        self.views = []
        
        if raw_data.dtype == 'object' or len(raw_data.shape) > 2:
            flat_data = raw_data.flatten()
            num_views = flat_data.shape[0]
            
            for i in range(num_views):
                view_data = flat_data[i]
                
                # 【关键修复】检查是否为稀疏矩阵，如果是，转为密集数组
                if scipy.sparse.issparse(view_data):
                    view_data = view_data.toarray()
                
                view_data = self._check_dims(view_data, self.num_samples)
                self.views.append(view_data)
        else:
            # 单个矩阵的情况
            view_data = raw_data
            
            # 【关键修复】检查是否为稀疏矩阵
            if scipy.sparse.issparse(view_data):
                view_data = view_data.toarray()
                
            view_data = self._check_dims(view_data, self.num_samples)
            self.views.append(view_data)

        # ---------------------------------------------------------
        # 3. 归一化处理
        # ---------------------------------------------------------
        scaler = MinMaxScaler()
        for i in range(len(self.views)):
            # 此时 self.views[i] 已经是普通的 numpy 数组了，不会报错
            self.views[i] = scaler.fit_transform(self.views[i]).astype(np.float32)

        print(f"Dataset '{dataset_name}' Loaded.")
        print(f"Samples: {self.num_samples}, Classes: {len(np.unique(self.labels))}, Views: {len(self.views)}")
        for i, v in enumerate(self.views):
            print(f"  - View {i} shape: {v.shape}")

    def _find_value_by_keys(self, data_dict, keys):
        for k in keys:
            if k in data_dict:
                return data_dict[k]
        return None

    def _check_dims(self, matrix, N):
        """强制转置矩阵，使得第一维度等于样本数 N"""
        if matrix.shape[0] == N:
            return matrix
        elif matrix.shape[1] == N:
            return matrix.T
        else:
            print(f"Warning: View shape {matrix.shape} does not match sample count {N} in either dim.")
            return matrix 

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data_list = [torch.from_numpy(v[idx]) for v in self.views]
        return data_list, self.labels[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset_name, data_path='./datasets/'):
    dataset = MultiViewDataset(data_path, dataset_name)
    dims = [v.shape[1] for v in dataset.views]
    view = len(dataset.views)
    data_size = dataset.num_samples
    class_num = len(np.unique(dataset.labels))
    return dataset, dims, view, data_size, class_num