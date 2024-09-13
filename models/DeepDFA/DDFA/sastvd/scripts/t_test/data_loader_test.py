import torch
from torch.utils.data import DataLoader, Dataset, Subset
import random
import dgl

# 假设有一个 get_epoch_indices 方法
def get_epoch_indices(dataset):
    return random.sample(range(len(dataset)), len(dataset))

# 示例 BigVulDatasetLineVD 类，处理图数据
class BigVulDatasetLineVD(Dataset):
    def __init__(self, data, partition, **kwargs):
        self.data = data
        self.partition = partition
        self.idx2id = {i: i for i in range(len(data))}

    def __getitem__(self, idx):
        return self.data[self.idx2id[idx]]

    def __len__(self):
        return len(self.data)

    def get_epoch_indices(self):
        return get_epoch_indices(self)

# 创建示例图数据
def create_example_graph(num_nodes, num_edges):
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g.add_edges(src, dst)
    g.ndata['_ABS_DATAFLOW'] = torch.randint(0, 5, (num_nodes,))
    return g

# 创建一些示例图数据
graph_data = [create_example_graph(num_nodes=3, num_edges=10) for _ in range(100)]
train_dataset = BigVulDatasetLineVD(graph_data, partition="train")

# 定义批次大小和工作线程数
batch_size = 10
train_workers = 4

# 创建 DataLoader
train_loader = DataLoader(
    Subset(train_dataset, train_dataset.get_epoch_indices()),
    shuffle=True,
    batch_size=batch_size,
    num_workers=train_workers,
    collate_fn=dgl.batch  # 使用 DGL 的 batch 函数来批处理图
)

# 迭代加载数据
for batch in train_loader:
    print(batch)
    print("Number of nodes in batch:", batch.number_of_nodes())
    print("Number of edges in batch:", batch.number_of_edges())
    print("batch.ndata['_ABS_DATAFLOW']", batch.ndata['_ABS_DATAFLOW'])
