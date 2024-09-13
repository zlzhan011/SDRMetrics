"""
Gated Graph Neural Network module for graph classification tasks
"""
import itertools
from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
import torch
from torch import nn

import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)
torch.set_printoptions(threshold=float('inf'))
from DDFA.code_gnn.models.base_module import BaseModule

from pytorch_lightning.utilities.cli import MODEL_REGISTRY

import logging

logger = logging.getLogger(__name__)

allfeats = [
    "api", "datatype", "literal", "operator",
]

@MODEL_REGISTRY
class FlowGNNGGNNModule(BaseModule):
    def __init__(self,
                feat,
                input_dim,
                hidden_dim,
                n_steps,
                num_output_layers,
                label_style="graph",
                concat_all_absdf=False,
                encoder_mode=False,
                **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if "_ABS_DATAFLOW" in feat:
            feat = "_ABS_DATAFLOW"
        self.feature_keys = {
            "feature": feat,
        }

        self.print_time = 0

        self.input_dim = input_dim
        self.concat_all_absdf = concat_all_absdf
        hidden_dim = hidden_dim
        # feature extractors
        embedding_dim = hidden_dim  # TODO: try varying embedding dim from hidden_dim
        if self.concat_all_absdf:
            print("input_dim:", input_dim)
            print("embedding_dim:", embedding_dim)
            self.all_embeddings = nn.ModuleDict({
                of: nn.Embedding(input_dim, embedding_dim) for of in allfeats
            })
            embedding_dim *= len(allfeats)
            hidden_dim *= len(allfeats)  # TODO: try compressing 4*embeding_dim to hidden_dim
        else:
            self.embedding = nn.Embedding(input_dim, embedding_dim)

        # graph stage
        self.ggnn = GatedGraphConv(in_feats=embedding_dim,
                                out_feats=hidden_dim,
                                n_steps=n_steps,
                                n_etypes=1)

        output_in_size = embedding_dim + hidden_dim

        self.out_dim = output_in_size

        if label_style == "graph":
            pooling_gate_nn = nn.Linear(output_in_size, 1)
            self.pooling = GlobalAttentionPooling(pooling_gate_nn)

        if not encoder_mode:
            output_layers = []
            for i in range(num_output_layers):
                if i == num_output_layers-1:
                    output_size = 1
                else:
                    output_size = output_in_size
                output_layers.append(nn.Linear(output_in_size, output_size))
                if i != num_output_layers-1:
                    output_layers.append(nn.ReLU())
            self.output_layer = nn.Sequential(*output_layers)

    def calculate_same_features(self, feat_embed):
        import numpy as np
        # 将 PyTorch 张量转换为 NumPy 数组
        feat_embed_np = feat_embed.cpu().detach().numpy()

        # 获取第一行
        feat_embed_first_row = feat_embed_np[3, :]

        # 比较所有行与第一行是否相同
        same_rows = np.all(feat_embed_np == feat_embed_first_row, axis=1)

        # 计算相同行的数量
        same_num = np.sum(same_rows)

        print(f"Number of rows that are the same as the first row: {same_num}")

    def forward(self, graph, extrafeats):
        # get embedding of feature
        if self.concat_all_absdf:
            cfeats = []
            for otherfeat in allfeats:
                feat = graph.ndata[f"_ABS_DATAFLOW_{otherfeat}"]
                if self.print_time <= 5:
                    print("otherfeat:", otherfeat)
                    print("feat shape:", feat.shape)
                self.print_time = self.print_time + 1
                cfeats.append(self.all_embeddings[otherfeat](feat))
            feat_embed = torch.cat(cfeats, dim=1)
            print("self.concat_all_absdf  True ")
        else:
            feat = graph.ndata[self.feature_keys["feature"]]
            feat_embed = self.embedding(feat)
            print("self.concat_all_absdf  not")
        print("feat_embed shape:", feat_embed.shape)


        self.calculate_same_features(feat_embed)
        all_same = torch.unique(feat_embed).numel() == 1
        print("all_same _ 2", all_same)
        print("graph:", graph)
        # graph learning stage
        ggnn_out = self.ggnn(graph, feat_embed)
        # print("ggnn_out:", ggnn_out)
        print("ggnn_out shape:", ggnn_out.shape)
        # concat input
        out = torch.cat([ggnn_out, feat_embed], -1)
        # print("out 00:", out)
        print("out 00 shape:", out.shape)
        # prediction stage
        if self.hparams.label_style == "graph":
            out = self.pooling(graph, out)

        if self.hparams.encoder_mode:
            logits = out
            print("logits shape:hparams.encoder_mode true ", logits.shape)
        else:
            logits = self.output_layer(out).squeeze()
            print("logits shape:hparams.encoder_mode false ", logits.shape)

        return logits
