import os.path
import pickle

import numpy as np
import pandas as pd
import DDFA.sastvd as svd
import torch as th
import tqdm
import functools

from dgl.data.utils import load_graphs
import dgl

# 假设nodes是你的DataFrame

# 设置Pandas，以便显示所有行
pd.set_option('display.max_rows', None)

# 设置Pandas，以便显示所有列
pd.set_option('display.max_columns', None)

import logging
logger = logging.getLogger(__name__)

allfeats = [
    "api", "datatype", "literal", "operator",
]


intermediate_result =  os.path.join(svd.processed_dir(), 'bigvul_intermediate_result')

def delete_duplicated_row_bak(nodes):
    nodes['combined'] = nodes['graph_id'].astype(str) + nodes['node_id'].astype(str)
    nodes['is_duplicated'] = nodes['combined'].duplicated(keep=False)
    # print("---------11111 -------nodes[nodes.is_duplicated==True] -----------------")
    # print(nodes[nodes.is_duplicated==True])
    # 使用duplicated()方法标记'combined'列中的重复值，然后使用~操作符选择非重复的行
    nodes = nodes[~nodes['combined'].duplicated()]

    # 如果不再需要'combined'列，可以选择删除它

    # print("---------2222 -------nodes[nodes.is_duplicated==True] -----------------")
    # print(nodes[nodes.is_duplicated == True])
    nodes = nodes.drop(columns=['combined'])
    nodes = nodes.drop(columns=['is_duplicated'])
    return nodes

def delete_duplicated_row(nodes):
    nodes['combined'] = nodes['graph_id'].astype(str) +"####"+ nodes['node_id'].astype(str)


    nodes['is_duplicated'] = nodes['combined'].duplicated(keep=False)
    nodes_sorted = nodes.sort_values(by='combined', ascending=True)
    print("---------11111 -------nodes[nodes.is_duplicated==True] -----------------")
    print(nodes_sorted[nodes_sorted.is_duplicated==True][:100])
    print(nodes_sorted[:100])
    # 使用duplicated()方法标记'combined'列中的重复值，然后使用~操作符选择非重复的行
    nodes = nodes[~nodes['combined'].duplicated()]

    # 如果不再需要'combined'列，可以选择删除它

    # print("---------2222 -------nodes[nodes.is_duplicated==True] -----------------")
    # print(nodes[nodes.is_duplicated == True])
    nodes = nodes.drop(columns=['combined'])
    nodes = nodes.drop(columns=['is_duplicated'])
    return nodes


@functools.cache
def get_nodes_df(dsname, sample_mode, feat, concat_all_absdf=False, load_features=True):
    sample_text = "_sample" if sample_mode else ""
    cols = ["Unnamed: 0", "graph_id", "node_id", "dgl_id", "vuln", "code", "_label"]
    nodes = pd.read_csv(svd.processed_dir() / dsname / f"nodes{sample_text}.csv", index_col=0, usecols=cols, dtype={"code": str}, na_values = [])
    nodes = nodes.reset_index(drop=True)
    nodes.code = nodes.code.astype(str)
    split = "fixed"
    print("000 nodes: shape:", nodes.shape)  #  这里是加载nodes, 看nodes有多少个
    nodes = delete_duplicated_row(nodes)     # 这是删除重复的nodes
    print("0000 nodes: shape:", nodes.shape)

    if load_features:
        if feat is not None:
            nodes = pd.merge(nodes, pd.read_csv(svd.processed_dir() / dsname / f"nodes_feat_{feat}_{split}{sample_text}.csv", index_col=0), how="left", on=["graph_id", "node_id"])
            nodes = delete_duplicated_row(nodes)
        if concat_all_absdf:
            prefix = "_ABS_DATAFLOW_"
            rest = feat[feat.index("_all"):]
            for otherfeat in allfeats:  # 这是遍历4种不同类型的features
                otherfeat_path = svd.processed_dir() / dsname / f"nodes_feat_{prefix}{otherfeat}{rest}_{split}{sample_text}.csv"
                print("otherfeat_path:", otherfeat_path)
                otherdf = pd.read_csv(otherfeat_path, index_col=0)  # 这是读取四个类别feature中的其中一个类别
                print("1 otherdf: shape:", otherdf.shape)
                otherdf = otherdf.rename(columns={next(c for c in otherdf.columns if c.startswith("_ABS_DATAFLOW")): f"_ABS_DATAFLOW_{otherfeat}"})
                print("2 otherdf: shape:", otherdf.shape)
                # print("other df", otherfeat)
                # print(otherdf)
                otherdf = delete_duplicated_row(otherdf)  # 可以看出otherdf里面也存在 重复的
                print("3 otherdf: shape:", otherdf.shape)
                nodes = pd.merge(nodes, otherdf, how="left", on=["graph_id", "node_id"])
                print("4 nodes: shape:", nodes.shape)
                nodes.to_csv(os.path.join(intermediate_result, 'function_name_get_nodes_df_variable_name_nodes_'+str(otherfeat)+'.csv'))
    nodes.to_csv(
        os.path.join(intermediate_result, 'function_name_get_nodes_df_variable_name_nodes_' + 'final_return' + '.csv'))
    return nodes

# @functools.cache
# def get_nodes_df(dsname, sample_mode, feat, concat_all_absdf=False, load_features=True):
#     sample_text = "_sample" if sample_mode else ""
#     cols = ["Unnamed: 0", "graph_id", "node_id", "dgl_id", "vuln", "code", "_label"]
#     nodes = pd.read_csv(svd.processed_dir() / dsname / f"nodes{sample_text}.csv", index_col=0, usecols=cols, dtype={"code": str}, na_values = [])
#     nodes = nodes.reset_index(drop=True)
#     nodes.code = nodes.code.astype(str)
#     split = "fixed"
#     if load_features:
#         if feat is not None:
#             nodes = pd.merge(nodes, pd.read_csv(svd.processed_dir() / dsname / f"nodes_feat_{feat}_{split}{sample_text}.csv", index_col=0), how="left", on=["graph_id", "node_id"])
#         if concat_all_absdf:
#             prefix = "_ABS_DATAFLOW_"
#             rest = feat[feat.index("_all"):]
#             for otherfeat in allfeats:
#                 otherdf = pd.read_csv(svd.processed_dir() / dsname / f"nodes_feat_{prefix}{otherfeat}{rest}_{split}{sample_text}.csv", index_col=0)
#                 otherdf = otherdf.rename(columns={next(c for c in otherdf.columns if c.startswith("_ABS_DATAFLOW")): f"_ABS_DATAFLOW_{otherfeat}"})
#                 # print("other df", otherfeat)
#                 # print(otherdf)
#                 nodes = pd.merge(nodes, otherdf, how="left", on=["graph_id", "node_id"])
#
#     return nodes


@functools.cache
def get_df_df(dsname, sample_mode):
    sample_text = "_sample" if sample_mode else ""
    df_df = pd.read_csv(svd.processed_dir() / dsname / f"nodes_feat_DF{sample_text}.csv", index_col=0)
    df_df = df_df[["graph_id", "node_id", "df_in"]]
    return df_df


@functools.cache
def get_graphs_by_id(dsname, sample_mode):
    sample_text = "_sample" if sample_mode else ""
    graphs, graph_labels = load_graphs(str(svd.processed_dir() / dsname / f"graphs{sample_text}.bin"))
    graphs_by_id = dict(zip(graph_labels["graph_id"].tolist(), graphs))
    return graphs_by_id


def get_graphs(dsname, nodes_df, sample_mode, feat, partition, concat_all_absdf, load_features):
    graphs_by_id = get_graphs_by_id(dsname, sample_mode)
    feats_init = []
    for i in range(len(graphs_by_id)):
        feats_init.append(dict())
    extrafeats_by_id = dict(zip(graphs_by_id.keys(), feats_init))
    if not load_features:
        return graphs_by_id, extrafeats_by_id

    # update graph features
    partition_graphs_by_id = {}
    skipped_df = 0
    node_len = []
    printed = 0
    was_vuln = []
    for graph_id, group in tqdm.tqdm(nodes_df.groupby("graph_id"), f"graphize {partition}"):
        # print("graph_id-----------------------:", graph_id)
        g: dgl.HeteroGraph = graphs_by_id[graph_id]
        g.ndata["_ABS_DATAFLOW"] = th.LongTensor(group[feat].tolist())
        if concat_all_absdf:
            for otherfeat in allfeats:
                g.ndata[f"_ABS_DATAFLOW_{otherfeat}"] = th.LongTensor(group[f"_ABS_DATAFLOW_{otherfeat}"].tolist())

        g.ndata["_VULN"] = th.Tensor(group["vuln"].tolist()).int()
        was_vuln.append(group["vuln"].max().item())

        if printed < 5:
            logger.debug("graph-------------- %d: %s\n%s", graph_id, g, g.ndata)
            # print("graph-------------- %d: %s\n%s", graph_id, g, g.ndata)
            printed += 1
            with open(os.path.join(intermediate_result, 'graph', f"{graph_id}.pkl"), 'wb') as f:
                graph_feature = {"graph_id":graph_id,
                                 "g":g,
                                 "g.ndata":g.ndata}
                pickle.dump(graph_feature, f)

        partition_graphs_by_id[graph_id] = g
    graphs_by_id = partition_graphs_by_id
    node_len = np.array(node_len)

    logger.info("percentage of vuln graphs: %s", np.average(was_vuln))
    logger.info("percentage of vuln nodes:\n%s", nodes_df.value_counts("vuln", normalize=True))
    logger.info("percentage of graphs with at least 1 vuln:\n%s", nodes_df.groupby("graph_id")["vuln"].agg(lambda g: 1 if g.any() else 0).value_counts(normalize=True))
    logger.info("skipped dataflow: %d", skipped_df)

    return graphs_by_id, extrafeats_by_id
