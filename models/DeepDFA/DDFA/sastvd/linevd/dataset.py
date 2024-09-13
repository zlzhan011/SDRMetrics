import logging
import os.path
import pickle

import DDFA.sastvd.helpers.dclass as svddc
from DDFA.sastvd.linevd.graphmogrifier import get_graphs, get_nodes_df

from DDFA.sastvd.helpers.datasets import parse_limits
import dgl

logger = logging.getLogger(__name__)
import DDFA.sastvd as svd
intermediate_result =  os.path.join(svd.processed_dir(), 'bigvul_intermediate_result')

class BigVulDatasetLineVD(svddc.BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(
        self,
        label_style="graph",
        gtype="cfg",
        feat="all",
        dsname="bigvul",
        sample_mode=False,
        concat_all_absdf=False,
        load_features=True,
        **kwargs,
    ):
        """Init."""
        self.label_style = label_style
        self.graph_type = gtype
        self.feat = feat
        self.limit_all = parse_limits(self.feat)[1]
        super(BigVulDatasetLineVD, self).__init__(sample_mode=sample_mode, dsname=dsname, **kwargs)

        nodes_df = get_nodes_df(dsname, sample_mode, feat, concat_all_absdf, load_features)
        if self.partition != "all":
            nodes_df = nodes_df[nodes_df["graph_id"].isin(self.df.id)]

        with open(os.path.join(intermediate_result, 'function_name_BigVulDatasetLineVD_variable_name_nodes_df.pkl'), 'wb') as f_write:
            pickle.dump(nodes_df, f_write)

        self.graphs_by_id, self.extrafeats_by_id = get_graphs(dsname, nodes_df, sample_mode, feat, self.partition, concat_all_absdf, load_features)

        print("self.graphs_by_id:", self.graphs_by_id)
        print("self.extrafeats_by_id:", self.extrafeats_by_id)
        with open(os.path.join(intermediate_result, 'function_name_BigVulDatasetLineVD_variable_name_self_graphs_by_id.pkl'), 'wb') as f_write:
            pickle.dump(self.graphs_by_id, f_write)
        with open(os.path.join(intermediate_result, 'function_name_BigVulDatasetLineVD_variable_name_self_extrafeats_by_id.pkl'), 'wb') as f_write:
            pickle.dump(self.extrafeats_by_id, f_write)

        # include only graphs with parsed nodes
        new_df = self.df[self.df.id.isin(self.graphs_by_id.keys())].reset_index(drop=True)

        with open(os.path.join(intermediate_result, 'function_name_BigVulDatasetLineVD_variable_name_new_df.pkl'), 'wb') as f_write:
            pickle.dump(new_df, f_write)

        if len(new_df) != len(self.df):
            logger.info("Reducing df len from %d to %d", len(self.df), len(new_df))
            self.df = new_df
            self.idx2id = self.get_idx2id()

        with open(os.path.join(intermediate_result, 'function_name_BigVulDatasetLineVD_variable_name_self_idx2id.pkl'), 'wb') as f_write:
            pickle.dump(self.idx2id, f_write)


    def item(self, _id):
        """Get item by id."""
        return self.graphs_by_id[_id], self.extrafeats_by_id[_id]

    def __getitem__(self, idx):
        """Override getitem."""
        return self.item(self.idx2id[idx])

    def __len__(self):
        """Get length of dataset."""
        return len(self.idx2id)

    def __iter__(self) -> dict:
        for i in self.idx2id:
            yield self[i]
    
    def get_indices(self, indices):
        ilist = indices.tolist()
        keep_idx = []
        graphs = []
        for i, idx in enumerate(ilist):
            try:
                graph = self.item(idx)[0]
                graphs.append(graph)
                keep_idx.append(i)
            except KeyError:
                logger.info("missing: index %d=%d", i, idx)
                continue
        # logger.info("keep idx %s", keep_idx)
        return dgl.batch(graphs).to(indices.device), keep_idx
