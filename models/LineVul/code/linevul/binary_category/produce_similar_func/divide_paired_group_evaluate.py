import json
import os.path
import pandas as pd


import pandas as pd
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt


def get_mertics(y_true, y_pred ):
    # calculate confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    #
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    metrics = {"tn": tn,
               "fp": fp,
               "fn": fn,
               "tp": tp,
               "recall": recall,
               "precision": precision,
               "f1": f1}
    return metrics

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_id_group_map(distribution_dir):
    paired_groups_distribution = os.path.join(distribution_dir, 'paired_groups_distribution.json')
    paired_groups_distribution = read_json(paired_groups_distribution)
    # print(paired_groups_distribution)
    id_group_map = {}
    for k, v in paired_groups_distribution.items():
        for id in v:
            id_group_map[int(id)] = k
    print(id_group_map)
    return  id_group_map

if __name__ == '__main__':
    distribution_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired_groups'
    predict_res_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired/LineVul'
    groups = ['d_0', 'd_0_01', 'd_0_02', 'd_0_03']
    id_group_map = get_id_group_map(distribution_dir)
    all_metrics = []
    for group in groups:
        print("\n\ngroup:", group)
        predict_path = os.path.join(predict_res_dir, group, 'before_after_paired.csv')
        predict_res = pd.read_csv(predict_path)
        print(predict_res.columns.values.tolist())
        predict_column_before = 'LineVul_' + group + '_Predictions_before'
        predict_column_after = 'LineVul_' + group + '_Predictions_after'
        predict_res['group'] = predict_res['all_inputs_ids'].apply(lambda x: id_group_map[x])
        print(predict_res['group'])
        for group_inner in groups:
            print("\ngroup_inner:", group_inner)
            predict_res_group = predict_res[predict_res['group'] == group_inner]
            y_true = len(predict_res_group['y_trues_before'].tolist()) * [1] + len(predict_res_group['y_trues_after'].tolist()) * [0]
            y_pred = predict_res_group[predict_column_before].tolist() + predict_res_group[predict_column_after].tolist()
            print("y_true:", y_true)
            print("y_pred:", y_pred)
            metrics = get_mertics(y_true, y_pred)
            metrics['training_add_group'] = group
            metrics['testing_group'] = group_inner
            all_metrics.append(metrics)
    all_metrics = pd.DataFrame(all_metrics)
    output_columns_name = ['training_add_group', 'testing_group', 'precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp']
    all_metrics = all_metrics[output_columns_name]
    all_metrics.to_csv(os.path.join(distribution_dir, 'group_paired_all_metrics.csv'))




