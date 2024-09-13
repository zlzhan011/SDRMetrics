import json
import os.path

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix



def not_use_error_graph(df):
    print("not_use_error_graph  df_shape:", df.shape)
    joern_error_index_list_file = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR/joern-error.txt'
    joern_error_index_list_file = '/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_MSR/external/joern-error_V2.txt'
    joern_error_index_list_file = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR/joern-error-c.txt'
    # joern_error_index_list_file = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR/joern-error-cpp.txt'
    # joern_error_index_list_file = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR/joern-missed-c.txt'
    joern_error_index_list_file = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR/joern-missed-cpp.txt'


    joern_error_index_list = []
    with open(joern_error_index_list_file, 'r') as f:
        for line in f:
            joern_error_index = int(line.strip().split("_")[0])
            if joern_error_index not in joern_error_index_list:
                joern_error_index_list.append(joern_error_index)

    df_v = df[df['target'] == 1]
    df_nv = df[df['target'] ==0]
    df_v_row = df_v.shape[0]
    df_nv_row = df_nv.shape[0]
    nv_v_ratio = df_nv_row / df_v_row

    df_filter = df[df['index'].isin(joern_error_index_list)]
    df_filter_row_num = df_filter.shape[0]
    print("df_filter_shape:", df_filter.shape)
    df_nv_version = 4
    if df_nv_version == 1:
        df_nv = df[df['target'] == 0]
        df_nv = df_nv.sample(n=501, random_state=42)
        # df_filter = pd.concat([df_filter, df_nv])
        df_filter = df_nv
    elif df_nv_version == 2:

        df_filter_joern_correct = df[~df['index'].isin(joern_error_index_list)]
        df_filter_joern_correct_v = df_filter_joern_correct[df_filter_joern_correct['target'] == 1]
        df_filter_joern_correct_v_501 = df_filter_joern_correct_v.sample(n=501, random_state=42)
        df_filter_joern_correct_v_501['func_before'] = df_filter_joern_correct_v_501['func_after']
        # df_filter = pd.concat([df_filter, df_filter_joern_correct_v_501])
        df_filter_joern_correct_v_501['target'] = 0
        df_filter = df_filter_joern_correct_v_501
    elif df_nv_version == 3:
        nv_v_ratio=501 * int(nv_v_ratio)

        df_nv = df[df['target'] == 0]
        df_nv = df_nv.sample(n=nv_v_ratio, random_state=42)
        # df_filter = pd.concat([df_filter, df_nv])
        df_filter = df_nv
    elif df_nv_version == 4:
        df_filter = df[~df['index'].isin(joern_error_index_list)]
    df_filter = df_filter.sample(frac=1, random_state=42).reset_index(drop=True)
    print("df_filter_shape 2:", df_filter.shape)

    return df_filter



def calculate_metrics(y_trues, y_preds):
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
    }


    for key in sorted(result.keys()):
        print("  %s = %s", key, str(round(result[key], 4)))



def select_random_non_vulnerable_samples(predict_dir):
    all_sample_file = 'val_func_before_test_prediction_joern_error_plus_ratio_after_fix_non_vul_all_sample.xlsx'
    all_sample_file = os.path.join(predict_dir, all_sample_file)
    all_sample_file = pd.read_excel(all_sample_file)
    all_sample_file_nv = all_sample_file[all_sample_file['y_trues'] == 0]
    all_sample_file_v = all_sample_file[all_sample_file['y_trues'] == 1]
    nv_v_ratio = all_sample_file_nv.shape[0] / all_sample_file_v.shape[0]
    all_sample_file_nv_ratio = all_sample_file_nv.sample(n=501 * int(nv_v_ratio), random_state=42)
    y_predict = all_sample_file_nv_ratio['DeepDFA_Without_LineVul_Predictions'].tolist()
    print(all_sample_file_nv_ratio.columns)
    print(sum(y_predict))
    y_true = all_sample_file_nv_ratio['y_trues'].tolist()
    return  y_predict, y_true, all_sample_file_nv_ratio


def read_joern_errors_list():
    joern_error_index_list_file = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR/joern-error.txt'
    joern_error_index_list = []
    with open(joern_error_index_list_file, 'r') as f:
        for line in f:
            joern_error_index = int(line.strip().split("_")[0])
            if joern_error_index not in joern_error_index_list:
                joern_error_index_list.append(joern_error_index)
    return joern_error_index_list
def read_after_fix_non_vulnerable_samples(predict_dir):
    joern_error_index_list = read_joern_errors_list()
    all_sample_file = 'val_func_before_test_prediction_joern_error_add_after_into_before.xlsx'
    all_sample_file = os.path.join(predict_dir, all_sample_file)
    all_sample_file = pd.read_excel(all_sample_file)
    all_sample_file_nv = all_sample_file[all_sample_file['y_trues'] == 0]
    all_sample_file_v = all_sample_file[all_sample_file['y_trues'] == 1]

    with open(os.path.join(predict_dir, 'id_map_add_after_to_before.json'), 'r', encoding='utf-8') as f:
        id_map_add_after_to_before = json.load(f)
    id_map_add_after_to_before = {value: key for key, value in id_map_add_after_to_before.items()}
    id_map_add_after_to_before_list = list(id_map_add_after_to_before.keys())
    all_sample_file_nv = all_sample_file_nv[all_sample_file_nv['all_inputs_ids'].isin(id_map_add_after_to_before_list)]
    all_sample_file_nv = all_sample_file_nv[~all_sample_file_nv['all_inputs_ids'].isin(joern_error_index_list)]


    all_sample_file_nv_random_after_fixed = all_sample_file_nv.sample(n=501, random_state=42)

    y_predict = all_sample_file_nv_random_after_fixed['DeepDFA_Without_LineVul_Predictions'].tolist()
    print(all_sample_file_nv_random_after_fixed.columns)
    print(sum(y_predict))
    y_true = all_sample_file_nv_random_after_fixed['y_trues'].tolist()
    return  y_predict, y_true, all_sample_file_nv_random_after_fixed


if __name__ == '__main__':

    option = 1
    if option == 1:
        c_root = '../../data/MSR/test.csv'
        df = pd.read_csv(c_root)
        not_use_error_graph(df)
    elif option == 2:
        predict_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired/DeepDFA/without_LineVul_add_after_to_before'
        predict_file = 'setting1_joern_error_instances_prediction_without_22_preprocessing_error_instances.xlsx'
        df_setting_1 = pd.read_excel(os.path.join(predict_dir, predict_file))
        y_trues = df_setting_1['y_trues'].tolist()
        Y_predict = df_setting_1['DeepDFA_Without_LineVul_Predictions'].tolist()
        y_predict_nv, y_true_nv, all_sample_file_nv_ratio = select_random_non_vulnerable_samples(predict_dir)
        y_trues = y_trues + y_true_nv
        Y_predict = Y_predict + y_predict_nv
        calculate_metrics(y_trues, Y_predict)

        df_setting_5 = pd.concat([df_setting_1, all_sample_file_nv_ratio])
        df_setting_5.to_excel(os.path.join(predict_dir, 'Setting_5.xlsx'))
    elif option == 3:
        predict_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired/DeepDFA/without_LineVul_add_after_to_before'
        predict_file = 'setting1_joern_error_instances_prediction_without_22_preprocessing_error_instances.xlsx'
        df_setting_1 = pd.read_excel(os.path.join(predict_dir, predict_file))
        y_trues = df_setting_1['y_trues'].tolist()
        y_trues_len = len(y_trues)
        Y_predict = df_setting_1['DeepDFA_Without_LineVul_Predictions'].tolist()
        y_predict_nv, y_true_nv, all_sample_file_nv_random_after_fixed = read_after_fix_non_vulnerable_samples(predict_dir)
        y_trues = y_trues + y_true_nv[:y_trues_len]
        Y_predict = Y_predict + y_predict_nv[:y_trues_len]
        calculate_metrics(y_trues, Y_predict)

        df_setting_6 = pd.concat([df_setting_1, all_sample_file_nv_random_after_fixed])
        df_setting_6.to_excel(os.path.join(predict_dir, 'Setting_6.xlsx'))