import os
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc


def generate_add_after_percent():
    c_dir = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR'
    files = ['test_add_after.csv', 'val_add_after.csv', 'train_add_after.csv']
    c_output_dir = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR/train_add_after_percent'
    for file in files:
        print("file:", file)
        file_path = os.path.join(c_dir, file)
        df = pd.read_csv(file_path)
        # df = df[['index', 'func_before', 'func_after', 'target']]
        df_vulnerable = df[df.target == 1]
        print("len_df", len(df))
        df_no_after = df[:-1 * len(df_vulnerable)]
        print("len(df_no_after):", len(df_no_after))
        print("\n\n")
        df_after = df[-1 * len(df_vulnerable):]
        df_after_len = len(df_after)
        # df_after_percent_10 = df_after[:round(df_after_len * 0.1)]
        # df_after_percent_30 = df_after[:round(df_after_len * 0.3)]
        # df_after_percent_50 = df_after[:round(df_after_len * 0.5)]
        # df_after_percent_70 = df_after[:round(df_after_len * 0.7)]
        # df_after_percent_90 = df_after[:round(df_after_len * 0.9)]
        for percent in [0.1, 0.3, 0.5, 0.7, 0.9]:
            df_after_percent =  df_after[:round(df_after_len * percent)]
            df_combine_add_after_percent = pd.concat([df_no_after, df_after_percent])
            # df_combine_add_after_percent = df_combine_add_after_percent.sample(frac=1)
            output_file_name = '_percent_' + str(int(100 * percent))
            df_combine_add_after_percent.to_csv(os.path.join(c_output_dir, file[:-4] + output_file_name +".csv") , index=False)



def calculate_metrics():
    c_dir = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR'
    files = ['test_add_after.csv']
    c_output_dir = '/data2/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR/train_add_after_percent'
    test_df_no_after_len = 0
    for file in files:
        print("file:", file)
        file_path = os.path.join(c_dir, file)
        df = pd.read_csv(file_path)
        df = df[['index', 'func_before', 'func_after', 'target']]
        df_vulnerable = df[df.target == 1]
        print("len_df", len(df))
        df_no_after = df[:-1 * len(df_vulnerable)]
        test_df_no_after_len = len(df_no_after)
        print("len(df_no_after):", len(df_no_after))
        print("\n\n")
    test_df_no_after_len = 18713
    c_output_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired/DeepDFA_LineVul/add_after_to_before'
    for dir in ['after_percent_30', 'after_percent_50', 'after_percent_70', 'after_percent_100']:
        print("sub_dir:", dir)
        sub_dir = os.path.join(c_output_dir, dir)
        files = ['test_func_before_prediction.xlsx', 'test_func_after_prediction.xlsx']
        df = pd.read_excel(os.path.join(sub_dir, files[0]))
        df_no_after = df[:test_df_no_after_len]
        columns = df_no_after.columns.tolist()
        y_trues = df_no_after['y_trues'].values.tolist()
        for column in columns:
            if 'Predictions' in column:
                y_pred = df_no_after[column].tolist()

        f1 = f1_score(y_trues, y_pred)
        precision = precision_score(y_trues, y_pred)
        recall = recall_score(y_trues, y_pred)
        print("f1:", f1, "precision:", precision, "recall:", recall)
        print("\n\n\n")



if __name__ == '__main__':
    generate_add_after_percent()
    # calculate_metrics()

    # sub_dir: after_percent_30
    # f1: 0.7558823529411764
    # precision: 0.7656405163853028
    # recall: 0.7463697967086157
    #
    # sub_dir: after_percent_50
    # f1: 0.7265193370165745
    # precision: 0.6843018213356461
    # recall: 0.774288518155054
    #
    # sub_dir: after_percent_70
    # f1: 0.6886749197615772
    # precision: 0.632687447346251
    # recall: 0.755533199195171
    #
    # sub_dir: after_percent_100
    # f1: 0.8475420037336652
    # precision: 0.9714693295292439
    # recall: 0.7516556291390728

