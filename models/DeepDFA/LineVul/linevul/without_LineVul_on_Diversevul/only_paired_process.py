import os.path

import pandas as pd
import numpy as np


if __name__ == '__main__':
    file_dir = '../../data/Diversevul'
    file_list_input = ['train.csv', 'test.csv', 'val.csv']
    file_list_output = ['train_add_after.csv', 'test_add_after.csv', 'val_add_after.csv']
    # file_list = ['val_add_after.csv']
    ttv = pd.read_csv('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_Diversevul_add_after_to_before/external/MSR_data_cleaned.csv')


    vulnerable_after = []
    for i, row in ttv.iterrows():
        index = row['index']
        index_original = row['index_original']
        if index != index_original:
            vulnerable_after.append(row)

    vulnerable_after = pd.DataFrame(vulnerable_after)
    vulnerable_after['target'] = vulnerable_after['vul']
    vulnerable_after['index']  = vulnerable_after['index_original']
    print(ttv.columns.tolist())
    i = 0
    for file in file_list_input:
        print("\n\n\n****************************")
        print(file)
        file_path = os.path.join(file_dir, file)
        df = pd.read_csv(file_path)
        df_v_before = df[df['target'] == 1]
        print(df.columns.tolist())
        print(df.shape)

        df_v_before_columns = ['func_before', 'target', 'func_after', 'after_target', 'commit_id', 'project', 'message', 'function_name', 'index']
        vulnerable_after = vulnerable_after[df_v_before_columns]
        print("444",vulnerable_after['target'].tolist())

        df_v_before_index = df_v_before['index'].tolist()
        print("df_v_before_index[:10]", df_v_before_index[:10])
        print("555555", vulnerable_after['index'])
        df_v_after = vulnerable_after[vulnerable_after['index'].isin(df_v_before_index)]

        print("df after")
        # df_merge = pd.merge(vulnerable_after, df, how='left',left_on='index', right_on='index_original')
        print(df_v_after.columns)
        print(df_v_after.shape)

        df_before_plus_after = pd.concat([df, df_v_after])
        df_before_plus_after = df_before_plus_after[df_before_plus_after['target'].isin([0,1,0.0,1.0])]
        print("df_before_plus_after")
        print("6666",df_before_plus_after.columns)
        print("7777",df_before_plus_after.shape)



        for one_column in df_before_plus_after.columns.tolist():
            df_before_plus_after = df_before_plus_after.dropna(subset=[one_column])
        df_before_plus_after['index'] = df_before_plus_after['index'].astype(int)
        df_before_plus_after.to_csv(os.path.join(file_dir, file[:-4]+ '_add_after.csv'), index=False)

