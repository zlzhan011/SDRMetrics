import os
import pandas as pd








if __name__ == '__main__':
    dataset= 'Diversevul'
    if dataset =='MSR':
        c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/'
        target_column = 'target'
        before_column = 'func_before'
        after_column = 'func_after'
    elif dataset == 'Diversevul':
        c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/Diversevul/ttv'
        target_column = 'before_target'
        before_column = 'before'
        after_column = 'after'
    out_dir = os.path.join(c_root, 'generate_non_paired_random_pair')
    files_list = ['valid.csv', 'test.csv', 'train.csv']
    df_nv_all = []
    df_v_all = []
    for file_name in files_list:
        print("file_name:", file_name)
        file_path = os.path.join(c_root, file_name)
        df = pd.read_csv(file_path)
        df_nv = df[df[target_column] == 0]
        df_v = df[df[target_column] == 1]
        print(len(df_nv))
        df_nv_all.append(df_nv)
        df_v_all.append(df_v)

    df_nv_all = pd.concat(df_nv_all)
    df_v_all = pd.concat(df_v_all)
    print("df_v_all_len:", len(df_v_all))
    print("df_nv_all_len:", len(df_nv_all))
    df_nv_all = df_nv_all.sample(frac=1).reset_index(drop=True)
    df_v_all = df_v_all.sample(frac=1).reset_index(drop=True)

    df_nv_all_random_pair = df_nv_all.sample(n=len(df_v_all)*2, random_state=42)
    print("df_nv_all_random_pair_len:", len(df_nv_all_random_pair))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir = os.path.join(out_dir, dataset)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_nv_all_random_pair = df_nv_all_random_pair.reset_index(drop=True)
    df_v_all_len = len(df_v_all) -1
    print("df_nv_all_random_pair_column:", len(df_nv_all_random_pair))
    for index, row in df_nv_all_random_pair.iterrows():
        print("index:", index)
        df_nv_all_random_pair.loc[index, after_column] = df_nv_all_random_pair.iloc[index+df_v_all_len][before_column]
        if index >= df_v_all_len:
            break
    df_nv_all_random_pair = df_nv_all_random_pair[:len(df_v_all)]
    print("df_nv_all_random_pair len:", len(df_nv_all_random_pair))
    df_nv_all_random_pair.to_csv(os.path.join(out_dir, 'df_nv_all_random_pair.csv'))
    df_v_all.to_csv(os.path.join(out_dir, 'df_v_all_paired.csv'))








