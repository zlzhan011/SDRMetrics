import copy
import os.path
import pandas as pd

def get_only_paired_func():
    input_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset'
    output_dir = os.path.join(input_dir, 'only_paired_func')
    files = ['train.csv', 'test.csv', 'valid.csv']
    for file in files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)
        df_vulnerable = df[df['target'] == 1]
        df_vulnerable_copy = copy.deepcopy(df_vulnerable)
        df_vulnerable_copy['func_before'] = df_vulnerable_copy['func_after']
        df_vulnerable_copy['processed_func'] = df_vulnerable_copy['func_after']
        df_vulnerable_copy['target'] = 0
        df_paired = pd.concat([df_vulnerable_copy, df_vulnerable])
        df_paired = df_paired.sample(frac=1).reset_index(drop=True)
        df_paired.to_csv(os.path.join(output_dir, file), index=False)


def add_after_into_before():
    input_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset'
    output_dir = '/data2/cs_lzhan011/vulnerability/dataset/MSR/add_after_into_before'
    output_dir_append = os.path.join(output_dir, 'add_after_into_before_append')
    output_dir_shuffle = os.path.join(output_dir, 'add_after_into_before_shuffle')
    files = ['train.csv', 'test.csv', 'valid.csv']
    for file in files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)
        df = df[['index', 'func_before', 'func_after', 'processed_func', 'target']]
        df_vulnerable = df[df['target'] == 1]
        df_non_vulnerable = df[df['target'] == 0]
        df_vulnerable_copy = copy.deepcopy(df_vulnerable)
        df_vulnerable_copy['func_before'] = df_vulnerable_copy['func_after']
        df_vulnerable_copy['processed_func'] = df_vulnerable_copy['func_after']
        df_vulnerable_copy['target'] = 0
        df_add_after = pd.concat([df, df_vulnerable_copy])
        # df_add_after = df_add_after.sample(frac=1).reset_index(drop=True)
        df_add_after.to_csv(os.path.join(output_dir_append, file), index=False)


def generate_add_after_percent():
    input_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset'
    output_dir = '/data2/cs_lzhan011/vulnerability/dataset/MSR/add_after_percent'
    files = ['train.csv', 'test.csv', 'valid.csv']
    for file in files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)
        df = df[['func_before', 'func_after', 'processed_func', 'target']]
        df_vulnerable = df[df['target'] == 1]
        df_non_vulnerable = df[df['target'] == 0]
        df_vulnerable_copy = copy.deepcopy(df_vulnerable)
        df_vulnerable_copy['func_before'] = df_vulnerable_copy['func_after']
        df_vulnerable_copy['processed_func'] = df_vulnerable_copy['func_after']
        df_vulnerable_copy['target'] = 0

        for percent in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # 随机选择20%的数据
            percent_multi = percent * 100
            percent_dir = 'add_after_percent_' + str(int(percent_multi))
            df_sampled = df_vulnerable_copy.sample(frac=percent)
            df_add_after = pd.concat([df, df_sampled])
            df_add_after = df_add_after.sample(frac=1).reset_index(drop=True)
            df_add_after.to_csv(os.path.join(output_dir, percent_dir, file), index=False)


if __name__ == '__main__':
    add_after_into_before()





