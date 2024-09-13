import copy
import os.path
import pandas as pd

def get_only_paired_func():
    pass


if __name__ == '__main__':
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
