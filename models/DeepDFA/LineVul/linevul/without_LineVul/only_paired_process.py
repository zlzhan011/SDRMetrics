import os.path

import pandas as pd
import numpy as np


if __name__ == '__main__':
    file_dir = '../../data/MSR'
    file_list = ['train_add_after.csv', 'test_add_after.csv', 'val_add_after.csv']
    # file_list = ['val_add_after.csv']
    output_dir = '../../data/MSR'
    for file in file_list:
        file_path = os.path.join(file_dir, file)
        df = pd.read_csv(file_path)
        df_vulner = df[df.target == 1]
        df_after = df[-1 * len(df_vulner):]
        df_combine = pd.concat([df_vulner, df_after])
        file_name = file.replace('_add_after.csv', '')
        df_combine = df_combine.sample(frac=1).reset_index(drop=True)
        df_combine.to_csv(os.path.join(output_dir, file_name+ '_only_paired.csv'), index=False)
        print(df.head())
