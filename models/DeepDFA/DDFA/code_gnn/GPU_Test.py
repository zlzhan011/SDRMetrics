


import os
import pickle
import pandas as pd
import DDFA.sastvd as svd

# 设置显示所有列
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.expand_frame_repr', False)


def read_pickle(pickle_path):
    # 打开文件（以二进制读取模式）
    with open(pickle_path, "rb") as file:
        # 使用pickle.load()方法从文件中读取对象
        loaded_data = pickle.load(file)

    print("\npickle_path：", pickle_path)
    print("loaded_data：", loaded_data)
    return loaded_data

intermediate_result =  os.path.join(svd.processed_dir(), 'bigvul_intermediate_result')

for file in os.listdir(intermediate_result):
    file_path = os.path.join(intermediate_result, file)
    if os.path.isdir(file_path):
        # for sub_file in os.listdir(file_path):
        #     sub_path = os.path.join(file_path, sub_file)
        #     read_pickle(sub_path)
        pass
    else:
        if file.endswith('.pkl'):
            print("\n\n-----------------")
            if "graphs_by_id" in file.lower():
                continue
            else:
                read_pickle(file_path)


