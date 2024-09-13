import json
import os
import pandas as pd
if __name__ == '__main__':
    c_root_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/Diversevul/ttv/paired_groups'
    # c_root_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired_groups'
    c_root = os.path.join(c_root_root, 'using_column_index')
    all_json = []
    for sub_dir in os.listdir(c_root):
        sub_dir = os.path.join(c_root, sub_dir)
        for file in os.listdir(sub_dir):
            if file.endswith('.json'):
                with open(os.path.join( sub_dir, file), 'r') as f:
                    data = json.load(f)
                    all_json.append(data)

    pd.DataFrame(all_json).to_csv(os.path.join(c_root_root, 'using_column_index.csv'))