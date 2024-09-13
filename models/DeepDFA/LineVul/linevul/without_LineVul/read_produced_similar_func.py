import json
import os
from tqdm import tqdm
def read_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)

    return content




class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 i):
        self.input_tokens = padding(input_tokens)
        self.input_ids = padding(input_ids)
        print("self.input_ids len:", len(self.input_ids))
        self.label=label
        self.index=i

def padding(seq):
    if len(seq) < 512:
        seq_len = len(seq)
        diff = 512 - seq_len
        for i in range(diff):
            seq.append(1)
    return seq


if __name__ == '__main__':
    c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/produce_similar_func'
    file_list = ['179639', '179281']
    examples = []
    for dir_index in file_list:
        file_dir = os.path.join(c_root, dir_index)
        for json_file in os.listdir(file_dir):
            if '.json' in json_file and 'knn' not in json_file:
                content = read_json(os.path.join(file_dir, json_file))
                for i in tqdm(range(len(content)), desc="load dataset"):
                    # print("content[i]:", content[i])
                    # print("content[i] len:", len(content[i]))
                    examples.append(InputFeatures(content[i], content[i], 1, int(dir_index)))
