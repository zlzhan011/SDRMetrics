
import argparse
import os
import pandas as pd
from cross_sections.paired.Levenshtein_distance import levenshtein_distance_with_intermediate_steps_list, levenshtein_distance
from tqdm import tqdm
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import numpy as np
import torch
import json
import copy

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 i):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.index=i


def count_trailing_ones(lst):
    count = 0
    for element in reversed(lst):
        if element == 1:
            count += 1
        else:
            break
    token_count = len(lst)-count
    return token_count

def write_json(filename, two_dim_list):
    # 使用 'with' 语句确保文件正确关闭
    with open(filename, 'w') as file:
        # 使用 json.dump() 方法将列表写入文件
        # ensure_ascii=False 允许写入非ASCII字符
        # indent 参数用于美化输出，使其更易于阅读
        json.dump(two_dim_list, file, ensure_ascii=False, indent=4)


def read_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)





def only_count_token_num(df):

    calcul_choice = 'v_plus_nv'  # v   v_plus_nv
    if calcul_choice=='v_plus_nv':
        indices_v = df['index'].tolist()
        func_before_with_target_v = df['before'].tolist()
    elif calcul_choice=='v':
        df = df[df.before_target == 1]
        indices_v = df['index'].tolist()
        func_before_with_target_v = df['before'].tolist()

    count_512 = 0
    for i in tqdm(range(len(indices_v)), desc="load dataset"):
        source_tokens_before, index, source_ids_before, label = convert_examples_to_features(
            func_before_with_target_v[i], indices_v[i], tokenizer, args, indices_v[i])

        token_count_before = count_trailing_ones(source_ids_before)
        if token_count_before == 512:
            count_512 += 1
            print("count_512:",count_512, " index:", index)

        distance = "#"
        intermediate_results = {
                                "token_count_before": token_count_before,

                                "func_before": func_before_with_target_v[i],
                                }
        args.output_dir_index = os.path.join(args.output_dir, 'nv', str(index))

        if not os.path.exists(args.output_dir_index):
            os.makedirs(args.output_dir_index)
        filename = os.path.join(args.output_dir_index, str(index) + "_" + str(distance) + ".json")
        write_json(filename, intermediate_results)
    print("count_512:", count_512)



def calcul_levenshtein_distance(df):
    func_before_with_target_v = df[df['before_target'] == 1]['before'].tolist()
    func_after_with_target_nv = df[df['before_target'] == 1]['after'].tolist()

    # indices_v = df[df['before_target'] == 1].index.tolist()
    # indices_nv = df[df['before_target'] == 0].index.tolist()

    if  args.using_column_index :
        indices_v = df[df['before_target'] == 1]['index'].tolist()
        indices_nv = df[df['before_target'] == 0]['index'].tolist()
    else:
        indices_v = df[df['before_target'] == 1].index.tolist()
        indices_nv = df[df['before_target'] == 0].index.tolist()


    # func_key = 'func_before'  # func_after  func_before

    for i in tqdm(range(len(indices_v)), desc="load dataset"):
        source_tokens_before, index, source_ids_before, label = convert_examples_to_features(
            func_before_with_target_v[i], indices_v[i], tokenizer, args, indices_v[i])
        source_tokens_after, index, source_ids_after, label = convert_examples_to_features(func_after_with_target_nv[i], indices_v[i], tokenizer, args, indices_v[i])

        distance = levenshtein_distance(source_ids_before, source_ids_after)

        token_count_before = count_trailing_ones(source_ids_before)
        token_count_after = count_trailing_ones(source_ids_after)

        distance_normal = distance / max(token_count_before, token_count_after)
        print("distance:", distance)

        index = int(index)
        intermediate_results = {"file_id":index,
                                "distance": distance,
                                "token_count_before": token_count_before,
                                "token_count_after": token_count_after,
                                "distance_normal": distance_normal,
                                "func_before": func_before_with_target_v[i],
                                "func_after": func_after_with_target_nv[i]}
        args.output_dir_index = os.path.join(args.output_dir, str(index))


        if not os.path.exists(args.output_dir_index):
            os.makedirs(args.output_dir_index)
        else:
            import shutil
            shutil.rmtree(args.output_dir_index)
            os.makedirs(args.output_dir_index)
        filename = os.path.join(args.output_dir_index, str(index) + "_" + str(distance) + ".json")
        write_json(filename, intermediate_results)

class TextDatasetProduceSimilar(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        args.file_type = file_type
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        print("args.input_dir", args.input_dir)
        print("file_path:", file_path)
        # if "holdout" in os.path.basename(file_path):
        #     df["split"].replace("holdout", "test")

        if file_path.endswith(".jsonl"):
            df = pd.read_json(file_path, orient="records", lines=True)
        elif file_path.endswith(".csv"):
            file_path = os.path.join(args.input_dir, file_path)
            print("file_path:", file_path)
            df = pd.read_csv(file_path)
            # df = re_shuffle_df(args, df)
        else:
            raise NotImplementedError(file_path)

        # if args.sample:
        #     df = df.sample(100)

        if "processed_func" in df.columns:
            func_key = "processed_func"
        elif "func" in df.columns:
            func_key = "func"

        # only_count_token_num(df)
        calcul_levenshtein_distance(df)



def convert_examples_to_features(func, label, tokenizer, args, index):
    if args.use_word_level_tokenizer:
        encoded = tokenizer.encode(func)
        encoded = encoded.ids
        if len(encoded) > 510:
            encoded = encoded[:510]
        encoded.insert(0, 0)
        encoded.append(2)
        if len(encoded) < 512:
            padding = 512 - len(encoded)
            for _ in range(padding):
                encoded.append(1)
        source_ids = encoded
        source_tokens = []
        return source_tokens, index, source_ids, label
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_tokens, index, source_ids, label




def show_levenshtein_distance_distribution():

    args.output_dir
    # 假设这是您的列表数据
    data = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95, 0.6, 0.4, 0.3, 0.05, 0.15, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    data = []
    different = []
    token_same = []

    data_larger_0 = []

    d_0 =  []
    d_0_01 = []
    d_0_02 = []
    d_0_03 = []
    args.output_dir = os.path.join(args.output_dir, 'nv')
    for subdir in os.listdir(args.output_dir):
        subdir_path = os.path.join(args.output_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for filename in os.listdir(subdir_path):
            if not str(filename[0]).isdigit():
                continue
            if 'nv' not in subdir_path:
                pass
            else:
                continue
            file_path = os.path.join(subdir_path, filename)
            print("file_path:", file_path)
            levenshtein_distance_res = read_json(file_path)
            levenshtein_distance_normalized = levenshtein_distance_res['distance_normal']
            distance = levenshtein_distance_res['distance']
            data.append(levenshtein_distance_normalized)
            different.append(distance)
            if distance == 0:
                token_same.append(subdir)
            if levenshtein_distance_normalized <= 0:
                d_0.append(subdir)
            elif levenshtein_distance_normalized <= 0.09765625:
                d_0_01.append(subdir)
            elif levenshtein_distance_normalized <= 0.3486820678416224:
                d_0_02.append(subdir)
            else:
                d_0_03.append(subdir)

            if levenshtein_distance_normalized <= 0:
                pass
            else:
                data_larger_0.append(levenshtein_distance_normalized)

    print("d_0：", len(d_0))
    print("d_0_01：", len(d_0_01))
    print("d_0_02：", len(d_0_02))
    print("d_0_03：", len(d_0_03))

    output_json = {"d_0": d_0,
                   "d_0_01": d_0_01,
                   "d_0_02":d_0_02,
                   "d_0_03":d_0_03}

    zero_count = different.count(0)
    print("zero_count:", zero_count)

    data_larger_0_sorted = sorted(data_larger_0)
    T1 = np.percentile(data_larger_0_sorted, 33.33)
    T2 = np.percentile(data_larger_0_sorted, 66.67)
    print("T1:", T1)
    print("T2:", T2)
    data = data_larger_0
    # 创建直方图
    plt.hist(data, bins=50, edgecolor='black')

    # 设置图表标题和标签
    plt.title('Distribution of similarity in paired functions')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')

    # 显示图表
    plt.show()

    draw_distribution_pie(data)

    c_output = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired_groups'
    output_file = os.path.join(c_output, 'paired_groups_distribution.json')
    write_json(output_file, output_json)




def draw_distribution_pie(data):
    import matplotlib.pyplot as plt
    import numpy as np

    # 示例数据
    # data = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # 定义区间
    bins = np.arange(0, 1.1, 0.1)  # 从0到1，每0.1一个区间

    # 计算每个区间的数量
    hist, bin_edges = np.histogram(data, bins=bins)

    # 定义区间标签
    labels = [f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}' for i in range(len(bin_edges) - 1)]

    # 绘制饼状图
    plt.figure(figsize=(8, 8))
    plt.pie(hist, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Proportion of Each Similarity Level')
    plt.axis('equal')  # 确保饼图是一个圆
    plt.show()


def add_divided_after_groups_into_before(c_input, c_output):
    paired_groups_distribution = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired_groups/paired_groups_distribution.json'
    paired_groups_distribution = read_json(paired_groups_distribution)

    file_list = ['valid.csv', 'test.csv','train.csv']
    for file_name in file_list:
        file_path = os.path.join(c_input, file_name)
        df = pd.read_csv(file_path)
        df = df[['index', 'func_after', 'func_before', 'processed_func', 'target']]
        df_v = df[df.target == 1]
        df_v_copy = copy.deepcopy(df_v)
        df_v_copy['func_before'] = df_v_copy['func_after']

        for k,v in paired_groups_distribution.items():
            v = [int(i) for i in v]

            filtered_k_df = df_v_copy[df_v_copy['index'].isin(v)]
            filtered_k_df['index'] = filtered_k_df['index'] + len(df)
            df_add_filtered_k_distribution = pd.concat([df, filtered_k_df])
            df_add_filtered_k_distribution = df_add_filtered_k_distribution.sample(frac=1).reset_index(drop=True)

            output_dir = os.path.join(c_output, k)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, file_name)
            df_add_filtered_k_distribution.to_csv(output_file)


def find_predict_res_for_before_and_after_token_are_same():
    predict_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired/LineVul/original'
    paired_compare = os.path.join(predict_dir, 'paired_compare.xlsx')

    paired_compare = pd.read_excel(paired_compare)
    result_dict = paired_compare.set_index('all_inputs_ids')['is_same'].to_dict()


    paired_groups_distribution = os.path.join( '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired_groups', 'paired_groups_distribution.json')
    paired_groups_distribution = read_json(paired_groups_distribution)

    ii = 0
    for k, v in paired_groups_distribution.items():
        # print("k:", k)
        # print("V:", v)
        if k == 'd_0':
            for id in v:
                id = int(id)
                if id in result_dict:
                    id_predict = result_dict[id]
                    print(id_predict)
                    ii += 1
    print("ii:", ii)

    # before_predict = os.path.join(predict_dir, 'val_func_before_prediction.xlsx')
    # after_predict = os.path.join(predict_dir, 'val_func_after_prediction.xlsx')
    # before_predict = pd.read_excel(before_predict)
    # after_predict = pd.read_excel(after_predict)
    # before_predict_v = before_predict[before_predict.y_trues == 1]
    # after_predict_v = after_predict[after_predict.y_trues == 1]
    # pd.merge(before_predict_v, after_predict_v, how= 'left', left_on='all_inputs_ids', right_on='all_inputs_ids')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check after not exist in before')
    # word-level tokenizer
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    # bpe non-pretrained tokenizer
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    args = parser.parse_args()
    args.c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/Diversevul/ttv'
    args.sample = True
    args.down_resample = False
    args.input_dir = args.c_root
    args.output_dir = os.path.join(args.input_dir, 'paired_groups')
    args.test_data_file = os.path.join(args.input_dir, 'valid.csv')    # test.cs result:2/1055      valid.csv  result:2/1055,  train.csv  29/8736
    args.eval_data_file = os.path.join(args.input_dir, 'test.csv')      # contact test, valid, train   36/10900
    args.train_data_file = os.path.join(args.input_dir, 'train.csv')
    args.using_column_index = True

    if args.using_column_index:
        args.output_dir = os.path.join(args.output_dir, 'using_column_index')
    else:
        args.output_dir = os.path.join(args.output_dir, 'using_virtual_index')


    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('../word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="../bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="../bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    # train_dataset = TextDataset(tokenizer, args, file_type='train')
    eval_dataset = TextDatasetProduceSimilar(tokenizer, args, file_type='train')

    # show_levenshtein_distance_distribution()
    # find_predict_res_for_before_and_after_token_are_same()

    # add_divided_after_groups_into_before(args.input_dir, args.output_dir)

