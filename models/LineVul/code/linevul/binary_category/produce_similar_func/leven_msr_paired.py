
import argparse
import os
import shutil
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
parent_parent2_dir = os.path.dirname(parent_parent_dir)
parent_parent3_dir = os.path.dirname(parent_parent2_dir)
parent_parent4_dir = os.path.dirname(parent_parent3_dir)
parent_parent5_dir = os.path.dirname(parent_parent4_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)
sys.path.append(parent_parent2_dir)
sys.path.append(parent_parent3_dir)
sys.path.append(parent_parent4_dir)
sys.path.append(parent_parent5_dir)
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
    if calcul_choice == 'v_plus_nv':
        indices_v = df['index'].tolist()
        func_before_with_target_v = df['func_before'].tolist()
    elif calcul_choice == 'v':
        df = df[df.target == 1]
        indices_v = df['index'].tolist()
        func_before_with_target_v = df['func_before'].tolist()

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



def calcul_levenshtein_distance(df, file_type):
    df = df.reset_index(drop=True)
    if file_type == 'non-paired':
        func_before_with_target_v = df[args.before_column].tolist()
        func_after_with_target_nv = df[args.after_column].tolist()

        indices_v = df.index.tolist()
        indices_nv = df.index.tolist()
    else:
        func_before_with_target_v = df[df[args.target_column] == 1][args.before_column].tolist()
        func_after_with_target_nv = df[df[args.target_column] == 1][args.after_column].tolist()

        indices_v = df[df[args.target_column] == 1].index.tolist()
        indices_nv = df[df[args.target_column] == 0].index.tolist()



    # func_key = 'func_before'  # func_after  func_before

    for i in tqdm(range(len(indices_v)), desc="load dataset"):
        # if i <= 5159:
        #     continue
        source_tokens_before, index, source_ids_before, label = convert_examples_to_features(
            func_before_with_target_v[i], indices_v[i], tokenizer, args, indices_v[i])
        source_tokens_after, index, source_ids_after, label = convert_examples_to_features(
            func_after_with_target_nv[i], indices_v[i], tokenizer, args, indices_v[i])

        distance = levenshtein_distance(source_ids_before, source_ids_after)

        token_count_before = count_trailing_ones(source_ids_before)
        token_count_after = count_trailing_ones(source_ids_after)

        distance_normal = distance / max(token_count_before, token_count_after)
        print("\ndistance:", distance, "distance normal:", distance_normal)


        intermediate_results = {"distance": distance,
                                "token_count_before": token_count_before,
                                "token_count_after": token_count_after,
                                "distance_normal": distance_normal,
                                "func_before": func_before_with_target_v[i],
                                "source_tokens_before": "###".join(source_tokens_before),
                                "func_after": func_after_with_target_nv[i],
                                "source_tokens_after": "###".join(source_tokens_after),}
        args.output_dir_index = os.path.join(args.output_dir, file_type, str(index))

        if not os.path.exists(args.output_dir_index):
            os.makedirs(args.output_dir_index)
        else:
            print("args.output_dir_index:", args.output_dir_index)
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
        elif file_type == "non-paired":
            file_path = args.non_paired_paired_data_file
        elif file_type == 'paired':
            file_path = args.paired_data_file
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
        calcul_levenshtein_distance(df, file_type)



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
    tokenizer_output = tokenizer.tokenize(str(func))
    tokenizer_output_length = len(tokenizer_output)
    if args.block_size == 512:

        code_tokens = tokenizer_output[:args.block_size - 2]
    else:
        code_tokens = tokenizer_output[:tokenizer_output_length - 2]
    print(" len code_tokens:", len(code_tokens))
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_tokens, index, source_ids, label



def calculate_distribution(data):
    import numpy as np

    print("data_len:", len(data))
    # 定义 bin 的边界
    bin_edges = [i * 0.1 for i in range(11)]
    print("bin_edges:", bin_edges)
    bin_edges = [-0.1] + bin_edges
    # 创建一个字典来存储分布信息
    distribution = {f"({bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]": 0 for i in range(len(bin_edges) - 1)}

    # 计算每个区间的元素个数
    for value in data:
        for i in range(1, len(bin_edges)):
            if bin_edges[i - 1] < value <= bin_edges[i]:
                key = f"({bin_edges[i - 1]:.1f}, {bin_edges[i]:.1f}]"
                distribution[key] += 1
                break

    # 输出结果
    for k, v in distribution.items():
        print(f"区间 {k}: {v} 个数")

    return distribution


def show_levenshtein_distance_distribution():
    args.output_dir_parent = copy.deepcopy(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.file_type )

    data = []
    different = []
    token_same = []

    data_larger_0 = []

    d_0 =  []
    d_0_01 = []
    d_0_02 = []
    d_0_03 = []
    print("os.listdir(args.output_dir)_len", len(os.listdir(args.output_dir)))

    for subdir in os.listdir(args.output_dir):
        subdir_path = os.path.join(args.output_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        i_cnt = 0
        for filename in os.listdir(subdir_path):
            if not str(filename[0]).isdigit():
                continue
            i_cnt = i_cnt + 1

            # if 'nv' not in subdir_path:
            #     pass
            # else:
            #     continue
            file_path = os.path.join(subdir_path, filename)
            if i_cnt >= 2:
                print("file_path:", file_path)
            # print("file_path:", file_path)
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
    print("i_cnt_----", i_cnt)
    output_json = {"d_0": d_0,
                   "d_0_01": d_0_01,
                   "d_0_02":d_0_02,
                   "d_0_03":d_0_03}

    zero_count = different.count(0)
    print("zero_count:", zero_count)


    distribution = calculate_distribution(data)
    # print(distribution)
    print(list(distribution.values()))
    categories = list(distribution.keys())
    values = list(distribution.values())
    print("categories:", categories)
    categories[0] = '[0.0, 0.0]'
    colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

    plt.bar(categories, values, color=colors, edgecolor='black')


    # 设置图表标题和标签
    plt.title('Distribution of distance in '+args.file_type+' functions')
    plt.xlabel('Normalized Levenshtein Distance')
    plt.ylabel('Number of instances')
    plt.xticks(ticks=np.arange(len(categories)), labels=categories, rotation=45, ha='right', color='black')
    for i, value in enumerate(values):
        plt.text(i, value + 0.10, str(value), ha='center', va='bottom')


    # 显示图表
    plt.tight_layout()

    c_output = args.output_dir_parent

    output_file = os.path.join(c_output, args.file_type + '_paired_groups_distribution.json')
    output_fig = os.path.join(c_output, args.file_type + '_distribution.pdf')
    plt.savefig(output_fig, format='pdf')
    plt.show()

    # draw_distribution_pie(data)


    write_json(output_file, output_json)




def read_paired_and_non_paired_distribution(args):
    args.output_dir_parent = copy.deepcopy(args.output_dir)
    args.output_dir_file_type = os.path.join(args.output_dir_parent, args.file_type)

    data = []
    different = []
    token_same = []

    data_larger_0 = []

    d_0 = []
    d_0_01 = []
    d_0_02 = []
    d_0_03 = []
    print("os.listdir(args.output_dir)_len", len(os.listdir(args.output_dir_file_type)))

    for subdir in os.listdir(args.output_dir_file_type):
        subdir_path = os.path.join(args.output_dir_file_type, subdir)
        if not os.path.isdir(subdir_path):
            continue
        i_cnt = 0
        for filename in os.listdir(subdir_path):
            if not str(filename[0]).isdigit():
                continue
            i_cnt = i_cnt + 1

            # if 'nv' not in subdir_path:
            #     pass
            # else:
            #     continue
            file_path = os.path.join(subdir_path, filename)
            if i_cnt >= 2:
                print("file_path:", file_path)
            # print("file_path:", file_path)
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
    print("i_cnt_----", i_cnt)
    output_json = {"d_0": d_0,
                   "d_0_01": d_0_01,
                   "d_0_02": d_0_02,
                   "d_0_03": d_0_03}

    zero_count = different.count(0)
    print("zero_count:", zero_count)

    distribution = calculate_distribution(data)
    # print(distribution)
    print(list(distribution.values()))
    categories = list(distribution.keys())
    values = list(distribution.values())
    print("categories:", categories)
    # categories[0] = '[0.0, 0.0]'
    return data, distribution, categories, values




def show_levenshtein_distance_distribution_one_dataset():
    args.output_dir = args.output_dir
    distribution_paired_types = {}
    for paired_type in ['paired', 'non-paired']:
        args.file_type = paired_type
        data, distribution, categories, values = read_paired_and_non_paired_distribution(args)
        distribution_paired_types[paired_type] = {"data":data,
                                                  "distribution": distribution,
                                                  "categories":categories,
                                                  "values":values}

    print(distribution_paired_types)
    fontsize = 24
    fontsize_2 = 16
    fontsize_3 = 22
    bins = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bins = np.array(bins)
    bins = np.array(bins) + 1e-10
    bins[0] -= 1e-10  # 第一个bin边界不需要调整

    paired_data = distribution_paired_types['paired']['data']
    non_paired_data = distribution_paired_types['non-paired']['data']

    counts1, bins1, patches1 = plt.hist(paired_data, bins=bins, edgecolor='black', alpha=0.5, color='blue', label='paired')
    counts2, bins2, patches2 = plt.hist(non_paired_data, bins=bins, edgecolor='black', alpha=0.5, color='red', label='non-paired')

    # 计算比例
    total_counts1 = sum(counts1)
    total_counts2 = sum(counts2)

    # 计算每个bin的百分比
    percentages1 = counts1 / total_counts1
    percentages2 = counts2 / total_counts2

    # 绘制百分比柱状图
    plt.clf()  # 清除之前的绘图
    width = bins[1] - bins[0]
    bins_mid = (bins[:-1] + bins[1:]) / 2
    plt.bar(bins_mid, percentages1, width=width, edgecolor='black', alpha=0.5, color='red', label='paired')
    plt.bar(bins_mid, percentages2, width=width, edgecolor='black', alpha=0.5, color='green', label='non-paired')

    # 在第一个数据集的每个柱子顶部添加比例标签
    for percentage, bin_edge in zip(percentages1, bins_mid):
        if f'{percentage:.2f}' == '0.00':
            pass
        else:
            plt.text(bin_edge, percentage, f'{percentage:.2f}', ha='center', va='bottom', color='red', fontsize=fontsize_2)

    # 在第二个数据集的每个柱子顶部添加比例标签
    for percentage, bin_edge in zip(percentages2, bins_mid):
        if f'{percentage:.2f}' == '0.00':
            pass
        else:
            plt.text(bin_edge, percentage, f'{percentage:.2f}', ha='center', va='bottom', color='green', fontsize=fontsize_2)

    # 添加图例

    plt.legend(fontsize=fontsize_3)
    # 设置图表标题和标签
    # plt.title('Distribution of distance in '+args.file_type+' functions')
    plt.xlabel('Normalized Levenshtein Distance',fontsize=fontsize_3)
    plt.ylabel('Ratio', fontsize=fontsize)
    # Adjust y-axis to show decimal values


    y_ticks = np.linspace(0, 1, num=11)
    plt.yticks(y_ticks, [f'{tick:.1f}' for tick in y_ticks],fontsize=fontsize)

    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, _: f'{x:.1f}')
    plt.gca().xaxis.set_major_formatter(formatter)

    bins_2 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(bins_2, [f'{tick:.1f}' for tick in bins_2], fontsize=fontsize)

    # # 显示图表
    # plt.tight_layout()

    c_output = args.output_dir_parent
    print("c_output:", c_output)
    output_fig = os.path.join(c_output, args.dataset+ '_distribution.pdf')
    plt.savefig(output_fig, format='pdf')

    output_fig = os.path.join(c_output, args.dataset + '_distribution.eps')
    plt.savefig(output_fig, format='eps', bbox_inches='tight')
    output_fig = os.path.join(c_output, args.dataset + '_distribution.svg')
    plt.savefig(output_fig, format='svg', bbox_inches='tight')
    plt.show()



def load_feature(cache_name, step_state, return_label=False, return_predict=False, return_file_ids=False):
    np_file = np.load(cache_name, allow_pickle=True)
    feat_log, score_log, label_log, predict_log, file_ids = np_file['arr_0'], np_file['arr_1'], np_file['arr_2'], np_file['arr_3'],np_file['arr_4']
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]
    # feature = feat_log[:, -770:-2]
    correct_log = [1 if label_log[i] == predict_log[i] else 0 for i in range(len(predict_log))]

    feature = feat_log[:, 768 * (step_state - 1):(step_state * 768)]
    if return_label:
        return feature, correct_log, label_log
    if return_predict:
        return feature, correct_log, predict_log
    if return_file_ids:
        files_id_embedding = {}
        for i in range(len(file_ids)):
            file_id = file_ids[i]
            print("file_id:", file_id)
            file_id = int(file_id)
            files_id_embedding[file_id] = feature[i]
        return files_id_embedding
    return feature, correct_log



def show_two_list(list1, list2, paired_type):
    import matplotlib.pyplot as plt
    import numpy as np

    # 示例数据
    # list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # list2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    # 检查两个列表是否长度相等
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度必须相等")

    # 创建图形和轴对象
    fig, ax1 = plt.subplots()

    # 创建 x 轴刻度 [0, 0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
    x_ticks = np.linspace(0, 1, num=len(list1))
    x_labels = ['[0]'] + [f'({tick:.1f}, {tick + 0.1:.1f}]' for tick in x_ticks[:-1]]
    print(x_labels)
    # 绘制第一个列表的折线图
    color1 = 'black'
    ax1.set_xlabel('Normalized Levenshtein Distance', fontsize='23')
    ax1.set_ylabel('Euclidean distance', color=color1, fontsize='23')
    ax1.plot(x_ticks, list1, label='Euclidean distance', color=color1, marker='o')
    ax1.tick_params(axis='y', labelcolor=color1)

    # 创建第二个 y 轴
    ax2 = ax1.twinx()

    # 绘制第二个列表的折线图
    color2 = 'tab:red'
    ax2.set_ylabel('Cosine similarity', color=color2, fontsize='23' )
    ax2.plot(x_ticks, list2, label='Cosine similarity', color=color2, marker='s')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 设置 x 轴刻度和标签
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize='23')  # 旋转标签以防止重叠

    # y_labels_2 = [f"{cosine_value:.1f}" for cosine_value in np.arange(0.0, 1.1, 0.1)]
    y_labels_2 = np.arange(0.0, 1.2, 0.2)
    y_labels_2_str = [f"{cosine_value:.1f}" for cosine_value in y_labels_2]
    # 设置x轴和y轴的刻度
    ax2.set_yticks(y_labels_2)
    ax2.set_yticklabels(y_labels_2_str, fontsize='23')

    if args.dataset == 'Diversevul':
        y_labels_1 = [euc_value for euc_value in range(0, 16, 2)]
        y_labels_1_str = [str(euc_value) for euc_value in range(0, 16, 2)]
    elif args.dataset == 'MSR':
        y_labels_1 = [euc_value for euc_value in range(0, 18, 2)]
        y_labels_1_str = [str(euc_value) for euc_value in range(0, 18, 2)]

    ax1.set_yticks(y_labels_1)
    ax1.set_yticklabels(y_labels_1_str, fontsize='23')

    # 添加图例
    fig.tight_layout()  # 调整布局，以防止标签重叠
    # fig.legend(loc='center', bbox_to_anchor=(0.26, 0.6), fontsize='23')  # 将图例放在中间

    # 保存图像

    file_name = os.path.join(args.output_dir, 'visual', args.dataset +'_token_distance_embedding_distance_distribution_'+paired_type+'_euclidean_cosine')
    file_name_png = file_name + ".png"
    file_name_pdf = file_name + ".pdf"
    file_name_eps = file_name + ".eps"
    file_name_svg = file_name + ".svg"
    plt.savefig(file_name_png, dpi=300, bbox_inches='tight')
    plt.savefig(file_name_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(file_name_eps, dpi=300, bbox_inches='tight')
    plt.savefig(file_name_svg, dpi=300, bbox_inches='tight',  format='svg')
    # 显示图像
    plt.show()


def show_one_list(list1, paired_type):




    # 创建图形和轴对象
    fig, ax1 = plt.subplots()
    x_ticks = np.linspace(0, 1, num=len(list1))


    # 绘制第一个列表的折线图
    color1 = 'black'
    ax1.set_xlabel('Normalized Levenshtein Distance')
    ax1.set_ylabel('loss', color=color1)
    ax1.plot(x_ticks, list1, label='loss', color=color1, marker='o')
    ax1.tick_params(axis='y', labelcolor=color1)


    # 添加图例
    fig.tight_layout()  # 调整布局，以防止标签重叠
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.savefig(os.path.join(args.output_dir, 'visual', 'token_distance_embedding_distance_distribution_'+paired_type+'_loss.png'))
    # 显示图像
    plt.show()


def show_one_list_v2(list1, paired_type):




    # 创建图形和轴对象
    fig, ax1 = plt.subplots()
    x_ticks = np.linspace(0, 1, num=len(list1))

    list1 = list1[:7]
    x_ticks = x_ticks[:7]
    # 绘制第一个列表的折线图
    color1 = 'black'
    ax1.set_xlabel('Normalized Levenshtein Distance')
    ax1.set_ylabel('loss', color=color1)
    ax1.plot(x_ticks, list1, label='loss', color=color1, marker='o')
    ax1.tick_params(axis='y', labelcolor=color1)


    # 添加图例
    fig.tight_layout()  # 调整布局，以防止标签重叠
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.savefig(os.path.join(args.output_dir, 'visual', 'token_distance_embedding_distance_distribution_'+paired_type+'_loss_only_7_elements.png'))
    # 显示图像
    plt.show()


def read_token_distribution_data():
    two_kinds_data = {}
    for paired_type in ['non-paired', 'paired']:
        args.file_type = paired_type

        args.output_dir_parent = copy.deepcopy(args.output_dir)
        args.output_dir_file_type = os.path.join(args.output_dir_parent, args.file_type)

        data = []

        for subdir in os.listdir(args.output_dir_file_type):
            subdir_path = os.path.join(args.output_dir_file_type, subdir)
            if not os.path.isdir(subdir_path):
                continue
            i_cnt = 0
            for filename in os.listdir(subdir_path):
                if not str(filename[0]).isdigit():
                    continue
                i_cnt = i_cnt + 1

                file_path = os.path.join(subdir_path, filename)
                if i_cnt >= 2:
                    print("file_path:", file_path)
                # print("file_path:", file_path)
                levenshtein_distance_res = read_json(file_path)
                levenshtein_distance_normalized = levenshtein_distance_res['distance_normal']
                distance = levenshtein_distance_res['distance']
                data.append({"file_id": subdir,
                             "levenshtein_distance_normalized": levenshtein_distance_normalized,
                             "distance": distance})
        two_kinds_data[paired_type] = data
    return two_kinds_data


def get_sector_embedding_distance(data, paired_type):

    if paired_type== 'paired':
        cache_name = os.path.join(args.output_dir, 'visual_cache',
                                  'df_v_all_paired_index_correct.csvfunc_after_last_second_layer_index_correct.npy.npz')
        step_state = 16
        files_id_embedding_func_after = load_feature(cache_name, step_state, return_label=False, return_predict=False,
                                                     return_file_ids=True)
        cache_name = os.path.join(args.output_dir, 'visual_cache',
                                  'df_v_all_paired_index_correct.csvfunc_before_last_second_layer_index_correct.npy.npz')
        files_id_embedding_func_before = load_feature(cache_name, step_state, return_label=False, return_predict=False,
                                                      return_file_ids=True)
        labels = torch.tensor([1, 0], dtype=torch.long)
    elif paired_type== 'non-paired':
        cache_name = os.path.join(args.output_dir, 'visual_cache',
                                  'df_nv_all_random_pair_index_correct.csvfunc_after_last_second_layer_index_correct.npy.npz')
        step_state = 16
        files_id_embedding_func_after = load_feature(cache_name, step_state, return_label=False, return_predict=False,
                                                     return_file_ids=True)
        cache_name = os.path.join(args.output_dir, 'visual_cache',
                                  'df_nv_all_random_pair_index_correct.csvfunc_before_last_second_layer_index_correct.npy.npz')
        files_id_embedding_func_before = load_feature(cache_name, step_state, return_label=False, return_predict=False,
                                                      return_file_ids=True)
        labels = torch.tensor([0, 0], dtype=torch.long)
    else:
        raise ValueError('not correct paired type')


    embedding_distance = {}
    for k, v in files_id_embedding_func_before.items():
        embedding_before = files_id_embedding_func_before[k]
        embedding_after = files_id_embedding_func_after[k]
        euclidean_distance = np.linalg.norm(embedding_before - embedding_after)

        cosine_similarity = np.dot(embedding_before, embedding_after) / (
                    np.linalg.norm(embedding_before) * np.linalg.norm(embedding_after))



        # before_after_logits = torch.tensor([embedding_before, embedding_after])
        # prob = torch.softmax(before_after_logits, dim=-1)
        #
        # from torch.nn import CrossEntropyLoss
        # loss_fct = CrossEntropyLoss()
        # loss = loss_fct(before_after_logits, labels)

        loss = [0]
        prob = 0

        embedding_distance[k] = {"euclidean_distance": euclidean_distance,
                                 "cosine_similarity": cosine_similarity,
                                 "loss":loss,
                                 'prob':prob}


    bins = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_sector_res = {}
    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i + 1]
        sector = str(start) + "_" + str(end)
        sector_files_id = []
        for i_inner in range(len(data)):
            file_id = data[i_inner]['file_id']
            levenshtein_distance_normalized = data[i_inner]['levenshtein_distance_normalized']
            # if i == 0:
            #     if levenshtein_distance_normalized >= start and levenshtein_distance_normalized <= end:
            #         sector_files_id.append(file_id)
            # else:
            #     if levenshtein_distance_normalized > start and levenshtein_distance_normalized <= end:
            #         sector_files_id.append(file_id)

            if levenshtein_distance_normalized > start and levenshtein_distance_normalized <= end:
                sector_files_id.append(file_id)

        all_sector_res[sector] = sector_files_id
    # print(all_sector_res)
    sector_embedding_distance = {}
    for k, v in all_sector_res.items():
        print("k number:", k, len(v))
        one_sector_embedding_distance = []
        for file_id in v:
            one_embedding_distance = embedding_distance[int(file_id)]
            one_sector_embedding_distance.append(one_embedding_distance)

        sector_embedding_distance[k] = one_sector_embedding_distance
    return sector_embedding_distance



def get_two_Kinds_distance(paired_sector_embedding_distance):
    euclidean_distance_avg_all = []
    cosine_similarity_avg_all = []
    loss_avg_all = []
    prob_age_all = []
    for k, v in paired_sector_embedding_distance.items():
        print("\n\nk:", k)
        if len(v)>0:
            euclidean_distance_sum = sum([item['euclidean_distance'] for item in v])
            cosine_similarity_sum = sum([item['cosine_similarity'] for item in v])
            # loss_sum = sum([item['loss'] for item in v])
            # prob_sum = sum([item['prob'] for item in v])
            euclidean_distance_avg = euclidean_distance_sum/len(v)
            cosine_similarity_avg  = cosine_similarity_sum/len(v)
            # loss_sum_avg = loss_sum/len(v)
            # prob_sum_avg = prob_sum/len(v)
            print("euclidean_distance_avg:", euclidean_distance_avg)
            print("cosine_similarity_avg:", cosine_similarity_avg)
        else:
            euclidean_distance_avg = 0
            cosine_similarity_avg = 0
        loss_sum_avg = 0
        prob_sum_avg = 0

        euclidean_distance_avg_all.append(euclidean_distance_avg)
        cosine_similarity_avg_all.append(cosine_similarity_avg)
        loss_avg_all.append(loss_sum_avg)
        prob_age_all.append(prob_sum_avg)

    return  euclidean_distance_avg_all, cosine_similarity_avg_all, loss_avg_all, prob_age_all

def show_paired_function_embedding_distance_trend():

    args.output_dir = args.output_dir
    distribution_paired_types = {}
    # ['paired', 'non-paired']


    two_kinds_data = read_token_distribution_data()
    paired_type = 'paired'
    data = two_kinds_data[paired_type]
    paired_sector_embedding_distance = get_sector_embedding_distance(data, paired_type)
    paired_type = 'non-paired'
    data = two_kinds_data[paired_type]
    non_paired_sector_embedding_distance = get_sector_embedding_distance(data, paired_type)
    both_paired_non_paired = {}
    for k, v in paired_sector_embedding_distance.items():
        both_paired_non_paired[k] = v + non_paired_sector_embedding_distance[k]


    paired_euclidean_distance_avg_all, paired_cosine_similarity_avg_all , paired_loss_sum_avg, paired_prob_sum_avg = get_two_Kinds_distance(paired_sector_embedding_distance)
    non_paired_euclidean_distance_avg_all, non_paired_cosine_similarity_avg_all, non_paired_loss_sum_avg, non_paired_prob_sum_avg = get_two_Kinds_distance(non_paired_sector_embedding_distance)
    both_paired_non_paired_euclidean_distance_avg_all, both_paired_non_paired_cosine_similarity_avg_all, both_paired_non_paired_loss_sum_avg, both_paired_non_paired_prob_sum_avg = get_two_Kinds_distance(both_paired_non_paired)

    print(paired_loss_sum_avg)
    print(paired_prob_sum_avg)
    show_two_list(paired_euclidean_distance_avg_all, paired_cosine_similarity_avg_all, paired_type='paired')
    # show_two_list(non_paired_euclidean_distance_avg_all, non_paired_cosine_similarity_avg_all, paired_type='non-paired')
    # show_two_list(both_paired_non_paired_euclidean_distance_avg_all, both_paired_non_paired_cosine_similarity_avg_all, paired_type='both')
    # show_one_list(paired_loss_sum_avg, paired_type='paired')
    # show_one_list(non_paired_loss_sum_avg, paired_type='non-paired')
    # show_one_list(both_paired_non_paired_loss_sum_avg,
    #               paired_type='both')
    #
    # show_one_list_v2(both_paired_non_paired_loss_sum_avg,
    #               paired_type='both')





    exit()
    print(distribution_paired_types)
    fontsize = 24
    fontsize_2 = 16
    fontsize_3 = 22
    bins = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bins = np.array(bins)
    bins = np.array(bins) + 1e-10
    bins[0] -= 1e-10  # 第一个bin边界不需要调整

    paired_data = distribution_paired_types['paired']['data']
    non_paired_data = distribution_paired_types['non-paired']['data']

    counts1, bins1, patches1 = plt.hist(paired_data, bins=bins, edgecolor='black', alpha=0.5, color='blue', label='paired')
    counts2, bins2, patches2 = plt.hist(non_paired_data, bins=bins, edgecolor='black', alpha=0.5, color='red', label='non-paired')

    # 计算比例
    total_counts1 = sum(counts1)
    total_counts2 = sum(counts2)

    # 计算每个bin的百分比
    percentages1 = counts1 / total_counts1
    percentages2 = counts2 / total_counts2


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
        print("file_name:", file_name)
        file_path = os.path.join(c_input, file_name)
        df = pd.read_csv(file_path)
        df = df[['index', 'func_after', 'func_before', 'processed_func', 'target']]
        df_v = df[df.target == 1]
        df_v_copy = copy.deepcopy(df_v)
        df_v_copy['func_before'] = df_v_copy['func_after']

        for k,v in paired_groups_distribution.items():
            print("\nk:", k)
            v = [int(i) for i in v]

            filtered_k_df = df_v_copy[df_v_copy['index'].isin(v)]
            print("len(filtered_k_df):", len(filtered_k_df))

            # filtered_k_df['index'] = filtered_k_df['index'] + len(df)
            # df_add_filtered_k_distribution = pd.concat([df, filtered_k_df])
            # df_add_filtered_k_distribution = df_add_filtered_k_distribution.sample(frac=1).reset_index(drop=True)
            #
            # output_dir = os.path.join(c_output, k)
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # output_file = os.path.join(output_dir, file_name)
            # df_add_filtered_k_distribution.to_csv(output_file)


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


    # before_predict = os.path.join(predict_dir, 'val_func_before_prediction.xlsx')
    # after_predict = os.path.join(predict_dir, 'val_func_after_prediction.xlsx')
    # before_predict = pd.read_excel(before_predict)
    # after_predict = pd.read_excel(after_predict)
    # before_predict_v = before_predict[before_predict.y_trues == 1]
    # after_predict_v = after_predict[after_predict.y_trues == 1]
    # pd.merge(before_predict_v, after_predict_v, how= 'left', left_on='all_inputs_ids', right_on='all_inputs_ids')


def visualize_test():
    import matplotlib.pyplot as plt
    import numpy as np

    # 示例数据
    data1 = [0.02, 0.15, 0.18, 0.23, 0.42, 0.57, 0.68, 0.72, 0.91, 0.99, 0.11, 0.45, 0.34, 0.21, 0.88]
    data2 = [0.05, 0.12, 0.17, 0.22, 0.31, 0.46, 0.58, 0.63, 0.75, 0.85, 0.95, 0.4, 0.33, 0.24, 0.76]

    # 自定义 bins 的边界
    bins = [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 绘制第一个数据集的直方图并获取柱子的属性
    counts1, bins1, patches1 = plt.hist(data1, bins=bins, edgecolor='black', alpha=0.5, color='skyblue', label='Data 1')
    # 绘制第二个数据集的直方图并获取柱子的属性
    counts2, bins2, patches2 = plt.hist(data2, bins=bins, edgecolor='black', alpha=0.5, color='orange', label='Data 2')

    # 在第一个数据集的每个柱子顶部添加数量标签
    for count, bin_edge in zip(counts1, bins1[:-1]):
        plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom', color='blue')

    # 在第二个数据集的每个柱子顶部添加数量标签
    for count, bin_edge in zip(counts2, bins2[:-1]):
        plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom', color='red')

    # 添加图例
    plt.legend()

    # 设置图表标题和标签
    plt.title('Histogram of Two Data Sets')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 显示图表
    plt.tight_layout()
    plt.show()

    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # 示例数据列表
    # data = [0.02, 0.15, 0.18, 0.23, 0.42, 0.57, 0.68, 0.72, 0.91, 0.99, 0.11, 0.45, 0.34, 0.21, 0.88]
    #
    # def calculate_distribution(data):
    #     bin_edges = [i * 0.1 for i in range(11)]
    #     distribution = {f"({bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]": 0 for i in range(len(bin_edges) - 1)}
    #     for value in data:
    #         for i in range(1, len(bin_edges)):
    #             if bin_edges[i - 1] < value <= bin_edges[i]:
    #                 key = f"({bin_edges[i - 1]:.1f}, {bin_edges[i]:.1f}]"
    #                 distribution[key] += 1
    #                 break
    #     return distribution
    #
    # distribution = calculate_distribution(data)
    #
    # categories = list(distribution.keys())
    # values = list(distribution.values())
    #
    # print("categories:", categories)
    #
    # plt.bar(categories, values, color='skyblue', edgecolor='black')
    #
    # # 设置图表标题和标签
    # plt.title('Distribution of similarity in paired functions')
    # plt.xlabel('Normalized Levenshtein Distance')
    # plt.ylabel('Number of instances')
    #
    # # 设置 x 轴标签为 categories 列表中的值
    # plt.xticks(rotation=45, ha='right', color='black')
    #
    # # 确保布局紧凑
    # plt.tight_layout()
    #
    # # 显示图表
    # plt.show()




def combine_paired_non_paired():
    two_kinds_data = {}
    for paired_type in ['non-paired', 'paired']:
        args.file_type = paired_type

        args.output_dir_parent = copy.deepcopy(args.output_dir)
        args.output_dir_file_type = os.path.join(args.output_dir_parent, args.file_type)

        data = []

        for subdir in os.listdir(args.output_dir_file_type):
            subdir_path = os.path.join(args.output_dir_file_type, subdir)
            if not os.path.isdir(subdir_path):
                continue
            i_cnt = 0
            for filename in os.listdir(subdir_path):
                if not str(filename[0]).isdigit():
                    continue
                i_cnt = i_cnt + 1

                file_path = os.path.join(subdir_path, filename)
                if i_cnt >= 2:
                    print("file_path:", file_path)
                # print("file_path:", file_path)
                levenshtein_distance_res = read_json(file_path)
                levenshtein_distance_normalized = levenshtein_distance_res['distance_normal']
                distance = levenshtein_distance_res['distance']
                func_before = levenshtein_distance_res['func_before']
                func_after = levenshtein_distance_res['func_after']
                data.append({"file_id": subdir,
                             "index":subdir,
                             "levenshtein_distance_normalized": levenshtein_distance_normalized,
                             "distance": distance,
                             "func_before":func_before,
                             "func_after":func_after})
        two_kinds_data[paired_type] = data

    paired_df = pd.DataFrame(two_kinds_data['paired'])
    paired_df['target'] = 1

    non_paired_df = pd.DataFrame(two_kinds_data['non-paired'])
    non_paired_df['target'] = 1

    paired_df.to_csv(os.path.join(args.output_dir , 'df_v_all_paired_index_correct.csv'))

    paired_df_distance_0 = paired_df[paired_df['levenshtein_distance_normalized'] == 0]
    paired_df_distance_0.to_csv(os.path.join(args.output_dir , 'paired_df_distance_0_index_correct.csv'))

    non_paired_df.to_csv(os.path.join(args.output_dir, 'df_nv_all_random_pair_index_correct.csv'))
    return two_kinds_data





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
    parser.add_argument("--block_size", default=513, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    args = parser.parse_args()
    args.sample = True
    args.down_resample = False


    args.file_type = 'paired'  # non-paired   paired

    args.dataset = 'MSR' # Diversevul  MSR
    if args.dataset == 'MSR':
        args.c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/'
        args.input_dir = args.c_root
        args.target_column = 'target'
        args.before_column = 'func_before'
        args.after_column = 'func_after'

    elif args.dataset == 'Diversevul':
        args.c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/Diversevul/ttv'
        args.input_dir = args.c_root

        args.target_column = 'before_target'
        args.before_column = 'before'
        args.after_column = 'after'


    if args.block_size == 512:
        output_subdir = 'generate_non_paired_random_pair'
    elif args.block_size > 512:
        output_subdir = 'generate_non_paired_random_pair_block_size_'+str(args.block_size)

    args.test_data_file = os.path.join(args.input_dir,'valid.csv')  # test.cs result:2/1055      valid.csv  result:2/1055,  train.csv  29/8736
    args.eval_data_file = os.path.join(args.input_dir, 'test.csv')  # contact test, valid, train   36/10900
    args.train_data_file = os.path.join(args.input_dir, 'train.csv')
    args.output_dir = os.path.join(args.input_dir, output_subdir, args.dataset)
    args.non_paired_paired_data_file = os.path.join(args.input_dir,
                                                    output_subdir, args.dataset,'df_nv_all_random_pair.csv')
    args.paired_data_file = os.path.join(args.input_dir,
                                         output_subdir, args.dataset,'df_v_all_paired.csv')

    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('../word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="../bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="../bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    # train_dataset = TextDataset(tokenizer, args, file_type='train')
    # eval_dataset = TextDatasetProduceSimilar(tokenizer, args, file_type=args.file_type )
    # visualize_test()
    # show_levenshtein_distance_distribution()
    # show_levenshtein_distance_distribution_one_dataset()
    # find_predict_res_for_before_and_after_token_are_same()

    # add_divided_after_groups_into_before(args.input_dir, args.output_dir)

    # combine_paired_non_paired()

    show_paired_function_embedding_distance_trend()

