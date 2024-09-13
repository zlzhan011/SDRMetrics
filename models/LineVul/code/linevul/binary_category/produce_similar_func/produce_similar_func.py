
import argparse
import os
import pandas as pd
from cross_sections.paired.Levenshtein_distance import levenshtein_distance_with_intermediate_steps_list
from tqdm import tqdm
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import numpy as np
import torch
import json

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



def write_json(filename, two_dim_list):
    # 使用 'with' 语句确保文件正确关闭
    with open(filename, 'w') as file:
        # 使用 json.dump() 方法将列表写入文件
        # ensure_ascii=False 允许写入非ASCII字符
        # indent 参数用于美化输出，使其更易于阅读
        json.dump(two_dim_list, file, ensure_ascii=False, indent=4)



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
            df = pd.read_csv(file_path, index_col=0)
            # df = re_shuffle_df(args, df)
        else:
            raise NotImplementedError(file_path)

        # if args.sample:
        #     df = df.sample(100)

        if "processed_func" in df.columns:
            func_key = "processed_func"
        elif "func" in df.columns:
            func_key = "func"

        func_before_with_target_v = df[df['target'] == 1]['func_before'].tolist()
        func_after_with_target_nv = df[df['target'] == 1]['func_after'].tolist()

        indices_v = df[df['target'] == 1].index.tolist()
        indices_nv = df[df['target'] == 1].index.tolist()


        # func_key = 'func_before'  # func_after  func_before
        # funcs_before = df[func_key].tolist()
        # labels = df["target"].astype(int).tolist()
        # print("labels. value:", set(labels))
        # print("labels.shape:", df.shape)
        # args.input_column = func_key
        # indices = df.index.astype(int).tolist()
        #
        # funcs_after = df['func_after'].tolist()


        for i in tqdm(range(len(indices_v)), desc="load dataset"):
            source_tokens_before, index, source_ids_before, label = convert_examples_to_features(func_before_with_target_v[i], indices_v[i], tokenizer, args, indices_v[i])
            source_tokens_after, index, source_ids_after, label = convert_examples_to_features(func_after_with_target_nv[i], indices_v[i], tokenizer, args, indices_v[i])

            distance, intermediate_results = levenshtein_distance_with_intermediate_steps_list(source_ids_before, source_ids_after)
            print("distance:", distance)
            print(intermediate_results[0])
            args.output_dir_index = os.path.join(args.output_dir, str(index))
            if not os.path.exists(args.output_dir_index):
                os.makedirs(args.output_dir_index)
            filename = os.path.join(args.output_dir_index, str(index)+"_"+str(distance)+".json" )
            write_json(filename, intermediate_results)



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




# def produce_similar_func(args, file_path):
#     input_df = pd.read_csv(file_path)
#     func_before_with_target_v = input_df[input_df['target'] == 1]['func_before'].tolist()
#     func_after_with_target_nv = input_df[input_df['target'] == 1]['func_after'].tolist()
#
#     indices_v = input_df[input_df['target'] == 1].index.tolist()
#     indices_nv = input_df[input_df['target'] == 1].index.tolist()
#     for i in tqdm(range(len(indices_v))):
#         index = indices_v[i]
#         v = func_before_with_target_v[i]
#         nv = func_after_with_target_nv[i]
#         distance, intermediate_results = levenshtein_distance_with_intermediate_steps_list(v, nv)
#         print("\ndistance:", distance)
#         index_dir = os.path.join(args.output_dir, str(index))
#         if not os.path.exists(index_dir):
#             os.makedirs(index_dir)
#         with open(os.path.join(index_dir,  str(index)+"_"+str(distance)+"_"+".csv"), 'w', encoding='utf-8') as f:
#             ii = 0
#             for str_t in intermediate_results:
#                 # print("ii:", ii)
#                 ii = ii +1
#                 f.write(str_t + '\n')



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
    args.c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/'
    args.sample = True
    args.down_resample = False
    args.input_dir = args.c_root
    args.output_dir = os.path.join(args.input_dir, 'produce_similar_func')
    args.test_data_file = os.path.join(args.input_dir, 'valid.csv')    # test.cs result:2/1055      valid.csv  result:2/1055,  train.csv  29/8736
    args.eval_data_file = os.path.join(args.input_dir, 'test.csv')      # contact test, valid, train   36/10900
    args.train_data_file = os.path.join(args.input_dir, 'train.csv')


    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('../word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="../bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="../bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    # train_dataset = TextDataset(tokenizer, args, file_type='train')
    eval_dataset = TextDatasetProduceSimilar(tokenizer, args, file_type='eval')

