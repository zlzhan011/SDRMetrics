import os

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def draw_auc(args, y_trues, logits):
    fpr, tpr, thresholds = roc_curve(y_trues, logits[:, 1])
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(args.MSR_processing_flag +"_" + args.input_column + '_roc_curve.png')
    plt.show()

def draw_auc_v2(args, y_trues, logits_all, y_trues_v2, logits_all_v2):
    fpr, tpr, thresholds = roc_curve(y_trues, logits_all)
    fpr2, tpr2, _ = roc_curve(y_trues_v2, logits_all_v2)

    roc_auc = auc(fpr, tpr)
    roc_auc_2 = auc(fpr2, tpr2)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve original (area = %0.2f)' % roc_auc)
    plt.plot(fpr2, tpr2, color='red', lw=2, label='ROC curve  add_after_to_before (area = %0.2f)' % roc_auc_2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if args.evaluate_only_paired_func:

        plt.title('ROC curve for paired')
        plt.legend(loc="lower right")
        plt.savefig('ROC_curve_for_paired.png')
        plt.show()
    else:
        plt.title('ROC curve for before func')
        plt.legend(loc="lower right")
        plt.savefig('ROC_curve_for_before_func.png')
        plt.show()




def read_label_logits(dir_name, args):
    y_trues_all = []
    logits_all = []
    if args.evaluate_only_paired_func:
        file_list = ['test_func_before_prediction.xlsx', 'test_func_after_prediction.xlsx']
    else:
        file_list = ['test_func_beforeevaluate_only_before_func_prediction.xlsx']
    for file in os.listdir(dir_name):
        if file in file_list:
            file_df = pd.read_excel(os.path.join(dir_name, file))
            y_trues = file_df['y_trues'].to_list()
            if 'after' in file:
                y_trues = [0] * len(y_trues)
            logits = file_df['logits'].to_list()
            y_trues_all = y_trues_all + y_trues
            logits_all = logits_all + logits

    return y_trues_all, logits_all

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
    args.c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/paired/LineVul/'
    args.c_root_original = os.path.join(args.c_root, 'original')
    args.c_root_add_after_to_before = os.path.join(args.c_root, 'add_after_to_before')
    args.evaluate_only_paired_func = False

    y_trues_all, logits_all = read_label_logits(args.c_root_original, args)
    y_trues_v2, logits_all_v2 = read_label_logits(args.c_root_add_after_to_before, args)


    draw_auc_v2(args, y_trues_all, logits_all, y_trues_v2, logits_all_v2)