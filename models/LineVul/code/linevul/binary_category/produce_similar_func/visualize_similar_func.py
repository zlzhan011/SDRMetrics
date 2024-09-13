import json
import os
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
# Re-importing necessary libraries after the reset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
def load_feature(cache_name, step_state, return_label=False, return_predict=False):
    np_file = np.load(cache_name, allow_pickle=True)
    feat_log, score_log, label_log, predict_log = np_file['arr_0'], np_file['arr_1'], np_file['arr_2'], np_file['arr_3']
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]
    # feature = feat_log[:, -770:-2]
    correct_log = [1 if label_log[i] == predict_log[i] else 0 for i in range(len(predict_log))]

    feature = feat_log[:, 769 * (step_state - 1):(step_state * 769)-1]
    if return_label:
        return feature, correct_log, label_log
    if return_predict:
        return feature, correct_log, predict_log
    return feature, correct_log


def read_front_back_correct_rate(json_path):
    with open(json_path, 'r') as f:
        front_back_correct_rate = json.load(f)
        front_results_correct_rate = front_back_correct_rate['front_results_correct_rate']
        back_results_correct_rate = front_back_correct_rate['back_results_correct_rate']
        inversions_front_results = front_back_correct_rate['inversions_front_results']
        inversions_back_results = front_back_correct_rate['inversions_back_results']
        front_results_correct_rate_cosine = front_back_correct_rate['front_results_correct_rate_cosine']
        back_results_correct_rate_cosine = front_back_correct_rate['back_results_correct_rate_cosine']
    return front_results_correct_rate, back_results_correct_rate, front_results_correct_rate_cosine, back_results_correct_rate_cosine, inversions_front_results, inversions_back_results


def visualize_similar_func(args, data, predict_log):


    # Assuming a shape of n*768 for the 2D array, using a smaller example for demonstration
    # n = 20  # Using a smaller number of points for illustration
    # data = np.random.rand(n, 768)  # Generating some random data

    # Using PCA to reduce the data to 2 dimensions for visualization
    # method_object = PCA(n_components=2)
    data_shape_0 = data.shape[0]

    perplexity_value = min(30, data_shape_0 - 1)
    method_object = TSNE(n_components=2, verbose=0, perplexity=perplexity_value, n_iter=300)

    reduced_data = method_object.fit_transform(data)

    # Plotting the scatter plot and annotating each point with its index
    plt.figure(figsize=(10, 8))
    shape_0 = reduced_data.shape[0]  - 1
    for i, point in enumerate(reduced_data):
        predict = predict_log[i]
        if predict == 1:
            color = 'red'
        else:
            color = 'black'
        if i  in [0, shape_0]:
            plt.scatter(point[0], point[1], marker='.', s=100, color=color)  # Using different markers for each point
            plt.text(point[0], point[1], str(i), fontsize=25, ha='right', va='top')
        else:
            plt.scatter(point[0], point[1], marker='.', s=100, color=color)  # Using different markers for each point
            plt.text(point[0], point[1], str(i), fontsize=9, ha='right', va='top')
    front_back_file_path = os.path.join(args.index_dir, args.index_dir_copy+"_front_back_correct_rate_knn.json")
    if os.path.isfile(front_back_file_path):
        front_func_correct_rate, back_func_correct_rate, front_results_correct_rate_cosine, back_results_correct_rate_cosine, inversions_front_results, inversions_back_results = read_front_back_correct_rate(front_back_file_path)
        print(front_func_correct_rate)
        # from inversions import inversions
        # front_results, back_results = inversions(feature)
        inversions_back_results = 1 - inversions_back_results
        text_note_1 = "front_func_correct_rate:" + str(round(front_func_correct_rate,2))
        text_note_2 = "back_func_correct_rate:" + str(round(back_func_correct_rate, 2))
        text_note_1_cosine = "front_func_correct_rate_cosine:" + str(round(front_results_correct_rate_cosine,2))
        text_note_2_cosine = "back_func_correct_rate_cosine:" + str(round(back_results_correct_rate_cosine, 2))
        text_note_3 = "inversions_front:" + str(round(inversions_front_results, 2))
        text_note_4 = "inversions_back:" + str(round(inversions_back_results, 2))
        text_note = text_note_1 +"\n\n\n"+ text_note_2
        text_note_cosine = text_note_1_cosine +"\n\n\n" + text_note_2_cosine
        text_note_inversions = text_note_3 +"\n\n\n" +  text_note_4
        x_max = plt.xlim()[1]  # 获取X轴的最大值
        y_max = plt.ylim()[1]  # 获取Y轴的最大值

        plt.text(0, y_max, text_note, fontsize=12, ha='center', va='top')
        plt.text(0, y_max * 0.9 , text_note_cosine, fontsize=12, ha='center', va='top')
        plt.text(0, y_max * 0.8, text_note_inversions, fontsize=12, ha='center', va='top')

    plt.title( str(args.index_dir_copy) + "   similar func reduce dimensions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig( os.path.join(args.index_dir,  args.index_dir_copy+"_produced_similar_func_distribution.png" ))
    plt.show()



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
    args.produce_similar_func = os.path.join(args.c_root, 'produce_similar_func')
    for index_dir in os.listdir(args.produce_similar_func):
            import copy

            index_dir_copy = copy.deepcopy(index_dir)
            args.index_dir_copy = index_dir_copy
            print("index_dir:", index_dir)
        # if index_dir in ['187532', 187532]:
            index_dir = os.path.join(args.produce_similar_func, index_dir)
            args.index_dir = index_dir
            for filename in os.listdir(index_dir):
                if '.npy.npz' in filename:
                    step_state = 1
                    filepath = os.path.join(index_dir, filename)
                    feature, correct_log, predict_log = load_feature(filepath, step_state, return_predict=True)
                    data_shape_0 = feature.shape[0]
                    if data_shape_0<2:
                        continue
                    visualize_similar_func(args, feature, predict_log)
                    time.sleep(3)
