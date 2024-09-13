import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
import json
import copy
import matplotlib.pyplot as plt
import numpy as np
def load_feature(cache_name, step_state, return_label=False, return_predict=False):
    np_file = np.load(cache_name, allow_pickle=True)
    feat_log, score_log, label_log, predict_log = np_file['arr_0'], np_file['arr_1'], np_file['arr_2'], np_file['arr_3']
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]
    # feature = feat_log[:, -770:-2]
    correct_log = [1 if label_log[i] == predict_log[i] else 0 for i in range(len(predict_log))]

    feature = feat_log[:, 769 * (step_state - 1):step_state * 768]
    if return_label:
        return feature, correct_log, label_log
    if return_predict:
        return feature, correct_log, predict_log
    return feature, correct_log
def find_k_closest_rows(data, k):
    # 计算与第一行的余弦相似度
    first_row_similarity = cosine_similarity([data[0]], data)[0]
    # 对余弦相似度进行排序，找出最近的k行的索引（除去第一行本身）
    closest_to_first = np.argsort(-first_row_similarity)[1:k + 1]

    # 计算与最后一行的余弦相似度
    last_row_similarity = cosine_similarity([data[-1]], data)[0]
    # 对余弦相似度进行排序，找出最近的k行的索引（除去最后一行本身）
    closest_to_last = np.argsort(-last_row_similarity)[1:k + 1]

    return closest_to_first, closest_to_last




def get_first_knn(data, k):

    # # 假设有一个形状为 n*768 的二维数组
    # n = 20  # 示例，总共有20行
    # data = np.random.rand(n, 768)  # 生成一个示例二维数组

    # 计算前10行的中心向量c
    c = np.mean(data[:k, :], axis=0)

    # 计算前10行中每一行与中心向量c的欧氏距离
    distances = np.linalg.norm(data[:k, :] - c, axis=1)

    # 找出这10行中距离中心向量c最远的距离farest_distance
    farest_distance = np.max(distances)

    # 计算其他行与中心向量c的距离
    other_distances = np.linalg.norm(data[k:, :] - c, axis=1)

    # 记录距离小于farest_distance的行的索引
    indexes = np.where(other_distances < farest_distance)[0]

    # 因为这些索引是基于第11行及之后的行计算的，所以需要加10得到原始数据中的行索引
    indexes += k
    print("first indexes:", indexes)
    k_list = [i for i in range(k)]
    k_list = k_list + indexes.tolist()
    print("first k_list:", k_list)
    return  indexes



def get_last_knn(data, k):

    # # 假设有一个形状为 n*768 的二维数组
    # n = 20  # 示例，总共有20行
    # data = np.random.rand(n, 768)  # 生成一个示例二维数组
    shape_0 = data.shape[0]
    k_list = [i for i in range(k, shape_0, 1)]
    # 计算前10行的中心向量c
    c = np.mean(data[-k:, :], axis=0)

    # 计算前10行中每一行与中心向量c的欧氏距离
    distances = np.linalg.norm(data[-k:, :] - c, axis=1)

    # 找出这10行中距离中心向量c最远的距离farest_distance
    farest_distance = np.max(distances)

    # 计算其他行与中心向量c的距离
    other_distances = np.linalg.norm(data[:-k, :] - c, axis=1)

    # 记录距离小于farest_distance的行的索引
    indexes = np.where(other_distances < farest_distance)[0]

    # 因为这些索引是基于第11行及之后的行计算的，所以需要加10得到原始数据中的行索引
    # indexes += k
    print("last indexes:", indexes)
    k_list = k_list + indexes.tolist()
    print("last k_list:", k_list)
    return  indexes



def draw_distribution(data, title=''):
    plt.hist(data, bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()



def calculate_relative_distance(data):
    first_array = data[0, :]
    last_array = data[-1, :]
    n = data.shape[0]
    # 计算前1/3数组与第一个、最后一个数组的距离
    front_third = data[1:int(n / 3), :]  # 排除第一个数组本身
    front_results = []

    any_element = data

    # 计算向量AB
    vector_AB = last_array - first_array

    # 计算向量CA和CB (对于数组中的每个any_element)
    vector_CA = first_array - any_element
    vector_CB = last_array - any_element

    # 计算CA与AB形成的角度的余弦值
    cosine_CA_AB = np.dot(vector_CA, vector_AB) / (np.linalg.norm(vector_CA, axis=1) * np.linalg.norm(vector_AB))
    cosine_CA_AB = cosine_CA_AB[1:int(n / 3)]
    cosine_CA_AB = np.where(cosine_CA_AB > 0, 1, 0.5)

    # 计算CB与BA（即AB的反方向，因此只需改变AB的符号）形成的角度的余弦值
    # 注意：BA向量即为-AB，但在计算余弦值时，向量方向的变化不影响结果，因此CB与AB的余弦值计算相同
    cosine_CB_BA = np.dot(vector_CB, -vector_AB) / (np.linalg.norm(vector_CB, axis=1) * np.linalg.norm(vector_AB))
    cosine_CB_BA =cosine_CB_BA[-int(n / 3):]
    cosine_CB_BA = np.where(cosine_CB_BA > 0, 1, 0.5)



    for array in front_third:
        dist_to_first = np.linalg.norm(array - first_array)
        dist_to_last = np.linalg.norm(array - last_array)
        front_results.append(1 if dist_to_first < dist_to_last else 0)

    # 计算后1/3数组与第一个、最后一个数组的距离
    back_third = data[-int(n / 3):, :]
    back_results = []

    for array in back_third:
        dist_to_first = np.linalg.norm(array - first_array)
        dist_to_last = np.linalg.norm(array - last_array)
        back_results.append(0 if dist_to_first < dist_to_last else 1)

    front_results_cosin = np.array(cosine_CA_AB) * np.array(front_results)
    distances_last_cosin = np.array(cosine_CB_BA) * np.array(back_results)

    return front_results, back_results, front_results_cosin, distances_last_cosin


def design_metrics(data):

    # 获取第一个和最后一个一维数组
    a = data[0, :]
    b = data[-1, :]

    def distance_ratio(a, b, c):
        return np.sqrt(np.sum((c - a) ** 2)) / np.sqrt(np.sum((c - b) ** 2))

    def distance_to_circle(a, b, c):
        m = (a + b) / 2
        return np.sqrt(np.sum((c - m) ** 2))

    def f(a, b, c):
        r = distance_ratio(a, b, c)
        d = distance_to_circle(a, b, c)
        return r * d

    # # Example usage
    # a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # b = np.array([9, 10, 11, 12, 13, 14, 15, 16])
    # c = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    #
    # metrics_value = f(a, b, c)

    n = data.shape[0]
    # 计算前1/3数组与第一个、最后一个数组的距离
    front_third = data[1:int(n / 3), :]  # 排除第一个数组本身
    front_results = []

    for array in front_third:
        metrics_value = f(a, b, array)
        front_results.append(metrics_value)

    back_third = data[-int(n / 3):, :]
    back_results = []

    for array in back_third:
        metrics_value = f(a, b, array)
        back_results.append(metrics_value)

    print("front_results-----:", front_results)
    print("back_results----:", back_results)
    exit()
    return n

from inversions import inversions


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
    args.produce_similar_func_v2 = os.path.join(args.c_root, 'produce_similar_func_v2')
    max_distance_first_all = []
    max_distance_last_all = []
    means_distance_all = []
    for index_dir in os.listdir(args.produce_similar_func):
            print("index_dir:", index_dir)

        # if index_dir in ['187532', 187532]:
            args.index_dir_copy = copy.deepcopy(index_dir)
            index_dir = os.path.join(args.produce_similar_func, index_dir)
            index_dir_v2 = os.path.join(args.produce_similar_func_v2, args.index_dir_copy)
            if not os.path.exists(index_dir_v2):
                os.makedirs(index_dir_v2)
            for filename in os.listdir(index_dir):
                if '.npy.npz' in filename:
                    step_state = 1
                    filepath = os.path.join(index_dir, filename)
                    feature, correct_log, predict_log = load_feature(filepath, step_state, return_predict=True)
                    data_shape_0 = feature.shape[0]
                    if data_shape_0<=2:
                        continue
                    k = int(data_shape_0/3)
                    if k <=1:
                        continue
                    print("k:", k)
                    if '184102' not in filename:
                        continue
                    # closest_to_first, closest_to_last = find_k_closest_rows(feature, k)
                    # indexes_first = get_first_knn(feature, k)
                    # indexes_last = get_last_knn(feature, k)
                    front_results, back_results,front_results_cosine, distances_last_cosine = calculate_relative_distance(feature)
                    front_results_correct_rate_cosine = sum(front_results_cosine)/len(front_results_cosine)
                    back_results_correct_rate_cosine =  sum(distances_last_cosine) / len(distances_last_cosine)
                    # front_results, back_results = design_metrics(feature)
                    inversions_front_results, inversions_back_results = inversions(feature)
                    print("inversions_front_results:", inversions_front_results)
                    print("inversions_back_results:", inversions_back_results)
                    closest_to_first = feature[:k,:]
                    closest_to_last = feature[-k:,:]
                    closest_to_first_center = np.mean(closest_to_first, axis=0)
                    closest_to_last_center = np.mean(closest_to_last, axis=0)

                    # 计算所有行与中心向量c的欧氏距离
                    distances_first = np.linalg.norm(closest_to_first - closest_to_first_center, axis=1)
                    # 找出与中心向量c的最大距离
                    max_distance_first = np.max(distances_first)
                    # print("max_distance_first:", max_distance_first)
                    max_distance_first_all.append(max_distance_first)

                    # 计算所有行与中心向量c的欧氏距离
                    distances_fast = np.linalg.norm(closest_to_last - closest_to_last_center, axis=1)
                    # 找出与中心向量c的最大距离
                    max_distance_last = np.max(distances_fast)
                    # print("max_distance_last:", max_distance_last)
                    max_distance_last_all.append(max_distance_last)

                    means_distance = np.linalg.norm(closest_to_first_center - closest_to_last_center)
                    # print("means_distance:", means_distance)
                    means_distance_all.append(means_distance)

                    # output_content = {"feature": feature.tolist(),
                    #                   "closest_to_first_center": closest_to_first_center.tolist(),
                    #                   "closest_to_last_center":closest_to_last_center.tolist(),
                    #                   "front_results":front_results,
                    #                   "back_results":back_results,
                    #                   "front_results_correct_rate":sum(front_results)/len(front_results),
                    #                   "back_results_correct_rate":sum(back_results)/len(back_results)}

                    output_content = {"feature": feature.tolist(),
                                      "closest_to_first_center": closest_to_first_center.tolist(),
                                      "closest_to_last_center": closest_to_last_center.tolist(),
                                      "front_results": front_results,
                                      "back_results": back_results,
                                      "front_results_correct_rate": sum(front_results)/len(front_results),
                                      "back_results_correct_rate": sum(back_results)/len(back_results),
                                      "inversions_front_results":inversions_front_results,
                                      "inversions_back_results":inversions_back_results,
                                      "front_results_correct_rate_cosine": front_results_correct_rate_cosine,
                                      "back_results_correct_rate_cosine": back_results_correct_rate_cosine}

                    output_path = os.path.join(index_dir, args.index_dir_copy+"_front_back_correct_rate_knn.json")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(output_content, f)


    # print(max_distance_first_all)
    # print(len(max_distance_first_all))
    # print(len(max_distance_last_all))
    # print(len(means_distance_all))

    draw_distribution(max_distance_first_all, title='diameter_close_to_before')
    draw_distribution(max_distance_last_all, title='diameter_close_to_after')
    draw_distribution(means_distance_all, title='distance_between_center_of_before_after')
                    # feature = np.vstack((feature, closest_to_first_center))
                    # feature = np.vstack((feature, closest_to_last_center))
                    # print(closest_to_first)
                    # print(closest_to_last)
                    # output_content = {"feature": feature.tolist(), "closest_to_first_center": closest_to_first_center.tolist(), "closest_to_last_center":closest_to_last_center.tolist()}
                    # output_path = os.path.join(index_dir_v2, filename.replace(".npy.npz", ".json"))
                    # with open(output_path, 'w', encoding='utf-8') as f:
                    #     json.dump(output_content, f)

