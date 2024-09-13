
import argparse
import os
import numpy as np
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def get_index_file_name_map(args):
    args.input_dir = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/binary_category/check_result_of_after_before/chen_files'
    file_path = os.path.join(args.input_dir, 'chen_function_content.csv')
    df = pd.read_csv(file_path)
    index_file_name_map = df.set_index('index')['file_name'].to_dict()
    return index_file_name_map

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


def get_cosine_distance(array1, array2):
    cosine_similarity = np.dot(array1, array2.T) / (np.linalg.norm(array1) * np.linalg.norm(array2))
    return cosine_similarity


def visualize(args, before, after, completed):
    # 步骤1: 拼接数组
    data = np.stack((before, after, completed))

    # 步骤2: t-SNE 降维
    tsne = TSNE(n_components=2, random_state=0, perplexity=2)
    data_2d = tsne.fit_transform(data)

    # 步骤3: 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1])
    plt.title( args.one_file_id + '_t-SNE visualization of before after completed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(args.embedding_output, 'visual', args.mode, args.one_file_id+".png"))
    plt.show()


def visualize_v2(args, all_vector):
    # 步骤1: 拼接数组
    data = all_vector
    # args.all_file_id_order = data[:, 0]
    # 步骤2: t-SNE 降维
    tsne = TSNE(n_components=2, random_state=0, perplexity=6)
    data_2d = tsne.fit_transform(data)

    # 步骤3: 可视化
    plt.figure(figsize=(8, 6))

    colors = ['r', 'b','g', 'y',  'm']  # 红色、绿色、蓝色
    markers = ['o', '^', 's', '*', 'd', '+']  # 圆形、三角形、正方形
    labels = args.all_file_id_order  # 文字标签
    label = ['before', 'after', 'completed']
    # for i in range(data.shape[0]):
    # plt.scatter(data_2d[:, 0], data_2d[:, 1])
    for i, (x, y) in enumerate(data_2d):
        m_l_index = i // 3
        # m_l_index = m_l_index - 1
        print("c_m_index:", m_l_index)
        l_index = i % 3
        label = l_index

        plt.scatter(x, y, color=colors[l_index], marker=markers[m_l_index], s=100)  # s控制点的大小
        plt.text(x, y, labels[m_l_index], fontsize=9, ha='right', va='bottom')  # 文字标签


    plt.title( args.mode + ' model t-SNE visualization of before-after-completed')

    x_max = plt.xlim()[1]  # 获取X轴的最大值
    y_max = plt.ylim()[1]  # 获取Y轴的最大值
    plt.text(0, y_max * 0.9, 'Red represent before', fontsize=12, ha='center', va='bottom')
    plt.text(0, y_max * 0.8, 'Black represent after', fontsize=12, ha='center', va='bottom')
    plt.text(0, y_max * 0.7, 'Green represent completed', fontsize=12, ha='center', va='bottom')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(args.embedding_output, 'visual', args.mode, args.mode+".png"))
    plt.show()


def measure_distance(args):
    files_id = [file_name.split("_")[0]  for file_name in os.listdir(args.embedding_output_mode)]
    files_id = list(set(files_id))
    file_feature = []
    i = 0
    for index, file  in args.index_file_name_map.items():
        file = file + ".npz"
        index_file_name = args.index_file_name_map[index]
        print("file:", file)
        print("index_file_name:", index_file_name)
        # exit()
        file_id = file.split('_')[0]
        before_after_completed = ""
        if "completed" in file:
            before_after_completed = 'completed'
        else:
            if 'after' in file:
                before_after_completed = 'after'
            elif 'before' in file:
                before_after_completed = 'before'
            else:
                print("please check the file name")
        # file_before = str(file_id) + "_CVE-2018-20815-before.c.npz"
        # file_before = str(file_id) + "_CVE-2018-20815-after.c.npz"
        file_path = os.path.join(args.embedding_output_mode, file)
        step_state = 1
        return_predict = True
        feature, correct_log, predict_log = load_feature(file_path, step_state, return_label=False, return_predict=return_predict)
        print("file:", file)
        # print(feature)
        print("feature.shape:", feature.shape)
        feature = feature[index,:]
        print("feature.shape_2:", feature.shape)
        print(feature)
        file_feature_dict = {"file":file, "feature":feature, "file_id":file_id, "before_after_completed":before_after_completed}
        file_feature.append(file_feature_dict)
        i = i + 1

    all_file_three_res = {}
    for one_file_id in files_id:
        one_file_three_res = {}
        for item in file_feature:
            if one_file_id == item['file_id']:
                before_after_completed = item['before_after_completed']
                one_file_three_res[before_after_completed] =item
        # print("one_file_three_res:", one_file_three_res)
        all_file_three_res[one_file_id] = one_file_three_res

    print(all_file_three_res)
    distances = []
    all_vector = []
    after_after_completed_vector = []
    args.all_file_id_order = []
    for one_file_id, one_file_three_res in all_file_three_res.items():
        args.one_file_id = one_file_id
        args.all_file_id_order.append(one_file_id)
        print("\n\n\n")
        print("one_file_id:", one_file_id)
        if one_file_id in [178176, '178176']:
            continue
        after = one_file_three_res['after']['feature']
        before = one_file_three_res['before']['feature']
        completed = one_file_three_res['completed']['feature']
        print("after_shape:", after.shape)
        if args.distance_method == 'euclidean':
            after_before = np.linalg.norm(after - before)
            before_completed = np.linalg.norm(before - completed)
            after_completed = np.linalg.norm(after - completed)
        elif args.distance_method == 'cosine':

            after_before = get_cosine_distance(after, before)
            before_completed = get_cosine_distance(before, completed)
            after_completed = get_cosine_distance(after,  completed)
        print("after_before:", after_before)
        print("before_completed:", before_completed)
        print("after_completed:", after_completed)
        distances.append({"file_id": one_file_id,
                          "distance_method": args.distance_method,
                          'model':"LineVul",
                          "before":before,
                          "after": after,
                          "completed":completed,
                          "model_version": args.mode,
                          "after_before":after_before,
                          "before_completed":before_completed,
                          "after_completed":after_completed,
                          })
        after_after_completed_vector = np.stack((before, after, completed))
        if all_vector == []:
            all_vector = after_after_completed_vector
        else:
            all_vector = np.concatenate((all_vector, after_after_completed_vector))
        # visualize(args, before, after, completed)
    visualize_v2(args, all_vector)
    distances_df = pd.DataFrame(distances)
    distances_df.to_csv(os.path.join(args.embedding_output, args.mode + "_" + args.distance_method + ".csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('chen_dir', default='D:\Research\paired_function\Dr_Chen_task\paired-completed')
    args = parser.parse_args()
    args.chen_dir_root = r'/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/binary_category/check_result_of_after_before/chen_files/'
    args.chen_dir = os.path.join(args.chen_dir_root, 'paired-completed')
    args.embedding_output =os.path.join(args.chen_dir_root, 'embedding_output')
    args.mode = 'add_after_to_before' # original  add_after_to_before
    args.embedding_output_mode = os.path.join(args.embedding_output, args.mode)

    index_file_name_map = get_index_file_name_map(args)
    args.index_file_name_map = index_file_name_map
    args.distance_method = 'cosine' # euclidean  cosine

    measure_distance(args)