import copy
import os
import pandas as pd
import os
import time
import torch

import numpy as np
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from visual_zoom_in import draw_one_group_add_background_similar_func_zoom_in_directly_v2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from calculate_llc import calculate_llc_core
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
import argparse
# Set display options to print all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Agg', 'Qt5Agg' 等
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def split_group_ids():
    c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/'
    output_dir = os.path.join(c_root, 'visual')
    raw_preds_synthetic = pd.read_csv(os.path.join(c_root, 'synthetic_testv3.csv'))
    group_ids = raw_preds_synthetic['group_id']
    for id in group_ids:
        df_id = raw_preds_synthetic[raw_preds_synthetic.group_id == id]
        targets = list(set(df_id['target'].tolist()))
        for target_no in targets:
            group_target_df = df_id[df_id.target==target_no]
            group_target_df.to_csv(os.path.join(output_dir, "raw_preds_synthetic_" + str(id) +"_"+str(target_no)+ ".csv"))



def split_target():
    c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/'
    output_dir = os.path.join(c_root, 'visual')
    raw_preds_synthetic = pd.read_csv(os.path.join(c_root, 'devign_test.csv'))
    # raw_preds_synthetic = raw_preds_synthetic.head(2000)
    targets = raw_preds_synthetic['target']
    targets = list(set(targets.tolist()))
    for id in targets:
        df_id = raw_preds_synthetic[raw_preds_synthetic.target == id]
        df_id.to_csv(os.path.join(output_dir, "devign_test_target_" + str(id) + ".csv"))


def split_target_cross_project():
    c_root = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/cross_sections/code/data/subsets/cross_project'
    for k_fold in range(6):
        dir_type_list = ['fold_'+str(k_fold)+'_holdout',
                    'fold_'+str(k_fold)+'_dataset']
        for dir_type in dir_type_list:

            output_dir = os.path.join(c_root, dir_type)
            if '_dataset' in dir_type:
                raw_preds_synthetic = pd.read_csv(os.path.join(output_dir, 'test.csv'))
            if '_holdout' in dir_type:
                raw_preds_synthetic = pd.read_csv(os.path.join(output_dir, 'holdout.csv'))

            # raw_preds_synthetic = raw_preds_synthetic.head(2000)
            targets = raw_preds_synthetic['target']
            targets = list(set(targets.tolist()))
            for id in targets:
                df_id = raw_preds_synthetic[raw_preds_synthetic.target == id]
                df_id.to_csv(os.path.join(output_dir, dir_type +'_target_'+ str(id) + ".csv"))


def load_feature(cache_name, step_state):
    print("cache_name", cache_name)
    np_file = np.load(cache_name, allow_pickle=True)
    feat_log, score_log, label_log, prediction_log = np_file['arr_0'], np_file['arr_1'], np_file['arr_2'], np_file['arr_3']
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]
    # feature = feat_log[:, -770:-2]
    feature = feat_log[:, 769 * (step_state - 1):(step_state * 769) -1]
    correct_log = [1 if label_log[i] == prediction_log[i] else 0 for i in range(len(prediction_log))]
    return feature, correct_log






def visualize():
    input_dir = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/cache_visual_test/'
    input_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_cache/'

    # target_file_list = ['multi_target_1_last_second_layer.npy.npz', 'multi_target_0_last_second_layer.npy.npz']
    target_file_list = ['multi_msr_test_target_1_last_second_layer.npy.npz', 'multi_msr_test_target_0_last_second_layer.npy.npz']
    # multi_target_1_last_second_layer.npy.npz must at the first, multi_target_0_last_second_layer.npy.npz must at the second
    # target_file is only the testing dataset file
    # target_file = [file for file in os.listdir(input_dir) if 'target' in file and file in target_file_list]
    target_file = target_file_list
    print("target_file:", target_file)
    file = target_file[0]

    # def single_function(layer_number):
    for layer_number in [15,  12,   7, 1, ]:
        print("layer_number:", layer_number)
        target_feature = []
        for file in target_file:
            cache_name = input_dir + file
            feature, file_correct_log = load_feature(cache_name, step_state=layer_number)
            target_feature.append(feature)

        file_list = os.listdir(input_dir)
        synthetic_tsne_1_list = []
        synthetic_tsne_0_list = []
        from tqdm import tqdm

        for file in tqdm(file_list):
            # if 'target' not in file: # the synthetic dataset
            if 'preds_synthetic' in file: # the synthetic dataset
                if '_1_' in file:
                    if "_1_1_" in file:
                        file_0 = file.replace("_1_1_", "_1_0_")
                    else:
                        file_0 = file.replace("_1_", "_0_")
                    group_id = file.replace("multi_raw_preds_synthetic_","")
                    group_id = group_id.replace("_1_last_second_layer.npy.npz", "")
                    if file_0 in file_list:
                        file_list.remove(file_0)
                        cache_name_1 = input_dir + file
                        cache_name_0 = input_dir + file_0
                        feature_1, correct_log_1 = load_feature(cache_name_1, step_state=layer_number)
                        feature_0, correct_log_0 = load_feature(cache_name_0, step_state=layer_number)
                        # combine the same group feature
                        feature = np.concatenate([feature_1, feature_0])
                        if feature.shape[0] >= 2:
                            print(feature.shape)
                            target_feature.append(feature)
                            X3_tsne_1, X3_tsne_0, vulnerable_back_ground_tsne, non_vulnerable_back_ground_tsne = visual_three_vector_v2(target_feature, feature_1)
                            p_number = X3_tsne_1.shape[0]
                            n_number = X3_tsne_0.shape[0]
                            draw_one_group(X3_tsne_1, X3_tsne_0, group_id, p_number, n_number,
                                           layer_number=layer_number,
                                           output_dir='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_in_group',
                                           correct_log_1=correct_log_1,
                                           correct_log_0=correct_log_0)

                            draw_one_group_add_background(X3_tsne_1, X3_tsne_0, group_id, p_number, n_number,
                                           layer_number=layer_number,
                                           output_dir='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_in_group',
                                           correct_log_1=correct_log_1,
                                           correct_log_0=correct_log_0,
                                           vulnerable_back_ground_tsne=vulnerable_back_ground_tsne,
                                           non_vulnerable_back_ground_tsne=non_vulnerable_back_ground_tsne)
                            synthetic_tsne_1_list.append(X3_tsne_1)
                            synthetic_tsne_0_list.append(X3_tsne_0)
                            target_feature.pop()

        # draw_picture(synthetic_tsne_1_list, synthetic_tsne_0_list,
        #              layer_number=layer_number,
        #              output_dir='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_in_group'
        # )




def visualize_add_similar_func():
    input_dir = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/cache_visual_test/'
    input_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_cache/'

    # target_file_list = ['multi_target_1_last_second_layer.npy.npz', 'multi_target_0_last_second_layer.npy.npz']
    target_file_list = ['multi_msr_test_target_1_last_second_layer.npy.npz', 'multi_msr_test_target_0_last_second_layer.npy.npz']
    # multi_target_1_last_second_layer.npy.npz must at the first, multi_target_0_last_second_layer.npy.npz must at the second
    # target_file is only the testing dataset file
    # target_file = [file for file in os.listdir(input_dir) if 'target' in file and file in target_file_list]
    target_file = target_file_list
    print("target_file:", target_file)
    file = target_file[0]

    # def single_function(layer_number):
    layer_numbers = [15,  12,   7, 1, ]
    layer_numbers = [15]
    for layer_number in layer_numbers:
        print("layer_number:", layer_number)
        target_feature = []
        for file in target_file:
            cache_name = input_dir + file
            # 这是加载 Vulnerable 和non-Vulnerable背景的部分
            feature, file_correct_log = load_feature(cache_name, step_state=layer_number)
            target_feature.append(feature)

        file_list = os.listdir(input_dir)
        synthetic_tsne_1_list = {}
        synthetic_tsne_0_list = {}
        correct_log_1_list = {}
        correct_log_0_list = {}
        from tqdm import tqdm

        similar_func_features_all_map, args = read_similar_func_features()
        for k, v in similar_func_features_all_map.items():

            args.index_dir_copy = k
            feature_similar_func = v['feature']
            correct_log = v['correct_log']
            predict_log = v['predict_log']

            half_produced_instance_num = int(feature_similar_func.shape[0] / 2)

            feature_1 = feature_similar_func[:half_produced_instance_num,:]
            feature_0 = feature_similar_func[half_produced_instance_num:,:]
            group_id = k
            correct_log_1 = correct_log[:half_produced_instance_num]
            correct_log_0 = correct_log[half_produced_instance_num:]
            # combine the same group feature
            feature_similar_func = np.concatenate([feature_1, feature_0])
            if feature_similar_func.shape[0] >= 2:
                print(feature_similar_func.shape)
                target_feature.append(feature_similar_func)
                X3_tsne_1, X3_tsne_0, vulnerable_back_ground_tsne, non_vulnerable_back_ground_tsne = visual_three_vector_v2(
                    target_feature, feature_1)
                p_number = X3_tsne_1.shape[0]
                n_number = X3_tsne_0.shape[0]

                output_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_produced_similar_func'
                args.output_dir = output_dir
                X3_tsne =  np.concatenate([X3_tsne_1, X3_tsne_0])
                draw_one_group_similar_func(args, X3_tsne , predict_log)




                if k in ['179281', 179281]:
                    X3_tsne_0_179639 = synthetic_tsne_0_list['179639']
                    X3_tsne_1_179639 = synthetic_tsne_1_list['179639']
                    correct_log_1_179639 = correct_log_1_list['179639']
                    correct_log_0_179639 = correct_log_0_list['179639']
                    X3_tsne_1 = np.concatenate([X3_tsne_1 , X3_tsne_1_179639])
                    X3_tsne_0 = np.concatenate([X3_tsne_0,  X3_tsne_0_179639])
                    correct_log_1 = correct_log_1 + correct_log_1_179639
                    correct_log_0 = correct_log_0 + correct_log_0_179639

                    draw_one_group_add_background_similar_func_zoom_in(X3_tsne_1, X3_tsne_0, group_id, p_number,
                                                                       n_number,
                                                                       layer_number=layer_number,
                                                                       output_dir='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_produced_similar_func',
                                                                       correct_log_1=correct_log_1,
                                                                       correct_log_0=correct_log_0,
                                                                       vulnerable_back_ground_tsne=vulnerable_back_ground_tsne,
                                                                       non_vulnerable_back_ground_tsne=non_vulnerable_back_ground_tsne,
                                                                       k=k)
                    draw_one_group_add_background_similar_func_zoom_in_2(X3_tsne_1, X3_tsne_0, group_id, p_number,
                                                                         n_number,
                                                                         layer_number=layer_number,
                                                                         output_dir='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_produced_similar_func',
                                                                         correct_log_1=correct_log_1,
                                                                         correct_log_0=correct_log_0,
                                                                         vulnerable_back_ground_tsne=vulnerable_back_ground_tsne,
                                                                         non_vulnerable_back_ground_tsne=non_vulnerable_back_ground_tsne,
                                                                         k=k)





                draw_one_group_add_background_similar_func(X3_tsne_1, X3_tsne_0, group_id, p_number, n_number,
                                              layer_number=layer_number,
                                              output_dir='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_produced_similar_func',
                                              correct_log_1=correct_log_1,
                                              correct_log_0=correct_log_0,
                                              vulnerable_back_ground_tsne=vulnerable_back_ground_tsne,
                                              non_vulnerable_back_ground_tsne=non_vulnerable_back_ground_tsne,
                                                           k=k)

                draw_one_group_add_background_similar_func_zoom_in_directly_v2(X3_tsne_1, X3_tsne_0, group_id, p_number, n_number,
                                                           layer_number=layer_number,
                                                           output_dir='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_produced_similar_func',
                                                           correct_log_1=correct_log_1,
                                                           correct_log_0=correct_log_0,
                                                           vulnerable_back_ground_tsne=vulnerable_back_ground_tsne,
                                                           non_vulnerable_back_ground_tsne=non_vulnerable_back_ground_tsne,
                                                           k=k)





                synthetic_tsne_1_list[k] = X3_tsne_1
                synthetic_tsne_0_list[k] = X3_tsne_0
                correct_log_1_list[k] = correct_log_1
                correct_log_0_list[k] = correct_log_0
                target_feature.pop()



def visual_three_vector(target_feature, feature_1, group_id):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # # 假设 X1, X2, X3 是你的三个 (n*768) 矩阵
    # X1 = np.random.rand(1000, 768)  # 这里我们创建了一个随机矩阵作为示例
    # X2 = np.random.rand(1000, 768)  # 第二个矩阵
    # X3 = np.random.rand(1000, 768)  # 第三个矩阵
    X1, X2, X3 = target_feature[0], target_feature[1], target_feature[-1]
    method = 'tsne_combination'
    if method == 'tsne':
        X1 = X1[:target_feature[-1].shape[0], :]
        X2 = X2[:target_feature[-1].shape[0], :]
        # 使用 TSNE 进行降维
        method_object = TSNE(n_components=2, verbose=1, perplexity=1, n_iter=300)
        X1_tsne = method_object.fit_transform(X1)
        X2_tsne = method_object.fit_transform(X2)
        X3_tsne = method_object.fit_transform(X3)
        X3_tsne_1 = X3_tsne[:feature_1.shape[0], :]
        X3_tsne_0 = X3_tsne[feature_1.shape[0]:, :]

    elif method == 'pca':
        method_object = PCA(n_components=2)
        X1 = X1[:target_feature[-1].shape[0], :]
        X2 = X2[:target_feature[-1].shape[0], :]

        X1_tsne = method_object.fit_transform(X1)
        X2_tsne = method_object.fit_transform(X2)
        X3_tsne = method_object.fit_transform(X3)
        X3_tsne_1 = X3_tsne[:feature_1.shape[0], :]
        X3_tsne_0 = X3_tsne[feature_1.shape[0]:, :]
    elif method == 'tsne_combination':
        X = np.concatenate([X1, X2, X3])
        method_object = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
        # method_object = PCA(n_components=2)
        # method_object = LDA(n_components=2)
        X_tsne = method_object.fit_transform(X)
        X1_tsne = X_tsne[:X1.shape[0], :]
        X1_X2 = (X1.shape[0]+X2.shape[0])
        X2_tsne = X_tsne[X1.shape[0]: X1_X2, :]
        X3_tsne = X_tsne[X1_X2:, :]
        print(X3_tsne.shape)
        X3_tsne_1 = X3_tsne[:feature_1.shape[0], :]
        X3_tsne_0 = X3_tsne[feature_1.shape[0]:, :]
        # X1_tsne = method_object.fit_transform(X1)
        # X2_tsne = method_object.fit_transform(X2)
        # X3_tsne = method_object.fit_transform(X3)

    # 设定不同的颜色和标记
    colors = ['orange', 'green', 'blue', 'red']
    markers = ['o', 's', '^', '<' ]  # 分别代表圆形、三角形和正方形
    markers = [
        ".",  # 点
        ",",  # 像素
        "o",  # 圆圈
        "v",  # 向下的三角形
        "^",  # 向上的三角形
        "<",  # 向左的三角形
        ">",  # 向右的三角形
        "1",  # 向下的三叉戟
        "2",  # 向上的三叉戟
        "3",  # 向左的三叉戟
        "4",  # 向右的三叉戟
        "s",  # 正方形
        "p",  # 五角星
        "*",  # 星号
        "h",  # 六边形1
        "H",  # 六边形2
        "+",  # 加号
        "x",  # 乘号
        "D",  # 菱形
        "d",  # 窄菱形
        "|",  # 垂直线
        "_"  # 水平线
    ]

    # 可视化展示
    plt.figure(figsize=(12, 8))

    # 为每个数据集绘制散点图
    plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1], c=colors[0], marker=markers[0], label='Test dataset, vulnerable')
    plt.scatter(X2_tsne[:, 0], X2_tsne[:, 1], c=colors[1], marker=markers[1], label='Test dataset, nonvulnerable')
    plt.scatter(X3_tsne_1[:, 0], X3_tsne_1[:, 1], c=colors[2], marker=markers[2], label='synthetic, vulnerable')
    plt.scatter(X3_tsne_0[:, 0], X3_tsne_0[:, 1], c=colors[3], marker=markers[3], label='synthetic, nonvulnerable')

    p_number = X3_tsne_1.shape[0]
    n_number = X3_tsne_0.shape[0]
    title = f"t-SNE: Group_id {group_id} Vulnerable number {p_number} Nonulnerable number {n_number}".format(group_id=group_id, p_number=p_number, n_number=n_number)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.show()


def draw_one_group(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id, p_number, n_number, layer_number, output_dir='',
                   correct_log_1=[], correct_log_0=[], step=-1):

    plt.figure(figsize=(12, 8))

    X3_tsne_1 = synthetic_tsne_1_list
    X3_tsne_0 = synthetic_tsne_0_list
    # plt.scatter(X3_tsne_1[:, 0], X3_tsne_1[:, 1], c='red',  marker='s',label='vulnerable', s=150)
    # plt.scatter(X3_tsne_0[:, 0], X3_tsne_0[:, 1], c='green',  marker="p",label="nonvulnerable", s=150)

    alpha = 1
    plt.figure(figsize=(12, 8))
    for i in tqdm(range(len(correct_log_1) - 1, -1, step), desc="correct_log_1 Plotting Progress"):
        color = 'purple' if correct_log_1[i] else 'red'
        plt.scatter(X3_tsne_1[i, 0], X3_tsne_1[i, 1], facecolors='none', edgecolors=color, marker='^', alpha=alpha)

    for i in tqdm(range(len(correct_log_0) - 1, -1, step), desc="correct_log_0 Plotting Progress"):
        color = 'green' if correct_log_0[i] else 'black'
        plt.scatter(X3_tsne_0[i, 0], X3_tsne_0[i, 1], facecolors='none', edgecolors=color, marker='o', alpha=alpha)
    print(" start to show")
    label_1 = "vulnerable"
    label_0 = "non-vulnerable"
    predict_correct_flag = " Predict Correct"
    predict_error_flag = " Predict InCorrect"
    plt.scatter([], [], facecolors='none', edgecolors='purple', marker='^', alpha=alpha,
                label=label_1 + predict_correct_flag)
    plt.scatter([], [], facecolors='none', edgecolors='red', marker='^', alpha=alpha,
                label=label_1 + predict_error_flag)
    plt.scatter([], [], facecolors='none', edgecolors='green', marker='o', alpha=alpha,
                label=label_0 + predict_correct_flag)
    plt.scatter([], [], facecolors='none', edgecolors='black', marker='o', alpha=alpha,
                label=label_0 + predict_error_flag)




    title = f"t-SNE: Group_id {group_id} Vulnerable number {p_number} Nonulnerable number {n_number} layer_number:{layer_number}".format(group_id=group_id, p_number=p_number, n_number=n_number, layer_number=layer_number)
    picture_name ="Group_id_"+str(group_id)+"_layer_number_"+str(layer_number)+".png"
    picture_name = picture_name.replace(":",'').replace(' ','_')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-10.0, 10.0])
    plt.ylim([-10.0, 10.0])
    plt.title(title)
    plt.legend()
    output_dir = os.path.join(output_dir, "layer_number_"+str(layer_number))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, picture_name),bbox_inches='tight', dpi=600)
    # plt.show()



def read_front_back_correct_rate(json_path):
    import json
    with open(json_path, 'r') as f:
        front_back_correct_rate = json.load(f)
        front_results_correct_rate = front_back_correct_rate['front_results_correct_rate']
        back_results_correct_rate = front_back_correct_rate['back_results_correct_rate']
        inversions_front_results = front_back_correct_rate['inversions_front_results']
        inversions_back_results = front_back_correct_rate['inversions_back_results']
        front_results_correct_rate_cosine = front_back_correct_rate['front_results_correct_rate_cosine']
        back_results_correct_rate_cosine = front_back_correct_rate['back_results_correct_rate_cosine']
    return front_results_correct_rate, back_results_correct_rate, front_results_correct_rate_cosine, back_results_correct_rate_cosine, inversions_front_results, inversions_back_results




def draw_one_group_similar_func(args, reduced_data, predict_log):

    print("reduced_data:",reduced_data)
    print("predict_log:", predict_log)
    plt.figure(figsize=(10, 8), dpi=100)
    shape_0 = reduced_data.shape[0] - 1
    predict_log_len = len(predict_log)
    predict_log_half_len = round(predict_log_len/2)
    for i, point in enumerate(reduced_data):
        print("point:", point)
        predict = predict_log[i]
        if predict == 1:
            color = 'red'
        else:
            color = 'black'

        if i < predict_log_half_len:
            color = 'purple'
            marker = 's'
        else:
            color = 'black'
            marker = '*'

        plt.scatter(point[0], point[1], marker=marker,
                    color=color, facecolors='none',alpha=1, s= 150)  # Using different markers for each point
        # plt.text(point[0], point[1], str(i), fontsize=25, ha='right', va='top')

        if i == 0:
            text = 'before-fixed'
        elif i == len(reduced_data) -1:
            text = 'after-fixed'
        # else:
        #     text = ''

        if i in [0, len(reduced_data) -1]:
            x, y = point[0], point[1]
            text_x, text_y = x + 0.0000003, y - 0.000008
            plt.annotate(
                text,
                xy=(x, y),
                xytext=(text_x, text_y),
                fontsize=30,
                ha='right',
                va='bottom',
                arrowprops=dict(facecolor='black', arrowstyle='->')
            )

        # if i in [0, shape_0]:
        #     plt.scatter(point[0], point[1], marker='s', s=100,
        #                 color=color)  # Using different markers for each point
        #     plt.text(point[0], point[1], str(i), fontsize=25, ha='right', va='top')
        # else:
        #     plt.scatter(point[0], point[1], marker='*', s=100,
        #                 color=color)  # Using different markers for each point
        #     plt.text(point[0], point[1], str(i), fontsize=9, ha='right', va='top')



    # plt.xlabel("x",  fontsize=20)
    # plt.ylabel("y",  fontsize=20)
    plt.xticks(fontsize=20,rotation=45)
    plt.yticks(fontsize=20)
    file_name = "Group_id_"+ args.index_dir_copy + "_produced_similar_func_distribution"
    file_name_png = file_name + ".png"
    file_name_pdf = file_name + ".pdf"
    file_name_eps = file_name + ".eps"
    # 调整图像布局以确保一致性
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # 获取当前图形
    fig = plt.gcf()

    # 强制重新绘制图形以确保一致性
    fig.canvas.draw()

    # 使用ScalarFormatter来格式化轴标签
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    plt.savefig(os.path.join(args.output_dir, file_name_pdf), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(args.output_dir, file_name_eps), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(args.output_dir, file_name_png), dpi=600, bbox_inches='tight')

    plt.show()
    # plt.savefig(os.path.join(args.output_dir, file_name_png), bbox_inches='tight', dpi=600)
    # plt.savefig(os.path.join(args.output_dir, file_name_pdf), bbox_inches='tight', dpi=600)
    # plt.savefig(os.path.join(args.output_dir, file_name_eps), bbox_inches='tight', dpi=600)


def draw_one_group_add_background(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id, p_number, n_number, layer_number, output_dir='',
                   correct_log_1=[], correct_log_0=[], step=-1, non_vulnerable_back_ground_tsne=[],vulnerable_back_ground_tsne=[]):

    plt.figure(figsize=(12, 8))

    X3_tsne_1 = synthetic_tsne_1_list
    X3_tsne_0 = synthetic_tsne_0_list

    alpha = 1
    plt.figure(figsize=(12, 8))
    for i in tqdm(range(len(correct_log_1) - 1, -1, step), desc="correct_log_1 Plotting Progress"):
        color = 'purple' if correct_log_1[i] else 'red'
        plt.scatter(X3_tsne_1[i, 0], X3_tsne_1[i, 1], facecolors='none', edgecolors=color, marker='^', alpha=alpha)

    for i in tqdm(range(len(correct_log_0) - 1, -1, step), desc="correct_log_0 Plotting Progress"):
        color = 'green' if correct_log_0[i] else 'black'
        plt.scatter(X3_tsne_0[i, 0], X3_tsne_0[i, 1], facecolors='none', edgecolors=color, marker='o', alpha=alpha)




    plt.scatter(vulnerable_back_ground_tsne[::10, 0], vulnerable_back_ground_tsne[::10, 1],
                color='red', marker='o', alpha=alpha, s=1)
    plt.scatter(non_vulnerable_back_ground_tsne[::10, 0], non_vulnerable_back_ground_tsne[::10, 1], color='green', marker='o', alpha=alpha, s=1)


    print(" start to show")
    label_1 = "vulnerable"
    label_0 = "non-vulnerable"
    predict_correct_flag = " Predict Correct"
    predict_error_flag = " Predict InCorrect"
    plt.scatter([], [], facecolors='none', edgecolors='purple', marker='^', alpha=alpha,
                label=label_1 + predict_correct_flag)
    plt.scatter([], [], facecolors='none', edgecolors='red', marker='^', alpha=alpha,
                label=label_1 + predict_error_flag)
    plt.scatter([], [], facecolors='none', edgecolors='green', marker='o', alpha=alpha,
                label=label_0 + predict_correct_flag)
    plt.scatter([], [], facecolors='none', edgecolors='black', marker='o', alpha=alpha,
                label=label_0 + predict_error_flag)
    plt.scatter([], [],  color='red', marker='o', alpha=alpha,
                label="vulnerable background")
    plt.scatter([], [],  color='green', marker='o', alpha=alpha,
                label="non-vulnerable background")








    title = f"t-SNE: Group_id {group_id} Vulnerable number {p_number} Nonulnerable number {n_number} layer_number:{layer_number}  added background".format(group_id=group_id, p_number=p_number, n_number=n_number, layer_number=layer_number)
    picture_name ="Group_id_"+str(group_id)+"_layer_number_"+str(layer_number)+"_add_background.png"
    picture_name = picture_name.replace(":",'').replace(' ','_')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-10.0, 10.0])
    plt.ylim([-10.0, 10.0])
    # plt.title(title)
    plt.legend()
    output_dir = os.path.join(output_dir, "layer_number_"+str(layer_number))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, picture_name),bbox_inches='tight', dpi=600)
    plt.show()




def draw_one_group_add_background_similar_func(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id, p_number, n_number, layer_number, output_dir='',
                   correct_log_1=[], correct_log_0=[], step=-1, non_vulnerable_back_ground_tsne=[],vulnerable_back_ground_tsne=[], k=0):

    plt.figure(figsize=(12, 8))

    X3_tsne_1 = synthetic_tsne_1_list
    X3_tsne_0 = synthetic_tsne_0_list

    alpha = 1




    plt.scatter(vulnerable_back_ground_tsne[::10, 0], vulnerable_back_ground_tsne[::10, 1],
                color='red', marker='o', alpha=alpha, s=20)
    plt.scatter(non_vulnerable_back_ground_tsne[::10, 0], non_vulnerable_back_ground_tsne[::10, 1],
                color='green', marker='o', alpha=alpha, s=20)
    # 获取当前轴对象
    ax = plt.gca()

    for i in tqdm(range(len(correct_log_1) - 1, -1, step), desc="correct_log_1 Plotting Progress"):
        color = 'purple' if correct_log_1[i] else 'red'
        color = 'purple'
        plt.scatter(X3_tsne_1[i, 0], X3_tsne_1[i, 1], facecolors='none', edgecolors=color, marker='s', alpha=alpha)

        if k in ['179281', 179281]:
            if i == 0:  # Only add text for the first plotted point
                plt.text(X3_tsne_1[i, 0], X3_tsne_1[i, 1] + 0.05, k, fontsize=15, ha='right', va='bottom')

            if i == p_number+1:  # Only add text for the first plotted point
                plt.text(X3_tsne_1[i, 0]+1, X3_tsne_1[i, 1] + 0.05, '179639', fontsize=15, ha='right', va='bottom')
                rect = patches.Rectangle((4.5, 8.7), 6.6 - 4.5, 9.5 - 8.7, linewidth=1, edgecolor='gray',
                                         facecolor='none')
                ax.add_patch(rect)
        elif k in ['179639', 179639]:
            if i == 0:  # Only add text for the first plotted point
                plt.text(X3_tsne_1[i, 0]+1, X3_tsne_1[i, 1] + 0.15, k, fontsize=15, ha='right', va='bottom')

                # 在图像上的指定位置添加灰色矩形框
                rect = patches.Rectangle((4.5, 8.7), 6.6 - 4.5, 9.5 - 8.7, linewidth=1, edgecolor='gray', facecolor='none')
                ax.add_patch(rect)


    for i in tqdm(range(len(correct_log_0) - 1, -1, step), desc="correct_log_0 Plotting Progress"):
        color = 'green' if correct_log_0[i] else 'black'
        color = 'black'
        plt.scatter(X3_tsne_0[i, 0], X3_tsne_0[i, 1], facecolors='none', edgecolors=color, marker='*', alpha=alpha)









    plt.scatter([], [], facecolors='none', edgecolors='purple', marker='s', alpha=alpha, label="before fix function")
    plt.scatter([], [], facecolors='none', edgecolors='black', marker='*', alpha=alpha, label="after fix function")
    plt.scatter([], [],  color='red', marker='o', alpha=alpha,
                label="vulnerable function")
    plt.scatter([], [],  color='green', marker='o', alpha=alpha,
                label="non-vulnerable function")





    # title = f"t-SNE: Group_id {group_id} Vulnerable number {p_number} Nonulnerable number {n_number} layer_number:{layer_number}  added background".format(group_id=group_id, p_number=p_number, n_number=n_number, layer_number=layer_number)
    picture_name ="Group_id_"+str(group_id)+"_layer_number_"+str(layer_number)+"_add_background.png"
    picture_name = picture_name.replace(":",'').replace(' ','_')
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([-8.0, 11.0])
    plt.ylim([-8.0, 11.0])
    # plt.title('fusion together')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=15)
    output_dir = os.path.join(output_dir, "layer_number_"+str(layer_number))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    total_similar_func = p_number +  n_number
    total_similar_func = str(total_similar_func)
    picture_name = f"Group_id_{group_id}_produced_similar_func_{total_similar_func}_layer_number_{layer_number}_added_background"
    picture_name_png = picture_name + ".png"
    picture_name_eps = picture_name + ".eps"
    picture_name_pdf = picture_name + ".pdf"
    picture_name_svg = picture_name + ".svg"
    plt.savefig(os.path.join(output_dir, picture_name_png),bbox_inches='tight', dpi=100)
    plt.savefig(os.path.join(output_dir, picture_name_eps),bbox_inches='tight', dpi=100)
    plt.savefig(os.path.join(output_dir, picture_name_pdf),bbox_inches='tight', dpi=100)
    plt.savefig(os.path.join(output_dir, picture_name_svg), bbox_inches='tight', dpi=100, format = 'svg')
    plt.show()




import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from tqdm import tqdm





def draw_one_group_add_background_similar_func_zoom_in(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id, p_number, n_number, layer_number, output_dir='',
                   correct_log_1=[], correct_log_0=[], step=-1, non_vulnerable_back_ground_tsne=[],vulnerable_back_ground_tsne=[], k=0):

    plt.figure(figsize=(12, 8))

    X3_tsne_1 = synthetic_tsne_1_list
    X3_tsne_0 = synthetic_tsne_0_list

    alpha = 1
    plt.figure(figsize=(12, 8))





    plt.scatter(vulnerable_back_ground_tsne[::10, 0], vulnerable_back_ground_tsne[::10, 1],
                color='red', marker='o', alpha=alpha)
    plt.scatter(non_vulnerable_back_ground_tsne[::10, 0], non_vulnerable_back_ground_tsne[::10, 1],
                color='green', marker='o', alpha=alpha)

    for i in tqdm(range(len(correct_log_1) - 1, -1, step), desc="correct_log_1 Plotting Progress"):
        color = 'purple' if correct_log_1[i] else 'red'
        color = 'purple'
        plt.scatter(X3_tsne_1[i, 0], X3_tsne_1[i, 1], facecolors='none', edgecolors=color, marker='s', alpha=alpha, s= 150)
        # if i == 0:  # Only add text for the first plotted point
        #     plt.text(X3_tsne_1[i, 0], X3_tsne_1[i, 1] + 0.05, k, fontsize=12, ha='right',  va='bottom')
        if k in ['179281', 179281]:
            if i == p_number+1:  # Only add text for the first plotted point
                x, y = X3_tsne_1[i, 0], X3_tsne_1[i, 1]
                text_x, text_y = x+0.2, y + 0.05
                plt.annotate(
                    'before-fixed',
                    xy=(x, y),
                    xytext=(text_x, text_y),
                    fontsize=30,
                    ha='right',
                    va='bottom',
                    arrowprops=dict(facecolor='black', arrowstyle='->')
                )



    for i in tqdm(range(len(correct_log_0) - 1, -1, step), desc="correct_log_0 Plotting Progress"):
        color = 'green' if correct_log_0[i] else 'black'
        color = 'black'
        plt.scatter(X3_tsne_0[i, 0], X3_tsne_0[i, 1], facecolors='none', edgecolors=color, marker='*', alpha=alpha, s= 150)

        if k in ['179281', 179281]:
            if i == p_number+1:  # Only add text for the first plotted point
                # plt.text(X3_tsne_0[i, 0] + 0.2, X3_tsne_0[i, 1] + 0.05, '179639', fontsize=12, ha='right', va='bottom')
                x, y = X3_tsne_0[i, 0], X3_tsne_0[i, 1]
                text_x, text_y = x + 0.2, y + 0.05
                plt.annotate(
                    'after-fixed',
                    xy=(x, y),
                    xytext=(text_x, text_y),
                    fontsize=30,
                    ha='right',
                    va='bottom',
                    arrowprops=dict(facecolor='black', arrowstyle='->')
                )






    plt.scatter([], [], facecolors='none', edgecolors='purple', marker='s', alpha=alpha, label="before fix function")
    plt.scatter([], [], facecolors='none', edgecolors='black', marker='*', alpha=alpha, label="after fix function")
    plt.scatter([], [],  color='red', marker='o', alpha=alpha,
                label="vulnerable function")
    plt.scatter([], [],  color='green', marker='o', alpha=alpha,
                label="non-vulnerable function")





    # title = f"t-SNE: Group_id {group_id} Vulnerable number {p_number} Nonulnerable number {n_number} layer_number:{layer_number}  added background".format(group_id=group_id, p_number=p_number, n_number=n_number, layer_number=layer_number)
    # picture_name ="Group_id_"+str(group_id)+"_layer_number_"+str(layer_number)+"_add_background.png"
    # picture_name = picture_name.replace(":",'').replace(' ','_')
    # plt.xlabel('x',  fontsize=20)
    # plt.ylabel('y',  fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    x_min = 4.5
    x_max = 6.6
    y_min = 8.7
    y_max = 9.5

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    # plt.title('draw_one_group_')
    # plt.legend(loc='lower left', bbox_to_anchor=(0, 0))
    output_dir = os.path.join(output_dir, "layer_number_"+str(layer_number))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    total_similar_func = p_number +  n_number
    total_similar_func = str(total_similar_func)
    picture_name = f"Group_id_{group_id}_produced_similar_func_{total_similar_func}_layer_number_{layer_number}_added_background_zoom_in"
    picture_name = picture_name + "_zoom_in_" + str(x_min) + "_" + str(x_max)+ "_" + str(y_min) + "_" + str(y_max)
    picture_name_png = picture_name + ".png"
    picture_name_eps = picture_name + ".eps"
    picture_name_pdf = picture_name + ".pdf"
    plt.savefig(os.path.join(output_dir, picture_name_png),bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(output_dir, picture_name_eps),bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(output_dir, picture_name_pdf),bbox_inches='tight', dpi=600)
    plt.show()



def draw_one_group_add_background_similar_func_zoom_in_2(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id, p_number, n_number, layer_number, output_dir='',
                   correct_log_1=[], correct_log_0=[], step=-1, non_vulnerable_back_ground_tsne=[],vulnerable_back_ground_tsne=[], k=0):

    plt.figure(figsize=(12, 8))

    X3_tsne_1 = synthetic_tsne_1_list
    X3_tsne_0 = synthetic_tsne_0_list

    alpha = 1
    plt.figure(figsize=(12, 8))





    plt.scatter(vulnerable_back_ground_tsne[::10, 0], vulnerable_back_ground_tsne[::10, 1],
                color='red', marker='o', alpha=alpha)
    plt.scatter(non_vulnerable_back_ground_tsne[::10, 0], non_vulnerable_back_ground_tsne[::10, 1],
                color='green', marker='o', alpha=alpha)

    for i in tqdm(range(len(correct_log_1) - 1, -1, step), desc="correct_log_1 Plotting Progress"):
        color = 'purple' if correct_log_1[i] else 'red'
        color = 'purple'
        plt.scatter(X3_tsne_1[i, 0], X3_tsne_1[i, 1], facecolors='none', edgecolors=color, marker='s', alpha=alpha, s= 150)
        if i == 0:  # Only add text for the first plotted point
            plt.text(X3_tsne_1[i, 0], X3_tsne_1[i, 1] + 0.05, k, fontsize=12, ha='right',  va='bottom')
        if k in ['179281', 179281]:
            if i == p_number+1:  # Only add text for the first plotted point
                plt.text(X3_tsne_1[i, 0], X3_tsne_1[i, 1] + 0.05, '179639', fontsize=12, ha='right', va='bottom')

    for i in tqdm(range(len(correct_log_0) - 1, -1, step), desc="correct_log_0 Plotting Progress"):
        color = 'green' if correct_log_0[i] else 'black'
        color = 'black'
        plt.scatter(X3_tsne_0[i, 0], X3_tsne_0[i, 1], facecolors='none', edgecolors=color, marker='*', alpha=alpha, s= 150)

        if k in ['179281', 179281]:
            if i == p_number+1:  # Only add text for the first plotted point
                plt.text(X3_tsne_0[i, 0] + 0.2, X3_tsne_0[i, 1] + 0.05, '179639', fontsize=12, ha='right', va='bottom')






    plt.scatter([], [], facecolors='none', edgecolors='purple', marker='s', alpha=alpha, label="before fix function")
    plt.scatter([], [], facecolors='none', edgecolors='black', marker='*', alpha=alpha, label="after fix function")
    plt.scatter([], [],  color='red', marker='o', alpha=alpha,
                label="vulnerable function")
    plt.scatter([], [],  color='green', marker='o', alpha=alpha,
                label="non-vulnerable function")





    # title = f"t-SNE: Group_id {group_id} Vulnerable number {p_number} Nonulnerable number {n_number} layer_number:{layer_number}  added background".format(group_id=group_id, p_number=p_number, n_number=n_number, layer_number=layer_number)
    # picture_name ="Group_id_"+str(group_id)+"_layer_number_"+str(layer_number)+"_add_background.png"
    # picture_name = picture_name.replace(":",'').replace(' ','_')
    plt.xlabel('x')
    plt.ylabel('y')
    x_min = 6
    x_max = 8.8
    y_min = 3.6
    y_max = 6.3

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    # plt.title('zoom_in_2')
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0))
    output_dir = os.path.join(output_dir, "layer_number_"+str(layer_number))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    total_similar_func = p_number +  n_number
    total_similar_func = str(total_similar_func)
    picture_name = f"Group_id_{group_id}_produced_similar_func_{total_similar_func}_layer_number_{layer_number}_added_background_zoom_in"
    picture_name = picture_name + "_zoom_in_" + str(x_min) + "_" + str(x_max)+ "_" + str(y_min) + "_" + str(y_max)
    picture_name_png = picture_name + ".png"
    picture_name_eps = picture_name + ".eps"
    picture_name_pdf = picture_name + ".pdf"
    plt.savefig(os.path.join(output_dir, picture_name_png),bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(output_dir, picture_name_eps),bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(output_dir, picture_name_pdf),bbox_inches='tight', dpi=600)
    plt.show()



def design_axin(ax, vulnerable_back_ground_tsne, non_vulnerable_back_ground_tsne, correct_log_1, correct_log_0, X3_tsne_1, X3_tsne_0
                ,p_number,  alpha, step, k):
    ax.scatter(vulnerable_back_ground_tsne[::10, 0], vulnerable_back_ground_tsne[::10, 1],
                color='red', marker='o', alpha=alpha)
    ax.scatter(non_vulnerable_back_ground_tsne[::10, 0], non_vulnerable_back_ground_tsne[::10, 1],
                color='green', marker='o', alpha=alpha)

    for i in tqdm(range(len(correct_log_1) - 1, -1, step), desc="correct_log_1 Plotting Progress"):
        color = 'purple' if correct_log_1[i] else 'red'
        color = 'purple'
        ax.scatter(X3_tsne_1[i, 0], X3_tsne_1[i, 1], facecolors='none', edgecolors=color, marker='s', alpha=alpha)
        if i == 0:  # Only add text for the first plotted point
            ax.text(X3_tsne_1[i, 0], X3_tsne_1[i, 1] + 0.05, k, fontsize=12, ha='right',  va='bottom')
        if k in ['179281', 179281]:
            if i == p_number+1:  # Only add text for the first plotted point
                ax.text(X3_tsne_1[i, 0], X3_tsne_1[i, 1] + 0.05, '179639', fontsize=12, ha='right', va='bottom')

    for i in tqdm(range(len(correct_log_0) - 1, -1, step), desc="correct_log_0 Plotting Progress"):
        color = 'green' if correct_log_0[i] else 'black'
        color = 'black'
        ax.scatter(X3_tsne_0[i, 0], X3_tsne_0[i, 1], facecolors='none', edgecolors=color, marker='*', alpha=alpha)

        if k in ['179281', 179281]:
            if i == p_number+1:  # Only add text for the first plotted point
                ax.text(X3_tsne_0[i, 0] + 0.2, X3_tsne_0[i, 1] + 0.05, '179639', fontsize=12, ha='right', va='bottom')


    return ax



def draw_one_group_add_background_similar_func_zoom_in_3(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id, p_number, n_number, layer_number, output_dir='',
                   correct_log_1=[], correct_log_0=[], step=-1, non_vulnerable_back_ground_tsne=[],vulnerable_back_ground_tsne=[], k=0):

    # plt.figure(figsize=(12, 8))

    X3_tsne_1 = synthetic_tsne_1_list
    X3_tsne_0 = synthetic_tsne_0_list

    alpha = 1
    # plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots()




    ax.scatter(vulnerable_back_ground_tsne[::10, 0], vulnerable_back_ground_tsne[::10, 1],
                color='red', marker='o', alpha=alpha)
    ax.scatter(non_vulnerable_back_ground_tsne[::10, 0], non_vulnerable_back_ground_tsne[::10, 1],
                color='green', marker='o', alpha=alpha)

    for i in tqdm(range(len(correct_log_1) - 1, -1, step), desc="correct_log_1 Plotting Progress"):
        color = 'purple' if correct_log_1[i] else 'red'
        color = 'purple'
        ax.scatter(X3_tsne_1[i, 0], X3_tsne_1[i, 1], facecolors='none', edgecolors=color, marker='s', alpha=alpha)
        if i == 0:  # Only add text for the first plotted point
            ax.text(X3_tsne_1[i, 0], X3_tsne_1[i, 1] + 0.05, k, fontsize=12, ha='right',  va='bottom')
        if k in ['179281', 179281]:
            if i == p_number+1:  # Only add text for the first plotted point
                ax.text(X3_tsne_1[i, 0], X3_tsne_1[i, 1] + 0.05, '179639', fontsize=12, ha='right', va='bottom')

    for i in tqdm(range(len(correct_log_0) - 1, -1, step), desc="correct_log_0 Plotting Progress"):
        color = 'green' if correct_log_0[i] else 'black'
        color = 'black'
        ax.scatter(X3_tsne_0[i, 0], X3_tsne_0[i, 1], facecolors='none', edgecolors=color, marker='*', alpha=alpha)

        if k in ['179281', 179281]:
            if i == p_number+1:  # Only add text for the first plotted point
                ax.text(X3_tsne_0[i, 0] + 0.2, X3_tsne_0[i, 1] + 0.05, '179639', fontsize=12, ha='right', va='bottom')




    ax.scatter([], [], facecolors='none', edgecolors='purple', marker='s', alpha=alpha, label="before fix function")
    ax.scatter([], [], facecolors='none', edgecolors='black', marker='*', alpha=alpha, label="after fix function")
    ax.scatter([], [],  color='red', marker='o', alpha=alpha,
                label="vulnerable function")
    ax.scatter([], [],  color='green', marker='o', alpha=alpha,
                label="non-vulnerable function")

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Zoom in insert')
    ax.legend()








    # 设置放大区域
    x1, x2, y1, y2 = 7.1, 7.11, 4.9, 5.0  # 放大区域的坐标范围

    # 添加插图
    axins = inset_axes(ax, width="30%", height="30%", loc='upper left')  # 设置插图的位置和大小

    axins = design_axin(axins, vulnerable_back_ground_tsne, non_vulnerable_back_ground_tsne, correct_log_1, correct_log_0,
                X3_tsne_1, X3_tsne_0
                , p_number, alpha, step, k)

    # axins.scatter(x, y)
    axins.set_xlim(x1, x2)  # 设置插图的 x 轴范围
    axins.set_ylim(y1, y2)  # 设置插图的 y 轴范围
    axins.set_xticklabels('')  # 隐藏插图的 x 轴标签
    axins.set_yticklabels('')  # 隐藏插图的 y 轴标签

    # 在主图上标出放大区域
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


    # 获取放大区域的中心点坐标
    zoom_center_x = 0.0
    zoom_center_y = 0.2

    # 将数据坐标转换为图坐标
    trans = ax.transData.transform
    inv_trans = fig.transFigure.inverted().transform

    # 被放大区域的中心点在图中的坐标
    zoom_center = inv_trans(trans((zoom_center_x, zoom_center_y)))

    # 插图的中心点在图中的坐标
    inset_center = inv_trans(trans((0.95, 0.95)))

    # 添加箭头从被放大区域指向插图
    ax.annotate(
        '', xy=inset_center, xytext=zoom_center, xycoords='figure fraction',
        textcoords='figure fraction',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='red', lw=2)
    )


    plt.legend(loc='lower left', bbox_to_anchor=(0, 0))
    output_dir = os.path.join(output_dir, "layer_number_"+str(layer_number))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    total_similar_func = p_number +  n_number
    total_similar_func = str(total_similar_func)
    picture_name = f"Group_id_{group_id}_produced_similar_func_{total_similar_func}_layer_number_{layer_number}_added_background_zoom_in"
    picture_name = picture_name + "_V3" + str(x1) + "_" + str(x2)+ "_" + str(y1) + "_" + str(y2)
    picture_name_png = picture_name + ".png"
    picture_name_eps = picture_name + ".eps"
    picture_name_pdf = picture_name + ".pdf"
    plt.savefig(os.path.join(output_dir, picture_name_png),bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(output_dir, picture_name_eps),bbox_inches='tight', dpi=600)
    plt.savefig(os.path.join(output_dir, picture_name_pdf),bbox_inches='tight', dpi=600)
    plt.show()



def visual_three_vector_v2(target_feature, feature_1):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # # 假设 X1, X2, X3 是你的三个 (n*768) 矩阵
    # X1 = np.random.rand(1000, 768)  # 这里我们创建了一个随机矩阵作为示例
    # X2 = np.random.rand(1000, 768)  # 第二个矩阵
    # X3 = np.random.rand(1000, 768)  # 第三个矩阵
    X1, X2, X3 = target_feature[0], target_feature[1], target_feature[-1]
    method = 'tsne_combination'
    if method == 'tsne':
        X1 = X1[:target_feature[-1].shape[0], :]
        X2 = X2[:target_feature[-1].shape[0], :]
        # 使用 TSNE 进行降维
        method_object = TSNE(n_components=2, verbose=0, perplexity=1, n_iter=300)
        X1_tsne = method_object.fit_transform(X1)
        X2_tsne = method_object.fit_transform(X2)
        X3_tsne = method_object.fit_transform(X3)
        X3_tsne_1 = X3_tsne[:feature_1.shape[0], :]
        X3_tsne_0 = X3_tsne[feature_1.shape[0]:, :]

    elif method == 'pca':
        method_object = PCA(n_components=2)
        X1 = X1[:target_feature[-1].shape[0], :]
        X2 = X2[:target_feature[-1].shape[0], :]

        X1_tsne = method_object.fit_transform(X1)
        X2_tsne = method_object.fit_transform(X2)
        X3_tsne = method_object.fit_transform(X3)
        X3_tsne_1 = X3_tsne[:feature_1.shape[0], :]
        X3_tsne_0 = X3_tsne[feature_1.shape[0]:, :]
    elif method == 'tsne_combination':
        X = np.concatenate([X1, X2, X3])
        method_object = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=300)
        # method_object = PCA(n_components=2)
        # method_object = LDA(n_components=2)
        X_tsne = method_object.fit_transform(X)
        X1_tsne = X_tsne[:X1.shape[0], :]
        X1_X2 = (X1.shape[0]+X2.shape[0])
        X2_tsne = X_tsne[X1.shape[0]: X1_X2, :]
        X3_tsne = X_tsne[X1_X2:, :]
        print(X3_tsne.shape)
        X3_tsne_1 = X3_tsne[:feature_1.shape[0], :]
        X3_tsne_0 = X3_tsne[feature_1.shape[0]:, :]
        # X1_tsne = method_object.fit_transform(X1)
        # X2_tsne = method_object.fit_transform(X2)
        # X3_tsne = method_object.fit_transform(X3)
    elif method == 'isomap':
        from sklearn.manifold import Isomap
        n_neighbors = 10  # k近邻数量
        n_components = 2  # 降维后的维度
        X = np.concatenate([X1, X2, X3])
        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        data_reduced = isomap.fit_transform(X)

        X_tsne = data_reduced
        X1_tsne = X_tsne[:X1.shape[0], :]
        X1_X2 = (X1.shape[0]+X2.shape[0])
        X2_tsne = X_tsne[X1.shape[0]: X1_X2, :]
        X3_tsne = X_tsne[X1_X2:, :]
        print(X3_tsne.shape)
        X3_tsne_1 = X3_tsne[:feature_1.shape[0], :]
        X3_tsne_0 = X3_tsne[feature_1.shape[0]:, :]
        # X1_tsne = method_object.fit_transform(X1)
        # X2_tsne = method_object.fit_transform(X2)
        # X3_tsne = method_object.fit_transform(X3)
    return  X3_tsne_1, X3_tsne_0, X1_tsne, X2_tsne

def draw_picture(synthetic_tsne_1_list, synthetic_tsne_0_list, layer_number, output_dir):
    colors = ['blue',  'green', 'red', 'cyan', 'magenta','yellow', 'black','white']

    # 可视化展示
    plt.figure(figsize=(12, 8))
    for i in  range(len(synthetic_tsne_1_list)):
        # if i >= len(markers)-1 or i>=7:
        #     break
        X3_tsne_1 = synthetic_tsne_1_list[i]
        X3_tsne_0 = synthetic_tsne_0_list[i]
        # 为每个数据集绘制散点图
        # plt.scatter(X3_tsne_1[:, 0], X3_tsne_1[:, 1], c=colors[i], marker=markers[i], s=150)
        # plt.scatter(X3_tsne_0[:, 0], X3_tsne_0[:, 1], c=colors[i+1], marker=markers[i], s=150)

        plt.scatter(X3_tsne_1[:, 0], X3_tsne_1[:, 1], c='red',  marker='s',label='vulnerable', s=150)
        plt.scatter(X3_tsne_0[:, 0], X3_tsne_0[:, 1], c='green',marker="p", label='nonvulnerable', s=150)

        # for i in range(len(X3_tsne_1[:, 0])):
        #     plt.text(X3_tsne_1[:, 0][i], X3_tsne_1[:, 1][i] + 0.6, str(group_id), fontsize=9, ha='center', va='top')
        #
        # for i in range(len(X3_tsne_0[:, 0])):
        #     plt.text(X3_tsne_0[:, 0][i], X3_tsne_0[:, 1][i] + 0.6, str(group_id), fontsize=9, ha='center', va='top')

    title = f"t-SNE: all groups Vulnerable and non-Vulnerable"
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    picture_name = "t-SNE_all_group_Vulnerable_and_nonVulnerable" + "_layer_number_" + str(layer_number) + ".png"
    picture_name = picture_name.replace(":", '').replace(' ', '_')

    output_dir = os.path.join(output_dir, "layer_number_"+str(layer_number))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, picture_name),bbox_inches='tight', dpi=600)

    # plt.show()


def load_similar_func_feature(cache_name, step_state, return_label=False, return_predict=False):
    np_file = np.load(cache_name, allow_pickle=True)
    feat_log, label_log, predict_log = np_file['arr_0'], np_file['arr_1'], np_file['arr_2']
    feat_log = feat_log.T.astype(np.float32)

    # feature = feat_log[:, -770:-2]
    correct_log = [1 if label_log[i] == predict_log[i] else 0 for i in range(len(predict_log))]

    feature = feat_log[:, 769 * (step_state - 1):(step_state * 769) -1]
    if return_label:
        return feature, correct_log, label_log
    if return_predict:
        return feature, correct_log, predict_log
    return feature, correct_log


def read_similar_func_features():
    import argparse
    parser = argparse.ArgumentParser(description='check after not exist in before')
    args = parser.parse_args()
    args.c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/'
    args.produce_similar_func = os.path.join(args.c_root, 'produce_similar_func')
    similar_func_features_all_map = {}

    files_id = ['179639', '179281']
    # files_id = os.listdir(args.produce_similar_func)

    for index_dir in files_id:
        import copy
        one_index_dir_res = {}
        index_dir_copy = copy.deepcopy(index_dir)
        args.index_dir_copy = index_dir_copy
        # if index_dir in ['187532', 187532]:
        index_dir = os.path.join(args.produce_similar_func, index_dir)
        args.index_dir = index_dir
        for filename in os.listdir(index_dir):
            if '.npy.npz' in filename:
                step_state = 1
                filepath = os.path.join(index_dir, filename)
                feature, correct_log, predict_log = load_similar_func_feature(filepath, step_state, return_predict=True)
                one_index_dir_res["feature"] = feature
                one_index_dir_res["correct_log"] = correct_log
                one_index_dir_res["predict_log"] = predict_log
        similar_func_features_all_map[index_dir_copy] = one_index_dir_res
    args.produce_similar_func = os.path.join(args.c_root, 'produce_similar_func')
    return  similar_func_features_all_map, args





def combine_two_figures():
    root_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_produced_similar_func'
    input_dir = os.path.join(root_dir, 'selected_updated')
    output_dir = os.path.join(root_dir, 'zoom_in')

    outer_figure = 'Group_id_179281_179639_produced_similar_func_13_layer_number_15_added_background.png'
    inner_figure = 'Group_id_179281_produced_similar_func_distribution.png'

    outer_figure_path = os.path.join(input_dir, outer_figure)
    inner_figure_path = os.path.join(input_dir, inner_figure)







    # 加载EPS图片作为主图
    fig, ax = plt.subplots()

    eps_img = mpimg.imread(outer_figure_path)
    ax.imshow(eps_img)
    ax.set_xlabel('X-axis')  # 显示主图的 x 轴标签
    ax.set_ylabel('Y-axis')  # 显示主图的 y 轴标签
    ax.set_title('Main Image')  # 显示主图的标题

    # 手动调整后的放大区域的坐标范围
    x1, x2, y1, y2 = 950, 970, 250, 270  # 这些值需要根据主图内容调整

    # 添加在大图外部的插图
    # 使用bbox_to_anchor来定位插图的位置，调整宽度和高度以放大插图
    axins = inset_axes(ax, width="30%", height="30%", bbox_to_anchor=(1.25, 0.5, 0.3, 0.3), bbox_transform=ax.transAxes,
                       loc="center left")
    # 加载PNG图片作为插图
    png_img = mpimg.imread(inner_figure_path)  # 使用相同的图片路径作为示例
    axins.imshow(png_img)
    axins.set_xticks([])  # 隐藏插图的 x 轴标签
    axins.set_yticks([])  # 隐藏插图的 y 轴标签

    # 找到图中数字 179281 的坐标（这些值需要根据具体图像内容调整）
    x_179281, y_179281 = 950, 250  # 这些值是示例，需根据实际情况调整

    # 获取放大区域的中心点坐标
    trans = ax.transData.transform
    inv_trans = fig.transFigure.inverted().transform

    # 插图的四个角在图中的坐标
    corner_coords = [
        inv_trans(axins.transAxes.transform((0, 0))),  # 左下角
        inv_trans(axins.transAxes.transform((1, 0))),  # 右下角
        inv_trans(axins.transAxes.transform((0, 1))),  # 左上角
        inv_trans(axins.transAxes.transform((1, 1)))  # 右上角
    ]

    # 添加四条灰色线条
    for corner in corner_coords:
        ax.annotate(
            '', xy=corner, xytext=(x_179281, y_179281), xycoords='figure fraction',
            textcoords='data',
            arrowprops=dict(arrowstyle="-", color='gray', lw=1)
        )
    plt.draw()
    # 显示图像
    plt.savefig(os.path.join(output_dir, 'output.png'), dpi=600, bbox_inches='tight')
    plt.show()



def find_knn(X):
    # 定义最近邻模型
    k = 5  # 最近邻的个数
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)

    # 计算每个样本的 k 个最近邻
    distances, indices = nbrs.kneighbors(X)

    # 显示结果
    for i in range(len(X)):
        print(f"向量 {i} 的最近邻是: {indices[i]}, 距离分别是: {distances[i]}")


def obtain_A_B_N(args, option, layer_number, target_file, input_dir):
    print("layer_number:", layer_number)
    target_feature = []
    all_label = []
    for file in target_file:
        cache_name = os.path.join(input_dir, file)
        # 这是加载 Vulnerable 和non-Vulnerable背景的部分
        if option == 1:
            feature, correct_log, label_log = load_similar_func_feature(cache_name, step_state=layer_number,
                                                                        return_label=True)
            target_feature.append(feature)
            all_label.append(label_log)
        elif option in [2, 3]:
            if 'test_add_after' in file:
                feature, correct_log, label_log = load_similar_func_feature(cache_name, step_state=layer_number,
                                                                            return_label=True)
                # target_feature.append(feature)
                # all_label.append(label_log)

                before_feature = []
                before_label = []
                non_feature = []
                non_label = []
                after_feature = []
                after_label = []
                after_section = len(label_log) - 946
                for ii in range(len(label_log)):
                    # if ii >= 17808:
                    #     break
                    i_label = label_log[ii]
                    if i_label == 1:
                        before_feature.append(feature[ii].tolist())
                        before_label.append(1)
                    elif ii >= after_section:
                        print("i_label:", i_label)
                        after_feature.append(feature[ii].tolist())
                        after_label.append(-1)
                    else:
                        non_feature.append(feature[ii].tolist())
                        non_label.append(-1)


                before_feature = np.array(before_feature)
                before_label = np.array(before_label)
                non_feature = np.array(non_feature)
                non_label = np.array(non_label)
                target_feature.append(before_feature)
                all_label.append(before_label)
                target_feature.append(non_feature)
                all_label.append(non_label)
                after_feature = np.array(after_feature)
                after_label = np.array(after_label)

            # elif 'test_add_after' in file:
            #     feature, correct_log, label_log = load_similar_func_feature(cache_name, step_state=layer_number,
            #                                                                 return_label=True)
            #     after_feature = []
            #     after_label = []
            #     after_section = len(label_log) - 946
            #     for ii in range(len(label_log)):
            #         if ii <= after_section:
            #             continue
            #         i_label = label_log[ii]
            #         if i_label == 0:
            #             after_feature.append(feature[ii].tolist())
            #             after_label.append(-1)
            #     after_feature = np.array(after_feature)
            #     after_label = np.array(after_label)
            #     target_feature.append(after_feature)
            #     all_label.append(after_label)


    print("before_feature shape:", before_feature.shape)
    print("after_feature shape:", after_feature.shape)
    print("non_feature shape:", non_feature.shape)
    args.b_cnt = len(before_label)
    args.a_cnt = len(after_label)
    args.non_cnt = len(non_label)


    A_B_N_feature = np.concatenate([before_feature, after_feature, non_feature])
    A_B_N_label = np.concatenate([before_label, after_label, non_label])
    A_B_N_feature = np.array(A_B_N_feature)
    A_B_N_label = np.array(A_B_N_label)

    return A_B_N_feature, A_B_N_label


def calculate_llc(args):




    # def single_function(layer_number):
    layer_numbers = [15, 12, 7, 1, ]
    layer_numbers = [1]
    args.v_cnt = 0
    args.nv_cnt = 0
    for layer_number in layer_numbers:
        A_B_N_feature, A_B_N_label = obtain_A_B_N(args, option, layer_number, target_file, input_dir)
        args.llc_type = 'A_B_N'
        draw_A_B_N(A_B_N_feature, A_B_N_label, args)


        # B_N_feature = np.concatenate([before_feature  , non_feature])
        # B_N_label = np.concatenate([before_label , non_label])
        # B_N_feature = np.array(B_N_feature)
        # B_N_label = np.array(B_N_label)
        # args.llc_type = 'B_N'
        # draw_B_N(B_N_feature, B_N_label, args)








def draw_A_B_N(features, labels, args):
    llc_sum = []

    # knn_list = [1, 2, 3, 4]
    for k in args.k:
        print("k:", k)
        b_llc_sum, a_llc_sum, non_paired_llc_sum, b_a_llc_sum = calculate_llc_core(features, labels, args, k=k, llc_type=args.llc_type)
        one_k_llc_sum = {"k": k,
                         "before_func_llc_avg": b_llc_sum,
                         "after_func_llc_avg": a_llc_sum,
                         # "before_after_llc_avg":b_a_llc_sum,
                         "non_paired_llc_avg": non_paired_llc_sum}
        llc_sum.append(one_k_llc_sum)
        print("one_k_llc_sum:", one_k_llc_sum)
    llc_sum = pd.DataFrame(llc_sum)
    print("args.llc_type:", args.llc_type)
    print(llc_sum)
    llc_sum.to_csv(args.dataset + "_llc_sum.csv")
    # visualize_llc(llc_sum, args)
    visualize_llc_v2(llc_sum, args)


def draw_B_N(features, labels, args):
    llc_sum = []
    for k in args.k:
        b_llc_sum, non_paired_llc_sum = calculate_llc_core(features, labels, args, k=k, llc_type=args.llc_type)
        one_k_llc_sum = {"k": k,
                         "before_func_llc_avg": b_llc_sum,
                         "non_paired_llc_avg": non_paired_llc_sum}
        llc_sum.append(one_k_llc_sum)
        # print("one_k_llc_sum:", one_k_llc_sum)
    llc_sum = pd.DataFrame(llc_sum)
    print("\n\nargs.llc_type:", args.llc_type)
    print(llc_sum)
    llc_sum.to_csv("llc_sum.csv")
    # visualize_llc(llc_sum, args)
    visualize_llc_v2(llc_sum, args)


def visualize_llc_v2(llc_sum, args):
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample data
    k = args.k
    # k = args.knn_list
    # before_func_llc_avg = [0.280011, 0.340734, 0.395933, 0.440462, 0.510902, 0.558882, 0.576808, 0.620629, 0.660826,
    #                        0.699580]
    # after_func_llc_avg = [0.105155, 0.136039, 0.153948, 0.153793, 0.169837, 0.164050, 0.176282, 0.174597, 0.193025,
    #                       0.192185]
    # non_paired_llc_avg = [0.010759, 0.014116, 0.017781, 0.019791, 0.019222, 0.020002, 0.019664, 0.020493, 0.020991,
    #                       0.021628]

    before_func_llc_avg = llc_sum['before_func_llc_avg'].tolist()
    after_func_llc_avg = llc_sum['after_func_llc_avg'].tolist()
    non_paired_llc_avg = llc_sum['non_paired_llc_avg'].tolist()

    # Custom scaling function for y-axis
    def custom_scale(y):
        return np.where(y < 2.25, y * 10, y)
        # return np.where(y < 0.1, y * 50, np.where(y <= 0.50, y * 10 + 0.2, y ))


    # Apply the custom scaling function
    before_func_llc_avg_scaled = custom_scale(np.array(before_func_llc_avg))
    after_func_llc_avg_scaled = custom_scale(np.array(after_func_llc_avg))
    non_paired_llc_avg_scaled = custom_scale(np.array(non_paired_llc_avg))

    # Plotting the data with custom scaled y-axis
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(k, before_func_llc_avg_scaled, marker='o', label='Before Function LLC Avg')
    ax.plot(k, after_func_llc_avg_scaled, marker='s', label='After Function LLC Avg')
    ax.plot(k, non_paired_llc_avg_scaled, marker='d', label='Non-Paired LLC Avg', color='green')



    custom_ticks = np.concatenate([np.linspace(27.4, 45.2, 5), np.linspace(0.783, 1.473, 3), np.linspace(1.307, 2.205, 3)])
    custom_tick_labels = [f'{tick:.3f}' for tick in np.linspace(27.4, 45.2, 5)] + [f'{tick:.3f}' for tick in
                                                                                    np.linspace(0.783,1.473, 3)] + [
                             f'{tick:.3f}' for tick in
                             np.linspace(1.307, 2.205, 3)]

    ax.set_yticks(custom_scale(custom_ticks))
    ax.set_yticklabels(custom_tick_labels, fontsize=30)

    ax.set_xticks(k)
    ax.set_xticklabels([str(item) for item in k], fontsize=30)

    # Add labels and title
    # ax.set_xlabel('K', fontsize=45)
    # ax.set_ylabel('LLC Average', fontsize=30)
    # ax.set_title('LLC Averages with Custom Scaled Y-Axis')
    # ax.legend()
    plt.grid(False)

    # 显示图表
    plt.tight_layout()
    if args.option == 3:
        file_name = 'DeepDFA_llc_diversevul'
    elif args.option == 2:
        file_name = 'DeepDFA_llc_msr'
    plt.savefig(os.path.join(args.output_dir, file_name + ".png"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, file_name + ".eps"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, file_name + ".svg"), dpi=300, format='svg')
    plt.show()



def visualize_llc(df, args):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    data = {
        'before_func_llc_avg': [0.280011, 0.340734, 0.395933, 0.440462, 0.510902, 0.558882, 0.576808, 0.620629,
                                0.660826, 0.699580],
        'after_func_llc_avg': [0.105155, 0.136039, 0.153948, 0.153793, 0.169837, 0.164050, 0.176282, 0.174597, 0.193025,
                               0.192185],
        'non_paired_llc_avg': [0.010759, 0.014116, 0.017781, 0.019791, 0.019222, 0.020002, 0.019664, 0.020493, 0.020991,
                               0.021628]
    }

    x = np.arange(1, len(data['before_func_llc_avg']) + 1)

    def custom_scale(y):
        return np.where(y < 0.022, y, y)

    before_func_llc_avg_scaled = custom_scale(np.array(data['before_func_llc_avg']))
    after_func_llc_avg_scaled = custom_scale(np.array(data['after_func_llc_avg']))
    non_paired_llc_avg_scaled = custom_scale(np.array(data['non_paired_llc_avg']))

    fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [4, 2, 2]})

    fontsize = 20

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.plot(x, before_func_llc_avg_scaled, label='Before Func LLC Avg', marker='o', color='blue')
    ax.set_ylim(0.25, 0.8)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)

    ax2.plot(x, after_func_llc_avg_scaled, label='After Func LLC Avg', marker='s', color='orange')
    ax2.set_ylim(0.09, 0.20)
    ax2.spines['bottom'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax2.xaxis.tick_top()
    ax2.tick_params(labeltop=False)

    ax3.plot(x, non_paired_llc_avg_scaled, label='Non Paired LLC Avg', marker='s', color='green')
    ax3.set_ylim(0.01, 0.022)
    ax3.xaxis.tick_bottom()

    d = .005  # how big to make the horizontal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (0, 0), **kwargs)
    ax.plot((-d, +d), (0, -0.025), **kwargs)
    ax.plot((-d, +d), (-0.05, -0.025), **kwargs)
    ax.plot((-d, +d), (-0.05, -0.075), **kwargs)
    ax.plot((-d, +d), (-0.10, -0.075), **kwargs)
    ax.plot((-d, +d), (-0.10, -0.125), **kwargs)
    ax.plot((1 - d, 1 + d), (0, 0), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1, 1), **kwargs)
    ax2.plot((1 - d, 1 + d), (1, 1 + 0.05), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 + 0.1, 1 + 0.05), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 + 0.1, 1 + 0.15), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 + 0.20, 1 + 0.15), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 + 0.20, 1 + 0.25), **kwargs)
    ax2.plot((1 - d, 1 + d), (1, 1), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (0, 0), **kwargs)
    ax2.plot((-d, +d), (0, -0.05), **kwargs)
    ax2.plot((-d, +d), (-0.10, -0.05), **kwargs)
    ax2.plot((-d, +d), (-0.10, -0.15), **kwargs)
    ax2.plot((-d, +d), (-0.20, -0.15), **kwargs)
    ax2.plot((-d, +d), (-0.20, -0.25), **kwargs)
    ax2.plot((1 - d, 1 + d), (0, 0), **kwargs)

    kwargs.update(transform=ax3.transAxes)
    ax3.plot((-d, +d), (1, 1), **kwargs)
    ax3.plot((1 - d, 1 + d), (1, 1 + 0.05), **kwargs)
    ax3.plot((1 - d, 1 + d), (1 + 0.1, 1 + 0.05), **kwargs)
    ax3.plot((1 - d, 1 + d), (1 + 0.1, 1 + 0.15), **kwargs)
    ax3.plot((1 - d, 1 + d), (1 + 0.20, 1 + 0.15), **kwargs)
    ax3.plot((1 - d, 1 + d), (1 + 0.20, 1 + 0.25), **kwargs)
    ax3.plot((1 - d, 1 + d), (1, 1), **kwargs)

    ax3.set_xticks(np.arange(1, len(x) + 1, 1))  # Example: showing every second tick
    plt.subplots_adjust(hspace=0.20)
    # plt.legend()
    # Set y-ticks for each subplot
    ax.set_yticks(np.arange(0.25, 0.85, 0.15))  # Example y-ticks for ax
    ax2.set_yticks(np.arange(0.09, 0.21, 0.05))  # Example y-ticks for ax2
    ax3.set_yticks(np.arange(0.01, 0.023, 0.005))  # Example y-ticks for ax3

    # 显示图表
    plt.tight_layout()
    if args.option == 3:
        file_name = 'llc_diversevul'
    elif args.option == 2:
        file_name = 'llc_msr'
    plt.savefig(os.path.join(args.output_dir, file_name +".png"), dpi = 300)
    plt.savefig(os.path.join(args.output_dir, file_name + ".eps"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, file_name + ".svg"), dpi=300, format='svg')
    plt.show()

    # 打印DataFrame以供参考
    print(df)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()


    option = 3
    args.option = option
    args.k = [1, 2, 3, 4,5, 6, 7, 8, 9, 10]
    if args.option == 2:
        input_dir = r'D:\Code\DeepDFA'
        args.input_dir = input_dir
        target_file = ['DeepDFA_on_MSR_last_second_layer_target_0_test.npy.npz_test.csv_setting_6.npz',
                       'DeepDFA_on_MSR_last_second_layer_target_0_and_1_func_before_add_after.npy.npz_setting_6_add_after.npz']
        args.output_dir = os.path.join(args.input_dir, 'llc')
        args.dataset = 'MSR'
        args.output_dir = os.path.join(args.output_dir, 'MSR')
        print("args.output_dir:", args.output_dir)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    elif args.option == 3:
        input_dir = r'D:\Code\DeepDFA'
        args.input_dir = input_dir
        target_file = ['DeepDFA_on_Diversevul_last_second_layer_target_0_and_1_func_before_test_add_after.npy.npz',
                       'DeepDFA_on_Diversevul_last_second_layer_target_0_and_1_func_before_test_add_after.npy.npz']
        args.output_dir = os.path.join(args.input_dir, 'llc')
        args.dataset = 'Diversevul'
        args.output_dir = os.path.join(args.output_dir, args.dataset )
        print("args.output_dir:", args.output_dir)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)


    # split_group_ids()
    # split_target_cross_project()
    # split_target()
    # visual_test()

    # visualize()
    # visualize_add_similar_func()
    # combine_two_figures()
    # calculate_llc(args)

    llc_sum = pd.read_csv(args.dataset+"_llc_sum.csv")
    visualize_llc_v2(llc_sum, args)



