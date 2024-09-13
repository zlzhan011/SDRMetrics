import os
import pandas as pd
import os
import time
import torch
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def load_feature(cache_name, step_state):
    np_file = np.load(cache_name, allow_pickle=True)
    feat_log, score_log, label_log, predict_log = np_file['arr_0'], np_file['arr_1'], np_file['arr_2'], np_file['arr_3']
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]
    # feature = feat_log[:, -770:-2]
    correct_lag = [1 if label_log[i] == predict_log[i] else 0 for i in range(len(predict_log))]

    feature = feat_log[:, 769 * (step_state - 1):step_state * 768]
    return feature, correct_lag


def split_target():
    c_root = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/'
    output_dir = os.path.join(c_root, 'visual')
    raw_preds_synthetic = pd.read_csv(os.path.join(c_root, 'devign_test.csv'))
    raw_preds_synthetic = raw_preds_synthetic.head(2000)
    targets = raw_preds_synthetic['target']
    targets = list(set(targets.tolist()))
    for id in targets:
        df_id = raw_preds_synthetic[raw_preds_synthetic.target == id]
        df_id.to_csv(os.path.join(output_dir, "target_" + str(id) + ".csv"))


def visualize_all_data(input_dir):
    devign_file = os.path.join(input_dir, 'multi_devign_test_last_second_layer.npy.npz')
    msr_file = os.path.join(input_dir, 'multi_test_MSR_last_second_layer.npy.npz')
    for step_state in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        msr_feature, msr_correct_lag = load_feature(msr_file, step_state)
        devign_feature, devign_correct_lag = load_feature(devign_file, step_state)
        visual_cross_dataset(devign_feature, msr_feature, step_state,
                             label_1='Devign', label_2='MSR',
                             cross_type='Cross Dataset',
                             only_vulnerable='vuln and non-vuln',
                             out_put_dir=input_dir.replace('cache', 'picture')+"/visualize_all_data_cross_dataset",
                             msr_correct_lag=msr_correct_lag,
                             devign_correct_lag=devign_correct_lag
                        )

def visualize_all_data_cross_project(input_dir):
    for k_fold in range(6):
        dataset_test_file = os.path.join(input_dir,
                                         'multi_fold_' + str(k_fold) + '_dataset_test_last_second_layer.npy.npz')
        holdout_file = os.path.join(input_dir,
                                    'multi_fold_' + str(k_fold) + '_holdout_holdout_last_second_layer.npy.npz')
        for step_state in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            dataset_feature, correct_lag = load_feature(dataset_test_file, step_state)
            holdout_feature, correct_lag = load_feature(holdout_file, step_state)
            step_state = str(step_state) + " k_fold: " + str(k_fold)
            visual_cross_dataset(dataset_feature, holdout_feature, step_state,
                                 label_1='Mix Project', label_2='Cross Project',
                                 cross_type='Cross Project',
                                 only_vulnerable='vuln and non-vuln'
                                 ,out_put_dir=input_dir.replace('cache', 'picture')+"/visualize_all_data_cross_project",
                                 k_fold=k_fold)


def visualize_all_data_four_category(input_dir):
    devign_file = os.path.join(input_dir, 'multi_devign_test_last_second_layer.npy.npz')
    msr_file = os.path.join(input_dir, 'multi_test_MSR_last_second_layer.npy.npz')
    for step_state in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        devign_feature, correct_lag = load_feature(devign_file, step_state)
        msr_feature, correct_lag = load_feature(msr_file, step_state)
        visual_cross_dataset(devign_feature, msr_feature, step_state)


def visualize_only_vulnerable(input_dir):
    devign_file = os.path.join(input_dir, 'multi_devign_test_target_1_last_second_layer.npy.npz')
    msr_file = os.path.join(input_dir, 'multi_msr_test_target_1_last_second_layer.npy.npz')
    for step_state in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        msr_feature, msr_correct_lag = load_feature(msr_file, step_state)
        devign_feature, devign_correct_lag = load_feature(devign_file, step_state)


        visual_cross_dataset(devign_feature, msr_feature, step_state,
                             label_1='Devign', label_2='MSR',
                             cross_type='Cross Dataset',
                             only_vulnerable='only vuln',
                             out_put_dir=input_dir.replace('cache','picture')+"/visualize_only_vulnerable_cross_dataset",
                             msr_correct_lag=msr_correct_lag,
                             devign_correct_lag=devign_correct_lag,
                             step = -1
                             )


def visualize_only_non_vulnerable(input_dir):
    devign_file = os.path.join(input_dir, 'multi_devign_test_target_0_last_second_layer.npy.npz')
    msr_file = os.path.join(input_dir, 'multi_msr_test_target_0_last_second_layer.npy.npz')
    for step_state in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        msr_feature, msr_correct_lag = load_feature(msr_file, step_state)
        devign_feature, devign_correct_lag = load_feature(devign_file, step_state)


        visual_cross_dataset(devign_feature, msr_feature, step_state,
                             label_1='Devign', label_2='MSR',
                             cross_type='Cross Dataset',
                             only_vulnerable='only non-vuln',
                             out_put_dir=input_dir.replace('cache','picture')+"/visualize_only_vulnerable_cross_dataset",
                             msr_correct_lag=msr_correct_lag,
                             devign_correct_lag=devign_correct_lag,
                             step = -10
                             )

def visualize_only_vulnerable_cross_project(input_dir):
    for k_fold in range(6):
        dataset_test_file = os.path.join(input_dir, 'multi_fold_' + str(k_fold) + '_dataset_fold_' + str(
            k_fold) + '_dataset_target_1_last_second_layer.npy.npz')
        holdout_file = os.path.join(input_dir, 'multi_fold_' + str(k_fold) + '_holdout_fold_' + str(
            k_fold) + '_holdout_target_1_last_second_layer.npy.npz')
        # for step_state in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        for step_state in [1, 7, 12,  15]:
            dataset_feature, dataset_correct_lag = load_feature(dataset_test_file, step_state)
            holdout_feature, holdout_correct_lag = load_feature(holdout_file, step_state)
            step_state = str(step_state) + " k_fold: " + str(k_fold)
            visual_cross_dataset(dataset_feature, holdout_feature, step_state,
                                 label_1='Mix Project', label_2='Cross Project',
                                 cross_type='Cross Project',
                                 only_vulnerable='only vuln',
                                 out_put_dir=input_dir.replace('cache',
                                                               'picture') + "/visualize_only_vulnerable_cross_project",
                                 k_fold=k_fold,
                                 msr_correct_lag=dataset_correct_lag,
                                 devign_correct_lag=holdout_correct_lag,
                                 step=-1
                                 )


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
        X1_X2 = (X1.shape[0] + X2.shape[0])
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
    markers = ['o', 's', '^', '<']  # 分别代表圆形、三角形和正方形
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
    title = f"t-SNE: Group_id {group_id} Vulnerable number {p_number} Nonulnerable number {n_number}".format(
        group_id=group_id, p_number=p_number, n_number=n_number)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.show()


def draw_one_group(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id, p_number, n_number):
    # 设定不同的颜色和标记
    # colors = ['orange', 'green', 'blue', 'red']
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']

    # markers = ['o', 's', '^', '<' ]  # 分别代表圆形、三角形和正方形
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

    X3_tsne_1 = synthetic_tsne_1_list
    X3_tsne_0 = synthetic_tsne_0_list
    plt.scatter(X3_tsne_1[:, 0], X3_tsne_1[:, 1], c='red', marker='s', s=150)
    plt.scatter(X3_tsne_0[:, 0], X3_tsne_0[:, 1], c='green', marker="p", s=150)
    # for i in range(len(X3_tsne_1[:, 0])):
    #     plt.text(X3_tsne_1[:, 0][i], X3_tsne_1[:, 1][i]+0.6, str(group_id), fontsize=9, ha='center', va='top')
    #
    # for i in range(len(X3_tsne_0[:, 0])):
    #     plt.text(X3_tsne_0[:, 0][i], X3_tsne_0[:, 1][i]+0.6, str(group_id), fontsize=9, ha='center', va='top')

    # p_number = X3_tsne_1.shape[0]
    # n_number = X3_tsne_0.shape[0]
    title = f"t-SNE: Group_id {group_id} Vulnerable number {p_number} Nonulnerable number {n_number}".format(
        group_id=group_id, p_number=p_number, n_number=n_number)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-10.0, 10.0])
    plt.ylim([-10.0, 10.0])
    plt.title(title)
    plt.legend()
    plt.show()


# 函数：随机抽样并降维
def sample_and_reduce(array, sample_size):
    sample = array[np.random.choice(array.shape[0], sample_size, replace=False), :]
    return TSNE(n_components=2, random_state=42).fit_transform(sample)


def combine_feature_tsne(X1, X2):
    X = np.concatenate([X1, X2])
    method_object = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
    # method_object = PCA(n_components=2)
    # method_object = LDA(n_components=2)
    X_tsne = method_object.fit_transform(X)
    X1_tsne = X_tsne[:X1.shape[0], :]
    X2_tsne = X_tsne[X1.shape[0]:, :]
    return X1_tsne, X2_tsne


def visual_cross_dataset(devign_feature, msr_feature,
                         step_state, label_1='Devign', label_2='MSR',
                         cross_type='cross dataset',
                         only_vulnerable='vuln and non-vuln',
                         out_put_dir='',
                         k_fold='',
                         devign_correct_lag=[],
                         msr_correct_lag=[],
                         step=-10):
    # 示例数据（使用您的实际数据替换）
    array1 = devign_feature
    array2 = msr_feature
    # 随机抽样并降维
    # array1 = sample_and_reduce(array1, 3318)  # 调整样本大小
    # array2 = sample_and_reduce(array2, 18864)
    array1, array2 = combine_feature_tsne(array1, array2)
    print("finish tsne")
    # 绘图
    alpha = 1
    plt.figure(figsize=(12, 8))
    for i in tqdm(range(len(msr_correct_lag)-1,-1,step),desc="msr_correct_lag Plotting Progress"):
        color = 'yellow' if msr_correct_lag[i] else 'red'
        plt.scatter(array2[i, 0], array2[i, 1], facecolors='none', edgecolors=color, marker='^', alpha=alpha)


    for i in tqdm(range(len(devign_correct_lag)-1,-1,step),desc="devign_correct_lag Plotting Progress"):
        color = 'green' if devign_correct_lag[i] else 'black'
        plt.scatter(array1[i, 0], array1[i, 1], facecolors='none', edgecolors=color, marker='o', alpha=alpha)
    print(" start to show")
    # plt.scatter(array1[:, 0], array1[:, 1], facecolors='none', edgecolors='blue', marker='o', alpha=0.3, label=label_1)
    # plt.scatter(array2[:, 0], array2[:, 1], facecolors='none', edgecolors='red', marker='^', alpha=0.3, label=label_2)
    # Adding legend entries
    predict_correct_flag= " Predict Correct"
    predict_error_flag = " Predict InCorrect"
    plt.scatter([], [], facecolors='none', edgecolors='yellow', marker='^', alpha=alpha, label=label_2+predict_correct_flag)
    plt.scatter([], [], facecolors='none', edgecolors='red', marker='^', alpha=alpha,
                label=label_2 + predict_error_flag)
    plt.scatter([], [], facecolors='none', edgecolors='green', marker='o', alpha=alpha, label=label_1+predict_correct_flag)
    plt.scatter([], [], facecolors='none', edgecolors='black', marker='o', alpha=alpha,
                label=label_1 + predict_error_flag)

    plt.legend()
    title = "t-SNE , " + only_vulnerable +" , " + cross_type + " , layer number:" + str(step_state)
    picture_name = "t-SNE_" + only_vulnerable +"_" + cross_type + "_layer number_" + str(step_state)+".png"
    picture_name = picture_name.replace(':','').replace(' ','_')
    plt.title(title)
    if k_fold!='':
        out_put_dir = os.path.join(out_put_dir, str(k_fold))
        if not os.path.isdir(out_put_dir):
            os.mkdir(out_put_dir)
    plt.savefig(os.path.join(out_put_dir, picture_name))
    plt.show()


if __name__ == '__main__':
    input_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_cache/'
    # visual_test()
    # visualize_all_data(input_dir)
    # visualize_only_vulnerable(input_dir)
    # visualize_only_non_vulnerable(input_dir)



    visualize_only_vulnerable_cross_project(input_dir)
    # visualize_all_data_cross_project(input_dir)
