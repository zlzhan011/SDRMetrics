import os
import pandas as pd
import os
import time
import torch
import faiss
import numpy as np
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    np_file = np.load(cache_name, allow_pickle=True)
    feat_log, score_log, label_log = np_file['arr_0'], np_file['arr_1'], np_file['arr_2']
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]
    # feature = feat_log[:, -770:-2]
    feature = feat_log[:, 769 * (step_state - 1):step_state * 768]
    return feature





def visualize():
    input_dir = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/cache_visual_test/'

    # target_file is only the testing dataset file
    target_file = [file for file in os.listdir(input_dir) if 'target' in file]
    print("target_file:", target_file)
    file = target_file[0]

    for layer_number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        target_feature = []
        for file in target_file:
            cache_name = input_dir + file
            feature = load_feature(cache_name, step_state=layer_number)
            target_feature.append(feature)

        file_list = os.listdir(input_dir)
        synthetic_tsne_1_list = []
        synthetic_tsne_0_list = []
        for file in file_list:
            if 'target' not in file: # the synthetic dataset
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
                        feature_1 = load_feature(cache_name_1, step_state=layer_number)
                        feature_0 = load_feature(cache_name_0, step_state=layer_number)
                        # combine the same group feature
                        feature = np.concatenate([feature_1, feature_0])
                        if feature.shape[0] >= 2:
                            print(feature.shape)
                            target_feature.append(feature)
                            X3_tsne_1, X3_tsne_0 = visual_three_vector_v2(target_feature, feature_1)
                            p_number = X3_tsne_1.shape[0]
                            n_number = X3_tsne_0.shape[0]
                            draw_one_group(X3_tsne_1, X3_tsne_0, group_id, p_number, n_number,
                                           layer_number=layer_number,
                                           output_dir='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_in_group')
                            synthetic_tsne_1_list.append(X3_tsne_1)
                            synthetic_tsne_0_list.append(X3_tsne_0)
                            target_feature.pop()

        draw_picture(synthetic_tsne_1_list, synthetic_tsne_0_list,
                     layer_number=layer_number,
                     output_dir='/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_in_group'
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



def draw_one_group(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id, p_number, n_number, layer_number, output_dir=''):
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
    plt.scatter(X3_tsne_1[:, 0], X3_tsne_1[:, 1], c='red',  marker='s',label='vulnerable', s=150)
    plt.scatter(X3_tsne_0[:, 0], X3_tsne_0[:, 1], c='green',  marker="p",label="nonvulnerable", s=150)
    # for i in range(len(X3_tsne_1[:, 0])):
    #     plt.text(X3_tsne_1[:, 0][i], X3_tsne_1[:, 1][i]+0.6, str(group_id), fontsize=9, ha='center', va='top')
    #
    # for i in range(len(X3_tsne_0[:, 0])):
    #     plt.text(X3_tsne_0[:, 0][i], X3_tsne_0[:, 1][i]+0.6, str(group_id), fontsize=9, ha='center', va='top')

    # p_number = X3_tsne_1.shape[0]
    # n_number = X3_tsne_0.shape[0]
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
    plt.savefig(os.path.join(output_dir, picture_name))
    # plt.show()



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
    return  X3_tsne_1, X3_tsne_0

def draw_picture(synthetic_tsne_1_list, synthetic_tsne_0_list, layer_number, output_dir):
    colors = ['blue',  'green', 'red', 'cyan', 'magenta','yellow', 'black','white']

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
    plt.savefig(os.path.join(output_dir, picture_name))

    # plt.show()

if __name__ == '__main__':
    # split_group_ids()
    # split_target_cross_project()
    # split_target()
    # visual_test()

    visualize()
