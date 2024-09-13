import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from tqdm import tqdm


def draw_one_group_add_background_similar_func_zoom_in_directly(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id,
                                                                p_number, n_number, layer_number, output_dir='',
                                                                correct_log_1=[], correct_log_0=[], step=-1,
                                                                non_vulnerable_back_ground_tsne=[],
                                                                vulnerable_back_ground_tsne=[], k=0):
    X3_tsne_1 = synthetic_tsne_1_list
    X3_tsne_0 = synthetic_tsne_0_list

    # 创建主图
    fig, ax = plt.subplots(figsize=(12, 8))

    alpha = 1

    ax.scatter(vulnerable_back_ground_tsne[::10, 0], vulnerable_back_ground_tsne[::10, 1],
               color='red', marker='o', alpha=alpha, s=20)
    ax.scatter(non_vulnerable_back_ground_tsne[::10, 0], non_vulnerable_back_ground_tsne[::10, 1],
               color='green', marker='o', alpha=alpha, s=20)

    # 绘制第一个列表的折线图
    for i in tqdm(range(len(correct_log_1) - 1, -1, step), desc="correct_log_1 Plotting Progress"):
        color = 'purple' if correct_log_1[i] else 'red'
        color = 'purple'
        ax.scatter(X3_tsne_1[i, 0], X3_tsne_1[i, 1], facecolors='none', edgecolors=color, marker='s', alpha=alpha)

        if k in ['179281', 179281]:
            if i == 0:  # Only add text for the first plotted point
                ax.text(X3_tsne_1[i, 0], X3_tsne_1[i, 1] + 0.05, k, fontsize=15, ha='right', va='bottom')

            if i == p_number + 1:  # Only add text for the first plotted point
                ax.text(X3_tsne_1[i, 0] + 1, X3_tsne_1[i, 1] + 0.05, '179639', fontsize=15, ha='right', va='bottom')
                rect = patches.Rectangle((4.5, 8.7), 6.6 - 4.5, 9.5 - 8.7, linewidth=1, edgecolor='gray',
                                         )
                ax.add_patch(rect)
        elif k in ['179639', 179639]:
            if i == 0:  # Only add text for the first plotted point
                ax.text(X3_tsne_1[i, 0] + 1, X3_tsne_1[i, 1] + 0.15, k, fontsize=15, ha='right', va='bottom')

                # 在图像上的指定位置添加灰色矩形框
                rect = patches.Rectangle((4.5, 8.7), 6.6 - 4.5, 9.5 - 8.7, linewidth=1, edgecolor='gray',
                                         )
                ax.add_patch(rect)

    for i in tqdm(range(len(correct_log_0) - 1, -1, step), desc="correct_log_0 Plotting Progress"):
        color = 'green' if correct_log_0[i] else 'black'
        color = 'black'
        ax.scatter(X3_tsne_0[i, 0], X3_tsne_0[i, 1], facecolors='none', edgecolors=color, marker='*', alpha=alpha)

    ax.scatter([], [], facecolors='none', edgecolors='purple', marker='s', alpha=alpha, label="before fix function")
    ax.scatter([], [], facecolors='none', edgecolors='black', marker='*', alpha=alpha, label="after fix function")
    ax.scatter([], [], color='red', marker='o', alpha=alpha, label="vulnerable function")
    ax.scatter([], [], color='green', marker='o', alpha=alpha, label="non-vulnerable function")

    # 设置子图的显示范围，放大主图的局部区域，确保覆盖179281附近
    x1, x2, y1, y2 = 0, 2, 0, 2  # 调整这些值以准确放大179281附近的区域

    # 添加子图到主图外部右侧
    axins = inset_axes(ax, width="50%", height="50%", bbox_to_anchor=(1.05, 0.5, 1, 1), bbox_transform=ax.transAxes,
                       loc="center left")

    axins.scatter(vulnerable_back_ground_tsne[::10, 0], vulnerable_back_ground_tsne[::10, 1],
                  color='red', marker='o', alpha=alpha, s=20)
    axins.scatter(non_vulnerable_back_ground_tsne[::10, 0], non_vulnerable_back_ground_tsne[::10, 1],
                  color='green', marker='o', alpha=alpha, s=20)

    for i in range(len(correct_log_1)):
        color = 'purple' if correct_log_1[i] else 'red'
        color = 'purple'
        axins.scatter(X3_tsne_1[i, 0], X3_tsne_1[i, 1], facecolors='none', edgecolors=color, marker='s', alpha=alpha)
    for i in range(len(correct_log_0)):
        color = 'green' if correct_log_0[i] else 'black'
        color = 'black'
        axins.scatter(X3_tsne_0[i, 0], X3_tsne_0[i, 1], facecolors='none', edgecolors=color, marker='*', alpha=alpha)

    # 在子图中添加注释
    axins.annotate('Zoomed Area', xy=(0, 0), xytext=(1, 1), fontsize=15, ha='right', va='bottom',
                   arrowprops=dict(facecolor='black', arrowstyle='->'))

    # 设置子图的坐标轴标签和刻度
    axins.tick_params(axis='both', which='major', labelsize=10)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    def format_func(value, tick_number):
        return f'{value:.5f}'

    axins.xaxis.set_major_formatter(FuncFormatter(format_func))
    axins.yaxis.set_major_formatter(FuncFormatter(format_func))

    # 高亮主图中的被放大区域
    rect_highlight = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='gray', facecolor='none')
    ax.add_patch(rect_highlight)

    # 用灰色的线将被放大区域与子图联系起来
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray")

    # 设置主图属性
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlim([-8.0, 11.0])
    ax.set_ylim([-8.0, 11.0])
    ax.set_title('zoom in directly together')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=15)

    # 保存图像
    output_dir = os.path.join(output_dir, "layer_number_" + str(layer_number))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    total_similar_func = p_number + n_number
    total_similar_func = str(total_similar_func)
    picture_name = f"test_Group_id_{group_id}_produced_similar_func_{total_similar_func}_layer_number_{layer_number}_added_background_zoom_in_directly_" + str(
        x1) + "_" + str(x2) + "_" + str(y1) + "_" + str(y2)
    fig.savefig(os.path.join(output_dir, picture_name + ".png"), bbox_inches='tight', dpi=100)
    fig.savefig(os.path.join(output_dir, picture_name + ".eps"), bbox_inches='tight', dpi=100)
    fig.savefig(os.path.join(output_dir, picture_name + ".pdf"), bbox_inches='tight', dpi=100)
    fig.savefig(os.path.join(output_dir, picture_name + ".svg"), bbox_inches='tight', dpi=100, format='svg')
    plt.show()


# 示例数据（请根据实际数据替换这些变量）
synthetic_tsne_1_list = np.random.rand(100, 2) * 10 - 5
synthetic_tsne_0_list = np.random.rand(100, 2) * 10 - 5
group_id = 1
p_number = 50
n_number = 50
layer_number = 1
output_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_picture/visualize_produced_similar_func/layer_number_15'
correct_log_1 = [True] * 100
correct_log_0 = [False] * 100
step = 1
non_vulnerable_back_ground_tsne = np.random.rand(1000, 2) * 10 - 5
vulnerable_back_ground_tsne = np.random.rand(1000, 2) * 10 - 5
k = 179281

# 调用函数
draw_one_group_add_background_similar_func_zoom_in_directly(synthetic_tsne_1_list, synthetic_tsne_0_list, group_id,
                                                            p_number, n_number, layer_number, output_dir,
                                                            correct_log_1, correct_log_0, step,
                                                            non_vulnerable_back_ground_tsne,
                                                            vulnerable_back_ground_tsne, k)
