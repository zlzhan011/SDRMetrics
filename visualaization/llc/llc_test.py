import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
from matplotlib.ticker import MultipleLocator, FuncFormatter

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
import copy
import os
#
#
# def visualize_llc(df, args):
#     df_copy = copy.deepcopy(df)
#
#     df = df[['before_func_llc_avg', 'after_func_llc_avg', 'non_paired_llc_avg']]
#
#     # 创建图表
#     plt.figure(figsize=(12, 6))
#     labels = ['before-fixed', 'after-fixed', 'non-paired']
#
#     # 为每一列创建一条折线
#     i = 0
#     for column in df.columns:
#         plt.plot(df.index, df[column], label=labels[i], marker='o')
#         i += 1
#
#     # 设置图表标题和轴标签
#     plt.xlabel('K', fontsize=40)
#     plt.ylabel('Average -In(LLC)', fontsize=40)
#
#     # 显示网格
#     plt.grid(True, alpha=0.2)
#
#     # 调整Y轴的范围，使图表更清晰
#     plt.ylim(0, max(df.max()) * 1.1)
#
#     # 设置 x 轴刻度
#     x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     x_ticks_v = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
#     plt.xticks(x_ticks, x_ticks_v)
#
#     # 设置 y 轴刻度
#     major_y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     minor_y_ticks = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
#     plt.yticks(major_y_ticks)
#
#     # 添加次要刻度
#     ax = plt.gca()
#     ax.yaxis.set_minor_locator(FixedLocator(minor_y_ticks))
#
#     plt.tick_params(axis='x', which='major', labelsize=40)
#     plt.tick_params(axis='y', which='major', labelsize=40)
#     plt.tick_params(axis='y', which='minor', labelsize=20)  # 次要刻度标签大小
#
#     # 显示图表
#     plt.tight_layout()
#     if args.option == 3:
#         file_name = 'llc_diversevul'
#     elif args.option == 2:
#         file_name = 'llc_msr'
#
#     plt.savefig(os.path.join(args.output_dir, file_name + ".png"), dpi=300)
#     plt.savefig(os.path.join(args.output_dir, file_name + ".eps"), dpi=300)
#     plt.savefig(os.path.join(args.output_dir, file_name + ".svg"), dpi=300, format='svg')
#     plt.show()
#
#     # 打印DataFrame以供参考
#     print(df)
#
#
# # 假设你有一个主函数或脚本部分来调用这个函数
# if __name__ == "__main__":
#     # 创建一个示例DataFrame（你需要用实际数据替换这个）
#     df = pd.DataFrame({
#         'before_func_llc_avg': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#         'after_func_llc_avg': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
#         'non_paired_llc_avg': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
#     })
#
#
#     # 创建一个参数对象（你需要根据实际情况调整这个）
#     class Args:
#         def __init__(self):
#             self.option = 3
#             self.output_dir = '.'
#
#
#     args = Args()
#
#     # 调用函数
#     visualize_llc(df, args)



import pandas as pd
import matplotlib.pyplot as plt



def visualize_llc_bak_good(df, args):
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample data
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    before_func_llc_avg = [0.280011, 0.340734, 0.395933, 0.440462, 0.510902, 0.558882, 0.576808, 0.620629, 0.660826,
                           0.699580]
    after_func_llc_avg = [0.105155, 0.136039, 0.153948, 0.153793, 0.169837, 0.164050, 0.176282, 0.174597, 0.193025,
                          0.192185]
    non_paired_llc_avg = [0.010759, 0.014116, 0.017781, 0.019791, 0.019222, 0.020002, 0.019664, 0.020493, 0.020991,
                          0.021628]

    # Custom scaling function for y-axis
    def custom_scale(y):
        return np.where(y < 0.022, y * 10, y + 0.22)

    # Apply the custom scaling function
    before_func_llc_avg_scaled = custom_scale(np.array(before_func_llc_avg))
    after_func_llc_avg_scaled = custom_scale(np.array(after_func_llc_avg))
    non_paired_llc_avg_scaled = custom_scale(np.array(non_paired_llc_avg))




    # Plotting the data with custom scaled y-axis
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k, before_func_llc_avg_scaled, marker='o', label='Before Function LLC Avg')
    ax.plot(k, after_func_llc_avg_scaled, marker='s', label='After Function LLC Avg')
    ax.plot(k, non_paired_llc_avg_scaled, marker='d', label='Non-Paired LLC Avg', color='green')

    custom_ticks = np.concatenate([np.linspace(0.01, 0.022, 2),
                                   np.linspace(0.1, 0.2, 2),
                                   np.linspace(0.2, 0.7, 5)])
    custom_tick_labels = ([f'{tick:.3f}' for tick in np.linspace(0.01, 0.022, 2)] +
                          [f'{tick:.3f}' for tick in np.linspace(0.1, 0.2, 2)] +
                          [ f'{tick:.1f}' for tick in np.linspace(0.2, 0.7, 5)])

    ax.set_yticks(custom_scale(custom_ticks))
    ax.set_yticklabels(custom_tick_labels, fontsize=45)

    ax.set_xticks(k)
    ax.set_xticklabels([str(item) for item in k], fontsize=45)

    # Add labels and title
    ax.set_xlabel('K', fontsize=45)
    ax.set_ylabel('LLC Average', fontsize=40)
    # ax.set_title('LLC Averages with Custom Scaled Y-Axis')
    # ax.legend()
    plt.grid(True, alpha=0.2)

    # 显示图表
    plt.tight_layout()
    if args.option == 3:
        file_name = 'llc_diversevul'
    elif args.option == 2:
        file_name = 'llc_msr'
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
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 显示图表
    plt.tight_layout()
    if args.option == 3:
        file_name = 'llc_diversevul'
    elif args.option == 2:
        file_name = 'llc_msr'
    plt.savefig(os.path.join(args.output_dir, file_name + ".png"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, file_name + ".eps"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, file_name + ".svg"), dpi=300, format='svg')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    option = 2
    args.option = option
    if option == 1:

        input_dir = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/cache_visual_test/'
        input_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_cache/'
        args.input_dir = input_dir

        target_file_list = ['multi_msr_test_target_1_last_second_layer.npy.npz',
                            'multi_msr_test_target_0_last_second_layer.npy.npz']
        # multi_target_1_last_second_layer.npy.npz must at the first, multi_target_0_last_second_layer.npy.npz must at the second

        target_file = target_file_list
        print("target_file:", target_file)
        file = target_file[0]
    elif option == 2:
        input_dir = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/visual_cache/'
        args.input_dir = input_dir
        target_file = ['test.csvfunc_before_last_second_layer_index_correct.npy.npz',
                       'test.csvfunc_after_last_second_layer_index_correct.npy.npz']
        args.output_dir = os.path.join(args.input_dir, 'llc')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    elif option == 3:
        input_dir = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/binary_category/check_result_of_after_before/diversevul_before_down_resample/10/visualize_loss_v2'
        args.input_dir = input_dir
        target_file = ['test.csvbefore_last_second_layer_index_correct.npy.npz',
                       'test.csvafter_last_second_layer_index_correct.npy.npz']
        args.output_dir = os.path.join(args.input_dir, 'llc')
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    # Data
    data = {
        'before_func_llc_avg': [0.280011, 0.340734, 0.395933, 0.440462, 0.510902, 0.558882, 0.576808, 0.620629,
                                0.660826,
                                0.699580],
        'after_func_llc_avg': [0.105155, 0.136039, 0.153948, 0.153793, 0.169837, 0.164050, 0.176282, 0.174597, 0.193025,
                               0.192185],
        'non_paired_llc_avg': [0.010759, 0.014116, 0.017781, 0.019791, 0.019222, 0.020002, 0.019664, 0.020493, 0.020991,
                               0.021628]
    }
    df = pd.DataFrame(data)
    visualize_llc(df, args)
