import argparse
import copy
import json
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter


def draw_loss(args):
    loss_df = pd.read_csv(args.input_file)
    df = loss_df
    df['batch_id'] = df['tr_num']
    df.reset_index(inplace=True)
    df['batch_id'] = df.index

    # 设置滚动窗口大小
    window_size = args.window_size

    # 计算滚动平均线
    df['smoothed_batch_loss'] = df['batch_loss'].rolling(window=window_size).mean()

    # 绘制 batch_loss 和 smoothed_batch_loss 的折线图
    # plt.figure(figsize=(30, 6))  # 增加图形的宽度
    # plt.scatter(df['batch_id'], df['smoothed_batch_loss'], marker='.', s =10, label='Smoothed Batch Loss', color='orange')
    # plt.title('Batch Loss Fluctuations with Smoothing')
    # plt.xlabel('Batch ID')
    # plt.ylabel('Batch Loss')
    # plt.title(args.MSR_processing_flag)
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join(args.output_dir_setting, 'loss_visualization.png'))
    # # plt.show()

    return df


from mpl_toolkits.axes_grid1.inset_locator import inset_axes




def set_scientific_notation(ax, num_ticks=5):
    # for axis in [ax.xaxis, ax.yaxis]:
    for axis in [ax.xaxis]:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        axis.set_major_formatter(formatter)
        if num_ticks != 5 :
            axis.set_major_locator(MaxNLocator(num_ticks))

        # Set font size for the scientific notation text
        axis.get_offset_text().set_fontsize(fontsize = 25)


def draw_combine_setting(all_loss_df):
    fig, ax = plt.subplots(figsize=(15, 8))  # 增加图形的宽度和高度
    colors = ['blue', 'green','red', 'purple', 'orange', 'black']
    setting_list = [ 'add_after_into_before_append', 'add_after_into_before_shuffle', 'original_save_loss']

    df_epoch_0_length = 0
    df_length = 0
    for i, setting in enumerate(setting_list):
        df = all_loss_df[setting]
        df = df.iloc[0:-303]
        df_copy = copy.deepcopy(df)
        if "original" not in setting:
            df = df[df.epoch <= 8]
            df_add_after_length = len(df)
            df_epoch_0_length = len(df[df.epoch == 0])
        else:
            df = df.head(df_add_after_length)
        df_length = len(df)

        # ax.scatter(df['batch_id'], df['smoothed_batch_loss'], marker='.', s=1, label=setting, color=colors[i])
        ax.plot(df['batch_id'], df['smoothed_batch_loss'], '-', linewidth=1, color=colors[i])

    # ax.set_title('Batch Loss Fluctuations with Smoothing')
    ax.set_xlabel('Batch ID', fontsize=25)
    ax.set_ylabel('Batch Loss', fontsize=25)
    ax.legend()
    ax.grid(True, linestyle='--')
    set_scientific_notation(ax, num_ticks = 11)
    # 添加第一个局部放大图
    axins1 = inset_axes(ax, width="40%", height="30%", loc='lower left',
                        bbox_to_anchor=(0.1, 0.65, 1, 1), bbox_transform=ax.transAxes)

    # 设置第一个要放大的 Batch ID 范围
    x1, x2 = df_epoch_0_length  - 700, df_epoch_0_length + 500
    y1, y2 = -0.02, 0.22

    # 添加第二个局部放大图
    axins2 = inset_axes(ax, width="40%", height="30%", loc='lower right',
                        bbox_to_anchor=(0.0, 0.65, 1, 1), bbox_transform=ax.transAxes)

    # 设置第二个要放大的 Batch ID 范围
    x3, x4 = df_length  - 700, df_length  # 这里设置第二个放大区域的 Batch ID 范围
    y3, y4 = -0.02, 0.32  # 这里设置第二个放大区域的 Loss 范围


    # setting_list = ['original_save_loss', 'add_after_into_before_append', 'add_after_into_before_shuffle' ]

    for i, setting in enumerate(setting_list):
        df = all_loss_df[setting]
        df = df.iloc[0:-303]
        if "original" not in setting:
            df = df[df.epoch <= 8]
            df_add_after_length = len(df)
            df_epoch_0_length = len(df[df.epoch == 0])
        else:
            df = df.head(df_add_after_length)

        # 为第一个子图绘制数据
        df_zoomed1 = df[(df['batch_id'] >= x1) & (df['batch_id'] <= x2)]
        # axins1.scatter(df_zoomed1['batch_id'], df_zoomed1['smoothed_batch_loss'],
        #                marker='.', s=1, color=colors[i])
        axins1.plot(df_zoomed1['batch_id'], df_zoomed1['smoothed_batch_loss'], '-',
                    linewidth=1, color=colors[i])

        # 为第二个子图绘制数据
        df_zoomed2 = df[(df['batch_id'] >= x3) & (df['batch_id'] <= x4)]
        # axins2.scatter(df_zoomed2['batch_id'], df_zoomed2['smoothed_batch_loss'],
        #                marker='.', s=1, color=colors[i])
        axins2.plot(df_zoomed2['batch_id'], df_zoomed2['smoothed_batch_loss'], '-',
                    linewidth=1, color=colors[i])


    # 设置第一个子图的范围和标签
    axins1.set_xlim(x1, x2)
    axins1.set_ylim(y1, y2)
    set_scientific_notation(axins1)
    axins1.set_xlabel('Batch ID', fontsize=25)
    axins1.set_ylabel('Batch Loss', fontsize=25)
    axins1.tick_params(axis='both', which='major', labelsize=25)
    axins1.xaxis.set_major_locator(plt.MaxNLocator(5))
    axins1.yaxis.set_major_locator(plt.MaxNLocator(5))

    # 设置第二个子图的范围和标签
    axins2.set_xlim(x3, x4)
    axins2.set_ylim(y3, y4)
    set_scientific_notation(axins2)
    axins2.set_xlabel('Batch ID', fontsize=25)
    axins2.set_ylabel('Batch Loss', fontsize=25)
    axins2.tick_params(axis='both', which='major', labelsize=25)
    axins2.xaxis.set_major_locator(plt.MaxNLocator(5))
    axins2.yaxis.set_major_locator(plt.MaxNLocator(5))

    # 在主图上标记放大区域
    ax.indicate_inset_zoom(axins1, edgecolor="black", linewidth=2)
    ax.indicate_inset_zoom(axins2, edgecolor="black", linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=25)
    title = 'three_settings_training_stage_loss'
    # plt.suptitle(title)
    plt.tight_layout()

    plt.savefig(os.path.join(args.output_dir_setting, title + '.png'),bbox_inches='tight', dpi = 300)
    plt.savefig(os.path.join(args.output_dir_setting, title + '.svg'),bbox_inches='tight', dpi = 300)
    plt.savefig(os.path.join(args.output_dir_setting, title + '.eps'),bbox_inches='tight', dpi = 300)
    import pickle
    with open(os.path.join(args.output_dir_setting, title + '.pkl'), 'wb') as file:
        pickle.dump(all_loss_df, file)

    plt.show()


def draw_combine_setting_bak(all_loss_df):
    plt.figure(figsize=(12, 6))  # 增加图形的宽度
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black']
    setting_list = ['original_save_loss', 'add_after_into_before_append', 'add_after_into_before_shuffle']

    for i in range(len(setting_list)):
        setting = setting_list[i]
        df  = all_loss_df[setting]
        df = df.iloc[0:-303]
        df = df[df.epoch <=8]
        plt.scatter(df['batch_id'], df['smoothed_batch_loss'], marker='.', s=10, label=setting_list[i],
                    color=colors[i])



    title = 'three_settings_training_stage'
    plt.title('Batch Loss Fluctuations with Smoothing')
    plt.xlabel('Batch ID')
    plt.ylabel('Batch Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir_setting, title+'.png'), bbox_inches='tight', dpi = 100)
    plt.savefig(os.path.join(args.output_dir_setting, title + '.svg'), bbox_inches='tight', dpi=100)
    plt.savefig(os.path.join(args.output_dir_setting, title + '.eps'), bbox_inches='tight', dpi=100)
    plt.show()



def draw_combine_setting_only_last_epoch(all_loss_df):

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black']
    setting_list = ['original_save_loss', 'add_after_into_before_append', 'add_after_into_before_shuffle']

    add_after_into_before_append = all_loss_df['add_after_into_before_append']
    df_epoch = add_after_into_before_append['epoch'].unique().tolist()
    for e in range(len(df_epoch)):
        plt.figure(figsize=(18, 6))  # 增加图形的宽度
        for i in range(len(setting_list)):
            setting = setting_list[i]
            df = all_loss_df[setting]
            df_e = df[df['epoch'] == e]
            df_e_copy = copy.deepcopy(df_e)
            df_e_copy.reset_index(drop=True, inplace=True)
            df_e_copy['batch_id'] = df_e_copy.index
            plt.scatter(df_e_copy['batch_id'], df_e_copy['smoothed_batch_loss'], marker='.', s=10, label=setting_list[i] + "_epoch:" + str(e),
                        color=colors[i])

        title = "three_setting_training_dataset_training_stage_epoch_" + str(e)
        plt.xlabel('Batch ID')
        plt.ylabel('Batch Loss')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir_setting, title+'.png'))
        plt.show()


def get_test_stage_loss(args):
    df = pd.read_csv(args.input_file_test_stage)
    df.reset_index(drop=True, inplace=True)
    df['batch_id'] = df.index
    # 设置滚动窗口大小
    window_size = args.window_size
    df['batch_loss'] = df['all_test_loss']
    # 计算滚动平均线
    df['smoothed_batch_loss'] = df['batch_loss'].rolling(window=window_size).mean()
    return df


def visualize_test_stage_loss(all_loss_df_test_stage):
    args.model_shuffle_test_append = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/binary_category/check_result_of_after_before/add_after_into_before_shuffle/visualize_loss_v2/batch_loss_training_dataset/add_after_into_before_shufflebatch_loss_9_test_model_shuffle_test_append.csv'
    args.input_file_test_stage = args.model_shuffle_test_append
    model_shuffle_test_data_append = get_test_stage_loss(args)



    plt.figure(figsize=(18, 6))  # 增加图形的宽度
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black']
    setting_list = ['original_save_loss', 'add_after_into_before_append', 'add_after_into_before_shuffle']

    for i in range(len(setting_list)):
        setting = setting_list[i]
        df  = all_loss_df_test_stage[setting]
        plt.scatter(df['batch_id'], df['smoothed_batch_loss'], marker='.', s=10, label=setting_list[i],
                    color=colors[i])

    plt.scatter(model_shuffle_test_data_append['batch_id'], model_shuffle_test_data_append['smoothed_batch_loss'], marker='.', s=10, label='model_shuffle_test_data_append',
                color=colors[i+1])
    title = "three_settings_test_dataset_test_stage_loss_epoch:" + str(args.epoch_choose)
    plt.xlabel('Batch ID')
    plt.ylabel('Batch Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir_setting, title + '.png'))
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    option = 1
    if option == 1:
        output_middle_dir = 'batch_loss_training_dataset'
    elif option == 2:
        output_middle_dir = 'batch_loss'


    args.input_dir = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/binary_category/check_result_of_after_before'
    args.MSR_processing_flag = 'original_save_loss'
    args.epoch_choose = '9'
    # args.MSR_processing_flag = 'add_after_into_before_append'
    # args.MSR_processing_flag = 'add_after_into_before_shuffle'
    all_loss_df = {}
    all_loss_df_test_stage = {}
    args.window_size = 300
    for setting in ['original_save_loss', 'add_after_into_before_append', 'add_after_into_before_shuffle']:
        args.MSR_processing_flag = setting
        args.input_dir_setting = os.path.join(args.input_dir, args.MSR_processing_flag)
        args.output_dir_setting = os.path.join(args.input_dir_setting, 'visualize_loss_v2', output_middle_dir)
        args.input_file = os.path.join(args.output_dir_setting, setting+'batch_loss_'+args.epoch_choose+'.csv')
        args.input_file_test_stage = os.path.join(args.output_dir_setting, setting+'batch_loss_'+args.epoch_choose+'_test.csv')
        loss_df = draw_loss(args)
        loss_df_test_stage = get_test_stage_loss(args)
        all_loss_df_test_stage[setting] = loss_df_test_stage
        all_loss_df[setting] = loss_df

    draw_combine_setting(all_loss_df)

    # draw_combine_setting_only_last_epoch(all_loss_df)
    #
    # visualize_test_stage_loss(all_loss_df_test_stage)




