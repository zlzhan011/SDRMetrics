import argparse
import os

import pandas as pd


def read_chen_files(args):

    chen_function_content = []
    i = 0
    for file in os.listdir(args.chen_dir):
        file_path = os.path.join(args.chen_dir, file)
        if 'before' in file_path:
            target = 1
        else:
            target = 0
        with open(file_path, 'r') as f:
            function_content = f.read()
            print(function_content)

            chen_function_content.append({"index":i, "file_name": file, "function_content": function_content, 'target':target})
        i = i + 1
    chen_function_content_df = pd.DataFrame(chen_function_content)
    chen_function_content_df.to_csv(os.path.join(args.chen_dir_root, "chen_function_content.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('chen_dir', default='D:\Research\paired_function\Dr_Chen_task\paired-completed')
    args = parser.parse_args()
    args.chen_dir_root = r'/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/linevul/binary_category/check_result_of_after_before/chen_files/'
    args.chen_dir = os.path.join(args.chen_dir_root, 'paired-completed')
    read_chen_files(args)

