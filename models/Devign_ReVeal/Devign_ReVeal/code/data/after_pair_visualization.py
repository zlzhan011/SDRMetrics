

import pandas as pd
import os
output_dir = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/MSR/predict_result/MSR'
input_dir = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/MSR'
input_path = os.path.join(input_dir, 'MSR_data_cleaned.csv')
input_df = pd.read_csv(input_path)


# 假设 'file1.csv' 和 'file2.csv' 是你的文件名
# 读取文件
file_path_before = os.path.join(output_dir, 'predict_result.xlsx')
file_path_after = os.path.join(output_dir, 'predict_result_after.xlsx')
df_before = pd.read_excel(file_path_before)
df_after = pd.read_excel(file_path_after)

df_before['func_before_after'] = 'func_before'
df_after['func_before_after'] = 'func_after'
df_before['file_id'] = df_before['files_name'].str.split('_').str[0]
df_after['file_id'] = df_after['files_name'].str.split('_').str[0]
df_before['project_name'] = df_before['files_name'].str.split('_').str[1]
df_after['project_name'] = df_after['files_name'].str.split('_').str[1]
df_before['commit_id'] = df_before['files_name'].str.split('_').str[2]
df_after['commit_id'] = df_after['files_name'].str.split('_').str[2]

input_df = input_df.reset_index().rename(columns={'index': 'file_id'})
input_df['file_id'] = input_df['file_id'].astype(str)
input_df_before = input_df[['file_id', 'func_before']]
df_before= pd.merge(df_before, input_df_before, on='file_id', how='left')
input_df_after = input_df[['file_id', 'func_after']]
df_after= pd.merge(df_after, input_df_after, on='file_id', how='left')

print(len(df_before))
print(len(df_after))
after_before_merge= pd.merge(df_before, df_after, on='file_id', how='left',suffixes=('_before', '_after'))
print(len(after_before_merge))
after_before_merge.to_excel(os.path.join(output_dir, 'after_before_merge.xlsx'))
print(after_before_merge.head())

df_not_null = after_before_merge[pd.notna(after_before_merge['commit_id_after'])]
df_not_null.to_excel(os.path.join(output_dir, 'df_not_null.xlsx'))
df_null  = after_before_merge[pd.isna(after_before_merge['commit_id_after'])]
df_null.to_excel(os.path.join(output_dir, 'df_null.xlsx'))
# # 合并 DataFrame
# combined_before_after = pd.concat([df_before, df_after])
#
# # 假设我们想按照 'ColumnName' 列排序
# # 这里的 'ColumnName' 应替换为你实际想要排序的列名
# sorted_df = combined_before_after.sort_values(by='files_name')
#
# # 显示排序后的 DataFrame
# print(sorted_df)
#
#
# sorted_df['ModifiedColumn'] = sorted_df['files_name'].str.replace('_after_', '').str.replace('_before_', '')
#
# # 找出 'ModifiedColumn' 中的重复行
# duplicates = sorted_df['ModifiedColumn'].duplicated(keep=False)
# non_duplicates = ~sorted_df['ModifiedColumn'].duplicated(keep=False)
# # 筛选出包含重复项的行
# after_before_filtered_df = sorted_df[duplicates]
#
# MSR_data_cleaned = pd.read_csv(os.path.join(input_dir, 'MSR_data_cleaned.csv'))
#
#
# after_before_merged_df = pd.merge(MSR_data_cleaned, after_before_filtered_df, on='commit_id', how='inner')
# only_before_merged_df = pd.merge(MSR_data_cleaned, non_duplicates, on='commit_id', how='inner')