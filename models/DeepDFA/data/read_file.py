import os
import pandas as pd



def read_original_label():
    file_dir ='/data/cs_lzhan011/vulnerability/DeepDFA_V2/DeepDFA/DDFA/storage/external'
    file_path = os.path.join(file_dir, 'MSR_data_cleaned.csv')
    df = pd.read_csv(file_path)
    df_new = df.set_index('Unnamed: 0')
    id_vul_dict = df_new['vul'].to_dict()
    return id_vul_dict

if __name__ == '__main__':
    read_original_label()

# if __name__ == '__main__':
#
#     file_dir ='/data/cs_lzhan011/vulnerability/DeepDFA_V2/DeepDFA/DDFA/storage/external'
#     file_path = os.path.join(file_dir, 'MSR_data_cleaned.csv')
#     df = pd.read_csv(file_path)
#     df_new = df.set_index('Unnamed: 0')
#     print(df.columns)
#     print("---")
    # new_df = []
    # for index, row in df.iterrows():
    #     if index % 3 == 0:
    #         print("333")
    #         row['vul'] = 3
    #     new_df.append(row)

    # new_df =pd.DataFrame(new_df)

    # new_df.to_csv(file_path)