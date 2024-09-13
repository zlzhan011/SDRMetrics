import os
import pandas as pd

CME_Group = True
CME_Group_value_error = True
group_id_map = {  # contain merge target
    "Buffer overflow": 1,
    "Input validation error": 2,
    "Privilege escalation": 3,
    "Resource error": 4,
    "Value Errors": 5,
    "Other": 6,

}


# if CME_Group:
#     group_id_map = {
#         "Buffer overflow":0,
#         "Input validation error":1,
#         "Privilege escalation":2,
#         "Resource error":3,
#         "Value Errors":4,
#         "Other":5,
#     }
#     # if CME_Group_value_error:
#     #     group_id_map = {
#     #         "Buffer overflow": 0,
#     #         "Input validation error": 1,
#     #         "Privilege escalation": 2,
#     #         "Resource error": 3,
#     #         "Value Errors": 4,
#     #         "Other": 5,
#     #
#     #     }


def read_CME_grouping(file_path):
    CME_DF = pd.read_excel(file_path)
    cmeid_group = {}
    for index, row in CME_DF.iterrows():
        group = row['Group']
        if str(group) == 'nan':
            group = group_tmp
        else:
            group = group.split('(')[0].strip()

            group_tmp = group
        CWE_ID = row['CWE ID']
        cmeid_group[CWE_ID] = group
    cmeid_group['nan'] = 'Other'
    return cmeid_group


def cmeid_convert(row, cmeid_group):
    x = str(row['CWE ID']).strip()
    if x in cmeid_group:
        group = cmeid_group[x]
    else:
        group = 'Other'
    if group not in group_id_map:
        print("group:", group)
    group_id = group_id_map[group]

    if not CME_Group:  # no merge
        if str(row['target']) == '0':
            group_id = 0
        else:
            group_id = group_id + 1

    return str(group_id)


def add_vulnerability_type(cme_group_file_path, input_dir, input_file_name, output_dir):
    file_path = os.path.join(input_dir, input_file_name)
    cmeid_group = read_CME_grouping(cme_group_file_path)
    print("cmeid_group:", cmeid_group)
    df = pd.read_csv(file_path)
    if "cleaned" in input_file_name:
        df["target"] = df["vul"]
    # df = df.head(1000)
    df['cme_category'] = df.apply(cmeid_convert, axis=1, args=(cmeid_group,))
    df['cme_category_to_target'] = df.apply(lambda row: row['cme_category'] if row['target'] == 1 else '0', axis=1)
    df['vul'] = df['cme_category_to_target']
    # df = df.head(10000)
    df.to_csv(os.path.join(output_dir, input_file_name[:-4] + "_cme_category_to_target.csv"))
    print("df.cme_category_to_target 1:", set(list(df['cme_category_to_target'])))
    # print("df head:")
    # print(df.head())
    # if not CME_Group:
    #     df_remove_6 = df[df.cme_category != '6']
    #     print("df.target 1:", set(list(df_remove_6['target'])))
    #     df_remove_6.to_csv(os.path.join(output_dir, input_file_name[:-4]+"_merge_multi_type.csv"))
    # else:
    #     df_remove_5 = df[df.cme_category != '5']
    #
    #     if CME_Group_value_error:
    #         df_remove_4_5 = df_remove_5[df_remove_5.cme_category != '4']
    #         print("df.target 2:", set(list(df_remove_4_5['target'])))
    #         df_remove_4_5.to_csv(os.path.join(output_dir, input_file_name[:-4] + "_CME_value_error_multi_type.csv"))
    #     else:
    #         print("df.target 3:", set(list(df_remove_5['target'])))
    #         df_remove_5.to_csv(os.path.join(output_dir, input_file_name[:-4] + "_CME_multi_type.csv"))


def analysis(file_path):
    cmeid_group = read_CME_grouping(cme_group_file_path)
    print("cmeid_group:", cmeid_group)
    df = pd.read_csv(file_path)
    """
    target:   total: 150908; type 1: 8736; type 0: 142172
    CME group: 
                                 type 1: 29780;
                                 type 2: 20415
                                 type 3: 26230
                                 type 4: 26912
                                 type 5: 35457
                                 type 6: 12114

    target 1 map into CME group: type 0： 142172；  
                                 type 1: 2374;
                                 type 2: 1106;
                                 type 3: 1348   ;
                                 type 4: 1235   ;
                                 type 5: 2008   ;
                                 type 6: 665  ;


    """


def concat_ood(file_path_train, file_path_test, file_path_valid):
    cmeid_group = read_CME_grouping(cme_group_file_path)
    train_df = pd.read_csv(file_path_train)
    test_df = pd.read_csv(file_path_test)
    valid_df = pd.read_csv(file_path_valid)
    df = pd.concat([train_df, test_df, valid_df])

    df['target'] = df.apply(cmeid_convert, axis=1, args=(cmeid_group,))
    # print("df head:")
    # print(df.head())
    if not CME_Group:
        df = df[df.target == '6']
        print("df.target 1:", set(list(df['target'])))
        df.to_csv(file_path_train[:-9] + "_ttv_ood_type.csv")
    else:

        if CME_Group_value_error:
            df_5 = df[df.target == '5']
            print("df.target 2:", set(list(df['target'])))
            df_5.to_csv(file_path_train[:-9] + "_CME_value_error_ttv_ood_type_5.csv")
            df_4 = df[df.target == '4']
            df_4.to_csv(file_path_train[:-9] + "_CME_value_error_ttv_ood_type_4.csv")
        else:
            df = df[df.target == '5']
            print("df.target 3:", set(list(df['target'])))
            df.to_csv(file_path_train[:-9] + "_CME_ttv_ood_type.csv")


def check_res(file_path):
    df = pd.read_csv(file_path)
    print("df.target:", set(list(df['target'])))


if __name__ == '__main__':
    cme_group_file_path = 'CWE_Groupings.xlsx'
    input_dir_1 = '/data/cs_lzhan011/vulnerability/data-package/models/LineVul/code/data/big-vul_dataset/'
    input_dir_2 = "/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/MSR"
    input_dir_1 = '/data/cs_lzhan011/vulnerability/DeepDFA_V2/DeepDFA/LineVul/data/MSR/'
    output_dir = os.path.join(input_dir_1, 'multi_category')
    input_dir_1 = '/data/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/DDFA/storage/external'
    output_dir = input_dir_1
    # add_vulnerability_type(cme_group_file_path, input_dir_2, 'MSR_data_cleaned.csv', output_dir)
    #
    # exit()

    # add_vulnerability_type(cme_group_file_path, input_dir_1, 'train.csv', output_dir)
    # train_file_path = os.path.join(datasets_dir, 'train.csv')
    # add_vulnerability_type(cme_group_file_path, train_file_path)
    add_vulnerability_type(cme_group_file_path, input_dir_1, 'MSR_data_cleaned_SAMPLE.csv_bak2', output_dir)
    # valid_file_path = os.path.join(datasets_dir, 'valid.csv')
    # add_vulnerability_type(cme_group_file_path, valid_file_path)

    # add_vulnerability_type(cme_group_file_path, input_dir_1, 'test.csv', output_dir)

    # concat_ood(train_file_path, test_file_path, valid_file_path)
    # check_res(os.path.join(datasets_dir, 'train_merge_multi_type.csv'))


