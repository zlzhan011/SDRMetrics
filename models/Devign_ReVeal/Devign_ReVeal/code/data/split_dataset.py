

import os
import pandas as pd

# In order to split the Devign and Reveal dataset into train, test, valid dataset
# The reason to split the dataset is that: the samples in train， test， valid are different with LineVul








if __name__ == '__main__':
    c_root = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/MSR'
    splits_file = os.path.join(c_root, 'full_experiment_real_data_processed/vMSR/splits.csv')
    MSR_data_cleaned = os.path.join(c_root, 'MSR_data_cleaned.csv')
    splits_df = pd.read_csv(splits_file)
    MSR_data_cleaned_df = pd.read_csv(MSR_data_cleaned)

    split_dict = {}
    for index_n, row in splits_df.iterrows():
        split_dict[row['index']] = row['split']

    train_df = []
    test_df = []
    valid_df = []
    for index, row in MSR_data_cleaned_df.iterrows():
        if index in split_dict:
            split_category = split_dict[index]
            if split_category == 'train':
                train_df.append(row)
            if split_category == 'test':
                test_df.append(row)
            if split_category == 'valid':
                valid_df.append(row)


        print(index)

    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)
    valid_df = pd.DataFrame(valid_df)
    train_df.to_csv(os.path.join(c_root, 'split_dataset', 'train.csv'))
    test_df.to_csv(os.path.join(c_root, 'split_dataset', 'test.csv'))
    valid_df.to_csv(os.path.join(c_root, 'split_dataset', 'valid.csv'))




