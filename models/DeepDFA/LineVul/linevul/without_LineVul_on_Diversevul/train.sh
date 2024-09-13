
data_path=/data/cs_lzhan011/vulnerability/DeepDFA_V3/DeepDFA/LineVul/data/MSR


# Setting 0
# storage_path = Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_Diversevul_only_before')
# Split_file = the original split file
# output_dir = 'saved_models_binary_category_without_LineVul_add_after_to_before_embedding_size_256/'
# Embedding size: 256
#--train_data_file=../../data/Diversevul/10/train.csv \
#--eval_data_file=../../data/Diversevul/10/val.csv \
#--test_data_file=../../data/Diversevul/10/test.csv \
#08/23/2024 05:17:55 - INFO - __main__ -     eval_f1 = 0.0477
#08/23/2024 05:17:55 - INFO - __main__ -     eval_precision = 0.1007
#08/23/2024 05:17:55 - INFO - __main__ -     eval_recall = 0.0312
#08/23/2024 05:17:55 - INFO - __main__ -     eval_threshold = 0.5




#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_binary_category_without_LineVul_add_after_to_before_embedding_size_256/ \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_test \
#--train_data_file=../../data/Diversevul/10/train.csv \
#--eval_data_file=../../data/Diversevul/10/val.csv \
#--test_data_file=../../data/Diversevul/10/test.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--seed 1



# Setting 1
# storage_path = Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_Diversevul_only_before')
# Split_file = the original split file
# output_dir = 'saved_models_binary_category_without_LineVul_add_after_to_before_embedding_size_256/'
# Embedding size: 256
#--train_data_file=../../data/Diversevul/train.csv \
#--eval_data_file=../../data/Diversevul/val.csv \
#--test_data_file=../../data/Diversevul/test.csv \
#08/23/2024 05:34:44 - INFO - __main__ -   ***** Eval results *****
#08/23/2024 05:34:44 - INFO - __main__ -     eval_f1 = 0.0359
#08/23/2024 05:34:44 - INFO - __main__ -     eval_precision = 0.0485
#08/23/2024 05:34:44 - INFO - __main__ -     eval_recall = 0.0285
#08/23/2024 05:34:44 - INFO - __main__ -     eval_threshold = 0.5



#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_binary_category_without_LineVul_add_after_to_before_embedding_size_256/ \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_test \
#--train_data_file=../../data/Diversevul/train.csv \
#--eval_data_file=../../data/Diversevul/val.csv \
#--test_data_file=../../data/Diversevul/test.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--seed 1



# Setting 2
# storage_path = Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_Diversevul_add_after_to_before/')
# Split_file = the original split file
# output_dir = 'saved_models_binary_category_without_LineVul_add_after_to_before_embedding_size_256_truely_add_after_to_before/'
# Embedding size: 256
#--train_data_file=../../data/Diversevul/train.csv \
#--eval_data_file=../../data/Diversevul/val.csv \
#--test_data_file=../../data/Diversevul/test.csv \
#08/24/2024 16:26:23 - INFO - __main__ -   ***** Eval results *****
#08/24/2024 16:26:23 - INFO - __main__ -     eval_f1 = 0.0219
#08/24/2024 16:26:23 - INFO - __main__ -     eval_precision = 0.0811
#08/24/2024 16:26:23 - INFO - __main__ -     eval_recall = 0.0127
#08/24/2024 16:26:23 - INFO - __main__ -     eval_threshold = 0.5



#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_binary_category_without_LineVul_add_after_to_before_embedding_size_256_truely_add_after_to_before/ \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_test \
#--train_data_file=../../data/Diversevul/train.csv \
#--eval_data_file=../../data/Diversevul/val.csv \
#--test_data_file=../../data/Diversevul/test.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--seed 1




# Setting 3
# storage_path = Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_Diversevul_add_after_to_before/')
# Split_file = the original split file
# output_dir = 'saved_models_binary_category_without_LineVul_add_after_to_before_embedding_size_128_truely_add_after_to_before_Setting_3/'
# Embedding size: 128
#--train_data_file=../../data/Diversevul/10/train_add_after.csv \
#--eval_data_file=../../data/Diversevul/10/val_add_after.csv \
#--test_data_file=../../data/Diversevul/10/test_add_after.csv \
#08/25/2024 11:23:41 - INFO - __main__ -   ***** Eval results *****
#08/25/2024 11:23:41 - INFO - __main__ -     eval_f1 = 0.0588
#08/25/2024 11:23:41 - INFO - __main__ -     eval_precision = 0.0838
#08/25/2024 11:23:41 - INFO - __main__ -     eval_recall = 0.0453
#08/25/2024 11:23:41 - INFO - __main__ -     eval_threshold = 0.5



python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
--output_dir=./saved_models_binary_category_without_LineVul_add_after_to_before_embedding_size_128_truely_add_after_to_before_Setting_3/ \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_test \
--train_data_file=../../data/Diversevul/10/train_add_after.csv \
--eval_data_file=../../data/Diversevul/val_add_after.csv \
--test_data_file=../../data/Diversevul/test_add_after.csv \
--epochs 100 \
--block_size 512 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--seed 1


#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_binary_category_without_LineVul_add_after_to_before_embedding_size_128_truely_add_after_to_before_Setting_3_func_before/ \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train  \
#--do_test \
#--train_data_file=../../data/Diversevul/10/train_add_after.csv \
#--eval_data_file=../../data/Diversevul/val.csv \
#--test_data_file=../../data/Diversevul/test_add_after.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--seed 1




#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_multi_category_without_LineVul_with_downsample \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--do_test \
#--train_data_file=../../data/MSR/multi_category/train_cme_category_to_target.csv \
#--eval_data_file=../../data/MSR/multi_category/test_cme_category_to_target.csv \
#--test_data_file=../../data/MSR/multi_category/val_cme_category_to_target.csv \
#--epochs 100 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--evaluate_during_training \
#--seed 1



#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_multi_category_without_LineVul_No_downsample \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--do_test \
#--train_data_file=../../data/MSR/multi_category/train_cme_category_to_target.csv \
#--eval_data_file=../../data/MSR/multi_category/test_cme_category_to_target.csv \
#--test_data_file=../../data/MSR/multi_category/val_cme_category_to_target.csv \
#--epochs 10 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--evaluate_during_training \
#--seed 1







#python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
#--output_dir=./saved_models_binary_category_without_LineVul_No_downsample \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--do_test  \
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/test.csv  \
#--test_data_file=../../data/MSR/val.csv  \
#--epochs 25  \
#--block_size 512  \
#--train_batch_size 16 \
#--eval_batch_size 16  \
#--learning_rate 2e-5  \
#--max_grad_norm 1.0  \
#--evaluate_during_training \
#--seed 1