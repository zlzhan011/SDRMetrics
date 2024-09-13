




# setting 3:
# data path: return Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_add_after_to_before_work_well_20240528_v2')  3GPU
# Split_file = the original split file
# output_dir = 'storage_add_after_to_before_work_well_20240528_embedding_32_original_write_loss_filter_joern_error_c'
# is_use_joern_error_instance_V2    No
# Embedding size: 32
# data file:
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/val.csv \
#--test_data_file=../../data/MSR/test.csv \
#evaluate test: 100%|██████████| 1154/1154 [00:08<00:00, 132.78it/s]
#08/25/2024 09:28:01 - INFO - __main__ -   4 items missing
#08/25/2024 09:28:01 - INFO - __main__ -   ***** Eval results *****
#08/25/2024 09:28:01 - INFO - __main__ -     eval_f1 = 0.0
#08/25/2024 09:28:01 - INFO - __main__ -     eval_precision = 0.0
#08/25/2024 09:28:01 - INFO - __main__ -     eval_recall = 0.0
#08/25/2024 09:28:01 - INFO - __main__ -     eval_threshold = 0.5


python linevul_main_without_LineVul_train_filter_joern_error_c.py --model_name=1_linevul.bin \
--output_dir=./storage_add_after_to_before_work_well_20240528_embedding_32_original_write_loss_filter_joern_error_c \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_train \
--do_test \
--train_data_file=../../data/MSR/train.csv \
--eval_data_file=../../data/MSR/val.csv \
--test_data_file=../../data/MSR/test.csv \
--epochs 50 \
--block_size 512 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--seed 1




# setting 3:
# data path: return Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_add_after_to_before_work_well_20240528_v2')  3GPU
# Split_file = the original split file
# output_dir = 'storage_add_after_to_before_work_well_20240528_embedding_32_original_write_loss_filter_joern_missed_c'
# is_use_joern_error_instance_V2    No
# Embedding size: 32
# data file:
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/val.csv \
#--test_data_file=../../data/MSR/test.csv \
#evaluate test: 100%|██████████| 1154/1154 [00:08<00:00, 132.78it/s]
#08/25/2024 09:28:01 - INFO - __main__ -   4 items missing
#08/25/2024 09:28:01 - INFO - __main__ -   ***** Eval results *****
#08/25/2024 09:28:01 - INFO - __main__ -     eval_f1 = 0.0
#08/25/2024 09:28:01 - INFO - __main__ -     eval_precision = 0.0
#08/25/2024 09:28:01 - INFO - __main__ -     eval_recall = 0.0
#08/25/2024 09:28:01 - INFO - __main__ -     eval_threshold = 0.5


#python linevul_main_without_LineVul_train_filter_joern_error_c.py --model_name=1_linevul.bin \
#--output_dir=./storage_add_after_to_before_work_well_20240528_embedding_32_original_write_loss_filter_joern_missed_c \
#--model_type=roberta \
#--tokenizer_name=microsoft/codebert-base \
#--model_name_or_path=microsoft/codebert-base \
#--do_train \
#--do_test \
#--train_data_file=../../data/MSR/train.csv \
#--eval_data_file=../../data/MSR/val.csv \
#--test_data_file=../../data/MSR/test.csv \
#--epochs 50 \
#--block_size 512 \
#--train_batch_size 16 \
#--eval_batch_size 16 \
#--learning_rate 2e-5 \
#--max_grad_norm 1.0 \
#--seed 1



