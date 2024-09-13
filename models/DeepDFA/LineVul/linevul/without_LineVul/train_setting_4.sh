# setting 4:
# data path: return Path('/data2/cs_lzhan011/vulnerability/DeepDFA_V2/DDFA/storage_add_after_to_before_work_well_20240528')  3GPU
# Split_file = the add after split file
# output_dir = 'storage_add_after_to_before_work_well_20240528_embedding_32_add_after_write_loss'
# is_use_joern_error_instance_V2    No
# Embedding size: 32
# data file:
#--train_data_file=../../data/MSR/train_add_after.csv \
#--eval_data_file=../../data/MSR/val_add_after.csv \
#--test_data_file=../../data/MSR/test_add_after.csv \
# is_shuffle dataset:  Yes

python linevul_main_without_LineVul_train.py --model_name=1_linevul.bin \
--output_dir=./storage_add_after_to_before_work_well_20240528_embedding_32_add_after_write_loss \
--model_type=roberta \
--tokenizer_name=microsoft/codebert-base \
--model_name_or_path=microsoft/codebert-base \
--do_test \
--train_data_file=../../data/MSR/train_add_after.csv \
--eval_data_file=../../data/MSR/val_add_after.csv \
--test_data_file=../../data/MSR/test_add_after.csv \
--epochs 100 \
--block_size 512 \
--train_batch_size 16 \
--eval_batch_size 16 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--seed 1